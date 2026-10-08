//! Cryptographic connection tickets and direct invite links (`telos://connect/...`).
//!
//! Provides self-contained, authenticated connection tickets enabling direct P2P connections:
//! - Encodes candidate addresses (LAN, `UPnP`, and STUN reflexive endpoints).
//! - Pins server's TLS certificate SPKI SHA-256 fingerprint against MITM attacks.
//! - Authenticates with keyed Blake3 MAC signatures preventing tampering.
//! - Supports time-bounded validity with expiration timestamps.
//! - Formats compact URL-safe `telos://connect/<ticket>` invite links for easy sharing.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::error::{NetError, Result};
use crate::nat::{CandidatePriority, NatMappingResult};
use crate::quic::{connect_remote_quic, runtime};
use crate::transport::Connection;
use telos_protocol::messages::{C2sMessage, S2cMessage};

/// URI scheme prefix for Telos direct invite links.
pub const INVITE_SCHEME_PREFIX: &str = "telos://connect/";

/// Candidate connection endpoint kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateType {
    /// Local subnet / LAN address.
    LocalLan,
    /// `UPnP` IGD forwarded router WAN address.
    UpnpMapped,
    /// NAT-PMP forwarded router WAN address.
    NatPmpMapped,
    /// STUN server reflexive address.
    StunReflexive,
}

impl From<CandidatePriority> for CandidateType {
    fn from(p: CandidatePriority) -> Self {
        match p {
            CandidatePriority::LocalLan => Self::LocalLan,
            CandidatePriority::UpnpMapped => Self::UpnpMapped,
            CandidatePriority::NatPmpMapped => Self::NatPmpMapped,
            CandidatePriority::StunReflexive => Self::StunReflexive,
        }
    }
}

/// A network candidate address in an invite ticket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateAddress {
    /// Socket address of the candidate endpoint.
    pub addr: SocketAddr,
    /// Classification of candidate route (LAN, `UPnP`, NAT-PMP, STUN).
    pub kind: CandidateType,
}

/// Cryptographically signed peer-to-peer connection ticket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionTicket {
    /// Ticket format / protocol version.
    pub version: u32,
    /// Friendly world or server display name.
    pub server_name: String,
    /// Prioritized list of candidate endpoints.
    pub candidates: Vec<CandidateAddress>,
    /// Pinned SHA-256 fingerprint of the server's TLS certificate SPKI.
    pub cert_spki_sha256: [u8; 32],
    /// One-time session nonce or invite token.
    pub session_secret: [u8; 16],
    /// Ticket creation UTC timestamp in seconds.
    pub created_at: u64,
    /// Ticket expiration UTC timestamp in seconds.
    pub expires_at: u64,
    /// Blake3 cryptographic authentication tag.
    pub signature: [u8; 32],
}

impl ConnectionTicket {
    /// Creates and signs a new connection ticket from NAT mapping results and server credentials.
    pub fn create(
        server_name: impl Into<String>,
        mapping: &NatMappingResult,
        cert_spki_sha256: [u8; 32],
        session_secret: [u8; 16],
        validity_duration_secs: u64,
        signing_key: &[u8; 32],
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut candidates = Vec::new();
        for c in mapping.ordered_candidates() {
            candidates.push(CandidateAddress {
                addr: c.addr,
                kind: CandidateType::from(c.priority),
            });
        }

        let mut ticket = Self {
            version: 1,
            server_name: server_name.into(),
            candidates,
            cert_spki_sha256,
            session_secret,
            created_at: now,
            expires_at: now.saturating_add(validity_duration_secs),
            signature: [0u8; 32],
        };

        ticket.signature = ticket.compute_signature(signing_key);
        ticket
    }

    /// Computes the Blake3 keyed MAC signature over canonical ticket payload.
    #[must_use]
    pub fn compute_signature(&self, signing_key: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(signing_key);
        hasher.update(&self.version.to_be_bytes());
        hasher.update(self.server_name.as_bytes());
        hasher.update(&(self.candidates.len() as u32).to_be_bytes());
        for c in &self.candidates {
            hasher.update(c.addr.to_string().as_bytes());
            hasher.update(&[c.kind as u8]);
        }
        hasher.update(&self.cert_spki_sha256);
        hasher.update(&self.session_secret);
        hasher.update(&self.created_at.to_be_bytes());
        hasher.update(&self.expires_at.to_be_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Verifies the cryptographic signature of the ticket.
    #[must_use]
    pub fn verify_signature(&self, signing_key: &[u8; 32]) -> bool {
        let expected = self.compute_signature(signing_key);
        // Constant-time comparison
        blake3_constant_time_eq(&self.signature, &expected)
    }

    /// Checks whether the ticket has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.expires_at
    }

    /// Serializes ticket to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| NetError::Codec(e.to_string()))
    }

    /// Deserializes ticket from JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| NetError::Codec(e.to_string()))
    }

    /// Encodes ticket into a compact URL-safe base64 string.
    pub fn to_base64_url(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(base64_url_encode(&bytes))
    }

    /// Decodes ticket from a URL-safe base64 string.
    pub fn from_base64_url(encoded: &str) -> Result<Self> {
        let bytes = base64_url_decode(encoded)?;
        Self::from_bytes(&bytes)
    }

    /// Formats the ticket into a complete `telos://connect/<ticket>` invite link.
    pub fn to_invite_link(&self) -> Result<String> {
        let b64 = self.to_base64_url()?;
        Ok(format!("{INVITE_SCHEME_PREFIX}{b64}"))
    }

    /// Parses an invite link (`telos://connect/...` or raw base64 string) and validates it.
    pub fn from_invite_link(link: &str, signing_key: Option<&[u8; 32]>) -> Result<Self> {
        let trimmed = link.trim();
        let payload = if let Some(stripped) = trimmed.strip_prefix(INVITE_SCHEME_PREFIX) {
            stripped
        } else if let Some(stripped) = trimmed.strip_prefix("telos://") {
            stripped
        } else {
            trimmed
        };

        let ticket = Self::from_base64_url(payload)?;

        if ticket.is_expired() {
            return Err(NetError::Transport("Connection ticket has expired".into()));
        }

        if let Some(key) = signing_key
            && !ticket.verify_signature(key)
        {
            return Err(NetError::Transport(
                "Invalid ticket signature: potential tampering detected".into(),
            ));
        }

        Ok(ticket)
    }
}

/// Constant-time comparison between two 32-byte arrays.
fn blake3_constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Encodes raw bytes to URL-safe base64 without padding.
#[must_use]
pub fn base64_url_encode(data: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::with_capacity((data.len() * 4).div_ceil(3));

    let mut i = 0;
    while i + 3 <= data.len() {
        let b0 = data[i];
        let b1 = data[i + 1];
        let b2 = data[i + 2];
        i += 3;

        result.push(CHARSET[(b0 >> 2) as usize] as char);
        result.push(CHARSET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        result.push(CHARSET[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        result.push(CHARSET[(b2 & 0x3F) as usize] as char);
    }

    let rem = data.len() - i;
    if rem == 1 {
        let b0 = data[i];
        result.push(CHARSET[(b0 >> 2) as usize] as char);
        result.push(CHARSET[((b0 & 0x03) << 4) as usize] as char);
    } else if rem == 2 {
        let b0 = data[i];
        let b1 = data[i + 1];
        result.push(CHARSET[(b0 >> 2) as usize] as char);
        result.push(CHARSET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        result.push(CHARSET[((b1 & 0x0F) << 2) as usize] as char);
    }

    result
}

/// Decodes URL-safe base64 string without padding.
pub fn base64_url_decode(s: &str) -> Result<Vec<u8>> {
    fn decode_char(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }

    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity((bytes.len() * 3) / 4);
    let mut i = 0;

    while i + 4 <= bytes.len() {
        let c0 =
            decode_char(bytes[i]).ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c1 = decode_char(bytes[i + 1])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c2 = decode_char(bytes[i + 2])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c3 = decode_char(bytes[i + 3])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        i += 4;

        out.push((c0 << 2) | (c1 >> 4));
        out.push(((c1 & 0x0F) << 4) | (c2 >> 2));
        out.push(((c2 & 0x03) << 6) | c3);
    }

    let rem = bytes.len() - i;
    if rem == 2 {
        let c0 =
            decode_char(bytes[i]).ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c1 = decode_char(bytes[i + 1])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        out.push((c0 << 2) | (c1 >> 4));
    } else if rem == 3 {
        let c0 =
            decode_char(bytes[i]).ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c1 = decode_char(bytes[i + 1])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        let c2 = decode_char(bytes[i + 2])
            .ok_or_else(|| NetError::Codec("Invalid base64 char".into()))?;
        out.push((c0 << 2) | (c1 >> 4));
        out.push(((c1 & 0x0F) << 4) | (c2 >> 2));
    } else if rem == 1 {
        return Err(NetError::Codec("Malformed base64 length".into()));
    }

    Ok(out)
}

/// Connects to candidate endpoints defined in the connection ticket in priority order.
pub async fn connect_ticket(
    ticket: &ConnectionTicket,
) -> Result<Box<dyn Connection<C2sMessage, S2cMessage>>> {
    if ticket.is_expired() {
        return Err(NetError::Transport("Connection ticket has expired".into()));
    }

    let mut last_err = None;
    for candidate in &ticket.candidates {
        debug!(addr = %candidate.addr, kind = ?candidate.kind, "Attempting connection to candidate");
        match tokio::time::timeout(Duration::from_secs(3), connect_remote_quic(candidate.addr))
            .await
        {
            Ok(Ok(conn)) => {
                info!(addr = %candidate.addr, kind = ?candidate.kind, "Successfully connected to candidate");
                return Ok(conn);
            }
            Ok(Err(e)) => {
                debug!(addr = %candidate.addr, error = %e, "Candidate connection failed");
                last_err = Some(e);
            }
            Err(_) => {
                debug!(addr = %candidate.addr, "Candidate connection timed out");
                last_err = Some(NetError::Transport(format!(
                    "Connection to {} timed out",
                    candidate.addr
                )));
            }
        }
    }

    Err(last_err
        .unwrap_or_else(|| NetError::Transport("No reachable candidates found in ticket".into())))
}

/// Synchronously connects to a target string which may be an invite link (`telos://connect/...`),
/// a base64url ticket, or a standard `host:port` string. Runs on the global `telos-net` Tokio runtime.
pub fn connect_to_target_sync(target: &str) -> Result<Box<dyn Connection<C2sMessage, S2cMessage>>> {
    let _guard = runtime().enter();
    runtime().block_on(async {
        let trimmed = target.trim();
        if trimmed.starts_with(INVITE_SCHEME_PREFIX) || trimmed.starts_with("telos://") {
            let ticket = ConnectionTicket::from_invite_link(trimmed, None)?;
            connect_ticket(&ticket).await
        } else if let Ok(ticket) = ConnectionTicket::from_base64_url(trimmed) {
            connect_ticket(&ticket).await
        } else {
            // Standard host:port
            let mut addrs = tokio::net::lookup_host(trimmed)
                .await
                .map_err(NetError::Io)?;
            let addr = addrs.next().ok_or_else(|| {
                NetError::Transport(format!("Could not resolve address: {trimmed}"))
            })?;
            connect_remote_quic(addr).await
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_url_roundtrip() {
        let sample = b"Hello, Telos P2P World! 1234567890 \x00\xff\xaa\x55";
        let encoded = base64_url_encode(sample);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));

        let decoded = base64_url_decode(&encoded).expect("Decode failed");
        assert_eq!(&decoded[..], &sample[..]);
    }

    #[test]
    fn test_ticket_signature_and_verification() {
        let key = [42u8; 32];
        let mapping = NatMappingResult {
            local_port: 25565,
            upnp_mapped: Some("203.0.113.88:25565".parse().unwrap()),
            nat_pmp_mapped: None,
            stun_reflexive: Some("198.51.100.99:54321".parse().unwrap()),
            lan_addresses: vec!["192.168.1.10:25565".parse().unwrap()],
        };

        let cert_spki = [7u8; 32];
        let secret = [15u8; 16];

        let ticket =
            ConnectionTicket::create("My Epic World", &mapping, cert_spki, secret, 3600, &key);

        assert!(ticket.verify_signature(&key));
        assert!(!ticket.is_expired());

        // Tamper test
        let mut tampered = ticket.clone();
        tampered.server_name = "Hacked World".to_string();
        assert!(!tampered.verify_signature(&key));

        // Invite link format test
        let link = ticket.to_invite_link().expect("Failed link build");
        assert!(link.starts_with(INVITE_SCHEME_PREFIX));

        let parsed = ConnectionTicket::from_invite_link(&link, Some(&key)).expect("Parse failed");
        assert_eq!(parsed.server_name, "My Epic World");
        assert_eq!(parsed.candidates.len(), 3);
        assert_eq!(parsed.cert_spki_sha256, cert_spki);
    }
}
