//! RFC 8489 / RFC 5389 STUN (Session Traversal Utilities for NAT) client.
//!
//! Provides reflexive public endpoint discovery over UDP:
//! - STUN Binding Request (`0x0001`).
//! - Magic Cookie: `0x2112A442`.
//! - Parses `XOR-MAPPED-ADDRESS` (`0x0020`) and `MAPPED-ADDRESS` (`0x0001`).
//! - Discovers public IP address and WAN port as seen by outside internet hosts.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;
use tracing::{debug, info};

use crate::error::{NetError, Result};

/// STUN Magic Cookie per RFC 5389 / 8489.
pub const STUN_MAGIC_COOKIE: u32 = 0x2112_A442;

/// STUN Binding Request message type.
pub const STUN_BINDING_REQUEST: u16 = 0x0001;
/// STUN Binding Success Response message type.
pub const STUN_BINDING_RESPONSE: u16 = 0x0101;

/// MAPPED-ADDRESS attribute type (RFC 3489).
pub const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
/// XOR-MAPPED-ADDRESS attribute type (RFC 5389 / 8489).
pub const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

/// Default public STUN servers for address discovery.
pub const DEFAULT_STUN_SERVERS: &[&str] = &[
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun2.l.google.com:19302",
];

/// Result from a successful STUN Binding query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StunEndpoint {
    /// Public reflexive IP and port.
    pub addr: SocketAddr,
    /// Local UDP socket bound for the query.
    pub local_port: u16,
}

/// STUN Client.
#[derive(Debug, Clone, Default)]
pub struct StunClient;

impl StunClient {
    /// Creates a new STUN client.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Queries the reflexive public address using a newly bound local UDP socket.
    pub async fn query_reflexive_address(
        &self,
        local_port: u16,
        timeout: Duration,
    ) -> Result<StunEndpoint> {
        let bind_addr = format!("0.0.0.0:{local_port}");
        let socket = tokio::net::UdpSocket::bind(&bind_addr).await.map_err(|e| {
            NetError::Transport(format!("Failed to bind local UDP socket for STUN: {e}"))
        })?;

        self.query_on_socket(&socket, timeout).await
    }

    /// Queries reflexive address using an existing already-bound UDP socket.
    pub async fn query_on_socket(
        &self,
        socket: &tokio::net::UdpSocket,
        timeout: Duration,
    ) -> Result<StunEndpoint> {
        let local_addr = socket
            .local_addr()
            .map_err(|e| NetError::Transport(format!("Failed to get local socket address: {e}")))?;

        for server_str in DEFAULT_STUN_SERVERS {
            debug!("Attempting STUN query to {server_str}");
            match tokio::net::lookup_host(server_str).await {
                Ok(mut addrs) => {
                    if let Some(target_addr) = addrs.next()
                        && let Ok(mapped) = self.query_server(socket, target_addr, timeout).await
                    {
                        info!("STUN discovered reflexive endpoint: {mapped} (via {server_str})");
                        return Ok(StunEndpoint {
                            addr: mapped,
                            local_port: local_addr.port(),
                        });
                    }
                }
                Err(e) => {
                    debug!("Failed DNS lookup for STUN server {server_str}: {e}");
                }
            }
        }

        Err(NetError::Transport(
            "All STUN servers failed or timed out".into(),
        ))
    }

    /// Performs a single STUN query against a target socket address.
    pub async fn query_server(
        &self,
        socket: &tokio::net::UdpSocket,
        server: SocketAddr,
        timeout: Duration,
    ) -> Result<SocketAddr> {
        let transaction_id = generate_transaction_id();
        let request_packet = build_stun_binding_request(transaction_id);

        socket.send_to(&request_packet, server).await.map_err(|e| {
            NetError::Transport(format!("Failed to send STUN request to {server}: {e}"))
        })?;

        let mut buf = [0u8; 1024];
        let deadline = tokio::time::Instant::now() + timeout;

        while tokio::time::Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await {
                Ok(Ok((len, from))) => {
                    if from == server
                        && let Some(mapped_addr) = parse_stun_response(&buf[..len], transaction_id)
                    {
                        return Ok(mapped_addr);
                    }
                }
                Ok(Err(e)) => return Err(NetError::Transport(format!("STUN socket error: {e}"))),
                Err(_) => break, // Timeout
            }
        }

        Err(NetError::Transport(format!(
            "STUN request to {server} timed out"
        )))
    }
}

/// Generates a random 96-bit (12-byte) transaction ID.
#[must_use]
pub fn generate_transaction_id() -> [u8; 12] {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let p1 = (now as u64).to_be_bytes();
    let p2 = ((now >> 64) as u32).to_be_bytes();
    let mut id = [0u8; 12];
    id[0..8].copy_from_slice(&p1);
    id[8..12].copy_from_slice(&p2);
    id
}

/// Constructs a 20-byte STUN Binding Request packet.
#[must_use]
pub fn build_stun_binding_request(transaction_id: [u8; 12]) -> [u8; 20] {
    let mut packet = [0u8; 20];
    packet[0..2].copy_from_slice(&STUN_BINDING_REQUEST.to_be_bytes()); // Type: 0x0001
    packet[2..4].copy_from_slice(&0u16.to_be_bytes()); // Length: 0
    packet[4..8].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes()); // Magic cookie: 0x2112A442
    packet[8..20].copy_from_slice(&transaction_id); // 96-bit transaction ID
    packet
}

/// Parses a STUN response packet, verifying magic cookie and transaction ID, and extracting mapped address.
#[must_use]
pub fn parse_stun_response(bytes: &[u8], expected_transaction_id: [u8; 12]) -> Option<SocketAddr> {
    if bytes.len() < 20 {
        return None;
    }

    let msg_type = u16::from_be_bytes([bytes[0], bytes[1]]);
    if msg_type != STUN_BINDING_RESPONSE {
        return None;
    }

    let msg_len = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    let magic_cookie = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if magic_cookie != STUN_MAGIC_COOKIE {
        return None;
    }

    if bytes[8..20] != expected_transaction_id {
        return None;
    }

    let attrs_slice = &bytes[20..std::cmp::min(bytes.len(), 20 + msg_len)];
    let mut offset = 0;

    let mut fallback_mapped: Option<SocketAddr> = None;

    while offset + 4 <= attrs_slice.len() {
        let attr_type = u16::from_be_bytes([attrs_slice[offset], attrs_slice[offset + 1]]);
        let attr_len =
            u16::from_be_bytes([attrs_slice[offset + 2], attrs_slice[offset + 3]]) as usize;
        offset += 4;

        if offset + attr_len > attrs_slice.len() {
            break;
        }

        let attr_val = &attrs_slice[offset..offset + attr_len];
        // Advance offset with 4-byte padding alignment per RFC 5389
        offset += (attr_len + 3) & !3;

        if attr_type == ATTR_XOR_MAPPED_ADDRESS && attr_len >= 8 {
            let family = attr_val[1];
            if family == 0x01 {
                // IPv4
                let x_port = u16::from_be_bytes([attr_val[2], attr_val[3]]);
                let port = x_port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);

                let x_ip = u32::from_be_bytes([attr_val[4], attr_val[5], attr_val[6], attr_val[7]]);
                let ip_u32 = x_ip ^ STUN_MAGIC_COOKIE;
                let ip = Ipv4Addr::from(ip_u32);

                return Some(SocketAddr::V4(SocketAddrV4::new(ip, port)));
            }
        } else if attr_type == ATTR_MAPPED_ADDRESS && attr_len >= 8 {
            let family = attr_val[1];
            if family == 0x01 {
                let port = u16::from_be_bytes([attr_val[2], attr_val[3]]);
                let ip = Ipv4Addr::new(attr_val[4], attr_val[5], attr_val[6], attr_val[7]);
                fallback_mapped = Some(SocketAddr::V4(SocketAddrV4::new(ip, port)));
            }
        }
    }

    fallback_mapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stun_packet_build_and_parse() {
        let tx_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let req = build_stun_binding_request(tx_id);
        assert_eq!(req.len(), 20);
        assert_eq!(u16::from_be_bytes([req[0], req[1]]), STUN_BINDING_REQUEST);
        assert_eq!(
            u32::from_be_bytes([req[4], req[5], req[6], req[7]]),
            STUN_MAGIC_COOKIE
        );
        assert_eq!(&req[8..20], &tx_id);

        // Synthesize a valid STUN Binding Success Response with XOR-MAPPED-ADDRESS
        let mut resp = Vec::new();
        resp.extend_from_slice(&STUN_BINDING_RESPONSE.to_be_bytes());
        resp.extend_from_slice(&12u16.to_be_bytes()); // Attribute len = 12 (type:2 + len:2 + val:8)
        resp.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        resp.extend_from_slice(&tx_id);

        // Attribute XOR-MAPPED-ADDRESS
        resp.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
        resp.extend_from_slice(&8u16.to_be_bytes());
        // Value: [0x00, 0x01 (IPv4), x_port(2), x_ip(4)]
        let target_port = 54321u16;
        let x_port = target_port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);
        let target_ip = Ipv4Addr::new(198, 51, 100, 42);
        let target_ip_u32 = u32::from(target_ip);
        let x_ip = target_ip_u32 ^ STUN_MAGIC_COOKIE;

        resp.push(0x00);
        resp.push(0x01); // IPv4
        resp.extend_from_slice(&x_port.to_be_bytes());
        resp.extend_from_slice(&x_ip.to_be_bytes());

        let parsed = parse_stun_response(&resp, tx_id).expect("Failed to parse STUN response");
        assert_eq!(
            parsed,
            SocketAddr::V4(SocketAddrV4::new(target_ip, target_port))
        );
    }
}
