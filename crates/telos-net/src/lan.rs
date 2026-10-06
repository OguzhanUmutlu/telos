//! LAN discovery subsystem conforming to Networking SKILL §15.
//!
//! Provides UDP multicast and broadcast beacon emission and listening:
//! - Multicast group: `239.255.118.120:47679`.
//! - Compact binary format: max 115 bytes (`"TLLAN"` magic, version 1, ports, SHA-256 cert fingerprint, player counts, MOTD).
//! - Stale server expiration after 5.0 seconds.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

use crate::error::{NetError, Result};

/// Magic bytes prefix identifying LAN beacons (`"TLLAN"`).
pub const LAN_MAGIC: &[u8; 5] = b"TLLAN";

/// Current LAN discovery beacon format version.
pub const LAN_VERSION: u8 = 1;

/// Default IPv4 multicast address for LAN server discovery.
pub const DEFAULT_LAN_MULTICAST_V4: Ipv4Addr = Ipv4Addr::new(239, 255, 118, 120);

/// Default UDP port for LAN server discovery beacons.
pub const DEFAULT_LAN_PORT: u16 = 47679;

/// Maximum allowable UTF-8 byte length for MOTD in LAN beacons.
pub const MAX_LAN_MOTD_BYTES: usize = 64;

/// Maximum allowable total byte length of a serialized LAN beacon.
pub const MAX_BEACON_BYTES: usize = 115;

/// Server beacon flags.
pub mod flags {
    /// Server requires authenticated login.
    pub const NEEDS_AUTH: u8 = 1 << 0;
    /// Server is password-protected.
    pub const HAS_PASSWORD: u8 = 1 << 1;
    /// Beacon is an immediate response to a client probe.
    pub const PROBE_RESPONSE: u8 = 1 << 2;
}

/// Structured payload of a LAN server discovery beacon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanBeacon {
    /// Bitflags describing server access requirements (`flags::*`).
    pub flags: u8,
    /// Wire protocol version (e.g. 1).
    pub protocol: u32,
    /// QUIC listener port for connecting to this server.
    pub quic_port: u16,
    /// SHA-256 fingerprint of the server's TLS certificate SPKI (pinned on connect).
    pub cert_spki_sha256: [u8; 32],
    /// Current number of online players.
    pub current_players: u16,
    /// Maximum allowed players.
    pub max_players: u16,
    /// Server game mode (0 = Survival, 1 = Creative, 2 = Adventure, 3 = Spectator).
    pub game_mode: u8,
    /// Server Message Of The Day (sanitized, <= 64 bytes).
    pub motd: String,
}

impl LanBeacon {
    /// Creates a new `LanBeacon` with default settings and specified port and MOTD.
    #[must_use]
    pub fn new(quic_port: u16, motd: impl Into<String>) -> Self {
        Self {
            flags: 0,
            protocol: 1,
            quic_port,
            cert_spki_sha256: [0u8; 32],
            current_players: 0,
            max_players: 64,
            game_mode: 0,
            motd: motd.into(),
        }
    }

    /// Encodes the beacon into a compact binary buffer (≤ 115 bytes).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let motd_bytes = sanitize_motd(&self.motd);
        let motd_len = motd_bytes.len().min(MAX_LAN_MOTD_BYTES) as u8;

        let mut buf = Vec::with_capacity(51 + motd_len as usize);
        buf.extend_from_slice(LAN_MAGIC);
        buf.push(LAN_VERSION);
        buf.push(self.flags);
        buf.extend_from_slice(&self.protocol.to_le_bytes());
        buf.extend_from_slice(&self.quic_port.to_le_bytes());
        buf.extend_from_slice(&self.cert_spki_sha256);
        buf.extend_from_slice(&self.current_players.to_le_bytes());
        buf.extend_from_slice(&self.max_players.to_le_bytes());
        buf.push(self.game_mode);
        buf.push(motd_len);
        buf.extend_from_slice(&motd_bytes.as_bytes()[..motd_len as usize]);

        buf
    }

    /// Decodes a beacon from raw wire bytes, validating magic and version.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 51 {
            return Err(NetError::InvalidPacket(
                "LAN beacon packet too short".into(),
            ));
        }

        if &bytes[0..5] != LAN_MAGIC {
            return Err(NetError::InvalidPacket("Invalid LAN beacon magic".into()));
        }

        if bytes[5] != LAN_VERSION {
            return Err(NetError::InvalidPacket(format!(
                "Unsupported LAN beacon version: {}",
                bytes[5]
            )));
        }

        let flags = bytes[6];
        let protocol = u32::from_le_bytes([bytes[7], bytes[8], bytes[9], bytes[10]]);
        let quic_port = u16::from_le_bytes([bytes[11], bytes[12]]);

        let mut cert_spki_sha256 = [0u8; 32];
        cert_spki_sha256.copy_from_slice(&bytes[13..45]);

        let current_players = u16::from_le_bytes([bytes[45], bytes[46]]);
        let max_players = u16::from_le_bytes([bytes[47], bytes[48]]);
        let game_mode = bytes[49];
        let motd_len = bytes[50] as usize;

        if bytes.len() < 51 + motd_len {
            return Err(NetError::InvalidPacket("Truncated LAN beacon MOTD".into()));
        }

        let raw_motd = std::str::from_utf8(&bytes[51..51 + motd_len])
            .map_err(|e| NetError::InvalidPacket(format!("Invalid UTF-8 in MOTD: {e}")))?;
        let motd = sanitize_motd(raw_motd);

        Ok(Self {
            flags,
            protocol,
            quic_port,
            cert_spki_sha256,
            current_players,
            max_players,
            game_mode,
            motd,
        })
    }
}

/// Sanitizes a string for display in MOTD by stripping control characters and trimming length.
#[must_use]
pub fn sanitize_motd(input: &str) -> String {
    let mut out = String::with_capacity(input.len().min(MAX_LAN_MOTD_BYTES));
    for c in input.chars() {
        if out.len() >= MAX_LAN_MOTD_BYTES {
            break;
        }
        if !c.is_control() {
            out.push(c);
        }
    }
    out
}

/// A discovered LAN server entry recorded by the listener.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredLanServer {
    /// Socket address where the UDP beacon arrived from.
    pub sender_addr: SocketAddr,
    /// Beacon details advertised by the server.
    pub beacon: LanBeacon,
    /// Timestamp when this server was last heard from.
    pub last_seen: Instant,
}

impl DiscoveredLanServer {
    /// Returns the target QUIC connect address by combining sender IP with server's advertised QUIC port.
    #[must_use]
    pub fn connect_addr(&self) -> SocketAddr {
        SocketAddr::new(self.sender_addr.ip(), self.beacon.quic_port)
    }
}

/// Listener scanning for LAN server discovery beacons over UDP multicast/broadcast.
pub struct LanDiscoveryListener {
    socket: UdpSocket,
    discovered: HashMap<SocketAddr, DiscoveredLanServer>,
    expire_duration: Duration,
}

impl LanDiscoveryListener {
    /// Binds a UDP listener on the default discovery port `47679` and joins multicast group.
    ///
    /// If port `47679` is already bound by another local process, binds to an ephemeral port
    /// so multiple local clients can still listen.
    pub fn new() -> std::io::Result<Self> {
        Self::with_port(DEFAULT_LAN_PORT)
    }

    /// Binds a UDP listener on the specified port.
    pub fn with_port(port: u16) -> std::io::Result<Self> {
        let socket = match UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port)) {
            Ok(s) => s,
            Err(_) => UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))?,
        };

        socket.set_nonblocking(true)?;
        let _ = socket.join_multicast_v4(&DEFAULT_LAN_MULTICAST_V4, &Ipv4Addr::UNSPECIFIED);

        Ok(Self {
            socket,
            discovered: HashMap::new(),
            expire_duration: Duration::from_secs(5),
        })
    }

    /// Creates a listener from a pre-configured UDP socket (useful for testing).
    #[must_use]
    pub fn from_socket(socket: UdpSocket) -> Self {
        let _ = socket.set_nonblocking(true);
        Self {
            socket,
            discovered: HashMap::new(),
            expire_duration: Duration::from_secs(5),
        }
    }

    /// Returns the local port bound by this listener.
    pub fn local_port(&self) -> std::io::Result<u16> {
        self.socket.local_addr().map(|addr| addr.port())
    }

    /// Polls for incoming UDP datagrams without blocking, updating discovered servers.
    pub fn poll(&mut self) {
        let mut buf = [0u8; 512];
        while let Ok((len, sender)) = self.socket.recv_from(&mut buf) {
            if let Ok(beacon) = LanBeacon::decode(&buf[..len]) {
                debug!(
                    sender = %sender,
                    server = beacon.motd.as_str(),
                    quic_port = beacon.quic_port,
                    "Discovered LAN server"
                );
                self.discovered.insert(
                    sender,
                    DiscoveredLanServer {
                        sender_addr: sender,
                        beacon,
                        last_seen: Instant::now(),
                    },
                );
            }
        }
    }

    /// Returns all currently active discovered servers, removing entries older than 5.0 seconds.
    pub fn active_servers(&mut self) -> Vec<DiscoveredLanServer> {
        self.poll();
        let now = Instant::now();
        self.discovered
            .retain(|_, s| now.duration_since(s.last_seen) < self.expire_duration);

        let mut list: Vec<DiscoveredLanServer> = self.discovered.values().cloned().collect();
        list.sort_by(|a, b| a.beacon.motd.cmp(&b.beacon.motd));
        list
    }
}

/// Periodic background emitter broadcasting LAN beacons over UDP multicast and broadcast.
pub struct LanBeaconEmitter {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LanBeaconEmitter {
    /// Spawns a background thread emitting the provided beacon every 1.5 seconds.
    pub fn spawn(beacon: LanBeacon, target_port: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))?;
        socket.set_broadcast(true)?;
        let _ = socket.set_multicast_loop_v4(true);
        let _ = socket.set_multicast_ttl_v4(1);

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let handle = std::thread::Builder::new()
            .name("telos-lan-beacon".into())
            .spawn(move || {
                let multicast_target =
                    SocketAddr::V4(SocketAddrV4::new(DEFAULT_LAN_MULTICAST_V4, target_port));
                let broadcast_target =
                    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::BROADCAST, target_port));
                let encoded = beacon.encode();

                while running_clone.load(Ordering::Relaxed) {
                    if let Err(e) = socket.send_to(&encoded, multicast_target) {
                        warn!("Failed to send LAN multicast beacon: {e}");
                    }
                    let _ = socket.send_to(&encoded, broadcast_target);

                    // Sleep for 1.5s with cooperative early-exit polling
                    for _ in 0..15 {
                        if !running_clone.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            })?;

        Ok(Self {
            running,
            handle: Some(handle),
        })
    }

    /// Stops the background beacon emitter thread.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for LanBeaconEmitter {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lan_beacon_roundtrip() {
        let beacon = LanBeacon {
            flags: flags::NEEDS_AUTH,
            protocol: 1,
            quic_port: 25565,
            cert_spki_sha256: [0x42; 32],
            current_players: 3,
            max_players: 8,
            game_mode: 0,
            motd: "Voxel World [LAN]".into(),
        };

        let encoded = beacon.encode();
        assert!(encoded.len() <= MAX_BEACON_BYTES);
        let decoded = LanBeacon::decode(&encoded).expect("Decoded valid beacon");

        assert_eq!(beacon, decoded);
    }

    #[test]
    fn test_lan_beacon_corrupt_rejection() {
        let mut encoded = LanBeacon {
            flags: 0,
            protocol: 1,
            quic_port: 12345,
            cert_spki_sha256: [0; 32],
            current_players: 1,
            max_players: 4,
            game_mode: 1,
            motd: "Test".into(),
        }
        .encode();

        // 1. Wrong magic
        encoded[0] = b'X';
        assert!(LanBeacon::decode(&encoded).is_err());

        // 2. Wrong version
        encoded[0] = b'V';
        encoded[5] = 99;
        assert!(LanBeacon::decode(&encoded).is_err());

        // 3. Truncated packet
        assert!(LanBeacon::decode(&encoded[..40]).is_err());
    }

    #[test]
    fn test_motd_sanitization() {
        let dirty = "Hello\x00\x07World\r\nTest! This is an extremely long MOTD string that definitely exceeds sixty-four bytes in total.";
        let clean = sanitize_motd(dirty);
        assert!(!clean.contains('\0'));
        assert!(!clean.contains('\r'));
        assert!(!clean.contains('\n'));
        assert!(clean.len() <= MAX_LAN_MOTD_BYTES);
        assert!(clean.starts_with("HelloWorldTest!"));
    }
}
