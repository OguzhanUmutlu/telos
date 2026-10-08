//! UDP hole-punching protocol and coordinator for direct peer-to-peer connections.
//!
//! When connecting directly behind NAT firewalls without dedicated port forwarding:
//! - Both peers exchange candidate socket addresses (LAN, `UPnP`, and STUN reflexive).
//! - Both peers send synchronized UDP probe packets (`"TLPUNCH"`) to each other's candidates.
//! - Outbound UDP packets create stateful NAT translation pinholes in home router firewalls.
//! - When bi-directional probes and ACKs are exchanged, the direct route is established.

use std::net::SocketAddr;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::error::{NetError, Result};

/// Magic bytes for hole-punching packets (`"TLPUNCH"`).
pub const HOLE_PUNCH_MAGIC: &[u8; 7] = b"TLPUNCH";

/// Outbound hole punching probe packet.
pub const PUNCH_TYPE_PROBE: u8 = 0x01;
/// Inbound acknowledgment packet confirming open hole.
pub const PUNCH_TYPE_ACK: u8 = 0x02;

/// Structured hole-punching packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HolePunchPacket {
    /// Packet message type (`PUNCH_TYPE_PROBE` or `PUNCH_TYPE_ACK`).
    pub msg_type: u8,
    /// 16-byte shared session rendezvous token.
    pub session_id: [u8; 16],
    /// Peer net ID.
    pub sender_id: u64,
    /// Sequential packet nonce.
    pub nonce: u32,
}

impl HolePunchPacket {
    /// Serializes packet to a 28-byte buffer.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 28] {
        let mut buf = [0u8; 28];
        buf[0..7].copy_from_slice(HOLE_PUNCH_MAGIC);
        buf[7] = self.msg_type;
        buf[8..24].copy_from_slice(&self.session_id);
        buf[24..28].copy_from_slice(&self.nonce.to_be_bytes());
        buf
    }

    /// Parses a 28-byte packet from wire buffer.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 28 || &bytes[0..7] != HOLE_PUNCH_MAGIC {
            return None;
        }

        let msg_type = bytes[7];
        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&bytes[8..24]);
        let nonce = u32::from_be_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);

        Some(Self {
            msg_type,
            session_id,
            sender_id: 0,
            nonce,
        })
    }
}

/// UDP Hole-Punching Coordinator.
#[derive(Debug)]
pub struct HolePunchCoordinator {
    session_id: [u8; 16],
    local_peer_id: u64,
}

impl HolePunchCoordinator {
    /// Creates a new hole punch coordinator with a designated session secret.
    #[must_use]
    pub fn new(session_id: [u8; 16], local_peer_id: u64) -> Self {
        Self {
            session_id,
            local_peer_id,
        }
    }

    /// Attempts hole punching against a list of remote candidate addresses.
    pub async fn punch_hole(
        &self,
        socket: &tokio::net::UdpSocket,
        candidates: &[SocketAddr],
        timeout: Duration,
    ) -> Result<SocketAddr> {
        if candidates.is_empty() {
            return Err(NetError::Transport("No candidates to punch hole to".into()));
        }

        info!(
            "Starting UDP hole punching across {} candidates for session {:?}",
            candidates.len(),
            &self.session_id[0..4]
        );

        let probe_packet = HolePunchPacket {
            msg_type: PUNCH_TYPE_PROBE,
            session_id: self.session_id,
            sender_id: self.local_peer_id,
            nonce: 1,
        }
        .to_bytes();

        let ack_packet = HolePunchPacket {
            msg_type: PUNCH_TYPE_ACK,
            session_id: self.session_id,
            sender_id: self.local_peer_id,
            nonce: 2,
        }
        .to_bytes();

        let mut recv_buf = [0u8; 128];
        let deadline = tokio::time::Instant::now() + timeout;
        let mut burst_interval = tokio::time::interval(Duration::from_millis(50));

        let mut confirmed_peer: Option<SocketAddr> = None;

        while tokio::time::Instant::now() < deadline {
            tokio::select! {
                _ = burst_interval.tick() => {
                    // Send probe burst to all candidates
                    for &target in candidates {
                        let _ = socket.send_to(&probe_packet, target).await;
                    }
                }
                recv_res = socket.recv_from(&mut recv_buf) => {
                    match recv_res {
                        Ok((len, from)) => {
                            if let Some(pkt) = HolePunchPacket::from_bytes(&recv_buf[..len])
                                && pkt.session_id == self.session_id
                            {
                                debug!("Received hole punch {:?} from {from}", pkt.msg_type);
                                if pkt.msg_type == PUNCH_TYPE_PROBE {
                                    // Send immediate ACK to confirm bi-directional hole
                                    let _ = socket.send_to(&ack_packet, from).await;
                                    confirmed_peer = Some(from);
                                } else if pkt.msg_type == PUNCH_TYPE_ACK {
                                    info!("Hole punch confirmed with {from} via ACK");
                                    return Ok(from);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Socket receive error during hole punching: {e}");
                        }
                    }
                }
            }

            if let Some(peer) = confirmed_peer {
                // If we got a probe, send another burst of ACKs and return
                for _ in 0..3 {
                    let _ = socket.send_to(&ack_packet, peer).await;
                }
                return Ok(peer);
            }
        }

        Err(NetError::Transport(
            "UDP hole punching timed out without establishing bidirectional connection".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hole_punch_packet_roundtrip() {
        let pkt = HolePunchPacket {
            msg_type: PUNCH_TYPE_PROBE,
            session_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            sender_id: 42,
            nonce: 999,
        };

        let bytes = pkt.to_bytes();
        assert_eq!(&bytes[0..7], HOLE_PUNCH_MAGIC);
        assert_eq!(bytes[7], PUNCH_TYPE_PROBE);

        let decoded = HolePunchPacket::from_bytes(&bytes).expect("Failed to decode");
        assert_eq!(decoded.msg_type, PUNCH_TYPE_PROBE);
        assert_eq!(decoded.session_id, pkt.session_id);
        assert_eq!(decoded.nonce, pkt.nonce);
    }
}
