//! NAT-PMP (NAT Port Mapping Protocol, RFC 6886) client.
//!
//! Provides lightweight UDP-based port forwarding directly to consumer gateways:
//! - Default port: UDP `5351`.
//! - Opcode 0: Request external public IPv4 address.
//! - Opcode 1: Map UDP port with specified lease lifetime.
//! - Result codes: Success, Unsupported Version, Not Authorized, Network Failure, Out of Resources.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;
use tracing::{debug, info};

use crate::error::{NetError, Result};

/// Default UDP port for NAT-PMP gateways.
pub const NAT_PMP_PORT: u16 = 5351;

/// Common gateway IP addresses for consumer subnets.
pub const COMMON_GATEWAYS: &[Ipv4Addr] = &[
    Ipv4Addr::new(192, 168, 1, 1),
    Ipv4Addr::new(192, 168, 0, 1),
    Ipv4Addr::new(10, 0, 0, 1),
    Ipv4Addr::new(172, 16, 0, 1),
];

/// NAT-PMP response result codes per RFC 6886.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatPmpResultCode {
    /// Request succeeded.
    Success,
    /// Unsupported protocol version.
    UnsupportedVersion,
    /// Gateway refused request (not authorized / firewall policy).
    NotAuthorized,
    /// Network failure (router cannot access WAN or state corrupted).
    NetworkFailure,
    /// Router out of port mapping resources.
    OutOfResources,
    /// Unsupported opcode.
    UnsupportedOpcode,
    /// Other vendor-specific error code.
    Other(u16),
}

impl From<u16> for NatPmpResultCode {
    fn from(code: u16) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::UnsupportedVersion,
            2 => Self::NotAuthorized,
            3 => Self::NetworkFailure,
            4 => Self::OutOfResources,
            5 => Self::UnsupportedOpcode,
            c => Self::Other(c),
        }
    }
}

/// Mapped port result from NAT-PMP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NatPmpMapping {
    /// Internal host port.
    pub internal_port: u16,
    /// Mapped public external port on gateway router.
    pub external_port: u16,
    /// Granted lease lifetime in seconds.
    pub lifetime_secs: u32,
    /// Gateway router IPv4 address.
    pub gateway: Ipv4Addr,
}

/// NAT-PMP Client.
#[derive(Debug, Clone)]
pub struct NatPmpClient {
    gateway: Option<Ipv4Addr>,
}

impl Default for NatPmpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl NatPmpClient {
    /// Creates a new NAT-PMP client.
    #[must_use]
    pub fn new() -> Self {
        Self { gateway: None }
    }

    /// Explicitly sets the gateway IP address.
    pub fn set_gateway(&mut self, gateway: Ipv4Addr) {
        self.gateway = Some(gateway);
    }

    /// Queries the external IP address from the gateway.
    pub async fn get_external_ip(&mut self, timeout: Duration) -> Result<Ipv4Addr> {
        let gateways_to_try: Vec<Ipv4Addr> = if let Some(gw) = self.gateway {
            vec![gw]
        } else {
            COMMON_GATEWAYS.to_vec()
        };

        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| {
                NetError::Transport(format!("Failed to bind UDP socket for NAT-PMP: {e}"))
            })?;

        // Packet for opcode 0: [version=0, opcode=0]
        let req = [0u8, 0u8];
        let mut buf = [0u8; 32];

        for gw in gateways_to_try {
            let dest = SocketAddr::V4(SocketAddrV4::new(gw, NAT_PMP_PORT));
            let _ = socket.send_to(&req, dest).await;

            let recv_res = tokio::time::timeout(timeout, socket.recv_from(&mut buf)).await;
            if let Ok(Ok((len, src))) = recv_res
                && len >= 12
                && buf[0] == 0
                && buf[1] == 128
            {
                let result_code = u16::from_be_bytes([buf[2], buf[3]]);
                if result_code == 0 {
                    let ip = Ipv4Addr::new(buf[8], buf[9], buf[10], buf[11]);
                    debug!("NAT-PMP external IP from {src}: {ip}");
                    self.gateway = Some(gw);
                    return Ok(ip);
                }
            }
        }

        Err(NetError::Transport(
            "NAT-PMP gateway did not respond to external IP query".into(),
        ))
    }

    /// Maps a UDP port on the gateway router.
    pub async fn map_udp_port(
        &mut self,
        internal_port: u16,
        external_port_suggestion: u16,
        lifetime_secs: u32,
        timeout: Duration,
    ) -> Result<NatPmpMapping> {
        let gw = if let Some(gw) = self.gateway {
            gw
        } else {
            // Find working gateway first
            self.get_external_ip(timeout).await?;
            self.gateway
                .ok_or_else(|| NetError::Transport("NAT-PMP gateway undetermined".into()))?
        };

        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| {
                NetError::Transport(format!("Failed to bind UDP socket for NAT-PMP map: {e}"))
            })?;

        // Packet for opcode 1 (map UDP): 12 bytes
        // [version=0, opcode=1, reserved=0,0, int_port(2), ext_port(2), lifetime(4)]
        let mut req = [0u8; 12];
        req[0] = 0; // version
        req[1] = 1; // opcode 1 = UDP
        req[2..4].copy_from_slice(&[0, 0]); // reserved
        req[4..6].copy_from_slice(&internal_port.to_be_bytes());
        req[6..8].copy_from_slice(&external_port_suggestion.to_be_bytes());
        req[8..12].copy_from_slice(&lifetime_secs.to_be_bytes());

        let dest = SocketAddr::V4(SocketAddrV4::new(gw, NAT_PMP_PORT));
        socket.send_to(&req, dest).await.map_err(|e| {
            NetError::Transport(format!("Failed to send NAT-PMP map request to {dest}: {e}"))
        })?;

        let mut buf = [0u8; 32];
        let recv_res = tokio::time::timeout(timeout, socket.recv_from(&mut buf)).await;

        match recv_res {
            Ok(Ok((len, _src))) => {
                if len >= 16 && buf[0] == 0 && buf[1] == 129 {
                    let result_code = u16::from_be_bytes([buf[2], buf[3]]);
                    if result_code == 0 {
                        let int_p = u16::from_be_bytes([buf[8], buf[9]]);
                        let ext_p = u16::from_be_bytes([buf[10], buf[11]]);
                        let granted_life = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);

                        info!(
                            "NAT-PMP mapping granted via {gw}: UDP {ext_p} -> {int_p} ({granted_life}s)"
                        );
                        return Ok(NatPmpMapping {
                            internal_port: int_p,
                            external_port: ext_p,
                            lifetime_secs: granted_life,
                            gateway: gw,
                        });
                    }
                    return Err(NetError::Transport(format!(
                        "NAT-PMP returned error code {result_code} ({:?})",
                        NatPmpResultCode::from(result_code)
                    )));
                }
                Err(NetError::Transport(
                    "Invalid response received from NAT-PMP gateway".into(),
                ))
            }
            Ok(Err(e)) => Err(NetError::Transport(format!("NAT-PMP socket error: {e}"))),
            Err(_) => Err(NetError::Transport("NAT-PMP request timed out".into())),
        }
    }

    /// Deletes a port mapping by sending a request with lifetime = 0.
    pub async fn unmap_udp_port(&self, internal_port: u16, timeout: Duration) -> Result<()> {
        let Some(gw) = self.gateway else {
            return Ok(());
        };

        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| {
                NetError::Transport(format!("Failed to bind UDP socket for NAT-PMP unmap: {e}"))
            })?;

        let mut req = [0u8; 12];
        req[0] = 0;
        req[1] = 1;
        req[2..4].copy_from_slice(&[0, 0]);
        req[4..6].copy_from_slice(&internal_port.to_be_bytes());
        req[6..8].copy_from_slice(&[0, 0]);
        req[8..12].copy_from_slice(&0u32.to_be_bytes()); // Lifetime 0 = delete

        let dest = SocketAddr::V4(SocketAddrV4::new(gw, NAT_PMP_PORT));
        let _ = socket.send_to(&req, dest).await;
        let mut buf = [0u8; 32];
        let _ = tokio::time::timeout(timeout, socket.recv_from(&mut buf)).await;
        debug!("NAT-PMP unmap UDP port {internal_port} sent to {gw}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_pmp_result_codes() {
        assert_eq!(NatPmpResultCode::from(0), NatPmpResultCode::Success);
        assert_eq!(
            NatPmpResultCode::from(1),
            NatPmpResultCode::UnsupportedVersion
        );
        assert_eq!(NatPmpResultCode::from(2), NatPmpResultCode::NotAuthorized);
        assert_eq!(NatPmpResultCode::from(99), NatPmpResultCode::Other(99));
    }

    #[test]
    fn test_nat_pmp_packet_construction() {
        let mut req = [0u8; 12];
        req[0] = 0;
        req[1] = 1;
        req[4..6].copy_from_slice(&25565u16.to_be_bytes());
        req[6..8].copy_from_slice(&25565u16.to_be_bytes());
        req[8..12].copy_from_slice(&3600u32.to_be_bytes());

        assert_eq!(req[0], 0);
        assert_eq!(req[1], 1);
        assert_eq!(u16::from_be_bytes([req[4], req[5]]), 25565);
        assert_eq!(u16::from_be_bytes([req[6], req[7]]), 25565);
        assert_eq!(u32::from_be_bytes([req[8], req[9], req[10], req[11]]), 3600);
    }
}
