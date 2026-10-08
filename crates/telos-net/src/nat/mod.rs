//! NAT traversal, `UPnP` IGD, NAT-PMP, STUN, and UDP hole punching.

pub mod hole_punch;
pub mod nat_pmp;
pub mod stun;
pub mod upnp;

pub use hole_punch::{HolePunchCoordinator, HolePunchPacket};
pub use nat_pmp::{NatPmpClient, NatPmpMapping};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
pub use stun::{StunClient, StunEndpoint};
use tracing::{debug, info};
pub use upnp::{UpnpClient, UpnpGateway};

/// Priority categories for connection candidate endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CandidatePriority {
    /// Local subnet / LAN direct connection (fastest, lowest latency).
    LocalLan = 100,
    /// Router port-forwarded via `UPnP` IGD.
    UpnpMapped = 80,
    /// Router port-forwarded via NAT-PMP.
    NatPmpMapped = 70,
    /// Discovered via STUN server reflexive address (may require hole punching).
    StunReflexive = 50,
}

/// A candidate socket address with priority classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NatCandidate {
    /// Candidate socket address.
    pub addr: SocketAddr,
    /// Priority level for route selection.
    pub priority: CandidatePriority,
}

/// Comprehensive result of automated NAT discovery and port forwarding.
#[derive(Debug, Clone, Default)]
pub struct NatMappingResult {
    /// Local socket port bound on this machine.
    pub local_port: u16,
    /// External WAN address mapped by `UPnP` IGD, if successful.
    pub upnp_mapped: Option<SocketAddr>,
    /// External WAN address mapped by NAT-PMP, if successful.
    pub nat_pmp_mapped: Option<SocketAddr>,
    /// External reflexive address reported by STUN.
    pub stun_reflexive: Option<SocketAddr>,
    /// Detected local LAN IPv4 addresses.
    pub lan_addresses: Vec<SocketAddr>,
}

impl NatMappingResult {
    /// Returns all available candidates ordered by highest priority first.
    #[must_use]
    pub fn ordered_candidates(&self) -> Vec<NatCandidate> {
        let mut candidates = Vec::new();

        for &lan in &self.lan_addresses {
            candidates.push(NatCandidate {
                addr: lan,
                priority: CandidatePriority::LocalLan,
            });
        }

        if let Some(upnp) = self.upnp_mapped {
            candidates.push(NatCandidate {
                addr: upnp,
                priority: CandidatePriority::UpnpMapped,
            });
        }

        if let Some(pmp) = self.nat_pmp_mapped
            && !candidates.iter().any(|c| c.addr == pmp)
        {
            candidates.push(NatCandidate {
                addr: pmp,
                priority: CandidatePriority::NatPmpMapped,
            });
        }

        if let Some(stun) = self.stun_reflexive
            && !candidates.iter().any(|c| c.addr == stun)
        {
            candidates.push(NatCandidate {
                addr: stun,
                priority: CandidatePriority::StunReflexive,
            });
        }

        // Sort descending by priority
        candidates.sort_by_key(|a| std::cmp::Reverse(a.priority));
        candidates
    }
}

/// Unified Port Mapper that orchestrates `UPnP`, NAT-PMP, and STUN concurrently.
#[derive(Debug)]
pub struct PortMapper {
    upnp: UpnpClient,
    nat_pmp: NatPmpClient,
    stun: StunClient,
    active_local_port: Option<u16>,
    active_upnp_port: Option<u16>,
    is_running: Arc<AtomicBool>,
}

impl Default for PortMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl PortMapper {
    /// Creates a new unified port mapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            upnp: UpnpClient::new(),
            nat_pmp: NatPmpClient::new(),
            stun: StunClient::new(),
            active_local_port: None,
            active_upnp_port: None,
            is_running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Automatically discovers and forwards the given local UDP port across available protocols.
    pub async fn map_port(
        &mut self,
        local_port: u16,
        lease_duration_secs: u32,
        timeout: Duration,
    ) -> NatMappingResult {
        let mut result = NatMappingResult {
            local_port,
            ..Default::default()
        };

        self.active_local_port = Some(local_port);

        // 1. Gather local network LAN IPs
        let local_ips = detect_local_lan_ips();
        for ip in &local_ips {
            result.lan_addresses.push(SocketAddr::new(*ip, local_port));
        }

        let primary_local_ip = local_ips
            .iter()
            .copied()
            .find(|ip| matches!(ip, IpAddr::V4(_)))
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));

        let local_v4: Ipv4Addr = match primary_local_ip {
            IpAddr::V4(v4) => v4,
            IpAddr::V6(_) => Ipv4Addr::LOCALHOST,
        };

        // 2. Concurrently attempt UPnP, NAT-PMP, and STUN
        let mut upnp_clone = self.upnp.clone();
        let upnp_fut = async {
            match upnp_clone.discover(timeout).await {
                Ok(_) => {
                    let ext_ip_res = upnp_clone.get_external_ip().await;
                    let map_res = upnp_clone
                        .add_port_mapping(
                            local_v4,
                            local_port,
                            local_port,
                            lease_duration_secs,
                            "Telos P2P Game Server",
                        )
                        .await;

                    if map_res.is_ok()
                        && let Ok(ext_ip) = ext_ip_res
                    {
                        return Some(SocketAddr::V4(SocketAddrV4::new(ext_ip, local_port)));
                    }
                    None
                }
                Err(e) => {
                    debug!("UPnP discovery failed: {e}");
                    None
                }
            }
        };

        let mut nat_pmp_clone = self.nat_pmp.clone();
        let nat_pmp_fut = async {
            match nat_pmp_clone.get_external_ip(timeout).await {
                Ok(ext_ip) => {
                    match nat_pmp_clone
                        .map_udp_port(local_port, local_port, lease_duration_secs, timeout)
                        .await
                    {
                        Ok(mapping) => Some(SocketAddr::V4(SocketAddrV4::new(
                            ext_ip,
                            mapping.external_port,
                        ))),
                        Err(e) => {
                            debug!("NAT-PMP port map failed: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    debug!("NAT-PMP discovery failed: {e}");
                    None
                }
            }
        };

        let stun_client = &self.stun;
        let stun_fut = async {
            match stun_client
                .query_reflexive_address(local_port, timeout)
                .await
            {
                Ok(endpoint) => Some(endpoint.addr),
                Err(e) => {
                    debug!("STUN discovery failed: {e}");
                    None
                }
            }
        };

        let (upnp_res, nat_pmp_res, stun_res) = tokio::join!(upnp_fut, nat_pmp_fut, stun_fut);

        result.upnp_mapped = upnp_res;
        if result.upnp_mapped.is_some() {
            self.active_upnp_port = Some(local_port);
            self.upnp = upnp_clone;
        }

        result.nat_pmp_mapped = nat_pmp_res;
        if result.nat_pmp_mapped.is_some() {
            self.nat_pmp = nat_pmp_clone;
        }

        result.stun_reflexive = stun_res;

        info!(
            "PortMapper completed for UDP {}: UPnP={:?}, NAT-PMP={:?}, STUN={:?}",
            local_port, result.upnp_mapped, result.nat_pmp_mapped, result.stun_reflexive
        );

        result
    }

    /// Teardown: removes active port mappings from the router.
    pub async fn unmap(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(port) = self.active_upnp_port.take() {
            let _ = self.upnp.delete_port_mapping(port).await;
        }

        if let Some(port) = self.active_local_port.take() {
            let _ = self
                .nat_pmp
                .unmap_udp_port(port, Duration::from_millis(500))
                .await;
        }

        debug!("PortMapper cleanly unmapped active ports");
    }
}

/// Helper to detect private LAN IP addresses for the local host.
#[must_use]
pub fn detect_local_lan_ips() -> Vec<IpAddr> {
    let mut ips = Vec::new();

    // Use a UDP socket connect probe trick (RFC 3493):
    // Connecting a UDP socket to an external address queries the OS routing table without sending traffic.
    let probe_targets = ["8.8.8.8:80", "1.1.1.1:80", "208.67.222.222:80"];

    for target in probe_targets {
        if let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0")
            && socket.connect(target).is_ok()
            && let Ok(local_addr) = socket.local_addr()
        {
            let ip = local_addr.ip();
            if !ip.is_loopback() && !ips.contains(&ip) {
                ips.push(ip);
            }
        }
    }

    if ips.is_empty() {
        ips.push(IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    ips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_candidates_priority() {
        let res = NatMappingResult {
            local_port: 25565,
            upnp_mapped: Some("203.0.113.10:25565".parse().unwrap()),
            nat_pmp_mapped: Some("203.0.113.10:25565".parse().unwrap()),
            stun_reflexive: Some("198.51.100.22:54321".parse().unwrap()),
            lan_addresses: vec!["192.168.1.50:25565".parse().unwrap()],
        };

        let ordered = res.ordered_candidates();
        assert!(!ordered.is_empty());
        assert_eq!(ordered[0].priority, CandidatePriority::LocalLan);
        assert_eq!(ordered[0].addr.to_string(), "192.168.1.50:25565");

        // UPnP should rank before STUN
        let upnp_idx = ordered
            .iter()
            .position(|c| c.priority == CandidatePriority::UpnpMapped)
            .unwrap();
        let stun_idx = ordered
            .iter()
            .position(|c| c.priority == CandidatePriority::StunReflexive)
            .unwrap();
        assert!(upnp_idx < stun_idx);
    }
}
