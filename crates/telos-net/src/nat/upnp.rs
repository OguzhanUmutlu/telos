//! `UPnP` IGD (Internet Gateway Device) port mapping client.
//!
//! Provides automatic WAN port forwarding via SSDP discovery and SOAP action requests:
//! - Multicast SSDP M-SEARCH discovery (`239.255.255.250:1900`).
//! - Parses XML device description for `WANIPConnection` or `WANPPPConnection` control URLs.
//! - SOAP `AddPortMapping` for opening incoming UDP ports.
//! - SOAP `DeletePortMapping` for clean teardown.
//! - SOAP `GetExternalIPAddress` for discovering public IPv4.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::error::{NetError, Result};

/// Default SSDP multicast address.
pub const SSDP_MULTICAST: Ipv4Addr = Ipv4Addr::new(239, 255, 255, 250);
/// Default SSDP port.
pub const SSDP_PORT: u16 = 1900;

/// Parsed `UPnP` gateway service information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpnpGateway {
    /// Control URL for SOAP requests.
    pub control_url: String,
    /// Service type (e.g. `urn:schemas-upnp-org:service:WANIPConnection:1`).
    pub service_type: String,
    /// Root URL base of the gateway router.
    pub url_base: String,
}

/// `UPnP` IGD Client.
#[derive(Debug, Clone)]
pub struct UpnpClient {
    gateway: Option<UpnpGateway>,
}

impl Default for UpnpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl UpnpClient {
    /// Creates a new `UPnP` IGD client.
    #[must_use]
    pub fn new() -> Self {
        Self { gateway: None }
    }

    /// Discovers an Internet Gateway Device on the local network via SSDP.
    pub async fn discover(&mut self, timeout: Duration) -> Result<UpnpGateway> {
        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| {
                NetError::Transport(format!("Failed to bind local UDP socket for SSDP: {e}"))
            })?;

        socket.set_broadcast(true).map_err(|e| {
            NetError::Transport(format!("Failed to enable UDP broadcast for SSDP: {e}"))
        })?;

        let search_targets = [
            "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
            "urn:schemas-upnp-org:service:WANIPConnection:1",
            "urn:schemas-upnp-org:service:WANPPPConnection:1",
        ];

        let dest = SocketAddr::V4(SocketAddrV4::new(SSDP_MULTICAST, SSDP_PORT));

        for target in search_targets {
            let msg = format!(
                "M-SEARCH * HTTP/1.1\r\n\
                 HOST: 239.255.255.250:1900\r\n\
                 MAN: \"ssdp:discover\"\r\n\
                 MX: 2\r\n\
                 ST: {target}\r\n\
                 \r\n"
            );
            let _ = socket.send_to(msg.as_bytes(), dest).await;
        }

        let mut buf = [0u8; 2048];
        let deadline = tokio::time::Instant::now() + timeout;

        while tokio::time::Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let recv_res = tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await;

            match recv_res {
                Ok(Ok((len, _addr))) => {
                    if let Ok(response) = std::str::from_utf8(&buf[..len])
                        && let Some(location) = extract_header_value(response, "LOCATION")
                    {
                        debug!("Found UPnP location: {location}");
                        if let Ok(gw) = fetch_gateway_info(&location).await {
                            info!("Discovered UPnP IGD gateway at {}", gw.control_url);
                            self.gateway = Some(gw.clone());
                            return Ok(gw);
                        }
                    }
                }
                Ok(Err(e)) => {
                    warn!("SSDP receive error: {e}");
                    break;
                }
                Err(_) => break, // Timeout
            }
        }

        Err(NetError::Transport(
            "No UPnP Internet Gateway Device discovered on network".into(),
        ))
    }

    /// Queries the router's external public IP address.
    pub async fn get_external_ip(&self) -> Result<Ipv4Addr> {
        let gw = self
            .gateway
            .as_ref()
            .ok_or_else(|| NetError::Transport("UPnP gateway not discovered yet".into()))?;

        let action = "GetExternalIPAddress";
        let body = format!("<u:{action} xmlns:u=\"{}\"></u:{action}>", gw.service_type);

        let resp = send_soap_request(gw, action, &body).await?;
        if let Some(ip_str) = extract_xml_tag_content(&resp, "NewExternalIPAddress") {
            let ip = ip_str.trim().parse::<Ipv4Addr>().map_err(|e| {
                NetError::Transport(format!("Failed to parse external IP '{ip_str}': {e}"))
            })?;
            Ok(ip)
        } else {
            Err(NetError::Transport(
                "UPnP GetExternalIPAddress response missing IP tag".into(),
            ))
        }
    }

    /// Adds a port mapping on the router for incoming UDP traffic.
    pub async fn add_port_mapping(
        &self,
        internal_ip: Ipv4Addr,
        internal_port: u16,
        external_port: u16,
        lease_duration_secs: u32,
        description: &str,
    ) -> Result<()> {
        let gw = self
            .gateway
            .as_ref()
            .ok_or_else(|| NetError::Transport("UPnP gateway not discovered yet".into()))?;

        let action = "AddPortMapping";
        let body = format!(
            "<u:{action} xmlns:u=\"{}\">\
             <NewRemoteHost></NewRemoteHost>\
             <NewExternalPort>{external_port}</NewExternalPort>\
             <NewProtocol>UDP</NewProtocol>\
             <NewInternalPort>{internal_port}</NewInternalPort>\
             <NewInternalClient>{internal_ip}</NewInternalClient>\
             <NewEnabled>1</NewEnabled>\
             <NewPortMappingDescription>{description}</NewPortMappingDescription>\
             <NewLeaseDuration>{lease_duration_secs}</NewLeaseDuration>\
             </u:{action}>",
            gw.service_type
        );

        let _ = send_soap_request(gw, action, &body).await?;
        info!(
            "UPnP port mapping added: WAN UDP {external_port} -> {internal_ip}:{internal_port} (lease: {lease_duration_secs}s)"
        );
        Ok(())
    }

    /// Deletes a previously added UDP port mapping.
    pub async fn delete_port_mapping(&self, external_port: u16) -> Result<()> {
        let gw = self
            .gateway
            .as_ref()
            .ok_or_else(|| NetError::Transport("UPnP gateway not discovered yet".into()))?;

        let action = "DeletePortMapping";
        let body = format!(
            "<u:{action} xmlns:u=\"{}\">\
             <NewRemoteHost></NewRemoteHost>\
             <NewExternalPort>{external_port}</NewExternalPort>\
             <NewProtocol>UDP</NewProtocol>\
             </u:{action}>",
            gw.service_type
        );

        let _ = send_soap_request(gw, action, &body).await?;
        info!("UPnP port mapping deleted: WAN UDP {external_port}");
        Ok(())
    }
}

/// Helper to parse HTTP header values case-insensitively.
#[must_use]
pub fn extract_header_value(raw: &str, header: &str) -> Option<String> {
    for line in raw.lines() {
        if let Some((k, v)) = line.split_once(':')
            && k.trim().eq_ignore_ascii_case(header)
        {
            return Some(v.trim().to_string());
        }
    }
    None
}

/// Helper to extract simple XML tag content without external XML parser dependency.
#[must_use]
pub fn extract_xml_tag_content(xml: &str, tag_name: &str) -> Option<String> {
    let open_tag = format!("<{tag_name}>");
    let close_tag = format!("</{tag_name}>");

    if let Some(start_idx) = xml.find(&open_tag) {
        let content_start = start_idx + open_tag.len();
        if let Some(end_idx) = xml[content_start..].find(&close_tag) {
            return Some(xml[content_start..content_start + end_idx].to_string());
        }
    }

    // Try with namespace prefix (e.g. `<u:NewExternalIPAddress>`)
    let colon_open = format!(":{tag_name}>");
    let colon_close = format!(":{tag_name}>");
    if let Some(open_idx) = xml.find(&colon_open) {
        let after_open = open_idx + colon_open.len();
        if let Some(close_idx) = xml[after_open..].find(&colon_close) {
            // Find preceding '</' before the colon
            if let Some(tag_end) = xml[..after_open + close_idx].rfind("</") {
                return Some(xml[after_open..tag_end].to_string());
            }
        }
    }

    None
}

/// Fetches device XML description and extracts WAN IP connection control URL.
async fn fetch_gateway_info(location_url: &str) -> Result<UpnpGateway> {
    let (host, port, path) = parse_http_url(location_url)?;
    let addr = format!("{host}:{port}");

    let mut stream = tokio::net::TcpStream::connect(&addr).await.map_err(|e| {
        NetError::Transport(format!("Failed to connect to UPnP location at {addr}: {e}"))
    })?;

    let req = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Connection: close\r\n\
         \r\n"
    );

    stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| NetError::Transport(format!("Failed to send GET to UPnP location: {e}")))?;

    let mut xml = String::new();
    stream
        .read_to_string(&mut xml)
        .await
        .map_err(|e| NetError::Transport(format!("Failed to read UPnP description XML: {e}")))?;

    // Look for WANIPConnection or WANPPPConnection service
    let service_types = [
        "urn:schemas-upnp-org:service:WANIPConnection:1",
        "urn:schemas-upnp-org:service:WANPPPConnection:1",
    ];

    for s_type in service_types {
        if let Some(idx) = xml.find(s_type) {
            let slice = &xml[idx..];
            if let Some(ctrl_url) = extract_xml_tag_content(slice, "controlURL") {
                let url_base = format!("http://{host}:{port}");
                let full_ctrl_url = if ctrl_url.starts_with('/') {
                    format!("{url_base}{ctrl_url}")
                } else if ctrl_url.starts_with("http") {
                    ctrl_url
                } else {
                    format!("{url_base}/{ctrl_url}")
                };

                return Ok(UpnpGateway {
                    control_url: full_ctrl_url,
                    service_type: s_type.to_string(),
                    url_base,
                });
            }
        }
    }

    Err(NetError::Transport(
        "No supported WANIPConnection or WANPPPConnection service found in UPnP XML".into(),
    ))
}

/// Sends a SOAP POST request to the `UPnP` control URL.
async fn send_soap_request(gw: &UpnpGateway, action: &str, soap_body: &str) -> Result<String> {
    let (host, port, path) = parse_http_url(&gw.control_url)?;
    let addr = format!("{host}:{port}");

    let mut stream = tokio::net::TcpStream::connect(&addr).await.map_err(|e| {
        NetError::Transport(format!(
            "Failed to connect to UPnP control URL at {addr}: {e}"
        ))
    })?;

    let envelope = format!(
        "<?xml version=\"1.0\"?>\r\n\
         <s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" \
                     s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">\r\n\
           <s:Body>\r\n\
             {soap_body}\r\n\
           </s:Body>\r\n\
         </s:Envelope>"
    );

    let soap_action = format!("\"{}#{action}\"", gw.service_type);
    let req = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Content-Type: text/xml; charset=\"utf-8\"\r\n\
         SOAPAction: {soap_action}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {envelope}",
        envelope.len()
    );

    stream.write_all(req.as_bytes()).await.map_err(|e| {
        NetError::Transport(format!("Failed to write SOAP request for {action}: {e}"))
    })?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp).await.map_err(|e| {
        NetError::Transport(format!("Failed to read SOAP response for {action}: {e}"))
    })?;

    if resp.contains("200 OK") {
        Ok(resp)
    } else {
        warn!("UPnP SOAP action {action} response: {resp}");
        Err(NetError::Transport(format!(
            "UPnP SOAP request for {action} returned non-200 status"
        )))
    }
}

/// Helper to parse a simple HTTP URL into (host, port, path).
fn parse_http_url(url: &str) -> Result<(String, u16, String)> {
    let stripped = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = stripped.split_once('/').unwrap_or((stripped, ""));
    let path = format!("/{path}");

    if let Some((h, p)) = host_port.split_once(':') {
        let port: u16 = p
            .parse()
            .map_err(|e| NetError::Transport(format!("Invalid port in URL '{url}': {e}")))?;
        Ok((h.to_string(), port, path))
    } else {
        Ok((host_port.to_string(), 80, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_header_value() {
        let raw = "HTTP/1.1 200 OK\r\nLocation: http://192.168.1.1:49152/rootdesc.xml\r\nServer: RouterOS\r\n";
        let loc = extract_header_value(raw, "location");
        assert_eq!(
            loc.as_deref(),
            Some("http://192.168.1.1:49152/rootdesc.xml")
        );
    }

    #[test]
    fn test_extract_xml_tag_content() {
        let xml = "<root><serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType><controlURL>/upnp/control/wanip</controlURL></root>";
        let ctrl = extract_xml_tag_content(xml, "controlURL");
        assert_eq!(ctrl.as_deref(), Some("/upnp/control/wanip"));

        let soap_resp = "<s:Envelope><s:Body><u:GetExternalIPAddressResponse xmlns:u=\"urn:schemas-upnp-org:service:WANIPConnection:1\"><NewExternalIPAddress>203.0.113.45</NewExternalIPAddress></u:GetExternalIPAddressResponse></s:Body></s:Envelope>";
        let ip = extract_xml_tag_content(soap_resp, "NewExternalIPAddress");
        assert_eq!(ip.as_deref(), Some("203.0.113.45"));
    }

    #[test]
    fn test_parse_http_url() {
        let (host, port, path) = parse_http_url("http://192.168.1.1:1900/desc.xml").unwrap();
        assert_eq!(host, "192.168.1.1");
        assert_eq!(port, 1900);
        assert_eq!(path, "/desc.xml");

        let (host2, port2, path2) = parse_http_url("http://router.local/").unwrap();
        assert_eq!(host2, "router.local");
        assert_eq!(port2, 80);
        assert_eq!(path2, "/");
    }
}
