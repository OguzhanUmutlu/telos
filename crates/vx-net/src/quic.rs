//! QUIC transport runtime powered by Quinn over TLS 1.3 with ALPN `vx/1`.

use crate::error::{NetError, Result};
use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use std::net::SocketAddr;
use std::sync::Arc;
use vx_protocol::ALPN_PROTOCOL;

/// Generates a self-signed TLS certificate and private key for local development or LAN hosting.
pub fn generate_self_signed_cert() -> Result<(Vec<u8>, Vec<u8>)> {
    let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let certified_key = rcgen::generate_simple_self_signed(subject_alt_names)
        .map_err(|e| NetError::Tls(e.to_string()))?;

    let cert_der = certified_key.cert.der().to_vec();
    let key_der = certified_key.key_pair.serialize_der();

    Ok((cert_der, key_der))
}

/// Custom certificate verifier that accepts any certificate (used for development and LAN TOFU).
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Creates a Quinn server configuration with the given DER certificate and private key.
pub fn create_server_config(cert_der: Vec<u8>, key_der: Vec<u8>) -> Result<ServerConfig> {
    let cert = CertificateDer::from(cert_der);
    let key = PrivateKeyDer::Pkcs8(key_der.into());

    let mut crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .map_err(|e| NetError::Tls(e.to_string()))?;

    crypto.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];

    let server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(crypto)
            .map_err(|e| NetError::Tls(e.to_string()))?,
    ));

    Ok(server_config)
}

/// Creates a Quinn client configuration configured for LAN/development servers.
pub fn create_client_config() -> Result<ClientConfig> {
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    crypto.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];

    let client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .map_err(|e| NetError::Tls(e.to_string()))?,
    ));

    Ok(client_config)
}

/// Server endpoint wrapping a bound Quinn UDP socket.
pub struct QuicServerEndpoint {
    endpoint: Endpoint,
}

impl QuicServerEndpoint {
    /// Binds a QUIC server on the specified local socket address with self-signed certificate.
    pub fn bind(addr: SocketAddr) -> Result<(Self, Vec<u8>)> {
        let (cert_der, key_der) = generate_self_signed_cert()?;
        let config = create_server_config(cert_der.clone(), key_der)?;

        let endpoint = Endpoint::server(config, addr).map_err(|e| NetError::Quic(e.to_string()))?;

        Ok((Self { endpoint }, cert_der))
    }

    /// Accesses the underlying Quinn endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Local socket address of the endpoint.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint.local_addr().map_err(NetError::Io)
    }

    /// Waits for an incoming connection and completes the QUIC handshake.
    pub async fn accept(&self) -> Result<quinn::Connection> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| NetError::Quic("Endpoint closed".to_string()))?;

        let conn = incoming.await.map_err(|e| NetError::Quic(e.to_string()))?;

        Ok(conn)
    }
}

/// Client endpoint wrapping an unbound or locally-bound Quinn UDP socket.
pub struct QuicClientEndpoint {
    endpoint: Endpoint,
}

impl QuicClientEndpoint {
    /// Binds a client QUIC endpoint to an ephemeral local port.
    pub fn bind(bind_addr: SocketAddr) -> Result<Self> {
        let mut endpoint =
            Endpoint::client(bind_addr).map_err(|e| NetError::Quic(e.to_string()))?;
        endpoint.set_default_client_config(create_client_config()?);
        Ok(Self { endpoint })
    }

    /// Accesses the underlying Quinn endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Connects to a remote QUIC server and awaits the handshake completion.
    pub async fn connect(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
    ) -> Result<quinn::Connection> {
        let connecting = self
            .endpoint
            .connect(server_addr, server_name)
            .map_err(|e| NetError::Quic(e.to_string()))?;

        let conn = connecting
            .await
            .map_err(|e| NetError::Quic(e.to_string()))?;

        Ok(conn)
    }
}
