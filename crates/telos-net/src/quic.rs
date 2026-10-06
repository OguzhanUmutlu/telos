use crate::error::{NetError, Result};
use crate::loopback::{ClientCodec, ServerCodec, WireCodec};
use crate::transport::{ConnStats, ConnStatsCounters, Connection, Incoming, Lane, Payload};
use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use telos_protocol::ALPN_PROTOCOL;
use telos_protocol::messages::{C2sMessage, ConnectionPhase, DisconnectReason, S2cMessage};
use tokio::sync::mpsc as tokio_mpsc;

/// Global Tokio runtime for background telos-net workers if none is active on current thread.
#[must_use]
pub fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("telos-net-worker")
            .build()
            .expect("Failed to initialize telos-net Tokio runtime")
    })
}

/// Spawns a future on the current or global background runtime.
pub fn spawn_task<F>(f: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(f);
    } else {
        runtime().spawn(f);
    }
}

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
        let _guard = runtime().enter();
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
        let _guard = runtime().enter();
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

    /// Connects to a remote server and opens bidirectional channels, returning a unified `Connection`.
    pub async fn connect_to(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
    ) -> Result<Box<dyn Connection<C2sMessage, S2cMessage>>> {
        let conn = self.connect(server_addr, server_name).await?;
        let (send, recv) = conn
            .open_bi()
            .await
            .map_err(|e| NetError::Quic(e.to_string()))?;
        let quic_conn =
            QuicConnection::<C2sMessage, S2cMessage, ClientCodec>::from_quinn(conn, send, recv);
        Ok(Box::new(quic_conn))
    }
}

/// Bidirectional QUIC connection wrapping Quinn streams and datagrams into the unified `Connection` trait.
pub struct QuicConnection<Tx, Rx, C: WireCodec<Tx, Rx>> {
    conn: quinn::Connection,
    outgoing_tx: tokio_mpsc::UnboundedSender<(Lane, Vec<u8>)>,
    inbox: Arc<Mutex<VecDeque<Vec<u8>>>>,
    unreliable_ring: Arc<Mutex<VecDeque<Vec<u8>>>>,
    local_closed: Arc<AtomicBool>,
    remote_closed: Arc<AtomicBool>,
    disconnect_reason: Arc<Mutex<Option<DisconnectReason>>>,
    stats: Arc<ConnStatsCounters>,
    _marker: PhantomData<(Tx, Rx, C)>,
}

impl<Tx: Send + Sync + 'static, Rx: Send + Sync + 'static, C: WireCodec<Tx, Rx>>
    QuicConnection<Tx, Rx, C>
{
    /// Creates a new `QuicConnection` from an established Quinn connection and bidirectional stream.
    pub fn from_quinn(
        conn: quinn::Connection,
        mut send_stream: quinn::SendStream,
        mut recv_stream: quinn::RecvStream,
    ) -> Self {
        let (outgoing_tx, mut outgoing_rx) = tokio_mpsc::unbounded_channel::<(Lane, Vec<u8>)>();
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let unreliable_ring = Arc::new(Mutex::new(VecDeque::new()));
        let local_closed = Arc::new(AtomicBool::new(false));
        let remote_closed = Arc::new(AtomicBool::new(false));
        let disconnect_reason = Arc::new(Mutex::new(None));
        let stats = Arc::new(ConnStatsCounters::default());

        // Background writer task
        let writer_conn = conn.clone();
        let writer_closed = local_closed.clone();
        let writer_stats = stats.clone();
        spawn_task(async move {
            while let Some((lane, buf)) = outgoing_rx.recv().await {
                if writer_closed.load(Ordering::Acquire) {
                    break;
                }
                let len = buf.len() as u64;
                match lane {
                    Lane::Unreliable => {
                        if writer_conn.send_datagram(bytes::Bytes::from(buf)).is_ok() {
                            writer_stats.inc_packets_sent();
                            writer_stats.add_bytes_sent(len);
                        }
                    }
                    Lane::Control | Lane::Chunk { .. } | Lane::Bulk => {
                        let len_bytes = (buf.len() as u32).to_be_bytes();
                        if send_stream.write_all(&len_bytes).await.is_err() {
                            break;
                        }
                        if send_stream.write_all(&buf).await.is_err() {
                            break;
                        }
                        writer_stats.inc_packets_sent();
                        writer_stats.add_bytes_sent(len);
                    }
                }
            }
        });

        // Background stream reader task
        let stream_inbox = inbox.clone();
        let stream_closed = remote_closed.clone();
        let stream_stats = stats.clone();
        spawn_task(async move {
            let mut len_buf = [0u8; 4];
            loop {
                if recv_stream.read_exact(&mut len_buf).await.is_err() {
                    stream_closed.store(true, Ordering::Release);
                    break;
                }
                let payload_len = u32::from_be_bytes(len_buf) as usize;
                if payload_len > 2 * 1024 * 1024 {
                    stream_closed.store(true, Ordering::Release);
                    break;
                }
                let mut buf = vec![0u8; payload_len];
                if recv_stream.read_exact(&mut buf).await.is_err() {
                    stream_closed.store(true, Ordering::Release);
                    break;
                }
                stream_stats.inc_packets_received();
                stream_stats.add_bytes_received(payload_len as u64);
                let mut q = stream_inbox.lock().unwrap();
                q.push_back(buf);
            }
        });

        // Background datagram reader task
        let dgram_conn = conn.clone();
        let dgram_ring = unreliable_ring.clone();
        let dgram_stats = stats.clone();
        spawn_task(async move {
            while let Ok(dgram) = dgram_conn.read_datagram().await {
                dgram_stats.inc_packets_received();
                dgram_stats.add_bytes_received(dgram.len() as u64);
                let mut q = dgram_ring.lock().unwrap();
                if q.len() >= 64 {
                    q.pop_front();
                }
                q.push_back(dgram.to_vec());
            }
        });

        Self {
            conn,
            outgoing_tx,
            inbox,
            unreliable_ring,
            local_closed,
            remote_closed,
            disconnect_reason,
            stats,
            _marker: PhantomData,
        }
    }
}

impl<Tx: Send + Sync + 'static, Rx: Send + Sync + 'static, C: WireCodec<Tx, Rx>> Connection<Tx, Rx>
    for QuicConnection<Tx, Rx, C>
{
    fn send(&self, lane: Lane, payload: Payload<Tx>) -> Result<()> {
        if self.is_closed() {
            return Err(NetError::Closed);
        }

        let mut buf = Vec::new();
        match payload {
            Payload::Msg(ref m) => {
                C::encode(m, &mut buf);
            }
            Payload::Shared(ref s) => {
                buf.extend_from_slice(s);
            }
        }

        self.outgoing_tx
            .send((lane, buf))
            .map_err(|_| NetError::Closed)?;

        Ok(())
    }

    fn try_recv(&self) -> Result<Option<Incoming<Rx>>> {
        let raw_opt = {
            let mut q = self.inbox.lock().unwrap();
            q.pop_front()
        }
        .or_else(|| {
            let mut ring = self.unreliable_ring.lock().unwrap();
            ring.pop_front()
        });

        if let Some(bytes) = raw_opt {
            let mut cursor = &bytes[..];
            let msg = C::decode(ConnectionPhase::Play, &mut cursor)?;
            return Ok(Some(Incoming::Msg(msg)));
        }

        if self.local_closed.load(Ordering::Acquire) {
            return Err(NetError::Closed);
        }
        if self.remote_closed.load(Ordering::Acquire) {
            let reason = self
                .disconnect_reason
                .lock()
                .unwrap()
                .unwrap_or(DisconnectReason::Normal);
            return Err(NetError::Disconnected {
                reason,
                message: "Closed by peer".to_string(),
            });
        }

        Ok(None)
    }

    fn stats(&self) -> ConnStats {
        #[allow(clippy::cast_possible_truncation)]
        self.stats.set_rtt(self.conn.rtt().as_millis() as u32);
        self.stats.snapshot()
    }

    fn close(&self, reason: DisconnectReason) {
        self.local_closed.store(true, Ordering::Release);
        let mut r = self.disconnect_reason.lock().unwrap();
        if r.is_none() {
            *r = Some(reason);
        }
        self.conn.close(quinn::VarInt::from_u32(0), b"closed");
    }

    fn is_closed(&self) -> bool {
        self.local_closed.load(Ordering::Acquire) || self.remote_closed.load(Ordering::Acquire)
    }
}

/// Listener accepting incoming QUIC connections from remote clients over TLS 1.3.
pub struct QuicListener {
    endpoint: Endpoint,
    local_addr: SocketAddr,
    incoming_rx: mpsc::Receiver<Box<dyn Connection<S2cMessage, C2sMessage>>>,
    closed: Arc<AtomicBool>,
}

impl QuicListener {
    /// Binds a QUIC server listener on the specified local socket address.
    pub fn bind(addr: SocketAddr) -> Result<Self> {
        let (server_endpoint, _cert) = QuicServerEndpoint::bind(addr)?;
        let local_addr = server_endpoint.local_addr()?;
        let endpoint = server_endpoint.endpoint;
        let (incoming_tx, incoming_rx) = mpsc::channel();
        let closed = Arc::new(AtomicBool::new(false));
        let task_closed = closed.clone();

        let accept_endpoint = endpoint.clone();
        spawn_task(async move {
            while !task_closed.load(Ordering::Acquire) {
                let Some(incoming) = accept_endpoint.accept().await else {
                    break;
                };

                let Ok(conn) = incoming.await else {
                    continue;
                };

                let conn_tx = incoming_tx.clone();
                spawn_task(async move {
                    if let Ok((send, recv)) = conn.accept_bi().await {
                        let quic_conn =
                            QuicConnection::<S2cMessage, C2sMessage, ServerCodec>::from_quinn(
                                conn, send, recv,
                            );
                        let boxed: Box<dyn Connection<S2cMessage, C2sMessage>> =
                            Box::new(quic_conn);
                        let _ = conn_tx.send(boxed);
                    }
                });
            }
        });

        Ok(Self {
            endpoint,
            local_addr,
            incoming_rx,
            closed,
        })
    }

    /// Bound local socket address.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Non-blocking check for a newly connected client.
    pub fn try_accept(&self) -> Option<Box<dyn Connection<S2cMessage, C2sMessage>>> {
        self.incoming_rx.try_recv().ok()
    }

    /// Closes the listener.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.endpoint.close(quinn::VarInt::from_u32(0), b"closed");
    }
}
