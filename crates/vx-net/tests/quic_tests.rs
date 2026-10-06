//! Integration tests for QUIC transport over Quinn and TLS 1.3.

use std::net::SocketAddr;
use vx_net::quic::{QuicClientEndpoint, QuicServerEndpoint};

#[tokio::test]
async fn test_quic_client_server_handshake_and_stream() {
    let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (server, _cert) =
        QuicServerEndpoint::bind(server_addr).expect("failed to bind QUIC server");
    let actual_addr = server.local_addr().expect("failed to get local addr");

    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client = QuicClientEndpoint::bind(client_addr).expect("failed to bind QUIC client");

    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server failed to accept");

        // Accept bidirectional stream
        let (mut send, mut recv) = conn.accept_bi().await.expect("failed to accept bi stream");
        let mut buf = vec![0u8; 12];
        recv.read_exact(&mut buf)
            .await
            .expect("failed to read stream");
        assert_eq!(&buf, b"Hello Server");

        send.write_all(b"Hello Client")
            .await
            .expect("failed to write stream");
        send.finish().expect("failed to finish stream");

        // Read datagram
        let dgram = conn.read_datagram().await.expect("failed to read datagram");
        assert_eq!(&dgram[..], b"unreliable ping");
    });

    let client_conn = client
        .connect(actual_addr, "localhost")
        .await
        .expect("client failed to connect");

    // Open bidirectional stream
    let (mut send, mut recv) = client_conn
        .open_bi()
        .await
        .expect("failed to open bi stream");
    send.write_all(b"Hello Server")
        .await
        .expect("failed to write stream");
    send.finish().expect("failed to finish stream");

    let mut buf = vec![0u8; 12];
    recv.read_exact(&mut buf)
        .await
        .expect("failed to read client stream");
    assert_eq!(&buf, b"Hello Client");

    // Send datagram
    client_conn
        .send_datagram(bytes::Bytes::from_static(b"unreliable ping"))
        .expect("failed to send datagram");

    server_task.await.expect("server task panicked");
}
