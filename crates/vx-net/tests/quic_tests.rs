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

#[tokio::test]
async fn test_quic_listener_and_connection_trait() {
    use std::time::{Duration, Instant};
    use vx_net::{Lane, Payload, QuicListener};
    use vx_protocol::bounded::BoundedString;
    use vx_protocol::messages::{
        C2sHello, C2sMessage, DisconnectReason, S2cHelloReply, S2cMessage,
    };

    let listener = QuicListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind listener");
    let addr = listener.local_addr();

    let client_ep =
        QuicClientEndpoint::bind("127.0.0.1:0".parse().unwrap()).expect("bind client ep");
    let client_conn = client_ep
        .connect_to(addr, "localhost")
        .await
        .expect("connect to server");

    // Client sends Hello right away over the bidirectional stream
    let hello = C2sMessage::Hello(C2sHello {
        protocol: 1,
        build: BoundedString::new("test").unwrap(),
        features: 0x42,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(hello))
        .expect("client send hello");

    // Wait for server to accept
    let start = Instant::now();
    let mut server_conn = None;
    while start.elapsed() < Duration::from_secs(3) {
        if let Some(conn) = listener.try_accept() {
            server_conn = Some(conn);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let server_conn = server_conn.expect("server should accept incoming connection");

    // Server receives Hello
    let start = Instant::now();
    let mut received_hello = false;
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(Some(incoming)) = server_conn.try_recv()
            && let Some(C2sMessage::Hello(h)) = incoming.into_msg()
        {
            assert_eq!(h.protocol, 1);
            assert_eq!(h.features, 0x42);
            received_hello = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(received_hello, "server must receive Hello message");

    // Server sends HelloReply
    let reply = S2cMessage::HelloReply(S2cHelloReply {
        protocol: 1,
        features: 0x42,
        server_id: [7u8; 16],
    });
    server_conn
        .send(Lane::Control, Payload::Msg(reply))
        .expect("server send reply");

    // Client receives HelloReply
    let start = Instant::now();
    let mut received_reply = false;
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(Some(incoming)) = client_conn.try_recv()
            && let Some(S2cMessage::HelloReply(r)) = incoming.into_msg()
        {
            assert_eq!(r.protocol, 1);
            assert_eq!(r.features, 0x42);
            assert_eq!(r.server_id, [7u8; 16]);
            received_reply = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(received_reply, "client must receive HelloReply message");

    client_conn.close(DisconnectReason::Normal);
    server_conn.close(DisconnectReason::Normal);
    listener.close();
}
