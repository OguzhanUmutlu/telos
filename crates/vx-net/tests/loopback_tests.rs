//! Integration tests for wire-serialized loopback transport.

use vx_net::loopback::loopback_pair;
use vx_net::transport::{Connection, Lane, Payload};
use vx_protocol::bounded::BoundedString;
use vx_protocol::messages::{
    AuthMode, C2sHello, C2sLoginStart, C2sMessage, ConnectionPhase, S2cHelloReply, S2cLoginSuccess,
    S2cMessage,
};

#[test]
fn test_loopback_serialized_handshake() {
    let (client, server) = loopback_pair(64);

    // Initial phase: Hello
    assert_eq!(client.phase(), ConnectionPhase::Hello);
    assert_eq!(server.phase(), ConnectionPhase::Hello);

    // Client -> Server Hello
    let hello = C2sMessage::Hello(C2sHello {
        protocol: 1,
        build: BoundedString::new("test-loopback").unwrap(),
        features: 0b101,
    });
    client
        .send(Lane::Control, Payload::msg(hello.clone()))
        .unwrap();

    let server_msg = server
        .try_recv()
        .unwrap()
        .expect("server received Hello")
        .into_msg()
        .unwrap();
    assert_eq!(server_msg, hello);

    // Server -> Client HelloReply
    let reply = S2cMessage::HelloReply(S2cHelloReply {
        protocol: 1,
        features: 0b101,
        server_id: [99u8; 16],
    });
    server
        .send(Lane::Control, Payload::msg(reply.clone()))
        .unwrap();

    let client_msg = client
        .try_recv()
        .unwrap()
        .expect("client received HelloReply")
        .into_msg()
        .unwrap();
    assert_eq!(client_msg, reply);

    // Transition to Login phase
    client.set_phase(ConnectionPhase::Login);
    server.set_phase(ConnectionPhase::Login);

    // Client -> Server LoginStart
    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new("LoopbackTester").unwrap(),
        mode: AuthMode::Offline,
    });
    client
        .send(Lane::Control, Payload::msg(login.clone()))
        .unwrap();

    let server_msg = server
        .try_recv()
        .unwrap()
        .expect("server received LoginStart")
        .into_msg()
        .unwrap();
    assert_eq!(server_msg, login);

    // Server -> Client LoginSuccess
    let login_ok = S2cMessage::LoginSuccess(S2cLoginSuccess {
        player_uuid: [123u8; 16],
        username: BoundedString::new("LoopbackTester").unwrap(),
    });
    server
        .send(Lane::Control, Payload::msg(login_ok.clone()))
        .unwrap();

    let client_msg = client
        .try_recv()
        .unwrap()
        .expect("client received LoginSuccess")
        .into_msg()
        .unwrap();
    assert_eq!(client_msg, login_ok);

    // Verify wire byte accounting in stats
    let c_stats = client.stats();
    let s_stats = server.stats();
    assert!(c_stats.bytes_sent > 0);
    assert_eq!(c_stats.bytes_sent, s_stats.bytes_received);
    assert_eq!(s_stats.bytes_sent, c_stats.bytes_received);
    assert_eq!(c_stats.packets_sent, 2);
    assert_eq!(s_stats.packets_sent, 2);
}
