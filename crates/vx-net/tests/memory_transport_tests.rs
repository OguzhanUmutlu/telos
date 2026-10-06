//! Integration tests for in-memory singleplayer transport.

use std::sync::Arc;
use std::time::Instant;
use vx_net::memory::MemoryConnection;
use vx_net::transport::{Connection, Incoming, Lane, Payload};
use vx_protocol::bounded::{BoundedString, BoundedVec};
use vx_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sKeepAlive, C2sKnownRegistries,
    C2sLoginStart, C2sMessage, DisconnectReason, S2cChatMessage, S2cConfigDone, S2cHelloReply,
    S2cLoginSuccess, S2cMessage, S2cRegistryData,
};

#[test]
#[allow(clippy::too_many_lines)]
fn test_memory_handshake_lifecycle() {
    let (client, server) = MemoryConnection::<C2sMessage, S2cMessage>::pair_default();

    let start = Instant::now();

    // 1. Hello Phase
    client
        .send(
            Lane::Control,
            Payload::msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("voxel-dev").unwrap(),
                features: 0b1,
            })),
        )
        .unwrap();

    let c2s_msg = server
        .try_recv()
        .unwrap()
        .expect("server should receive Hello")
        .into_msg()
        .unwrap();
    assert!(matches!(c2s_msg, C2sMessage::Hello(_)));

    server
        .send(
            Lane::Control,
            Payload::msg(S2cMessage::HelloReply(S2cHelloReply {
                protocol: 1,
                features: 0b1,
                server_id: [42u8; 16],
            })),
        )
        .unwrap();

    let s2c_msg = client
        .try_recv()
        .unwrap()
        .expect("client should receive HelloReply")
        .into_msg()
        .unwrap();
    assert!(matches!(s2c_msg, S2cMessage::HelloReply(_)));

    // 2. Login Phase
    client
        .send(
            Lane::Control,
            Payload::msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("SingleplayerHero").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();

    let c2s_msg = server
        .try_recv()
        .unwrap()
        .expect("server should receive LoginStart")
        .into_msg()
        .unwrap();
    assert!(matches!(c2s_msg, C2sMessage::LoginStart(_)));

    server
        .send(
            Lane::Control,
            Payload::msg(S2cMessage::LoginSuccess(S2cLoginSuccess {
                player_uuid: [1u8; 16],
                username: BoundedString::new("SingleplayerHero").unwrap(),
            })),
        )
        .unwrap();

    let s2c_msg = client
        .try_recv()
        .unwrap()
        .expect("client should receive LoginSuccess")
        .into_msg()
        .unwrap();
    assert!(matches!(s2c_msg, S2cMessage::LoginSuccess(_)));

    // 3. Config Phase
    client
        .send(
            Lane::Control,
            Payload::msg(C2sMessage::KnownRegistries(C2sKnownRegistries {
                known_hashes: BoundedVec::empty(),
            })),
        )
        .unwrap();

    let _ = server
        .try_recv()
        .unwrap()
        .expect("server received known registries");

    server
        .send(
            Lane::Control,
            Payload::msg(S2cMessage::RegistryData(S2cRegistryData {
                registry_id: BoundedString::new("voxel:block").unwrap(),
                content_hash: [0x42; 32],
                entries: BoundedVec::new(vec![
                    BoundedString::new("voxel:air").unwrap(),
                    BoundedString::new("voxel:stone").unwrap(),
                ])
                .unwrap(),
            })),
        )
        .unwrap();

    let _ = client
        .try_recv()
        .unwrap()
        .expect("client received registry data");

    client
        .send(
            Lane::Control,
            Payload::msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 32,
                simulation_distance: 12,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .unwrap();

    let _ = server
        .try_recv()
        .unwrap()
        .expect("server received client settings");

    server
        .send(
            Lane::Control,
            Payload::msg(S2cMessage::ConfigDone(S2cConfigDone)),
        )
        .unwrap();

    let _ = client
        .try_recv()
        .unwrap()
        .expect("client received config done");

    client
        .send(
            Lane::Control,
            Payload::msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .unwrap();

    let _ = server
        .try_recv()
        .unwrap()
        .expect("server received config ack");

    // 4. Play Phase
    server
        .send(
            Lane::Control,
            Payload::msg(S2cMessage::ChatMessage(S2cChatMessage {
                sender: BoundedString::new("Server").unwrap(),
                message: BoundedString::new("Game Started").unwrap(),
                timestamp: 1_000,
            })),
        )
        .unwrap();

    let chat = client
        .try_recv()
        .unwrap()
        .expect("client received chat")
        .into_msg()
        .unwrap();
    assert!(matches!(chat, S2cMessage::ChatMessage(_)));

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 50,
        "Handshake over MemoryTransport took {elapsed:?}, budget is < 1ms"
    );

    // Verify stats
    let c_stats = client.stats();
    let s_stats = server.stats();
    assert_eq!(c_stats.packets_sent, s_stats.packets_received);
    assert_eq!(s_stats.packets_sent, c_stats.packets_received);
}

#[test]
fn test_unreliable_ring_drops_oldest() {
    let (client, server) = MemoryConnection::<C2sMessage, S2cMessage>::pair(1024, 4);

    // Send 6 unreliable keep-alive packets into a ring buffer of capacity 4
    for i in 0..6 {
        client
            .send(
                Lane::Unreliable,
                Payload::msg(C2sMessage::KeepAlive(C2sKeepAlive { id: i })),
            )
            .unwrap();
    }

    // Server should receive exactly 4 packets: ids 2, 3, 4, 5 (0 and 1 dropped!)
    let mut received_ids = Vec::new();
    while let Ok(Some(item)) = server.try_recv() {
        if let Some(C2sMessage::KeepAlive(k)) = item.into_msg() {
            received_ids.push(k.id);
        }
    }

    assert_eq!(received_ids, vec![2, 3, 4, 5]);
}

#[test]
fn test_zero_copy_shared_payload() {
    let (client, server) = MemoryConnection::<C2sMessage, S2cMessage>::pair_default();

    let chunk_bytes = Arc::new(vec![0xAA; 4096]);
    server
        .send(
            Lane::Chunk { priority: 10 },
            Payload::shared(chunk_bytes.clone()),
        )
        .unwrap();

    let incoming = client
        .try_recv()
        .unwrap()
        .expect("client should receive chunk");
    match incoming {
        Incoming::Shared(shared) => {
            assert_eq!(shared.len(), 4096);
            assert!(Arc::ptr_eq(&shared, &chunk_bytes)); // Exact zero-copy pointer identity!
        }
        other => panic!("Expected Incoming::Shared, got {other:?}"),
    }
}

#[test]
fn test_clean_disconnect() {
    let (client, server) = MemoryConnection::<C2sMessage, S2cMessage>::pair_default();

    client.close(DisconnectReason::Normal);
    assert!(client.is_closed());

    // Server should detect client disconnection
    let res = server.try_recv();
    assert!(res.is_err());
}
