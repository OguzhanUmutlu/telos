//! Integration tests for authoritative server session management, chunk streaming, and eviction.

use vx_core::coords::ChunkPos;
use vx_net::{Connection, Lane, MemoryConnection, Payload};
use vx_protocol::bounded::BoundedString;
use vx_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerPosition, S2cMessage,
};
use vx_server::{Server, ServerConfig};

#[test]
fn test_server_handshake_and_chunk_streaming() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));
    assert_eq!(session_id, 1);
    assert_eq!(server.session_count(), 1);

    // 1. Client sends Hello
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        )
        .expect("send Hello");

    server.tick();

    // Client receives HelloReply
    let incoming = client_conn.try_recv().unwrap().expect("recv HelloReply");
    match incoming.into_msg().unwrap() {
        S2cMessage::HelloReply(reply) => {
            assert_eq!(reply.protocol, 1);
        }
        other => panic!("expected HelloReply, got {other:?}"),
    }

    // 2. Client sends LoginStart
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("Player1").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");

    server.tick();

    // Client receives LoginSuccess
    let incoming = client_conn.try_recv().unwrap().expect("recv LoginSuccess");
    match incoming.into_msg().unwrap() {
        S2cMessage::LoginSuccess(succ) => {
            assert_eq!(succ.username.as_str(), "Player1");
        }
        other => panic!("expected LoginSuccess, got {other:?}"),
    }

    // 3. Client sends ClientSettings + ConfigAck
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 2,
                simulation_distance: 2,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .expect("send ClientSettings");
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .expect("send ConfigAck");

    server.tick();

    // Client receives JoinGame
    let incoming = client_conn.try_recv().unwrap().expect("recv JoinGame");
    match incoming.into_msg().unwrap() {
        S2cMessage::JoinGame(join) => {
            assert_eq!(join.entity_id, 1);
            assert_eq!(join.view_distance, 2);
        }
        other => panic!("expected JoinGame, got {other:?}"),
    }

    // 4. Server ticks again to deliver chunks
    server.tick();

    let mut received_chunks = 0;
    let mut received_lod_nodes = 0;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        match incoming.into_msg().unwrap() {
            S2cMessage::ChunkData(chunk) => {
                received_chunks += 1;
                assert!(chunk.chunk_y >= -2 && chunk.chunk_y <= 2);
            }
            S2cMessage::UniformChunk(_) => {
                received_chunks += 1;
            }
            S2cMessage::LodNodeData(lod) => {
                received_lod_nodes += 1;
                assert!(lod.level >= 1);
            }
            other => panic!("unexpected message during chunk delivery: {other:?}"),
        }
    }

    assert!(
        received_chunks > 0,
        "Expected server to deliver initial chunks"
    );
    assert!(
        received_lod_nodes > 0,
        "Expected server to deliver initial far-field LOD nodes"
    );
}

#[test]
fn test_player_movement_and_chunk_eviction() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 100,
        ..Default::default()
    };
    let mut server = Server::new(42, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    // Fast-forward handshake
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv();

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("Steve").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv();

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 2,
                simulation_distance: 2,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .unwrap();
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv();

    // Deliver all chunks around (0, 0, 0)
    for _ in 0..5 {
        server.tick();
    }
    // Drain delivered chunks
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Now player teleports far away: x = 320.0 (chunk x = 10)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 320.0,
                y: 64.0,
                z: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();

    server.tick();

    // We should receive S2cChunkUnload messages for chunks around (0, 0, 0)
    let mut unloads = Vec::new();
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChunkUnload(unload)) = incoming.into_msg() {
            unloads.push(ChunkPos::new(
                unload.chunk_x,
                unload.chunk_y,
                unload.chunk_z,
            ));
        }
    }

    assert!(
        !unloads.is_empty(),
        "Expected chunk unloads after moving far away"
    );
    // Spawn chunk should be unloaded
    assert!(
        unloads.contains(&ChunkPos::new(4, 1, 5)),
        "Expected spawn chunk (4, 1, 5) to be unloaded"
    );
}
