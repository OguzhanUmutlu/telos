//! Integration tests for embeddable Server API, `MultiWorldManager`, and /world command transfers.

use glam::DVec3;
use vx_core::coords::BlockPos;
use vx_net::{Connection, Lane, MemoryConnection, Payload};
use vx_protocol::bounded::BoundedString;
use vx_protocol::messages::{
    AuthMode, C2sChatMessage, C2sConfigAck, C2sLoginStart, C2sMessage, S2cMessage,
};
use vx_server::{GeneratorKind, Server, ServerConfig, WorldConfig};
use vx_voxel::coords::split_block_pos;

fn complete_handshake(
    server: &mut Server,
    client_conn: &MemoryConnection<C2sMessage, S2cMessage>,
    username: &str,
) {
    // 1. Hello
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(vx_protocol::messages::C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        )
        .expect("send Hello");
    server.tick();
    let _ = client_conn.try_recv();

    // 2. Login
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new(username).unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();
    let _ = client_conn.try_recv();

    // 3. ConfigAck
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck {})),
        )
        .expect("send ConfigAck");
    server.tick();

    // Drain JoinGame and initial sync packets
    while let Ok(Some(_)) = client_conn.try_recv() {}
}

#[test]
fn test_server_builder_fluent_api() {
    let server = Server::builder()
        .seed(9999)
        .motd("Telos Embedded Engine")
        .view_distance(6)
        .vertical_view_distance(8)
        .max_players(128)
        .world("mining", 8888, GeneratorKind::Flat)
        .world("sky", 7777, GeneratorKind::Void)
        .build()
        .expect("Failed to build server");

    assert_eq!(server.config().motd, "Telos Embedded Engine");
    assert_eq!(server.config().view_distance, 6);
    assert_eq!(server.config().max_players, 128);

    // 2 custom worlds specified in builder replace the default single overworld
    assert_eq!(server.worlds().len(), 2);
    assert!(server.worlds().contains("mining"));
    assert!(server.worlds().contains("sky"));

    assert_eq!(
        server.world_named("mining").unwrap().generator_kind(),
        GeneratorKind::Flat
    );
    assert_eq!(
        server.world_named("sky").unwrap().generator_kind(),
        GeneratorKind::Void
    );
}

#[test]
fn test_server_config_toml_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_path = tmp.path().join("server.toml");

    let original = ServerConfig {
        motd: "Custom MOTD Roundtrip".to_string(),
        view_distance: 14,
        worlds: vec![
            WorldConfig {
                name: "lobby".to_string(),
                seed: 123,
                generator: GeneratorKind::Void,
                save_directory: None,
            },
            WorldConfig {
                name: "arena".to_string(),
                seed: 456,
                generator: GeneratorKind::Flat,
                save_directory: None,
            },
        ],
        ..Default::default()
    };

    original.save_to_file(&cfg_path).expect("save_to_file");
    let loaded = ServerConfig::load_from_file(&cfg_path).expect("load_from_file");

    assert_eq!(loaded.motd, "Custom MOTD Roundtrip");
    assert_eq!(loaded.view_distance, 14);
    assert_eq!(loaded.worlds.len(), 2);
    assert_eq!(loaded.worlds[0].name, "lobby");
    assert_eq!(loaded.worlds[0].generator, GeneratorKind::Void);
    assert_eq!(loaded.worlds[1].name, "arena");
    assert_eq!(loaded.worlds[1].generator, GeneratorKind::Flat);
}

#[test]
fn test_multi_world_concurrent_ticking_and_player_transfer() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 16,
        worlds: vec![
            WorldConfig {
                name: "overworld".to_string(),
                seed: 1337,
                generator: GeneratorKind::Standard,
                save_directory: None,
            },
            WorldConfig {
                name: "flat".to_string(),
                seed: 42,
                generator: GeneratorKind::Flat,
                save_directory: None,
            },
        ],
        ..Default::default()
    };

    let mut server = Server::new(1337, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));

    complete_handshake(&mut server, &client_conn, "Explorer");

    // Deliver initial chunks for overworld
    server.tick();

    let mut received_chunks = 0;
    while let Ok(Some(msg)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChunkData(_) | S2cMessage::UniformChunk(_)) = msg.into_msg() {
            received_chunks += 1;
        }
    }
    assert!(received_chunks > 0, "Client must receive overworld chunks");

    // Transfer player to "flat" dimension
    server
        .transfer_player_world(session_id, "flat", Some(DVec3::new(0.0, 5.0, 0.0)))
        .expect("transfer_player_world");

    // Client receives unloads for previous dimension chunks immediately on transfer
    let mut received_unloads = 0;
    while let Ok(Some(msg)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChunkUnload(_)) = msg.into_msg() {
            received_unloads += 1;
        }
    }
    assert!(
        received_unloads > 0,
        "Client must receive ChunkUnload for overworld chunks on world transfer"
    );

    // Tick server to stream flat world chunks
    server.tick();

    let mut received_flat_chunks = 0;
    while let Ok(Some(msg)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChunkData(data)) = msg.into_msg() {
            received_flat_chunks += 1;
            // Verify chunk at origin has flat layer content
            if data.chunk_x == 0 && data.chunk_y == 0 && data.chunk_z == 0 {
                let vx_protocol::messages::ChunkPayload::Snapshot(snap) = data.payload else {
                    panic!("Expected snapshot");
                };
                let bedrock = snap.blocks().get(split_block_pos(BlockPos::new(0, 0, 0)).1);
                let dirt = snap.blocks().get(split_block_pos(BlockPos::new(0, 1, 0)).1);
                let grass = snap.blocks().get(split_block_pos(BlockPos::new(0, 4, 0)).1);
                let air = snap.blocks().get(split_block_pos(BlockPos::new(0, 5, 0)).1);

                assert_ne!(bedrock.0, 0, "Bedrock layer must not be air");
                assert_ne!(dirt.0, 0, "Dirt layer must not be air");
                assert_ne!(grass.0, 0, "Grass layer must not be air");
                assert_eq!(air.0, 0, "Layer above grass must be air");
                assert_ne!(bedrock, dirt);
                assert_ne!(dirt, grass);
            }
        }
    }
    assert!(received_flat_chunks > 0, "Must receive flat world chunks");
}

#[test]
fn test_world_command_list_and_teleport() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        worlds: vec![
            WorldConfig {
                name: "overworld".to_string(),
                seed: 1337,
                generator: GeneratorKind::Standard,
                save_directory: None,
            },
            WorldConfig {
                name: "mining".to_string(),
                seed: 42,
                generator: GeneratorKind::Flat,
                save_directory: None,
            },
        ],
        ..Default::default()
    };

    let mut server = Server::new(1337, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let _session_id = server.add_connection(Box::new(server_conn));

    complete_handshake(&mut server, &client_conn, "CmdPlayer");

    // 1. Send /world list command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/world list").unwrap(),
            })),
        )
        .expect("send /world list");

    server.tick();

    let mut found_list_reply = false;
    while let Ok(Some(msg)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChatMessage(chat)) = msg.into_msg()
            && chat.message.as_str().contains("Loaded worlds (2):")
            && chat.message.as_str().contains("mining (0 players)")
            && chat.message.as_str().contains("overworld (1 players)")
        {
            found_list_reply = true;
        }
    }
    assert!(
        found_list_reply,
        "Server must respond to /world list with active dimensions"
    );

    // 2. Send /world tp mining
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/world tp mining 0 10 0").unwrap(),
            })),
        )
        .expect("send /world tp mining");

    server.tick();

    let mut found_tp_reply = false;
    while let Ok(Some(msg)) = client_conn.try_recv() {
        if let Some(S2cMessage::ChatMessage(chat)) = msg.into_msg()
            && chat
                .message
                .as_str()
                .contains("Transferred to world 'mining'")
        {
            found_tp_reply = true;
        }
    }
    assert!(
        found_tp_reply,
        "Server must confirm /world tp transfer to mining"
    );
}

#[test]
fn test_server_quic_listener_accepts_remote_client() {
    use std::time::{Duration, Instant};
    use vx_net::quic::QuicClientEndpoint;
    use vx_protocol::messages::C2sHello;

    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        ..Default::default()
    };

    let mut server = Server::new(4242, config);
    // Bind to ephemeral port
    server
        .listen_addr("127.0.0.1:0".parse().unwrap())
        .expect("listen_addr");

    let server_addr = server
        .listener
        .as_ref()
        .expect("listener must exist")
        .local_addr();

    // Spawn async client in vx_net quic runtime
    let rt = vx_net::quic::runtime();
    let client_conn = rt.block_on(async move {
        let client_ep =
            QuicClientEndpoint::bind("127.0.0.1:0".parse().unwrap()).expect("bind client");
        let conn = client_ep
            .connect_to(server_addr, "localhost")
            .await
            .expect("connect");

        let hello = C2sMessage::Hello(C2sHello {
            protocol: 1,
            build: BoundedString::new("0.1.0").unwrap(),
            features: 0,
        });
        conn.send(Lane::Control, Payload::Msg(hello))
            .expect("send hello");

        conn
    });

    // Tick server to accept connection and process Hello
    let start = Instant::now();
    while server.session_count() == 0 && start.elapsed() < Duration::from_secs(3) {
        server.tick();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        server.session_count(),
        1,
        "Server must accept remote client into sessions"
    );

    // Next tick processes Hello and replies HelloReply
    server.tick();

    // Client receives HelloReply
    let start = Instant::now();
    let mut got_reply = false;
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(Some(msg)) = client_conn.try_recv()
            && let Some(S2cMessage::HelloReply(_)) = msg.into_msg()
        {
            got_reply = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        got_reply,
        "Remote client must receive HelloReply from server"
    );
}
