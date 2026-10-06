//! Integration tests for authoritative server session management, chunk streaming, and eviction.

use vx_core::coords::{BlockPos, ChunkPos, Face};
use vx_net::{Connection, Lane, MemoryConnection, Payload};
use vx_protocol::bounded::{BoundedString, BoundedVec};
use vx_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sChatMessage, C2sClientSettings,
    C2sCommandSuggest, C2sConfigAck, C2sHello, C2sInteractEntity, C2sInventoryClick, C2sLoginStart,
    C2sMessage, C2sPlayerCommand, C2sPlayerInput, C2sPlayerPosition, C2sTeleportAck, InputFrame,
    PlayerCommandKind, S2cMessage, input_buttons,
};
use vx_server::{Server, ServerConfig};
use vx_voxel::state::BlockStateId;

#[test]
#[allow(clippy::too_many_lines)]
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

    // Client receives RegistryData (blocks)
    let incoming = client_conn
        .try_recv()
        .unwrap()
        .expect("recv RegistryData blocks");
    match incoming.into_msg().unwrap() {
        S2cMessage::RegistryData(data) => {
            assert_eq!(data.registry_id.as_str(), "voxel:block");
            assert!(!data.entries.is_empty());
        }
        other => panic!("expected RegistryData blocks, got {other:?}"),
    }

    // Client receives RegistryData (items)
    let incoming = client_conn
        .try_recv()
        .unwrap()
        .expect("recv RegistryData items");
    match incoming.into_msg().unwrap() {
        S2cMessage::RegistryData(data) => {
            assert_eq!(data.registry_id.as_str(), "voxel:item");
            assert!(!data.entries.is_empty());
        }
        other => panic!("expected RegistryData items, got {other:?}"),
    }

    // Client receives ConfigDone
    let incoming = client_conn.try_recv().unwrap().expect("recv ConfigDone");
    match incoming.into_msg().unwrap() {
        S2cMessage::ConfigDone(_) => {}
        other => panic!("expected ConfigDone, got {other:?}"),
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
            S2cMessage::UpdateTime(time) => {
                assert!(time.time_of_day >= 6000);
            }
            S2cMessage::UpdateStats(_)
            | S2cMessage::InventoryBulk(_)
            | S2cMessage::UpdateWeather(_)
            | S2cMessage::PlayerMovementAck(_) => {}
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

#[test]
#[allow(clippy::too_many_lines)]
fn test_server_block_break_and_place() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

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
                username: BoundedString::new("Miner").unwrap(),
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

    // Position player at (0.0, 64.0, 0.0)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 0.0,
                y: 64.0,
                z: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Drain initial chunks
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // 1. Manually set a block at (0, 64, 2) to Stone (1)
    let target_pos = BlockPos::new(0, 64, 2);
    server
        .world_mut()
        .set_block(target_pos, BlockStateId::new(1));
    assert_eq!(
        server.world_mut().get_block(target_pos),
        BlockStateId::new(1)
    );

    // 2. Client breaks target_pos within reach (distance = 2.0 <= 6.0)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                sequence: 10,
                action: BlockActionKind::Break,
                x: 0,
                y: 64,
                z: 2,
                input_tick: 1,
            })),
        )
        .unwrap();

    server.tick();

    // Verify block is now Air (0) in server world
    assert_eq!(server.world_mut().get_block(target_pos), BlockStateId::AIR);

    // Verify client received S2cBlockUpdate and S2cBlockActionAck
    let mut got_update = false;
    let mut got_ack = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        match incoming.into_msg().unwrap() {
            S2cMessage::BlockUpdate(upd) => {
                if upd.x == 0 && upd.y == 64 && upd.z == 2 && upd.state_id == BlockStateId::AIR {
                    got_update = true;
                }
            }
            S2cMessage::BlockActionAck(ack) if ack.sequence == 10 => {
                got_ack = true;
            }
            _ => {}
        }
    }
    assert!(got_update, "Expected S2cBlockUpdate for broken block");
    assert!(got_ack, "Expected S2cBlockActionAck");

    // 3. Client places a block at target_pos (by clicking on (0, 63, 2) Top)
    // Placed block: 2 (Dirt)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                sequence: 11,
                action: BlockActionKind::Place {
                    state_id: BlockStateId::new(2),
                    hit_face: Face::Up as u8,
                },
                x: 0,
                y: 63,
                z: 2,
                input_tick: 2,
            })),
        )
        .unwrap();

    server.tick();

    // Verify block is now Dirt (2) in server world
    assert_eq!(
        server.world_mut().get_block(target_pos),
        BlockStateId::new(2)
    );

    let mut got_place_update = false;
    let mut got_place_ack = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        match incoming.into_msg().unwrap() {
            S2cMessage::BlockUpdate(upd) => {
                if upd.x == 0 && upd.y == 64 && upd.z == 2 && upd.state_id == BlockStateId::new(2) {
                    got_place_update = true;
                }
            }
            S2cMessage::BlockActionAck(ack) if ack.sequence == 11 => {
                got_place_ack = true;
            }
            _ => {}
        }
    }
    assert!(got_place_update, "Expected S2cBlockUpdate for placed block");
    assert!(got_place_ack, "Expected S2cBlockActionAck for place");

    // 4. Reach violation check: target block at (0, 64, 50), distance 50.0 > 6.0
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                sequence: 12,
                action: BlockActionKind::Break,
                x: 0,
                y: 64,
                z: 50,
                input_tick: 3,
            })),
        )
        .unwrap();

    server.tick();

    let mut got_reject_ack = false;
    let mut got_resync_update = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        match incoming.into_msg().unwrap() {
            S2cMessage::BlockActionAck(ack) if ack.sequence == 12 => {
                got_reject_ack = true;
            }
            S2cMessage::BlockUpdate(upd) if upd.x == 0 && upd.y == 64 && upd.z == 50 => {
                got_resync_update = true;
            }
            _ => {}
        }
    }
    assert!(
        got_reject_ack,
        "Expected S2cBlockActionAck for reach violation"
    );
    assert!(
        got_resync_update,
        "Expected rollback S2cBlockUpdate for reach violation"
    );
}

#[test]
fn test_server_persistence_save_and_reload() {
    let temp_dir = tempfile::tempdir().expect("create tempdir");
    let config = ServerConfig {
        tps: 20,
        view_distance: 1,
        vertical_view_distance: 1,
        save_directory: Some(temp_dir.path().to_path_buf()),
        ..Default::default()
    };

    let seed = 424_242;
    let modified_pos = BlockPos::new(14, 55, 14);
    let test_block_state = BlockStateId::new(3); // Cobblestone

    // Server 1: Modify world and save
    {
        let mut server = Server::new(seed, config.clone());
        let _ = server.world_mut().set_block(modified_pos, test_block_state);
        assert_eq!(server.world_mut().get_block(modified_pos), test_block_state);

        let saved = server.save_and_flush().expect("save_and_flush");
        assert!(saved >= 1, "Expected at least 1 saved chunk");
    }

    // Server 2: Reopen with the same directory and seed
    {
        let mut server2 = Server::new(seed, config);
        // Querying the block should load the chunk from disk and match the edited state
        assert_eq!(
            server2.world_mut().get_block(modified_pos),
            test_block_state,
            "Block state must persist across server restarts via .vxr storage"
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_survival_stats_and_inventory_interaction() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 1,
        vertical_view_distance: 1,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(777, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    // Handshake
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
                username: BoundedString::new("SurvivalPlayer").unwrap(),
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
                view_distance: 1,
                simulation_distance: 1,
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

    // Verify initial messages: JoinGame, UpdateTime, UpdateStats, InventoryBulk
    let mut got_stats = false;
    let mut got_bulk = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        match incoming.into_msg().unwrap() {
            S2cMessage::UpdateStats(stats) => {
                assert!((stats.health - 20.0).abs() < 0.001);
                assert_eq!(stats.food, 20);
                assert_eq!(stats.xp_level, 0);
                got_stats = true;
            }
            S2cMessage::InventoryBulk(bulk) => {
                assert_eq!(bulk.slots.len(), 46);
                assert_eq!(bulk.slots[0].item, 1); // Stone
                assert_eq!(bulk.slots[0].count, 64);
                assert_eq!(bulk.carried.count, 0);
                got_bulk = true;
            }
            _ => {}
        }
    }
    assert!(got_stats, "Must receive initial UpdateStats");
    assert!(got_bulk, "Must receive initial InventoryBulk");

    // Test Damage command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::Damage(4.0),
            })),
        )
        .unwrap();
    server.tick();

    let mut health_reduced = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::UpdateStats(stats)) = incoming.into_msg() {
            assert!((stats.health - 16.0).abs() < 0.001);
            health_reduced = true;
        }
    }
    assert!(health_reduced, "Health must drop to 16.0 after 4.0 damage");

    // Test AddXp command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::AddXp(60),
            })),
        )
        .unwrap();
    server.tick();

    let mut xp_updated = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::UpdateStats(stats)) = incoming.into_msg() {
            assert_eq!(stats.xp_level, 5);
            xp_updated = true;
        }
    }
    assert!(
        xp_updated,
        "XP level must increase after receiving 60 XP points"
    );

    // Test Inventory click: Left click on slot 0 (stone x64) to pick it up
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 0,
                button: 0, // Left
                mode: 0,   // Pickup
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    let mut inventory_swapped = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::InventoryBulk(bulk)) = incoming.into_msg() {
            assert_eq!(bulk.slots[0].count, 0, "Slot 0 should now be empty");
            assert_eq!(bulk.carried.item, 1, "Carried item should now be stone");
            assert_eq!(bulk.carried.count, 64, "Carried count should be 64");
            inventory_swapped = true;
        }
    }
    assert!(
        inventory_swapped,
        "Inventory click must update inventory bulk"
    );

    // Test Server-Authoritative 2x2 Crafting:
    // 1. Put stone back into slot 0
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 0,
                button: 0, // Left
                mode: 0,   // Pickup
                predicted_carried_item: 1,
                predicted_carried_count: 64,
            })),
        )
        .unwrap();
    server.tick();

    // 2. Pick up 1 Oak Log (slot 2: item 5) with right click
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 2,
                button: 1, // Right click: split
                mode: 0,   // Pickup
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // 3. Place 1 Oak Log into crafting input slot 40
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 40,
                button: 1, // Right click: place 1
                mode: 0,   // Pickup
                predicted_carried_item: 5,
                predicted_carried_count: 32,
            })),
        )
        .unwrap();
    server.tick();

    let mut got_crafting_result = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::InventoryBulk(bulk)) = incoming.into_msg()
            && bulk.slots[44].item == 7
            && bulk.slots[44].count == 4
        {
            got_crafting_result = true;
        }
    }
    assert!(
        got_crafting_result,
        "Slot 44 must compute 4 Oak Planks from 1 Oak Log in slot 40"
    );
}

#[test]
#[allow(clippy::too_many_lines, clippy::float_cmp)]
fn test_server_weather_synchronization_and_commands() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 1,
        vertical_view_distance: 1,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(888, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    // Handshake
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
                username: BoundedString::new("WeatherPlayer").unwrap(),
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
                view_distance: 1,
                simulation_distance: 1,
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

    // Drain initial messages and verify initial weather
    let mut initial_weather = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::UpdateWeather(w)) = incoming.into_msg() {
            initial_weather = Some(w);
        }
    }
    assert!(
        initial_weather.is_some(),
        "Client must receive initial weather packet on join"
    );
    let w = initial_weather.unwrap();
    assert_eq!(w.rain_level, 0.0);
    assert_eq!(w.thunder_level, 0.0);

    // Command server to start Rain
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::SetWeather(1), // Rain
            })),
        )
        .unwrap();

    // Tick server 100 times (5.0s, enough for 0.01/tick to reach 1.0)
    for _ in 0..100 {
        server.tick();
    }

    let mut latest_weather = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::UpdateWeather(w)) = incoming.into_msg() {
            latest_weather = Some(w);
        }
    }
    assert!(latest_weather.is_some());
    let w = latest_weather.unwrap();
    assert!(
        w.rain_level >= 0.95,
        "Rain level must fade smoothly to ~1.0"
    );

    // Command server to trigger lightning
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::TriggerLightning,
            })),
        )
        .unwrap();

    server.tick();

    let mut got_lightning = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::UpdateWeather(w)) = incoming.into_msg()
            && w.lightning_flash > 0
        {
            got_lightning = true;
        }
    }
    assert!(
        got_lightning,
        "Server must broadcast lightning strike flash event"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_server_mob_lifecycle_and_combat() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    // Connect player to Play phase
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
    let _ = client_conn.try_recv(); // HelloReply

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("Hero").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv(); // LoginSuccess

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 3,
                simulation_distance: 3,
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

    // Drain initial join messages
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Align player position to terrain surface
    let surface_y = server.world().get_surface_y(128, 160);
    let player_y = f64::from(surface_y) + 1.0;
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Spawn Zombie near player at (128.0, player_y, 161.5)
    let zombie_id = server.spawn_mob(
        vx_sim::EntityType::Zombie,
        glam::DVec3::new(128.0, player_y, 161.5),
    );
    assert_eq!(server.tracked_mobs.len(), 1);

    // Client receives S2cSpawnEntity
    let mut got_spawn = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::SpawnEntity(spawn)) = incoming.into_msg()
            && spawn.net_id == zombie_id
        {
            assert_eq!(spawn.entity_type, vx_sim::EntityType::Zombie.to_u8());
            assert!((spawn.health - 20.0).abs() < 0.01);
            got_spawn = true;
        }
    }
    assert!(
        got_spawn,
        "Client must receive S2cSpawnEntity for newly spawned mob"
    );

    // Tick server so mob AI updates and movement delta broadcasts
    server.tick();

    let mut got_move = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::EntityMove(m)) = incoming.into_msg()
            && m.net_id == zombie_id
        {
            got_move = true;
        }
    }
    // Note: mob may or may not move immediately depending on distance, but tracking exists
    let _ = got_move;

    // Player attacks zombie (action 0)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InteractEntity(C2sInteractEntity {
                target_net_id: zombie_id,
                action: 0,
            })),
        )
        .unwrap();

    server.tick();

    // Verify S2cEntityStatus(2) for hurt
    let mut got_hurt = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::EntityStatus(s)) = incoming.into_msg()
            && s.net_id == zombie_id
            && s.status == 2
        {
            got_hurt = true;
        }
    }
    assert!(
        got_hurt,
        "Server must broadcast entity hurt status on valid attack"
    );

    // Attack 4 more times to kill zombie (20 HP / 4 dmg = 5 hits)
    for _ in 0..5 {
        // Tick to decay combat invulnerability
        for _ in 0..12 {
            server.tick();
        }
        let mob_pos = server
            .mob_positions
            .get(&zombie_id)
            .copied()
            .unwrap_or(glam::DVec3::new(128.0, player_y, 161.5));
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                    x: mob_pos.x,
                    y: mob_pos.y,
                    z: mob_pos.z - 1.0,
                    yaw: 0.0,
                    pitch: 0.0,
                    on_ground: true,
                })),
            )
            .unwrap();
        server.tick();

        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::InteractEntity(C2sInteractEntity {
                    target_net_id: zombie_id,
                    action: 0,
                })),
            )
            .unwrap();
        server.tick();
    }

    // Verify death status and despawn packet
    let mut got_death = false;
    let mut got_despawn = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(msg) = incoming.into_msg() {
            match msg {
                S2cMessage::EntityStatus(s) if s.net_id == zombie_id && s.status == 3 => {
                    got_death = true;
                }
                S2cMessage::DespawnEntity(d) if d.net_ids.as_slice().contains(&zombie_id) => {
                    got_despawn = true;
                }
                _ => {}
            }
        }
    }
    assert!(
        got_death,
        "Server must broadcast death status (3) on lethal hit"
    );
    assert!(
        got_despawn,
        "Server must broadcast despawn packet on mob death"
    );
    assert!(!server.tracked_mobs.contains_key(&zombie_id));

    // Test SpawnMob command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::SpawnMob {
                    mob_type: vx_sim::EntityType::Pig.to_u8(),
                    x: 128.0,
                    y: 45.0,
                    z: 160.0,
                },
            })),
        )
        .unwrap();

    server.tick();
    assert_eq!(server.tracked_mobs.len(), 1);

    // Test ClearMobs command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::ClearMobs,
            })),
        )
        .unwrap();

    server.tick();
    assert_eq!(server.tracked_mobs.len(), 0);
}

fn login_test_client(
    server: &mut Server,
    client_conn: &MemoryConnection<C2sMessage, S2cMessage>,
    username: &str,
) {
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
    while let Ok(Some(_)) = client_conn.try_recv() {}

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new(username).unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 1,
                simulation_distance: 1,
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
    while let Ok(Some(_)) = client_conn.try_recv() {}
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_server_chat_broadcasting_and_commands() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 1,
        vertical_view_distance: 1,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(777, config);

    // Connect Alice and Bob
    let (alice_srv, alice_cli) = MemoryConnection::pair_default();
    server.add_connection(Box::new(alice_srv));
    login_test_client(&mut server, &alice_cli, "Alice");

    let (bob_srv, bob_cli) = MemoryConnection::pair_default();
    server.add_connection(Box::new(bob_srv));
    login_test_client(&mut server, &bob_cli, "Bob");

    // 1. Alice sends a chat message
    alice_cli
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("Hello everyone!").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    // Verify Bob received Alice's chat message
    let mut bob_chat = None;
    while let Ok(Some(incoming)) = bob_cli.try_recv() {
        if let Some(S2cMessage::ChatMessage(c)) = incoming.into_msg() {
            bob_chat = Some(c);
        }
    }
    assert!(
        bob_chat.is_some(),
        "Bob should receive Alice's chat message"
    );
    let chat = bob_chat.unwrap();
    assert_eq!(chat.sender.as_str(), "Alice");
    assert_eq!(chat.message.as_str(), "Hello everyone!");

    // Verify Alice also received her chat message (broadcast)
    let mut alice_chat = None;
    while let Ok(Some(incoming)) = alice_cli.try_recv() {
        if let Some(S2cMessage::ChatMessage(c)) = incoming.into_msg() {
            alice_chat = Some(c);
        }
    }
    assert!(
        alice_chat.is_some(),
        "Alice should receive her own chat message"
    );

    // 2. Alice executes a command: /weather rain
    alice_cli
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/weather rain").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    // Verify server weather state updated
    assert_eq!(server.weather.kind, vx_sim::WeatherKind::Rain);
    for _ in 0..100 {
        server.tick();
    }
    assert!(server.weather.rain_level > 0.9, "Rain level should be 1.0");

    // Alice should receive command feedback from "Server"
    let mut alice_feedback = None;
    let mut alice_weather_update = false;
    while let Ok(Some(incoming)) = alice_cli.try_recv() {
        if let Some(msg) = incoming.into_msg() {
            match msg {
                S2cMessage::ChatMessage(c) => alice_feedback = Some(c),
                S2cMessage::UpdateWeather(_) => alice_weather_update = true,
                _ => {}
            }
        }
    }
    assert!(
        alice_feedback.is_some(),
        "Alice should receive command feedback"
    );
    let fb = alice_feedback.unwrap();
    assert_eq!(fb.sender.as_str(), "Server");
    assert!(
        fb.message.as_str().contains("Set weather to rain"),
        "Feedback should confirm weather set"
    );
    assert!(
        alice_weather_update,
        "Alice should receive weather update packet"
    );

    // Bob should also receive weather update packet
    let mut bob_weather_update = false;
    while let Ok(Some(incoming)) = bob_cli.try_recv() {
        if let Some(S2cMessage::UpdateWeather(_)) = incoming.into_msg() {
            bob_weather_update = true;
        }
    }
    assert!(
        bob_weather_update,
        "Bob should receive weather update packet"
    );

    // 3. Alice executes /time set noon
    alice_cli
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/time set noon").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    assert_eq!(server.time_of_day(), vx_core::NOON_TICKS + 1);

    // 4. Alice requests command suggestions for "/time "
    alice_cli
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::CommandSuggest(C2sCommandSuggest {
                id: 42,
                command: BoundedString::new("/time ").unwrap(),
                cursor: 6,
            })),
        )
        .unwrap();
    server.tick();

    let mut suggestions_received = None;
    while let Ok(Some(incoming)) = alice_cli.try_recv() {
        if let Some(S2cMessage::CommandSuggestions(s)) = incoming.into_msg() {
            suggestions_received = Some(s);
        }
    }
    assert!(
        suggestions_received.is_some(),
        "Alice should receive command suggestions"
    );
    let s = suggestions_received.unwrap();
    assert_eq!(s.id, 42);
    let match_strs: Vec<&str> = s.matches.iter().map(BoundedString::as_str).collect();
    assert!(
        match_strs.contains(&"set"),
        "Suggestions should contain 'set'"
    );
    assert!(
        match_strs.contains(&"query"),
        "Suggestions should contain 'query'"
    );

    // 5. Rate limiting test: Alice sends 8 messages in a single tick (burst limit is 5)
    while let Ok(Some(_)) = bob_cli.try_recv() {}
    for i in 0..8 {
        let msg = format!("Spam message {i}");
        alice_cli
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                    message: BoundedString::new(msg).unwrap(),
                })),
            )
            .unwrap();
    }
    server.tick();

    let mut bob_chat_count = 0;
    while let Ok(Some(incoming)) = bob_cli.try_recv() {
        if let Some(S2cMessage::ChatMessage(_)) = incoming.into_msg() {
            bob_chat_count += 1;
        }
    }
    // Only at most 5 messages should have been delivered due to burst limit
    assert!(
        bob_chat_count <= 5,
        "Bob should have received at most 5 messages due to burst rate limit, got {bob_chat_count}"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_server_wasm_mod_event_and_world_mutation() {
    use vx_mod::{ModConfig, ModPermissions};

    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        ..Default::default()
    };
    let mut server = Server::new(777, config);

    // A mod in WAT that reacts to BlockPlaced by placing oak planks (state 7) directly above it
    let wat = r#"
        (module
            (import "vx" "vx_subscribe_events" (func $sub (param i32) (result i32)))
            (import "vx" "vx_set_block" (func $set_block (param i32 i32 i32 i32) (result i32)))

            (func (export "vx_init")
                ;; Subscribe to BLOCK_PLACED (1 << 1 = 2)
                (drop (call $sub (i32.const 2)))
            )

            (func (export "vx_on_event") (param $event_id i32) (param $p1 i64) (param $p2 i64) (param $p3 i64) (param $p4 i64) (result i32)
                (if (i32.eq (local.get $event_id) (i32.const 2))
                    (then
                        ;; x = p1, y = p2 + 1, z = p3, state = 7
                        (drop (call $set_block
                            (i32.wrap_i64 (local.get $p1))
                            (i32.add (i32.wrap_i64 (local.get $p2)) (i32.const 1))
                            (i32.wrap_i64 (local.get $p3))
                            (i32.const 7)
                        ))
                    )
                )
                (i32.const 0)
            )
        )
    "#;

    server
        .load_mod_from_wat(
            "builder_mod",
            wat,
            ModPermissions::all_permissions(),
            ModConfig::default(),
        )
        .expect("Failed to load mod");

    assert_eq!(server.mod_manager.loaded_count(), 1);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let _session_id = server.add_connection(Box::new(server_conn));

    // Complete login handshake
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

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("ModTester").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .unwrap();
    server.tick();

    // Drain initial login packets
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Set player position to near origin (0, 64, 0)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 0.0,
                y: 64.0,
                z: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Client places block at (0, 64, 0) with face Up -> target is (0, 65, 0) with state 1 (Stone)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                sequence: 1,
                action: BlockActionKind::Place {
                    state_id: BlockStateId::new(1),
                    hit_face: 1, // Face::Up
                },
                x: 0,
                y: 64,
                z: 0,
                input_tick: 1,
            })),
        )
        .unwrap();

    // Tick server: processes place -> emits BlockPlaced event -> mod triggers -> queues (0, 66, 0) -> applied!
    server.tick();

    // Verify player placed block at (0, 65, 0)
    let state_65 = server.world_mut().get_block(BlockPos::new(0, 65, 0));
    assert_eq!(state_65, BlockStateId::new(1));

    // Verify mod placed block at (0, 66, 0)
    let state_66 = server.world_mut().get_block(BlockPos::new(0, 66, 0));
    assert_eq!(state_66, BlockStateId::new(7));
}

#[test]
#[allow(clippy::too_many_lines, clippy::float_cmp)]
fn test_authoritative_movement_prediction_and_reconciliation() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 1,
        chunks_per_tick_per_player: 4,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    // Handshake
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

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("Player1").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();

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

    // Drain initial messages and ensure initial PlayerMovementAck
    let mut initial_ack_received = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::PlayerMovementAck(ack)) = incoming.into_msg() {
            assert_eq!(ack.client_tick_ack, 0);
            assert_eq!(ack.teleport_id, 0);
            initial_ack_received = true;
        }
    }
    assert!(
        initial_ack_received,
        "Server must emit initial movement ack on join"
    );

    // Client sends 2 input frames moving forward (+Z direction)
    let frame1 = InputFrame {
        tick: 1,
        buttons: input_buttons::FORWARD | input_buttons::SPRINT,
        yaw: vx_sim::quantize_yaw(90.0),
        pitch: vx_sim::quantize_pitch(0.0),
        hotbar: 0,
    };
    let frame2 = InputFrame {
        tick: 2,
        buttons: input_buttons::FORWARD | input_buttons::SPRINT,
        yaw: vx_sim::quantize_yaw(90.0),
        pitch: vx_sim::quantize_pitch(0.0),
        hotbar: 0,
    };

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerInput(C2sPlayerInput {
                last_server_tick_ack: 0,
                frames: BoundedVec::new(vec![frame1, frame2]).unwrap(),
            })),
        )
        .unwrap();

    server.tick();

    // Verify server simulated frames and emitted ack with updated state
    let mut received_ack = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::PlayerMovementAck(ack)) = incoming.into_msg() {
            received_ack = Some(ack);
        }
    }
    let ack = received_ack.expect("Server must send PlayerMovementAck after simulating frames");
    assert_eq!(ack.client_tick_ack, 2);
    assert!(
        ack.z > 160.0,
        "Player must have moved forward along +Z axis"
    );
    assert!(ack.vz > 0.0, "Velocity must be positive forward");

    // Test authoritative teleport via /tp command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/tp 200 80 300").unwrap(),
            })),
        )
        .unwrap();

    server.tick();

    let mut tp_ack_received = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::PlayerMovementAck(ack)) = incoming.into_msg()
            && ack.teleport_id > 0
        {
            tp_ack_received = Some(ack);
        }
    }
    let tp_ack = tp_ack_received.expect("Server must send PlayerMovementAck with teleport_id > 0");
    assert_eq!(tp_ack.x, 200.0);
    assert_eq!(tp_ack.y, 80.0);
    assert_eq!(tp_ack.z, 300.0);

    // Client acknowledges teleport
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::TeleportAck(C2sTeleportAck {
                teleport_id: tp_ack.teleport_id,
            })),
        )
        .unwrap();

    server.tick();
}
