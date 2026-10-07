//! Monster spawner proximity activation, mob generation & deactivation tests.

use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerPosition, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::EntityType;
use telos_voxel::state::BlockStateId;

fn setup_server_and_player() -> (Server, MemoryConnection<C2sMessage, S2cMessage>, u64, f64) {
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

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("SpawnerTester").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    while let Ok(Some(_)) = client_conn.try_recv() {}

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

    while let Ok(Some(_)) = client_conn.try_recv() {}

    let surface_y = f64::from(server.world().get_surface_y(128, 160));
    (server, client_conn, session_id, surface_y)
}

#[test]
fn test_monster_spawner_player_proximity_activation() {
    let (mut server, client_conn, _session_id, player_y) = setup_server_and_player();

    // Spawner position 6 blocks away from player (within 16 blocks radius)
    let spawner_pos = BlockPos::new(128, player_y as i32, 166);
    let spawner_id = server
        .registries
        .block_registry()
        .get(&telos_core::ident::Identifier::new("telos", "monster_spawner").unwrap())
        .unwrap()
        .default_state();

    server.world_mut().set_block(spawner_pos, spawner_id);
    server.register_spawner(spawner_pos, EntityType::Zombie);

    // Initial spawner state has delay = 100
    assert!(server.active_spawners.contains_key(&spawner_pos));
    let initial_delay = server
        .active_spawners
        .get(&spawner_pos)
        .unwrap()
        .spawn_delay;
    assert_eq!(initial_delay, 100);

    // Position player at (128.0, player_y, 160.0) - 6 blocks away from spawner
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

    // Tick server 10 times - delay must count down since player is nearby
    for _ in 0..10 {
        server.tick();
    }

    let updated_delay = server
        .active_spawners
        .get(&spawner_pos)
        .unwrap()
        .spawn_delay;
    assert!(
        updated_delay < initial_delay,
        "Spawner delay must count down when player is within 16 blocks"
    );

    // Force delay to 1 tick and tick server to trigger spawn
    server
        .active_spawners
        .get_mut(&spawner_pos)
        .unwrap()
        .spawn_delay = 1;

    // Ensure space around spawner is suitable for spawning: solid stone below, air candidate
    let cand_pos = BlockPos::new(128, player_y as i32, 165);
    let below_pos = BlockPos::new(128, player_y as i32 - 1, 165);
    let stone_id = server
        .registries
        .block_registry()
        .get(&telos_core::ident::Identifier::new("telos", "stone").unwrap())
        .unwrap()
        .default_state();
    server.world_mut().set_block(cand_pos, BlockStateId::AIR);
    server.world_mut().set_block(below_pos, stone_id);

    server.tick();

    // Spawner should reset delay back to 200
    let delay_after = server
        .active_spawners
        .get(&spawner_pos)
        .unwrap()
        .spawn_delay;
    assert_eq!(
        delay_after, 200,
        "Spawner must reset countdown after spawn attempt"
    );

    let _ = client_conn;
}

#[test]
fn test_monster_spawner_player_out_of_range() {
    let (mut server, client_conn, _session_id, player_y) = setup_server_and_player();

    // Move player far away (> 16 blocks, e.g. 100 blocks)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 300.0,
                y: player_y,
                z: 300.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Spawner at original position (128, player_y, 160)
    let spawner_pos = BlockPos::new(128, player_y as i32, 160);
    let spawner_id = server
        .registries
        .block_registry()
        .get(&telos_core::ident::Identifier::new("telos", "monster_spawner").unwrap())
        .unwrap()
        .default_state();

    server.world_mut().set_block(spawner_pos, spawner_id);
    server.register_spawner(spawner_pos, EntityType::Zombie);

    // Advance 20 ticks
    for _ in 0..20 {
        server.tick();
    }

    // Delay must NOT have changed since player is far away
    let delay = server
        .active_spawners
        .get(&spawner_pos)
        .unwrap()
        .spawn_delay;
    assert_eq!(
        delay, 100,
        "Spawner delay must not count down when no player is within 16 blocks"
    );
}

#[test]
fn test_monster_spawner_broken_deactivation() {
    let (mut server, _client_conn, _session_id, player_y) = setup_server_and_player();

    let spawner_pos = BlockPos::new(128, player_y as i32, 162);
    let spawner_id = server
        .registries
        .block_registry()
        .get(&telos_core::ident::Identifier::new("telos", "monster_spawner").unwrap())
        .unwrap()
        .default_state();

    server.world_mut().set_block(spawner_pos, spawner_id);
    server.register_spawner(spawner_pos, EntityType::Zombie);
    assert!(server.active_spawners.contains_key(&spawner_pos));

    // Breaking the spawner by setting it to AIR
    server.world_mut().set_block(spawner_pos, BlockStateId::AIR);

    // Next server tick will detect the missing block and remove the spawner
    server.tick();

    assert!(
        !server.active_spawners.contains_key(&spawner_pos),
        "Broken spawner must be removed from active spawners"
    );
}
