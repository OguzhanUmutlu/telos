//! Integration tests for server-authoritative mob pathfinding and 3D navigation.

use glam::DVec3;
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerPosition, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::{EntityType, PathFollower, Position};
use telos_voxel::state::BlockStateId;

fn setup_server_and_player() -> (Server, MemoryConnection<C2sMessage, S2cMessage>, f64) {
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let _session_id = server.add_connection(Box::new(server_conn));

    // Handshake: Hello
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

    // LoginStart
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("PathPlayer").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    // Drain login messages
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // ClientSettings + ConfigAck
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

    // Drain initial join and chunk messages
    while let Ok(Some(_)) = client_conn.try_recv() {}

    let surface_y = f64::from(server.world().get_surface_y(128, 160));
    (server, client_conn, surface_y)
}

#[test]
fn test_server_zombie_pathfinding_around_wall() {
    let (mut server, client_conn, surface_y) = setup_server_and_player();
    let sy = surface_y as i32;

    // 1. Prepare flat test arena at (120..=136, 155..=170)
    let stone = BlockStateId::new(1);
    let air = BlockStateId::AIR;
    for x in 120..=136 {
        for z in 155..=170 {
            let _ = server.world_mut().set_block(BlockPos::new(x, sy, z), stone);
            for y in 1..=4 {
                let _ = server
                    .world_mut()
                    .set_block(BlockPos::new(x, sy + y, z), air);
            }
        }
    }

    // 2. Build obstacle wall at z = 163 from x = 126 to x = 130, height 2 blocks
    for x in 126..=130 {
        let _ = server
            .world_mut()
            .set_block(BlockPos::new(x, sy + 1, 163), stone);
        let _ = server
            .world_mut()
            .set_block(BlockPos::new(x, sy + 2, 163), stone);
    }

    // 3. Position player at (128.0, sy + 1, 160.0)
    let player_y = surface_y + 1.0;
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

    // 4. Spawn Zombie at (128.0, sy + 1, 166.0) - behind the wall
    let zombie_id = server.spawn_mob(EntityType::Zombie, DVec3::new(128.0, player_y, 166.0));
    let zombie_entity = *server.tracked_mobs.get(&zombie_id).unwrap();

    // 5. Tick server to run pathfinding
    server.tick();

    // 6. Inspect PathFollower component on the zombie
    let follower = server
        .ecs_world()
        .get::<PathFollower>(zombie_entity)
        .expect("Zombie must have PathFollower");

    assert!(
        follower.path.is_some(),
        "Zombie should calculate a path towards player"
    );

    let path = follower.path.as_ref().unwrap();
    assert!(!path.waypoints().is_empty(), "Path must have waypoints");

    // Verify none of the waypoints walk directly through the solid wall at z = 163
    for wp in path.waypoints() {
        if wp.z() == 163 {
            assert!(
                wp.x() < 126 || wp.x() > 130,
                "Waypoint {wp:?} must not intersect the obstacle wall at x in 126..=130, z=163"
            );
        }
    }

    // 7. Tick multiple times and verify zombie steps around the wall
    let initial_pos = server.ecs_world().get::<Position>(zombie_entity).unwrap().0;

    for _ in 0..15 {
        server.tick();
    }

    let moved_pos = server.ecs_world().get::<Position>(zombie_entity).unwrap().0;

    assert!(
        moved_pos.distance_squared(initial_pos) > 0.05,
        "Zombie must move along its navigation path (initial: {initial_pos}, moved: {moved_pos})"
    );
}

#[test]
fn test_server_zombie_jump_up_elevation() {
    let (mut server, client_conn, surface_y) = setup_server_and_player();
    let sy = surface_y as i32;

    let stone = BlockStateId::new(1);
    let air = BlockStateId::AIR;
    // Prepare base ground
    for x in 125..=131 {
        for z in 155..=170 {
            let _ = server.world_mut().set_block(BlockPos::new(x, sy, z), stone);
            for y in 1..=4 {
                let _ = server
                    .world_mut()
                    .set_block(BlockPos::new(x, sy + y, z), air);
            }
        }
    }

    // Elevated terrace where player stands: z in 155..=162 at sy + 1
    for x in 125..=131 {
        for z in 155..=162 {
            let _ = server
                .world_mut()
                .set_block(BlockPos::new(x, sy + 1, z), stone);
        }
    }

    // Position player on elevated terrace at (128.0, sy + 2, 160.0)
    let player_y = surface_y + 2.0;
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

    // Spawn Zombie on lower ground at (128.0, sy + 1, 166.0)
    let zombie_id = server.spawn_mob(
        EntityType::Zombie,
        DVec3::new(128.0, surface_y + 1.0, 166.0),
    );
    let zombie_entity = *server.tracked_mobs.get(&zombie_id).unwrap();

    server.tick();

    let follower = server
        .ecs_world()
        .get::<PathFollower>(zombie_entity)
        .expect("Zombie must have PathFollower");

    assert!(
        follower.path.is_some(),
        "Zombie should find path up terrace"
    );
    let path = follower.path.as_ref().unwrap();

    // Verify path contains an elevated waypoint at sy + 2
    let has_step_up = path.waypoints().iter().any(|wp| wp.y() == sy + 2);
    assert!(has_step_up, "Path must climb step up to sy + 2");

    // Tick server to allow zombie to advance and jump up
    for _ in 0..25 {
        server.tick();
    }

    let current_pos = server.ecs_world().get::<Position>(zombie_entity).unwrap().0;

    assert!(
        current_pos.z < 166.0,
        "Zombie should have advanced forward towards player (pos: {current_pos})"
    );
}
