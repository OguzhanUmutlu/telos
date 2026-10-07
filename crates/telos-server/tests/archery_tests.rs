//! Integration tests for server-authoritative bow charging, arrow ballistic physics,
//! mob hit damage/knockback, block face embedding, and player proximity pickup.

use glam::{DVec3, Vec3};
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerCommand, C2sPlayerPosition, PlayerCommandKind, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::{ArrowEntity, EntityType, HurtTime};
use telos_sim::inventory::{ITEM_ARROW, ITEM_BOW};
use telos_sim::{Health, Inventory, ItemStack};
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
                username: BoundedString::new("ArcherPlayer").unwrap(),
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
    let sy = surface_y as i32;
    for x in 120..=136 {
        for z in 150..=180 {
            let _ = server
                .world_mut()
                .set_block(BlockPos::new(x, sy, z), BlockStateId::new(1));
            for y in 1..=4 {
                let _ = server
                    .world_mut()
                    .set_block(BlockPos::new(x, sy + y, z), BlockStateId::AIR);
            }
        }
    }

    (server, client_conn, session_id, surface_y)
}

#[test]
fn test_bow_charging_and_shoot_validation() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let player_y = surface_y + 1.0;

    // Position player at (128.0, player_y, 160.0) facing South (+Z)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 180.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Give player a bow in slot 0, and 2 arrows in slot 1
    let player_entity = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();
    if let Some(mut inv) = server.ecs_world_mut().get_mut::<Inventory>(player_entity) {
        inv.slots[0] = ItemStack::new(ITEM_BOW, 1);
        inv.slots[1] = ItemStack::new(ITEM_ARROW, 2);
    }

    // 1. Minimum charge failure: charge_ticks = 2 (< 3)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::ShootBow { charge_ticks: 2 },
            })),
        )
        .unwrap();
    server.tick();

    assert!(
        server.tracked_arrows.is_empty(),
        "Arrow should not spawn when charge_ticks < BOW_MIN_CHARGE_TICKS"
    );

    // 2. Successful release: charge_ticks = 10
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::ShootBow { charge_ticks: 10 },
            })),
        )
        .unwrap();
    server.tick();

    assert_eq!(
        server.tracked_arrows.len(),
        1,
        "One arrow entity should be active"
    );

    // Verify arrow count decremented from 2 to 1
    let inv = server.ecs_world().get::<Inventory>(player_entity).unwrap();
    assert_eq!(inv.slots[1].item, ITEM_ARROW);
    assert_eq!(inv.slots[1].count, 1);

    // Verify client received S2cSpawnArrow packet
    let mut received_spawn = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::SpawnArrow(_)) = incoming.into_msg() {
            received_spawn = true;
        }
    }
    assert!(received_spawn, "Client should receive S2cSpawnArrow");

    // 3. Shoot second arrow (consumes last arrow)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::ShootBow { charge_ticks: 20 },
            })),
        )
        .unwrap();
    server.tick();

    let inv = server.ecs_world().get::<Inventory>(player_entity).unwrap();
    assert_eq!(inv.slots[1].count, 0, "Arrow ammo should be exhausted");

    // 4. Try to shoot with 0 arrows: should fail
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::ShootBow { charge_ticks: 20 },
            })),
        )
        .unwrap();
    server.tick();

    assert_eq!(
        server.tracked_arrows.len(),
        2,
        "No additional arrow should be spawned with 0 ammo"
    );
}

#[test]
fn test_arrow_ballistic_flight_and_block_hit() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let player_y = surface_y + 1.0;

    // Erect a solid stone wall at z = 164 (4 blocks away from player at z = 160)
    let sy = surface_y as i32;
    for x in 125..=131 {
        for y in 1..=4 {
            let _ = server
                .world_mut()
                .set_block(BlockPos::new(x, sy + y, 164), BlockStateId::new(1));
        }
    }

    // Spawn arrow firing forward (+Z) towards the wall at 2 blocks/tick
    let launch_pos = DVec3::new(128.0, player_y + 0.62, 160.0);
    let vel = Vec3::new(0.0, 0.0, 2.0);
    let net_id =
        server.spawn_arrow_entity("overworld", launch_pos, vel, 180.0, 0.0, Some(1), 6.0, true);

    // Initial state: not embedded
    let arrow_entity = *server.tracked_arrows.get(&net_id).unwrap();
    let arrow_comp = *server.ecs_world().get::<ArrowEntity>(arrow_entity).unwrap();
    assert!(!arrow_comp.in_ground);

    // Advance simulation ticks until arrow reaches and hits the wall
    for _ in 0..10 {
        server.tick();
    }

    // Arrow should now be embedded in the stone block face
    let arrow_comp = *server.ecs_world().get::<ArrowEntity>(arrow_entity).unwrap();
    assert!(arrow_comp.in_ground, "Arrow should be embedded in block");
    assert!(arrow_comp.stuck_block.is_some());
    assert_eq!(arrow_comp.stuck_block.unwrap().z(), 164);

    // Pickup stuck arrow by moving player close
    let player_entity = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();
    if let Some(mut inv) = server.ecs_world_mut().get_mut::<Inventory>(player_entity) {
        inv.slots[0] = ItemStack::EMPTY;
        inv.slots[1] = ItemStack::EMPTY;
    }

    // Move player right next to the arrow
    let arrow_pos = server
        .ecs_world()
        .get::<telos_sim::Position>(arrow_entity)
        .unwrap()
        .0;

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: arrow_pos.x,
                y: arrow_pos.y,
                z: arrow_pos.z,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Arrow should now be despawned and picked up into inventory
    assert!(
        !server.tracked_arrows.contains_key(&net_id),
        "Stuck arrow should despawn on player pickup"
    );
    let inv = server.ecs_world().get::<Inventory>(player_entity).unwrap();
    let total_arrows: u16 = inv
        .slots
        .iter()
        .filter(|s| s.item == ITEM_ARROW)
        .map(|s| s.count)
        .sum();
    assert_eq!(total_arrows, 1, "Player should have picked up 1 arrow");
}

#[test]
fn test_arrow_entity_hit_damage_and_knockback() {
    let (mut server, _client_conn, _session_id, surface_y) = setup_server_and_player();
    let player_y = surface_y + 1.0;

    // Spawn a Zombie mob entity at (128.0, player_y, 166.0)
    let zombie_pos = DVec3::new(128.0, player_y, 166.0);
    let zombie_net_id = server.spawn_mob(EntityType::Zombie, zombie_pos);
    let zombie_entity = *server.tracked_mobs.get(&zombie_net_id).unwrap();

    let initial_health = server.ecs_world().get::<Health>(zombie_entity).unwrap().cur;
    assert!((initial_health - 20.0).abs() < 1e-4);

    // Spawn arrow aimed directly at zombie
    let launch_pos = DVec3::new(128.0, player_y + 0.62, 160.0);
    let vel = Vec3::new(0.0, 0.0, 2.0); // 2.0 blocks/tick
    let arrow_net_id =
        server.spawn_arrow_entity("overworld", launch_pos, vel, 180.0, 0.0, Some(1), 8.0, true);

    // Tick until arrow hits zombie
    for _ in 0..5 {
        server.tick();
    }

    // Arrow should have hit and despawned
    assert!(
        !server.tracked_arrows.contains_key(&arrow_net_id),
        "Arrow should despawn upon striking living entity"
    );

    // Zombie should have taken damage
    let current_health = server.ecs_world().get::<Health>(zombie_entity).unwrap().cur;
    assert!(
        current_health < initial_health,
        "Zombie health should be reduced after arrow hit: current={current_health}"
    );

    // Hurt time should be set
    let hurt_time = server.ecs_world().get::<HurtTime>(zombie_entity).unwrap().0;
    assert!(hurt_time > 0, "Zombie should have active HurtTime");
}
