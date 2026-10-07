//! Integration tests for server-authoritative dropped item entities,
//! physics simulation, proximity stack merging, and inventory pickup.

use glam::{DVec3, Vec3};
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerCommand, C2sPlayerPosition, PlayerCommandKind, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::ItemEntity;
use telos_sim::{Inventory, ItemStack};
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
                username: BoundedString::new("ItemPlayer").unwrap(),
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
        for z in 150..=175 {
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
fn test_server_drop_item_command() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
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

    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();

    // Populate hotbar slot 0 with 5 stone blocks (item_id 1)
    {
        let mut inv = server
            .ecs_world_mut()
            .get_mut::<Inventory>(player_ecs)
            .unwrap();
        inv.slots[0] = ItemStack::new(1, 5);
    }

    // Drop single item (entire_stack: false)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::DropItem {
                    entire_stack: false,
                },
            })),
        )
        .unwrap();
    server.tick();

    // Inventory slot 0 should now contain 4 items
    {
        let inv = server.ecs_world().get::<Inventory>(player_ecs).unwrap();
        assert_eq!(inv.slots[0].count, 4);
        assert_eq!(inv.slots[0].item, 1);
    }

    // Exactly 1 dropped item entity should exist in the world
    assert_eq!(server.tracked_items.len(), 1);
    let (&_net_id_1, &ecs_ent_1) = server.tracked_items.iter().next().unwrap();
    let item_comp_1 = *server.ecs_world().get::<ItemEntity>(ecs_ent_1).unwrap();
    assert_eq!(item_comp_1.stack.count, 1);
    assert_eq!(item_comp_1.stack.item, 1);

    // Drop remaining entire stack (entire_stack: true)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::DropItem { entire_stack: true },
            })),
        )
        .unwrap();
    server.tick();

    // Inventory slot 0 should now be empty
    {
        let inv = server.ecs_world().get::<Inventory>(player_ecs).unwrap();
        assert_eq!(inv.slots[0].count, 0);
    }

    // Two item entities should exist (or merged if close enough)
    let total_dropped: u16 = server
        .tracked_items
        .values()
        .map(|&e| server.ecs_world().get::<ItemEntity>(e).unwrap().stack.count)
        .sum();
    assert_eq!(total_dropped, 5, "Total dropped items must equal 5");
}

#[test]
fn test_server_item_pickup() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
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

    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();

    // Spawn an item entity 0.5 blocks away from the player with pickup_delay = 5
    let net_id = server.spawn_item_entity(
        "overworld",
        DVec3::new(128.0, player_y, 160.5),
        Vec3::ZERO,
        ItemStack::new(2, 10), // 10 dirt blocks
        5,
    );
    assert_eq!(server.tracked_items.len(), 1);

    // Advance 4 ticks: pickup delay counts down (5 -> 1), item must not be collected yet
    for _ in 0..4 {
        server.tick();
    }
    assert_eq!(
        server.tracked_items.len(),
        1,
        "Item must still exist while pickup_delay > 0"
    );

    // Advance 2 more ticks: pickup delay expires (reaches 0), player collects item
    for _ in 0..2 {
        server.tick();
    }
    assert!(
        !server.tracked_items.contains_key(&net_id),
        "Item should be despawned after pickup"
    );

    // Inventory must now contain the 10 dirt blocks
    let inv = server.ecs_world().get::<Inventory>(player_ecs).unwrap();
    let has_dirt = inv.slots.iter().any(|s| s.item == 2 && s.count == 10);
    assert!(has_dirt, "Player should have received 10 dirt blocks");
}

#[test]
fn test_server_item_merging() {
    let (mut server, _client_conn, _session_id, surface_y) = setup_server_and_player();
    let y = surface_y + 1.0;

    // Spawn two matching items within 0.5 blocks (threshold is <= 1.5 blocks)
    let net_a = server.spawn_item_entity(
        "overworld",
        DVec3::new(128.0, y, 160.0),
        Vec3::ZERO,
        ItemStack::new(1, 15),
        100,
    );
    let net_b = server.spawn_item_entity(
        "overworld",
        DVec3::new(128.3, y, 160.3),
        Vec3::ZERO,
        ItemStack::new(1, 20),
        100,
    );
    assert_eq!(server.tracked_items.len(), 2);

    // Tick server to run proximity stack merging
    server.tick();

    // One of the entities should be despawned, remaining entity has combined count 35
    assert_eq!(
        server.tracked_items.len(),
        1,
        "Matching item entities should merge into 1"
    );
    let (&surviving_id, &surviving_ent) = server.tracked_items.iter().next().unwrap();
    assert!(surviving_id == net_a || surviving_id == net_b);

    let comp = *server.ecs_world().get::<ItemEntity>(surviving_ent).unwrap();
    assert_eq!(comp.stack.item, 1);
    assert_eq!(comp.stack.count, 35);

    // Spawn non-matching item entity nearby
    let _net_c = server.spawn_item_entity(
        "overworld",
        DVec3::new(128.2, y, 160.2),
        Vec3::ZERO,
        ItemStack::new(2, 5), // dirt
        100,
    );
    assert_eq!(server.tracked_items.len(), 2);

    server.tick();

    // Dissimilar items must NOT merge
    assert_eq!(
        server.tracked_items.len(),
        2,
        "Dissimilar items must not merge"
    );
}
