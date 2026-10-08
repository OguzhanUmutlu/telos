//! Integration tests for server-authoritative chest containers,
//! block entity side tables, container interactions, persistence, and auto-closing.

use glam::DVec3;
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sClientSettings, C2sConfigAck, C2sHello,
    C2sInventoryClick, C2sLoginStart, C2sMessage, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::ItemEntity;
use telos_sim::{ChestInventory, Inventory, ItemStack};
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
                username: BoundedString::new("ChestPlayer").unwrap(),
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

    {
        let session = server.get_session_mut(session_id).unwrap();
        session.position = DVec3::new(128.0, surface_y + 1.0, 160.0);
        session.move_state.pos = session.position;
    }

    (server, client_conn, session_id, surface_y)
}

#[test]
fn test_chest_right_click_open_and_reach_validation() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();

    let chest_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_chest(id))
        .expect("chest block state registered");

    let chest_pos = BlockPos::new(128, surface_y as i32 + 1, 162);
    server.world_mut().set_block(chest_pos, chest_state);

    // Drain initial login packets
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // 1. Right-click chest within reach (distance ~2.0 blocks)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: chest_pos.x(),
                y: chest_pos.y(),
                z: chest_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    let mut received_open = false;
    let mut received_block_event = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(msg) = payload.into_msg() {
            match msg {
                S2cMessage::OpenContainer(open) => {
                    assert_eq!(open.x, chest_pos.x());
                    assert_eq!(open.y, chest_pos.y());
                    assert_eq!(open.z, chest_pos.z());
                    assert_eq!(open.window_id, 1);
                    assert_eq!(open.slots.len(), 27);
                    received_open = true;
                }
                S2cMessage::BlockEvent(evt)
                    if evt.x == chest_pos.x()
                        && evt.y == chest_pos.y()
                        && evt.z == chest_pos.z() =>
                {
                    assert_eq!(evt.action, 1);
                    assert_eq!(evt.param, 1);
                    received_block_event = true;
                }
                _ => {}
            }
        }
    }
    assert!(received_open, "Expected S2cOpenContainer packet");
    assert!(received_block_event, "Expected S2cBlockEvent open packet");

    // Verify session active container is tracked
    let session = server.get_session(session_id).expect("session exists");
    assert!(session.active_container.is_some());
    assert_eq!(session.active_container.unwrap().block_pos, chest_pos);

    // 2. Far chest beyond reach (> 6.0 blocks away)
    let far_chest_pos = BlockPos::new(128, surface_y as i32 + 1, 180);
    server.world_mut().set_block(far_chest_pos, chest_state);

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: far_chest_pos.x(),
                y: far_chest_pos.y(),
                z: far_chest_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 2,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    let mut received_far_open = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::OpenContainer(open)) = payload.into_msg()
            && open.z == far_chest_pos.z()
        {
            received_far_open = true;
        }
    }
    assert!(!received_far_open, "Should not open chest beyond reach");
}

#[test]
fn test_container_item_transfer_and_shift_click() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();

    let chest_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_chest(id))
        .expect("chest block state registered");

    let chest_pos = BlockPos::new(128, surface_y as i32 + 1, 161);
    server.world_mut().set_block(chest_pos, chest_state);

    // Give player some cobblestone (item 4, count 32) in hotbar slot 0
    let entity = server.get_session(session_id).unwrap().ecs_entity.unwrap();
    {
        let mut inv = server.ecs_world_mut().get_mut::<Inventory>(entity).unwrap();
        inv.slots = [ItemStack::EMPTY; telos_sim::PLAYER_INVENTORY_SLOTS];
        inv.slots[0] = ItemStack::new(4, 32);
    }

    // Open the chest
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: chest_pos.x(),
                y: chest_pos.y(),
                z: chest_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Slot 54 in container window corresponds to hotbar slot 0
    // Perform shift-click (QuickMove) to transfer hotbar stack into chest
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 54,
                button: 0,
                mode: 1, // QuickMove
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify chest in world storage received the 32 cobblestone
    let chest_be = server
        .world()
        .get_block_entity(chest_pos)
        .expect("chest block entity exists");
    let chest_inv = ChestInventory::from_block_entity(chest_be);
    assert_eq!(chest_inv.slots[0].item, 4);
    assert_eq!(chest_inv.slots[0].count, 32);

    // Player hotbar slot 0 is now empty
    let inv = server.ecs_world().get::<Inventory>(entity).unwrap();
    assert!(inv.slots[0].is_empty());

    // Shift-click chest slot 0 (slot index 0) to transfer items back to player
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 0,
                button: 0,
                mode: 1, // QuickMove
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify items moved back to player storage
    let inv_after = server.ecs_world().get::<Inventory>(entity).unwrap();
    let total_cobble: u16 = inv_after
        .slots
        .iter()
        .filter(|s| s.item == 4)
        .map(|s| s.count)
        .sum();
    assert_eq!(total_cobble, 32);
}

#[test]
fn test_chest_persistence_across_world_save_and_reload() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 10,
        save_directory: Some(temp_dir.path().to_path_buf()),
        ..Default::default()
    };
    let mut server = Server::new(12345, config.clone());

    let chest_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_chest(id))
        .expect("chest block state registered");

    let chest_pos = BlockPos::new(128, 64, 160);
    server.world_mut().set_block(chest_pos, chest_state);

    // Insert items into chest block entity
    if let Some(be) = server.world_mut().get_block_entity_mut(chest_pos) {
        let mut inv = ChestInventory::from_block_entity(be);
        inv.slots[3] = ItemStack::new(20, 16); // 16 Diamonds
        inv.slots[12] = ItemStack::new(52, 64); // 64 Iron Ingots
        *be = inv.to_block_entity();
    }

    // Save world chunks to disk
    server.save_and_flush().expect("save chunks");

    // Reopen server with same save directory
    let mut reloaded_server = Server::new(12345, config);
    // Trigger loading chunk at chest_pos
    let _ = reloaded_server.world_mut().get_block(chest_pos);
    let be_reloaded = reloaded_server
        .world()
        .get_block_entity(chest_pos)
        .expect("reloaded chest block entity exists");
    let inv_reloaded = ChestInventory::from_block_entity(be_reloaded);

    assert_eq!(inv_reloaded.slots[3].item, 20);
    assert_eq!(inv_reloaded.slots[3].count, 16);
    assert_eq!(inv_reloaded.slots[12].item, 52);
    assert_eq!(inv_reloaded.slots[12].count, 64);
}

#[test]
fn test_container_auto_close_on_distance_exceed() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();

    let chest_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_chest(id))
        .expect("chest block state registered");

    let chest_pos = BlockPos::new(128, surface_y as i32 + 1, 161);
    server.world_mut().set_block(chest_pos, chest_state);

    // Open chest
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: chest_pos.x(),
                y: chest_pos.y(),
                z: chest_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    assert!(
        server
            .get_session(session_id)
            .unwrap()
            .active_container
            .is_some()
    );

    // Move player far away (> 5.0m)
    let session = server.get_session_mut(session_id).unwrap();
    session.position = DVec3::new(128.0, surface_y + 1.0, 180.0);
    session.move_state.pos = session.position;

    server.tick();

    // Client should receive S2cCloseContainer and S2cBlockEvent (param 0)
    let mut received_close = false;
    let mut received_close_event = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(msg) = payload.into_msg() {
            match msg {
                S2cMessage::CloseContainer(close) => {
                    assert_eq!(close.window_id, 1);
                    received_close = true;
                }
                S2cMessage::BlockEvent(evt)
                    if evt.x == chest_pos.x()
                        && evt.y == chest_pos.y()
                        && evt.z == chest_pos.z() =>
                {
                    assert_eq!(evt.action, 1);
                    assert_eq!(evt.param, 0);
                    received_close_event = true;
                }
                _ => {}
            }
        }
    }

    assert!(received_close, "Expected S2cCloseContainer packet");
    assert!(received_close_event, "Expected S2cBlockEvent close packet");
    assert!(
        server
            .get_session(session_id)
            .unwrap()
            .active_container
            .is_none()
    );
}

#[test]
fn test_chest_break_drops_stored_items() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();

    let chest_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_chest(id))
        .expect("chest block state registered");

    let chest_pos = BlockPos::new(128, surface_y as i32 + 1, 161);
    server.world_mut().set_block(chest_pos, chest_state);

    // Put 16 golden ingots into the chest
    if let Some(be) = server.world_mut().get_block_entity_mut(chest_pos) {
        let mut inv = ChestInventory::from_block_entity(be);
        inv.slots[5] = ItemStack::new(53, 16);
        *be = inv.to_block_entity();
    }

    // Open chest so player is an active viewer
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: chest_pos.x(),
                y: chest_pos.y(),
                z: chest_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Break the chest
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: chest_pos.x(),
                y: chest_pos.y(),
                z: chest_pos.z(),
                action: BlockActionKind::Break,
                sequence: 2,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify chest block is now Air
    assert!(server.world_mut().get_block(chest_pos).is_air());
    // Block entity removed
    assert!(server.world().get_block_entity(chest_pos).is_none());
    // Active container closed
    assert!(
        server
            .get_session(session_id)
            .unwrap()
            .active_container
            .is_none()
    );

    // Verify dropped item entity spawned in ECS world with the 16 golden ingots
    let mut query = server.ecs_world_mut().query::<&ItemEntity>();
    let gold_entities: Vec<&ItemEntity> = query
        .iter(server.ecs_world())
        .filter(|item| item.stack.item == 53)
        .collect();
    assert_eq!(gold_entities.len(), 1);
    assert_eq!(gold_entities[0].stack.count, 16);
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_furnace_interaction_and_combustion_ticking() {
    let (mut server, client_conn, _session_id, surface_y) = setup_server_and_player();
    let furnace_pos = BlockPos::new(128, surface_y as i32 + 1, 162);

    let furnace_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| {
            server.world().registry().is_furnace(id)
                && !server.world().registry().is_lit_furnace(id)
        })
        .expect("furnace block state registered");
    server.world_mut().set_block(furnace_pos, furnace_state);

    // Drain initial login packets
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Right click / interact with furnace
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: furnace_pos.x(),
                y: furnace_pos.y(),
                z: furnace_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify client received OpenContainer with kind 1 (Furnace)
    let mut opened = false;
    let mut properties_received = 0;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(msg) = payload.into_msg() {
            match msg {
                S2cMessage::OpenContainer(open) => {
                    assert_eq!(open.container_kind, 1);
                    assert_eq!(open.slots.len(), 3);
                    opened = true;
                }
                S2cMessage::ContainerProperty(_) => {
                    properties_received += 1;
                }
                _ => {}
            }
        }
    }
    assert!(opened, "Client must receive OpenContainer for furnace");
    assert!(
        properties_received >= 4,
        "Must receive initial 4 container properties"
    );

    // Populate furnace with 2 Raw Beef (item 50) in slot 0, and 1 Coal (item 54) in slot 1
    if let Some(telos_voxel::block_entity::BlockEntityData::Furnace { items, .. }) =
        server.world_mut().get_block_entity_mut(furnace_pos)
    {
        items[0] = telos_voxel::block_entity::BlockEntitySlot::new(0, 50, 2);
        items[1] = telos_voxel::block_entity::BlockEntitySlot::new(1, 54, 1);
    }

    // Tick the server once -> combustion starts! Coal is consumed, block transitions to lit_furnace (83)
    server.tick();

    let cur_block = server.world_mut().get_block(furnace_pos);
    assert!(
        server.world().registry().is_lit_furnace(cur_block),
        "Furnace must become lit_furnace"
    );

    // Simulate 200 ticks of smelting (10 seconds)
    for _ in 0..200 {
        server.tick();
    }

    // Verify cooked beef (item 66) is in output slot
    let be = server
        .world()
        .get_block_entity(furnace_pos)
        .expect("furnace entity exists");
    if let telos_voxel::block_entity::BlockEntityData::Furnace { items, .. } = be {
        assert_eq!(items[0].count, 1, "1 raw beef remaining in input");
        assert_eq!(items[2].item, 66, "Cooked beef in output slot");
        assert_eq!(items[2].count, 1, "1 cooked beef produced");
    } else {
        panic!("expected furnace block entity");
    }

    // Break the furnace
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: furnace_pos.x(),
                y: furnace_pos.y(),
                z: furnace_pos.z(),
                action: BlockActionKind::Break,
                sequence: 2,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify block is air and furnace entity was removed
    assert!(server.world_mut().get_block(furnace_pos).is_air());
    assert!(server.world().get_block_entity(furnace_pos).is_none());

    // Verify items dropped in world: cooked beef (66) and remaining raw beef (50)
    let mut query = server.ecs_world_mut().query::<&ItemEntity>();
    let dropped: Vec<&ItemEntity> = query.iter(server.ecs_world()).collect();
    assert!(
        dropped
            .iter()
            .any(|item| item.stack.item == 66 && item.stack.count == 1)
    );
    assert!(
        dropped
            .iter()
            .any(|item| item.stack.item == 50 && item.stack.count == 1)
    );
}
