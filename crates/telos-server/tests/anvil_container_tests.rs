//! Integration tests for server-authoritative anvil containers,
//! combining and repairing items, experience deduction, and refunding on close.

use glam::DVec3;
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sClientSettings, C2sCloseContainer, C2sConfigAck,
    C2sHello, C2sInventoryClick, C2sLoginStart, C2sMessage, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::{
    CompactEnchantments, EnchantmentKind, Experience, ITEM_ENCHANTED_BOOK, ITEM_IRON_SWORD,
    Inventory, ItemStack,
};
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
                username: BoundedString::new("AnvilTester").unwrap(),
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
    let anvil_pos = BlockPos::new(128, surface_y as i32 + 1, 161);
    let anvil_state = (0..server.world().registry().total_states() as u32)
        .map(BlockStateId::new)
        .find(|&id| server.world().registry().is_anvil(id))
        .expect("telos:anvil registered");
    let _ = server.world_mut().set_block(anvil_pos, anvil_state);

    {
        let session = server.get_session_mut(session_id).unwrap();
        session.position = DVec3::new(128.0, surface_y + 1.0, 160.0);
        session.move_state.pos = session.position;
    }

    (server, client_conn, session_id, surface_y)
}

#[test]
fn test_anvil_right_click_open_and_close() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let anvil_pos = BlockPos::new(128, surface_y as i32 + 1, 161);

    // Right-click anvil to open
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: anvil_pos.x(),
                y: anvil_pos.y(),
                z: anvil_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    let mut opened_anvil = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::OpenContainer(open)) = payload.into_msg() {
            assert_eq!(open.container_kind, 3, "container_kind must be 3 for Anvil");
            assert_eq!(open.slots.len(), 3, "Anvil container has 3 slots");
            opened_anvil = true;
        }
    }
    assert!(opened_anvil, "Expected OpenContainer for anvil");

    // Close container
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::CloseContainer(C2sCloseContainer {
                window_id: 1,
            })),
        )
        .unwrap();
    server.tick();

    let session = server.get_session(session_id).unwrap();
    assert!(
        session.active_container.is_none(),
        "Container should be closed"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_anvil_combining_and_xp_deduction() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let anvil_pos = BlockPos::new(128, surface_y as i32 + 1, 161);

    // Give player an iron sword in hotbar 0 (inv slot 0), an enchanted book in hotbar 1 (inv slot 1),
    // and 160 total XP points (level 10).
    let entity = server.get_session(session_id).unwrap().ecs_entity.unwrap();
    {
        let mut inv = server.ecs_world_mut().get_mut::<Inventory>(entity).unwrap();
        inv.slots[0] = ItemStack::new(ITEM_IRON_SWORD, 1);
        let mut compact = CompactEnchantments::EMPTY;
        assert!(compact.set_enchantment(EnchantmentKind::Sharpness, 1));
        inv.slots[1] = ItemStack::new_enchanted(ITEM_ENCHANTED_BOOK, 1, compact);
    }
    {
        let mut exp = server
            .ecs_world_mut()
            .get_mut::<Experience>(entity)
            .unwrap();
        exp.total_xp = 160;
        assert!(exp.level() >= 10);
    }

    // Open anvil
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: anvil_pos.x(),
                y: anvil_pos.y(),
                z: anvil_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // In dual anvil window:
    // Slot 0: left input
    // Slot 1: right input
    // Slot 2: result
    // Slot 3..29: storage
    // Slot 30: player hotbar 0 (Iron Sword)
    // Slot 31: player hotbar 1 (Enchanted Book)

    // Click slot 30 to pickup iron sword into carried
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 30,
                button: 0,
                mode: 0, // Pickup
                predicted_carried_item: ITEM_IRON_SWORD,
                predicted_carried_count: 1,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Click slot 0 (left input) to place iron sword
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 0,
                button: 0,
                mode: 0, // Pickup / place
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Click slot 31 to pickup enchanted book into carried
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 31,
                button: 0,
                mode: 0,
                predicted_carried_item: ITEM_ENCHANTED_BOOK,
                predicted_carried_count: 1,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Click slot 1 (right input) to place enchanted book
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 1,
                button: 0,
                mode: 0,
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Check anvil state on server:
    let session = server.get_session(session_id).unwrap();
    assert_eq!(session.active_anvil.left.item, ITEM_IRON_SWORD);
    assert_eq!(session.active_anvil.right.item, ITEM_ENCHANTED_BOOK);
    assert_eq!(session.active_anvil.result.item, ITEM_IRON_SWORD);
    assert_eq!(
        session
            .active_anvil
            .result
            .enchantments
            .get_level(EnchantmentKind::Sharpness),
        1
    );
    assert!(session.active_anvil.level_cost > 0);
    let cost = session.active_anvil.level_cost;

    let pre_level = server
        .ecs_world()
        .get::<Experience>(entity)
        .unwrap()
        .level();

    // Pick up result from slot 2 into carried
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 2,
                button: 0,
                mode: 0,
                predicted_carried_item: ITEM_IRON_SWORD,
                predicted_carried_count: 1,
            })),
        )
        .unwrap();
    server.tick();

    // Verify result is now in carried, input left and right consumed, and level deducted
    let post_level = server
        .ecs_world()
        .get::<Experience>(entity)
        .unwrap()
        .level();
    assert_eq!(
        post_level,
        pre_level - cost,
        "Levels must be deducted by anvil cost"
    );

    let inv = server.ecs_world().get::<Inventory>(entity).unwrap();
    assert_eq!(inv.carried.item, ITEM_IRON_SWORD);
    assert_eq!(
        inv.carried
            .enchantments
            .get_level(EnchantmentKind::Sharpness),
        1,
        "Enchanted item should be in carried"
    );
}

#[test]
fn test_anvil_refunds_inputs_on_close() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let anvil_pos = BlockPos::new(128, surface_y as i32 + 1, 161);

    // Give player iron sword in inv slot 0
    let entity = server.get_session(session_id).unwrap().ecs_entity.unwrap();
    {
        let mut inv = server.ecs_world_mut().get_mut::<Inventory>(entity).unwrap();
        inv.slots[0] = ItemStack::new(ITEM_IRON_SWORD, 1);
    }

    // Open anvil
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                x: anvil_pos.x(),
                y: anvil_pos.y(),
                z: anvil_pos.z(),
                action: BlockActionKind::Interact,
                sequence: 1,
                input_tick: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Move iron sword from hotbar (slot 30) to anvil slot 0
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 30,
                button: 0,
                mode: 0,
                predicted_carried_item: ITEM_IRON_SWORD,
                predicted_carried_count: 1,
            })),
        )
        .unwrap();
    server.tick();
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: 0,
                button: 0,
                mode: 0,
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify slot 0 in anvil has sword
    let session = server.get_session(session_id).unwrap();
    assert_eq!(session.active_anvil.left.item, ITEM_IRON_SWORD);

    // Close anvil
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::CloseContainer(C2sCloseContainer {
                window_id: 1,
            })),
        )
        .unwrap();
    server.tick();

    // Verify iron sword is refunded back to player inventory
    let inv = server.ecs_world().get::<Inventory>(entity).unwrap();
    let has_sword = inv.slots.iter().any(|s| s.item == ITEM_IRON_SWORD);
    assert!(
        has_sword,
        "Iron sword must be refunded to player inventory on anvil close"
    );
}
