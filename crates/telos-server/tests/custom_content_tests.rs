//! Integration tests for server-defined custom blocks, items, wire synchronization,
//! dynamic light emission, placement, and survival drop pickup.

use std::fs;
use std::sync::Arc;
use telos_content::{DiscoveredPack, ModSide, RegistryBuilder, discover_packs, resolve_load_order};
use telos_core::coords::BlockPos;
use telos_core::ident::Identifier;
use telos_net::{Connection, Incoming, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sClientSettings, C2sConfigAck, C2sHello,
    C2sLoginStart, C2sMessage, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use tempfile::tempdir;

#[test]
#[allow(clippy::too_many_lines)]
fn test_custom_content_handshake_sync_placement_and_drops() {
    let temp = tempdir().unwrap();
    let packs_dir = temp.path().join("datapacks");
    let sample_pack_dir = packs_dir.join("ruby_mod");

    // 1. Setup sample mod pack structure
    let block_dir = sample_pack_dir.join("data").join("ruby").join("block");
    let item_dir = sample_pack_dir.join("data").join("ruby").join("item");

    fs::create_dir_all(&block_dir).unwrap();
    fs::create_dir_all(&item_dir).unwrap();

    let manifest_content = r#"
        [mod]
        id = "ruby_mod"
        version = "1.0.0"
        name = "Ruby Mod"
        description = "Test custom block and item"
        namespaces = ["ruby"]

        [dependencies]
        telos = ">=0.1.0, <0.2.0"
    "#;
    fs::write(sample_pack_dir.join("mod.toml"), manifest_content).unwrap();

    // Custom block with emission 12 and base color [220, 20, 60, 255]
    let ruby_block_ron = r#"
        Block(
            shape: FullCube,
            render_layer: Opaque,
            opacity: Opaque,
            light_emission: 12,
            hardness: 3.5,
            blast_resistance: 10.0,
            tool: Some("telos:pickaxe"),
            sound: Some("telos:stone"),
            item: Auto,
            base_color: Some((220, 20, 60, 255)),
        )
    "#;
    fs::write(block_dir.join("ruby_block.ron"), ruby_block_ron).unwrap();

    // Custom item
    let ruby_item_ron = r#"
        Item(
            name: "Ruby Crystal",
            max_stack_size: 64,
            item_type: Generic,
        )
    "#;
    fs::write(item_dir.join("ruby_crystal.ron"), ruby_item_ron).unwrap();

    // 2. Discover and resolve registries
    let discovered = discover_packs(&[packs_dir]).unwrap();
    assert_eq!(discovered.len(), 1);

    let core_manifest = telos_content::ModManifest::from_toml_str(
        r#"
        [mod]
        id = "telos"
        version = "0.1.0"
        name = "Telos Engine Core"
    "#,
    )
    .unwrap();
    let mut all_packs = vec![DiscoveredPack::new(
        core_manifest,
        temp.path().join("core"),
        true,
    )];
    all_packs.extend(discovered);

    let resolved = resolve_load_order(all_packs, ModSide::Both).unwrap();
    let mut builder = RegistryBuilder::new();
    builder.load_core_pack().unwrap();
    builder.load_pack(&resolved[1]).unwrap();

    let registries = Arc::new(builder.freeze().unwrap());

    let ruby_block_ident = Identifier::new("ruby", "ruby_block").unwrap();
    let ruby_block_state = registries
        .get_block_state(&ruby_block_ident)
        .expect("ruby_block state exists");
    assert_ne!(ruby_block_state.as_u32(), 0);
    assert_eq!(
        registries.block_registry().light_emission(ruby_block_state),
        12
    );

    // 3. Start server with configured registries
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::with_registries(42, config, registries);

    // 4. Connect client and verify handshake packets
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let _session_id = server.add_connection(Box::new(server_conn));

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
                username: BoundedString::new("ContentTester").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    // Inspect packets sent by server during config handshake
    let mut received_registry_block = false;
    let mut received_registry_item = false;
    let mut received_content_manifest = false;
    let mut received_config_done = false;

    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Incoming::Msg(msg) = incoming {
            match msg {
                S2cMessage::RegistryData(reg) => {
                    if reg.registry_id.as_str() == "telos:block" {
                        received_registry_block = true;
                    } else if reg.registry_id.as_str() == "telos:item" {
                        received_registry_item = true;
                    }
                }
                S2cMessage::ContentManifest(manifest) => {
                    received_content_manifest = true;
                    // Verify custom block in manifest
                    let ruby_wire = manifest
                        .custom_blocks
                        .iter()
                        .find(|b| b.identifier.as_str() == "ruby:ruby_block")
                        .expect("manifest contains ruby:ruby_block");
                    assert_eq!(ruby_wire.state_id, ruby_block_state.as_u32());
                    assert_eq!(ruby_wire.light_emission, 12);
                    assert_eq!(ruby_wire.base_color, [220, 20, 60, 255]);
                    assert_eq!(ruby_wire.shape_kind, 0); // FullCube

                    // Verify custom item in manifest
                    let item_wire = manifest
                        .custom_items
                        .iter()
                        .find(|i| i.identifier.as_str() == "ruby:ruby_crystal")
                        .expect("manifest contains ruby:ruby_crystal");
                    assert_eq!(item_wire.name.as_str(), "Ruby Crystal");
                    assert_eq!(item_wire.max_stack_size, 64);
                }
                S2cMessage::ConfigDone(_) => {
                    received_config_done = true;
                }
                _ => {}
            }
        }
    }

    assert!(received_registry_block, "Must receive telos:block registry");
    assert!(received_registry_item, "Must receive telos:item registry");
    assert!(
        received_content_manifest,
        "Must receive S2cContentManifest with custom content"
    );
    assert!(received_config_done, "Must receive S2cConfigDone");

    // Complete config handshake to enter Play phase
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 3,
                simulation_distance: 3,
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

    // 5. Test Block Placement and Light Level 12
    let world = server.world_mut();
    let target_pos = BlockPos::new(128, 45, 159);

    // Place ruby_block at (128, 45, 159)
    let outcome = world.set_block(target_pos, ruby_block_state);
    assert!(outcome.is_some(), "set_block must succeed");

    // Verify chunk light emission is exactly 12 (not the old hardcoded 14)
    let (_sky, block_light) = world.get_light(target_pos);
    assert_eq!(
        block_light, 12,
        "World block light at placed custom block must equal 12"
    );

    // 6. Test Survival Block Break & Item Drop
    // Send BlockAction::Break from player
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::BlockAction(C2sBlockAction {
                sequence: 1,
                action: BlockActionKind::Break,
                x: target_pos.x(),
                y: target_pos.y(),
                z: target_pos.z(),
                input_tick: 1,
            })),
        )
        .expect("send BlockAction::Break");
    server.tick();

    // Verify block is now Air
    let cur_state = server.world().get_loaded_block(target_pos);
    assert_eq!(cur_state, telos_voxel::state::BlockStateId::AIR);

    // Verify an ItemEntity was spawned and broadcast / picked up into inventory
    let ruby_item_id = server
        .registries
        .item_registry()
        .get_by_ident(&ruby_block_ident)
        .expect("ruby block item id");

    let mut received_spawn_item = false;
    let mut received_inventory_with_item = false;

    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Incoming::Msg(msg) = incoming {
            match msg {
                S2cMessage::SpawnItem(spawn) => {
                    if spawn.item_id == ruby_item_id {
                        received_spawn_item = true;
                    }
                }
                S2cMessage::InventoryBulk(bulk)
                    if bulk.slots.iter().any(|s| s.item == ruby_item_id) =>
                {
                    received_inventory_with_item = true;
                }
                _ => {}
            }
        }
    }
    assert!(
        received_spawn_item || received_inventory_with_item,
        "Breaking ruby_block must spawn ItemEntity for ruby_block item id"
    );
}
