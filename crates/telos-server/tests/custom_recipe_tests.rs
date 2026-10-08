//! Integration tests for server-defined custom crafting recipes, smelting recipes,
//! combustible fuels, network recipe manifest synchronization, and container interactions.

use std::fs;
use std::sync::Arc;
use telos_content::{DiscoveredPack, ModSide, RegistryBuilder, discover_packs, resolve_load_order};
use telos_core::ident::Identifier;
use telos_net::{Connection, Incoming, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sInventoryClick, C2sLoginStart,
    C2sMessage, ConnectionPhase, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use tempfile::tempdir;

#[test]
#[allow(clippy::too_many_lines)]
fn test_server_custom_recipes_manifest_and_crafting() {
    let temp = tempdir().unwrap();
    let packs_dir = temp.path().join("datapacks");
    let sample_pack_dir = packs_dir.join("ruby_crafting");

    // 1. Setup sample mod pack structure with block, items, recipes, and fuels
    let block_dir = sample_pack_dir.join("data").join("ruby").join("block");
    let item_dir = sample_pack_dir.join("data").join("ruby").join("item");
    let recipe_dir = sample_pack_dir.join("data").join("ruby").join("recipes");
    let fuel_dir = sample_pack_dir.join("data").join("ruby").join("fuels");

    fs::create_dir_all(&block_dir).unwrap();
    fs::create_dir_all(&item_dir).unwrap();
    fs::create_dir_all(&recipe_dir).unwrap();
    fs::create_dir_all(&fuel_dir).unwrap();

    let manifest_content = r#"
        [mod]
        id = "ruby_crafting"
        version = "1.0.0"
        name = "Ruby Crafting Pack"
        description = "Test custom recipes and fuels"
        namespaces = ["ruby"]

        [dependencies]
        telos = ">=0.1.0, <0.2.0"
    "#;
    fs::write(sample_pack_dir.join("mod.toml"), manifest_content).unwrap();

    // Custom items: ruby, raw_ruby, fire_stone
    fs::write(
        item_dir.join("ruby.ron"),
        r#"Item(name: "Ruby Gem", max_stack_size: 64, item_type: Generic)"#,
    )
    .unwrap();
    fs::write(
        item_dir.join("raw_ruby.ron"),
        r#"Item(name: "Raw Ruby Ore", max_stack_size: 64, item_type: Generic)"#,
    )
    .unwrap();
    fs::write(
        item_dir.join("fire_stone.ron"),
        r#"Item(name: "Fire Stone", max_stack_size: 64, item_type: Generic)"#,
    )
    .unwrap();

    // Custom block: ruby_block
    fs::write(
        block_dir.join("ruby_block.ron"),
        r#"Block(
            shape: FullCube,
            render_layer: Opaque,
            opacity: Opaque,
            light_emission: 10,
            hardness: 4.0,
            blast_resistance: 12.0,
            tool: Some("telos:pickaxe"),
            sound: Some("telos:stone"),
            item: Auto,
        )"#,
    )
    .unwrap();

    // Shaped recipe: 3x3 ruby -> ruby_block
    let shaped_json = r#"{
        "type": "shaped",
        "pattern": [
            "RRR",
            "RRR",
            "RRR"
        ],
        "key": {
            "R": "ruby:ruby"
        },
        "result": {
            "item": "ruby:ruby_block",
            "count": 1
        }
    }"#;
    fs::write(recipe_dir.join("ruby_block.json"), shaped_json).unwrap();

    // Shapeless recipe: 1 ruby_block -> 9 ruby
    let shapeless_json = r#"{
        "type": "shapeless",
        "ingredients": [
            "ruby:ruby_block"
        ],
        "result": {
            "item": "ruby:ruby",
            "count": 9
        }
    }"#;
    fs::write(recipe_dir.join("ruby_unpack.json"), shapeless_json).unwrap();

    // Smelting recipe: raw_ruby -> ruby
    let smelting_json = r#"{
        "type": "smelting",
        "ingredient": "ruby:raw_ruby",
        "result": {
            "item": "ruby:ruby",
            "count": 1
        },
        "cook_duration": 100,
        "experience": 0.8
    }"#;
    fs::write(recipe_dir.join("smelt_ruby.json"), smelting_json).unwrap();

    // Fuel definition: fire_stone -> 2400 ticks burn duration
    let fuel_json = r#"{
        "item": "ruby:fire_stone",
        "burn_ticks": 2400
    }"#;
    fs::write(fuel_dir.join("fire_stone.json"), fuel_json).unwrap();

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

    let ruby_item_id = registries
        .item_registry()
        .get_by_ident(&Identifier::new("ruby", "ruby").unwrap())
        .expect("ruby item registered");
    let raw_ruby_id = registries
        .item_registry()
        .get_by_ident(&Identifier::new("ruby", "raw_ruby").unwrap())
        .expect("raw_ruby item registered");
    let fire_stone_id = registries
        .item_registry()
        .get_by_ident(&Identifier::new("ruby", "fire_stone").unwrap())
        .expect("fire_stone item registered");
    let ruby_block_id = registries
        .item_registry()
        .get_by_ident(&Identifier::new("ruby", "ruby_block").unwrap())
        .expect("ruby_block item registered");

    // 3. Start server with custom registries
    let config = ServerConfig {
        save_directory: None,
        lan_broadcast: false,
        ..Default::default()
    };
    let mut server = Server::with_registries(12345, config, registries);

    // Verify server registries were populated
    assert!(server.smelting_recipes.has_recipe(raw_ruby_id));
    let smelt_def = server.smelting_recipes.find_recipe(raw_ruby_id).unwrap();
    assert_eq!(smelt_def.output_item, ruby_item_id);
    assert_eq!(smelt_def.cook_duration, 100);

    assert_eq!(server.fuel_registry.burn_duration(fire_stone_id), 2400);

    // 4. Connect mock client and perform handshake
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));

    // Hello phase
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

    // Login phase
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("RubyCrafter").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();

    // Drain messages on client side to inspect manifests
    let mut got_recipe_manifest = false;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Incoming::Msg(S2cMessage::RecipeManifest(manifest)) = incoming {
            got_recipe_manifest = true;
            assert_eq!(manifest.shaped_recipes.len(), 1);
            assert_eq!(manifest.shaped_recipes[0].result_item, ruby_block_id);
            assert_eq!(manifest.shaped_recipes[0].width, 3);
            assert_eq!(manifest.shaped_recipes[0].height, 3);

            assert_eq!(manifest.shapeless_recipes.len(), 1);
            assert_eq!(manifest.shapeless_recipes[0].result_item, ruby_item_id);
            assert_eq!(manifest.shapeless_recipes[0].result_count, 9);

            assert_eq!(manifest.smelting_recipes.len(), 1);
            assert_eq!(manifest.smelting_recipes[0].input_item, raw_ruby_id);
            assert_eq!(manifest.smelting_recipes[0].output_item, ruby_item_id);
            assert_eq!(manifest.smelting_recipes[0].cook_duration, 100);

            assert_eq!(manifest.fuels.len(), 1);
            assert_eq!(manifest.fuels[0].item_id, fire_stone_id);
            assert_eq!(manifest.fuels[0].burn_duration_ticks, 2400);
        }
    }
    assert!(
        got_recipe_manifest,
        "Must receive S2cRecipeManifest during Config handshake"
    );

    // Acknowledge config and enter Play
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 4,
                simulation_distance: 4,
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

    let session = server.get_session(session_id).unwrap();
    assert_eq!(session.phase, ConnectionPhase::Play);

    // 5. Test player 2x2 live crafting execution:
    // Put 1 ruby_block into 2x2 crafting input slot (slot 40 in player inventory)
    let reg = server.recipe_registry.clone();
    {
        let mut inv = server.player_inventory_mut(session_id).unwrap();
        inv.slots[40] = telos_sim::ItemStack::new(ruby_block_id, 1);
        inv.update_crafting_with_registry(&reg);

        assert_eq!(
            inv.slots[telos_sim::CRAFTING_RESULT_SLOT],
            telos_sim::ItemStack::new(ruby_item_id, 9),
            "2x2 craft preview must yield 9 rubies from 1 ruby block"
        );
    }

    // Click crafting result slot 44 via C2sInventoryClick
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::InventoryClick(C2sInventoryClick {
                slot: telos_sim::CRAFTING_RESULT_SLOT as u16,
                button: 0, // Left click
                mode: 0,   // Pickup
                predicted_carried_item: 0,
                predicted_carried_count: 0,
            })),
        )
        .unwrap();
    server.tick();

    // Verify result was taken into cursor carried stack
    let inv = server.player_inventory(session_id).unwrap();
    assert_eq!(inv.carried, telos_sim::ItemStack::new(ruby_item_id, 9));
    assert!(inv.slots[40].is_empty());
    assert!(inv.slots[telos_sim::CRAFTING_RESULT_SLOT].is_empty());
}
