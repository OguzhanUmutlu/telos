//! Integration tests for data pack loading, custom blocks/items, and save-ID lifecycle.

use std::fs;
use telos_core::ident::Identifier;
use telos_voxel::storage::Blocks;
use tempfile::tempdir;

use telos_content::{
    BlockShapeDef, DiscoveredPack, ModSide, RegistryBuilder, WorldRegistryMap, discover_packs,
    resolve_load_order,
};

#[test]
#[allow(clippy::too_many_lines)]
fn test_custom_data_pack_lifecycle_and_persistence() {
    let temp = tempdir().unwrap();
    let packs_dir = temp.path().join("datapacks");
    let sample_pack_dir = packs_dir.join("sample_mod");

    // 1. Setup sample mod pack structure
    let block_dir = sample_pack_dir.join("data").join("sample").join("block");
    let item_dir = sample_pack_dir.join("data").join("sample").join("item");
    let tag_dir = sample_pack_dir
        .join("data")
        .join("sample")
        .join("tags")
        .join("block")
        .join("mineable");

    fs::create_dir_all(&block_dir).unwrap();
    fs::create_dir_all(&item_dir).unwrap();
    fs::create_dir_all(&tag_dir).unwrap();

    // mod.toml
    let manifest_content = r#"
        [mod]
        id = "sample_mod"
        version = "1.0.0"
        name = "Sample Mod"
        description = "Test custom block and item"
        namespaces = ["sample"]

        [dependencies]
        telos = ">=0.1.0, <0.2.0"
    "#;
    fs::write(sample_pack_dir.join("mod.toml"), manifest_content).unwrap();

    // Block: data/sample/block/ruby_block.ron
    let ruby_block_ron = r#"
        Block(
            shape: FullCube,
            render_layer: Opaque,
            opacity: Opaque,
            light_emission: 12,
            hardness: 3.0,
            blast_resistance: 9.0,
            tool: Some("telos:pickaxe"),
            sound: Some("telos:stone"),
            item: Auto,
            material_texture_index: Some(25),
        )
    "#;
    fs::write(block_dir.join("ruby_block.ron"), ruby_block_ron).unwrap();

    // Item: data/sample/item/ruby.ron
    let ruby_item_ron = r#"
        Item(
            name: "Ruby Gem",
            max_stack_size: 64,
            item_type: Generic,
        )
    "#;
    fs::write(item_dir.join("ruby.ron"), ruby_item_ron).unwrap();

    // Tag: data/sample/tags/block/mineable/pickaxe.json
    let tag_json = r#"
        {
            "replace": false,
            "values": [
                "sample:ruby_block"
            ]
        }
    "#;
    fs::write(tag_dir.join("pickaxe.json"), tag_json).unwrap();

    // 2. Discover packs
    let discovered = discover_packs(&[packs_dir]).unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id(), "sample_mod");

    // Add virtual core pack to discovered list
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

    // 3. Resolve load order with Kahn's algorithm
    let resolved = resolve_load_order(all_packs, ModSide::Both).unwrap();
    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved[0].id(), "telos");
    assert_eq!(resolved[1].id(), "sample_mod");

    // 4. Assemble and freeze registry
    let mut builder = RegistryBuilder::new();
    builder.load_core_pack().unwrap();
    builder.load_pack(&resolved[1]).unwrap();

    let frozen = builder.freeze().unwrap();

    let ruby_block_ident = Identifier::new("sample", "ruby_block").unwrap();
    let ruby_item_ident = Identifier::new("sample", "ruby").unwrap();

    let ruby_block_state = frozen
        .get_block_state(&ruby_block_ident)
        .expect("ruby_block registered");
    assert_ne!(ruby_block_state.as_u32(), 0);

    let ruby_def = frozen
        .get_block_def(&ruby_block_ident)
        .expect("ruby_block def");
    assert_eq!(ruby_def.light_emission, 12);
    assert_eq!(ruby_def.shape, BlockShapeDef::FullCube);

    // Auto-registered block item
    let ruby_block_item_id = frozen
        .item_registry()
        .get_by_ident(&ruby_block_ident)
        .expect("Auto block item");
    assert_eq!(
        frozen.item_registry().item_name(ruby_block_item_id),
        "Ruby Block"
    );

    // Explicit item
    let ruby_gem_item_id = frozen
        .item_registry()
        .get_by_ident(&ruby_item_ident)
        .expect("Ruby gem item");
    assert_eq!(
        frozen.item_registry().item_name(ruby_gem_item_id),
        "Ruby Gem"
    );
    assert_eq!(frozen.item_registry().max_stack_size(ruby_gem_item_id), 64);

    // Tag check
    let tag_ident = Identifier::new("sample", "block/mineable/pickaxe").unwrap();
    assert!(frozen.is_in_tag(&tag_ident, &ruby_block_ident));

    // Verify custom_blocks and custom_items
    let custom_blocks = frozen.custom_blocks();
    assert_eq!(custom_blocks.len(), 1);
    assert_eq!(custom_blocks[0].0, &ruby_block_ident);
    assert_eq!(custom_blocks[0].1.light_emission, 12);

    let custom_items = frozen.custom_items();
    assert_eq!(custom_items.len(), 2);
    assert!(
        custom_items
            .iter()
            .any(|(_, ident, _)| *ident == &ruby_block_ident)
    );
    assert!(
        custom_items
            .iter()
            .any(|(_, ident, _)| *ident == &ruby_item_ident)
    );

    // Verify block_registry light emission
    assert_eq!(frozen.block_registry().light_emission(ruby_block_state), 12);

    // Content hash determinism
    let hash1 = frozen.content_hash_hex();
    assert!(!hash1.is_empty());

    // 5. Test World Save-ID Mapping & Persistence
    let world_dir = temp.path().join("world");
    let save_map = WorldRegistryMap::open_or_create(&world_dir, &frozen).unwrap();
    let remap = save_map.build_remap(&frozen);

    let ruby_save_id = remap.runtime_to_save(ruby_block_state);
    assert_ne!(ruby_save_id, 0);
    assert_eq!(remap.save_to_runtime(ruby_save_id), ruby_block_state);

    // Create a chunk with ruby block
    let chunk_blocks = Blocks::Uniform(ruby_block_state);
    let disk_blocks = remap.remap_blocks_runtime_to_save(&chunk_blocks);

    // 6. Simulate removing sample_mod: load world with core only
    let mut core_only_builder = RegistryBuilder::new();
    core_only_builder.load_core_pack().unwrap();
    let core_only_frozen = core_only_builder.freeze().unwrap();

    let core_remap = save_map.build_remap(&core_only_frozen);
    let missing_ident = Identifier::new("telos", "missing").unwrap();
    let missing_state = core_only_frozen.get_block_state(&missing_ident).unwrap();

    // When loading the chunk without sample_mod, ruby block safely remaps to missing placeholder
    let remapped_missing_chunk = core_remap.remap_blocks_save_to_runtime(disk_blocks.clone());
    match remapped_missing_chunk {
        Blocks::Uniform(state) => assert_eq!(state, missing_state),
        Blocks::Packed(_) => panic!("Expected uniform missing chunk"),
    }

    // 7. Simulate re-enabling sample_mod: chunk restores perfectly!
    let restored_chunk = remap.remap_blocks_save_to_runtime(disk_blocks);
    match restored_chunk {
        Blocks::Uniform(state) => assert_eq!(state, ruby_block_state),
        Blocks::Packed(_) => panic!("Expected uniform restored ruby chunk"),
    }
}
