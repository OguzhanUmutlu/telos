//! Comprehensive test suite for the `telos-sdk` crate.

use semver::Version;
use telos_sdk::builder::{BlockDefBuilder, ItemDefBuilder, RecipeBuilder, TagBuilder};
use telos_sdk::data::{
    ArmorSlotDef, BlockDef, BlockShapeDef, ItemDef, ItemTypeDef, OpacityDef, RecipeDef,
    RenderLayerDef, TagDef,
};
use telos_sdk::events::{EventFilter, ModEvent};
use telos_sdk::export_mod;
use telos_sdk::guest::{LogLevel, TelosMod, log};
use telos_sdk::js::{generate_starter_plugin_js, generate_typescript_declarations};
use telos_sdk::manifest::{ModManifest, ModSide};

#[test]
fn test_minimal_manifest() {
    let toml = r#"
        [mod]
        id = "simple_mod"
        version = "0.1.0"
        name = "Simple Mod"
    "#;
    let manifest = ModManifest::from_toml_str(toml).expect("Valid minimal manifest");
    assert_eq!(manifest.info.id, "simple_mod");
    assert_eq!(manifest.info.version, Version::parse("0.1.0").unwrap());
    assert_eq!(manifest.info.side, ModSide::Both);
    assert!(manifest.validate().is_ok());
}

#[test]
fn test_full_manifest_roundtrip() {
    let toml = r#"
        [mod]
        id = "complex_mod"
        version = "1.2.3"
        name = "Complex Mod"
        description = "A full featured test mod."
        authors = ["Developer"]
        license = "MIT"
        side = "server"
        namespaces = ["complex_mod", "custom_ns"]

        [api]
        server = "telos:server@^0.1"
        data = "^1"

        [dependencies]
        core_lib = "^0.2"

        [ordering]
        after = ["core_lib"]
        incompatible = { broken_mod = "<1.0" }

        [permissions]
        server = ["world.read", "world.write", "command.register"]
    "#;
    let manifest = ModManifest::from_toml_str(toml).expect("Valid complex manifest");
    assert_eq!(manifest.info.id, "complex_mod");
    assert_eq!(manifest.info.side, ModSide::Server);
    assert_eq!(manifest.info.namespaces.len(), 2);
    assert_eq!(manifest.permissions.server.len(), 3);
    assert!(manifest.validate().is_ok());

    let serialized = manifest.to_toml_string().expect("Serialization succeeded");
    let reparsed = ModManifest::from_toml_str(&serialized).expect("Reparsed successfully");
    assert_eq!(manifest.info.id, reparsed.info.id);
    assert_eq!(manifest.info.version, reparsed.info.version);
}

#[test]
fn test_manifest_validation_failures() {
    // Empty ID
    let mut manifest = ModManifest::new("", Version::parse("0.1.0").unwrap(), "Empty");
    assert!(manifest.validate().is_err());

    // Invalid uppercase character
    manifest.info.id = "Invalid_Name".to_string();
    assert!(manifest.validate().is_err());

    // Invalid symbol
    manifest.info.id = "invalid-name!".to_string();
    assert!(manifest.validate().is_err());

    // Unknown server permission
    manifest.info.id = "valid_mod".to_string();
    manifest
        .permissions
        .server
        .push("unauthorized.root".to_string());
    assert!(manifest.validate().is_err());
}

#[test]
fn test_block_builder_ron_and_json() {
    let block = BlockDefBuilder::new()
        .shape(BlockShapeDef::FullCube)
        .render_layer(RenderLayerDef::Opaque)
        .opacity(OpacityDef::Opaque)
        .light_emission(14)
        .hardness(2.5)
        .blast_resistance(10.0)
        .tool("telos:pickaxe")
        .sound("telos:stone")
        .base_color([255, 215, 0, 255])
        .build();

    assert_eq!(block.light_emission, 14);
    assert!((block.hardness - 2.5).abs() < f32::EPSILON);
    assert_eq!(block.tool.as_deref(), Some("telos:pickaxe"));

    // Serialize to RON
    let ron_str = ron::ser::to_string_pretty(&block, ron::ser::PrettyConfig::default())
        .expect("RON serialization");
    assert!(ron_str.contains("FullCube"));
    assert!(ron_str.contains("2.5"));

    // Deserialize back from RON
    let reparsed: BlockDef = ron::from_str(&ron_str).expect("RON deserialization");
    assert_eq!(reparsed.light_emission, 14);
    assert_eq!(reparsed.base_color, Some([255, 215, 0, 255]));

    // Serialize to JSON
    let json_str = serde_json::to_string_pretty(&block).expect("JSON serialization");
    let json_reparsed: BlockDef = serde_json::from_str(&json_str).expect("JSON deserialization");
    assert_eq!(json_reparsed.light_emission, 14);
}

#[test]
fn test_item_builder_ron_and_json() {
    // Generic Item
    let _gem = ItemDefBuilder::new("Ruby Gem").max_stack_size(64).build();
    let gem_ron = ItemDefBuilder::new("Ruby Gem")
        .max_stack_size(64)
        .to_ron_string()
        .expect("RON item");
    let gem_reparsed: ItemDef = ron::from_str(&gem_ron).expect("RON item reparsed");
    assert_eq!(gem_reparsed.name, "Ruby Gem");
    assert_eq!(gem_reparsed.max_stack_size, 64);
    assert_eq!(gem_reparsed.item_type, ItemTypeDef::Generic);

    // Armor Item
    let helmet = ItemDefBuilder::new("Ruby Helmet")
        .armor(ArmorSlotDef::Helmet, 3, 1.0)
        .build();
    assert_eq!(helmet.max_stack_size, 1);
    match helmet.item_type {
        ItemTypeDef::Armor {
            slot,
            defense,
            toughness,
        } => {
            assert_eq!(slot, ArmorSlotDef::Helmet);
            assert_eq!(defense, 3);
            assert!((toughness - 1.0).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Armor item type"),
    }

    // Tool Item
    let pickaxe = ItemDefBuilder::new("Ruby Pickaxe")
        .tool("pickaxe", "ruby", 8.0, 4.0)
        .build();
    match pickaxe.item_type {
        ItemTypeDef::Tool {
            kind,
            tier,
            mining_speed,
            attack_damage,
        } => {
            assert_eq!(kind, "pickaxe");
            assert_eq!(tier, "ruby");
            assert!((mining_speed - 8.0).abs() < f32::EPSILON);
            assert!((attack_damage - 4.0).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Tool item type"),
    }
}

#[test]
fn test_recipe_builder() {
    // Shaped Recipe
    let shaped = RecipeBuilder::shaped(
        ["#T#", "TGT", "#T#"],
        [
            ('#', "telos:stone"),
            ('T', "telos:torch"),
            ('G', "telos:gold_ingot"),
        ],
        "custom:lantern",
        2,
    );
    let shaped_json = serde_json::to_string(&shaped).expect("Serialize shaped recipe");
    let shaped_reparsed: RecipeDef =
        serde_json::from_str(&shaped_json).expect("Deserialize shaped recipe");
    if let RecipeDef::Shaped(s) = shaped_reparsed {
        assert_eq!(s.pattern.len(), 3);
        assert_eq!(s.key.len(), 3);
        assert_eq!(s.result.item, "custom:lantern");
        assert_eq!(s.result.count, 2);
    } else {
        panic!("Expected shaped recipe");
    }

    // Shapeless Recipe
    let shapeless = RecipeBuilder::shapeless(
        ["custom:lantern", "telos:iron_ingot"],
        "custom:reinforced_lantern",
        1,
    );
    let shapeless_json = serde_json::to_string(&shapeless).expect("Serialize shapeless recipe");
    let shapeless_reparsed: RecipeDef =
        serde_json::from_str(&shapeless_json).expect("Deserialize shapeless recipe");
    if let RecipeDef::Shapeless(s) = shapeless_reparsed {
        assert_eq!(s.ingredients.len(), 2);
        assert_eq!(s.result.item, "custom:reinforced_lantern");
    } else {
        panic!("Expected shapeless recipe");
    }

    // Smelting Recipe
    let smelting = RecipeBuilder::smelting("custom:raw_ore", "custom:ingot", 1, 150, 2.0);
    let smelting_json = serde_json::to_string(&smelting).expect("Serialize smelting");
    let smelting_reparsed: RecipeDef =
        serde_json::from_str(&smelting_json).expect("Deserialize smelting");
    if let RecipeDef::Smelting(s) = smelting_reparsed {
        assert_eq!(s.ingredient, "custom:raw_ore");
        assert_eq!(s.cook_duration, 150);
        assert!((s.experience - 2.0).abs() < f32::EPSILON);
    } else {
        panic!("Expected smelting recipe");
    }

    // Fuel
    let fuel = RecipeBuilder::fuel("custom:magma_chunk", 1600);
    let fuel_json = serde_json::to_string(&fuel).expect("Serialize fuel");
    let fuel_reparsed: RecipeDef = serde_json::from_str(&fuel_json).expect("Deserialize fuel");
    if let RecipeDef::Fuel(f) = fuel_reparsed {
        assert_eq!(f.item, "custom:magma_chunk");
        assert_eq!(f.burn_ticks, 1600);
    } else {
        panic!("Expected fuel");
    }
}

#[test]
fn test_tag_builder() {
    let tag = TagBuilder::new()
        .replace(false)
        .add("custom:ruby_block")
        .add("#telos:base_stone")
        .add_optional("other:optional_block")
        .build();

    let json_str = serde_json::to_string_pretty(&tag).expect("JSON tag serialization");
    let reparsed: TagDef = serde_json::from_str(&json_str).expect("JSON tag deserialization");
    assert!(!reparsed.replace);
    assert_eq!(reparsed.values.len(), 3);
    assert_eq!(reparsed.values[0].id, "custom:ruby_block");
    assert!(reparsed.values[0].required);
    assert_eq!(reparsed.values[2].id, "other:optional_block");
    assert!(!reparsed.values[2].required);
}

#[test]
fn test_event_abi_decoding() {
    // 1: BlockBroken
    let ev1 = ModEvent::from_abi(1, 10, 20, 30, 42);
    assert_eq!(
        ev1,
        Some(ModEvent::BlockBroken {
            pos: (10, 20, 30),
            old_state: 42,
            actor_id: None,
        })
    );

    // 2: BlockPlaced
    let ev2 = ModEvent::from_abi(2, -5, 64, 12, 105);
    assert_eq!(
        ev2,
        Some(ModEvent::BlockPlaced {
            pos: (-5, 64, 12),
            new_state: 105,
            actor_id: None,
        })
    );

    // 3: EntityDamage
    let dmg = 4.5f32;
    let ev3 = ModEvent::from_abi(3, 100, i64::from(dmg.to_bits()), 200, 0);
    assert_eq!(
        ev3,
        Some(ModEvent::EntityDamage {
            target_id: 100,
            damage: 4.5,
            attacker_id: Some(200),
        })
    );

    // 4: Tick
    let ev4 = ModEvent::from_abi(4, 12345, 0, 0, 0);
    assert_eq!(ev4, Some(ModEvent::Tick { tick: 12345 }));

    // 5: PlayerJoined
    let ev5 = ModEvent::from_abi(5, 7, 0, 0, 0);
    assert_eq!(ev5, Some(ModEvent::PlayerJoined { player_id: 7 }));

    // 6: PlayerLeft
    let ev6 = ModEvent::from_abi(6, 7, 0, 0, 0);
    assert_eq!(ev6, Some(ModEvent::PlayerLeft { player_id: 7 }));
}

#[test]
fn test_event_filter_flags() {
    let f1 = EventFilter::BLOCK_BROKEN.union(EventFilter::BLOCK_PLACED);
    assert!(f1.contains(EventFilter::BLOCK_BROKEN));
    assert!(f1.contains(EventFilter::BLOCK_PLACED));
    assert!(!f1.contains(EventFilter::TICK));
    assert!(EventFilter::ALL.contains(EventFilter::TICK));
}

#[derive(Default)]
struct DummyTestMod {
    initialized: bool,
    events_received: usize,
}

impl TelosMod for DummyTestMod {
    fn init(&mut self) -> Result<(), String> {
        self.initialized = true;
        log(LogLevel::Info, "DummyTestMod initialized");
        Ok(())
    }

    fn on_event(&mut self, _event: &ModEvent) {
        self.events_received += 1;
    }

    fn on_command(&mut self, _cmd: &str, _args: &str) -> Result<String, String> {
        Ok("Command executed".to_string())
    }
}

export_mod!(DummyTestMod);

#[test]
fn test_macro_export_mod_lifecycle() {
    telos_init();
    assert_eq!(telos_on_event(1, 0, 0, 0, 1), 0);
    assert_eq!(telos_on_command(12345, 0), 0);
}

#[test]
fn test_javascript_definitions_generator() {
    let dts = generate_typescript_declarations();
    assert!(dts.contains("onPlayerChat"));
    assert!(dts.contains("onBlockBreak"));
    assert!(dts.contains("onTick"));

    let starter_js = generate_starter_plugin_js("my_test_plugin");
    assert!(starter_js.contains("my_test_plugin"));
    assert!(starter_js.contains("onPlayerChat"));
    assert!(starter_js.contains("onBlockBreak"));
}
