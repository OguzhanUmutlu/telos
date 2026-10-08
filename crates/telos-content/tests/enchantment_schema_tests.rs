//! Tests for data-driven enchantment schema parsing and pack discovery.

use std::fs;
use telos_content::{
    DiscoveredPack, EnchantmentDef, EnchantmentRarityDef, EnchantmentTargetDef, ModManifest,
    RegistryBuilder,
};
use telos_core::ident::Identifier;

#[test]
fn test_enchantment_json_deserialization() {
    let json_str = r#"{
        "id": "minecraft:sharpness",
        "name": "Sharpness",
        "max_level": 5,
        "weight": 10,
        "rarity": "common",
        "target": "weapon",
        "conflicts": ["minecraft:smite", "minecraft:bane_of_arthropods"]
    }"#;

    let def = EnchantmentDef::from_json_str(json_str).expect("Valid JSON enchantment definition");
    assert_eq!(def.id, "minecraft:sharpness");
    assert_eq!(def.name, "Sharpness");
    assert_eq!(def.max_level, 5);
    assert_eq!(def.weight, 10);
    assert_eq!(def.rarity, EnchantmentRarityDef::Common);
    assert_eq!(def.target, EnchantmentTargetDef::Weapon);
    assert_eq!(def.conflicts.len(), 2);
    assert!(def.conflicts_with("minecraft:smite"));
    assert!(!def.conflicts_with("minecraft:unbreaking"));
}

#[test]
fn test_enchantment_ron_deserialization() {
    let ron_str = r#"EnchantmentDef(
        id: "mymod:frost_walker",
        name: "Frost Walker",
        max_level: 2,
        weight: 2,
        rarity: rare,
        target: boots,
        conflicts: ["minecraft:depth_strider"],
    )"#;

    let def = EnchantmentDef::from_ron_str(ron_str).expect("Valid RON enchantment definition");
    assert_eq!(def.id, "mymod:frost_walker");
    assert_eq!(def.name, "Frost Walker");
    assert_eq!(def.max_level, 2);
    assert_eq!(def.weight, 2);
    assert_eq!(def.rarity, EnchantmentRarityDef::Rare);
    assert_eq!(def.target, EnchantmentTargetDef::Boots);
    assert!(def.conflicts_with("minecraft:depth_strider"));
}

#[test]
fn test_pack_enchantment_discovery() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let pack_dir = temp_dir.path().join("enchant_pack");
    fs::create_dir_all(pack_dir.join("data").join("magic").join("enchantments"))
        .expect("create dirs");

    let manifest_toml = r#"
[mod]
id = "magic"
name = "Magic Mod"
version = "1.0.0"
"#;
    fs::write(pack_dir.join("mod.toml"), manifest_toml).expect("write manifest");

    let ench_json = r#"{
        "id": "magic:lifesteal",
        "name": "Life Steal",
        "max_level": 3,
        "weight": 5,
        "rarity": "rare",
        "target": "weapon",
        "conflicts": []
    }"#;
    fs::write(
        pack_dir
            .join("data")
            .join("magic")
            .join("enchantments")
            .join("lifesteal.json"),
        ench_json,
    )
    .expect("write enchantment");

    let manifest: ModManifest = toml::from_str(manifest_toml).expect("parse manifest");
    let discovered = DiscoveredPack::new(manifest, pack_dir, false);

    let mut builder = RegistryBuilder::new();
    builder.load_core_pack().expect("load core");
    builder.load_pack(&discovered).expect("load pack");
    let frozen = builder.freeze().expect("freeze");

    let magic_ident = Identifier::new("magic", "lifesteal").expect("ident");
    let def = frozen
        .get_enchantment_def(&magic_ident)
        .expect("enchantment should be registered");
    assert_eq!(def.name, "Life Steal");
    assert_eq!(def.max_level, 3);
    assert_eq!(def.rarity, EnchantmentRarityDef::Rare);

    let custom_enchs = frozen.custom_enchantments();
    assert_eq!(custom_enchs.len(), 1);
    assert_eq!(custom_enchs[0].0, magic_ident);
}
