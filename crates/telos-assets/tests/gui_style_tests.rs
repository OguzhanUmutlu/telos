//! Integration tests for data-driven container layouts, style sheets, and resource pack skinning.

use std::fs;
use telos_assets::{
    ContainerLayoutDef, GuiStyleSheet, NineSliceBorderDef, ResourcePackStack, SlotLayoutDef,
};
use tempfile::tempdir;

#[test]
fn test_default_container_layouts() {
    let inv = ContainerLayoutDef::default_inventory();
    assert_eq!(inv.width, 176);
    assert_eq!(inv.height, 166);
    assert_eq!(inv.slots.len(), 46);
    assert_eq!(inv.slot_pos(0), Some([8, 142])); // Hotbar slot 0
    assert_eq!(inv.slot_pos(44), Some([154, 28])); // Crafting result
    assert_eq!(inv.slot_pos(45), Some([77, 62])); // Offhand

    let chest = ContainerLayoutDef::default_chest();
    assert_eq!(chest.width, 176);
    assert_eq!(chest.height, 166);
    assert_eq!(chest.slots.len(), 63);
    assert_eq!(chest.slot_pos(0), Some([8, 18])); // Chest top-left
    assert_eq!(chest.slot_pos(26), Some([8 + 8 * 18, 18 + 2 * 18]));
    assert_eq!(chest.slot_pos(54), Some([8, 142])); // Player hotbar in chest

    let furnace = ContainerLayoutDef::default_furnace();
    assert_eq!(furnace.slots.len(), 39);
    assert_eq!(furnace.slot_pos(0), Some([56, 17])); // Input
    assert_eq!(furnace.slot_pos(1), Some([56, 53])); // Fuel
    assert_eq!(furnace.slot_pos(2), Some([116, 35])); // Output
    assert_eq!(furnace.flame_pos, Some([56, 36]));
    assert_eq!(furnace.arrow_pos, Some([79, 34]));

    let crafting = ContainerLayoutDef::default_crafting_table();
    assert_eq!(crafting.slots.len(), 46);
    assert_eq!(crafting.slot_pos(0), Some([124, 35])); // Output
    assert_eq!(crafting.slot_pos(1), Some([30, 17])); // Top-left grid
    assert_eq!(crafting.slot_pos(9), Some([30 + 2 * 18, 17 + 2 * 18])); // Bottom-right grid
    assert_eq!(crafting.arrow_pos, Some([90, 35]));
}

#[test]
fn test_container_layout_serialization_roundtrip_json() {
    let custom = ContainerLayoutDef {
        width: 200,
        height: 180,
        title_pos: [10, 8],
        inventory_title_pos: Some([10, 80]),
        background_texture: Some("textures/gui/container/custom_chest.png".to_string()),
        nine_slice: Some(NineSliceBorderDef::uniform(4).with_stretch_inner(true)),
        slots: vec![
            SlotLayoutDef::new(0, 12, 20),
            SlotLayoutDef::with_size(1, 32, 20, 18, 18),
        ],
        flame_pos: None,
        arrow_pos: None,
    };

    let json = serde_json::to_string(&custom).expect("serialize json");
    let deserialized = ContainerLayoutDef::from_json_str(&json).expect("deserialize json");
    assert_eq!(custom, deserialized);
    assert_eq!(deserialized.slot_pos(0), Some([12, 20]));
    assert_eq!(deserialized.slot_pos(1), Some([32, 20]));
    assert_eq!(deserialized.nine_slice.unwrap().left, 4);
    assert!(deserialized.nine_slice.unwrap().stretch_inner);
}

#[test]
fn test_container_layout_serialization_roundtrip_ron() {
    let custom = ContainerLayoutDef {
        width: 190,
        height: 170,
        title_pos: [6, 6],
        inventory_title_pos: None,
        background_texture: None,
        nine_slice: Some(NineSliceBorderDef::new(2, 3, 2, 4)),
        slots: vec![SlotLayoutDef::new(0, 8, 16)],
        flame_pos: None,
        arrow_pos: None,
    };

    let ron_str = ron::to_string(&custom).expect("serialize ron");
    let deserialized = ContainerLayoutDef::from_ron_str(&ron_str).expect("deserialize ron");
    assert_eq!(custom, deserialized);
    assert_eq!(deserialized.width, 190);
    assert_eq!(deserialized.nine_slice.unwrap().bottom, 4);
}

#[test]
fn test_resource_pack_stack_load_gui_style_sheet() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    // Create assets/telos/gui/containers/chest.json
    let chest_dir = root.join("assets/telos/gui/containers");
    fs::create_dir_all(&chest_dir).expect("create dir");

    let custom_chest_json = r#"{
        "width": 210,
        "height": 190,
        "title_pos": [12, 10],
        "inventory_title_pos": [12, 90],
        "nine_slice": {
            "left": 6,
            "top": 6,
            "right": 6,
            "bottom": 6,
            "stretch_inner": false
        },
        "slots": [
            { "index": 0, "x": 16, "y": 24, "size": [16, 16] }
        ]
    }"#;
    fs::write(chest_dir.join("chest.json"), custom_chest_json).expect("write chest.json");

    // Create a 9-slice sprite mcmeta: textures/gui/sprites/widget/button.png.mcmeta
    let button_dir = root.join("textures/gui/sprites/widget");
    fs::create_dir_all(&button_dir).expect("create button dir");
    let button_mcmeta = r#"{
        "gui": {
            "scaling": {
                "type": "nine_slice",
                "width": 200,
                "height": 20,
                "border": 3
            }
        }
    }"#;
    fs::write(button_dir.join("button.png.mcmeta"), button_mcmeta).expect("write button mcmeta");

    let mut stack = ResourcePackStack::new();
    stack.add_root(root);

    let style = stack.load_gui_style_sheet();

    // Overridden chest layout
    assert_eq!(style.chest.width, 210);
    assert_eq!(style.chest.height, 190);
    assert_eq!(style.chest.slot_pos(0), Some([16, 24]));
    assert_eq!(style.chest.nine_slice.unwrap().left, 6);

    // Fallback untouched inventory layout
    assert_eq!(style.inventory.width, 176);
    assert_eq!(style.inventory.slots.len(), 46);

    // 9-slice sprite registered
    assert_eq!(
        style.nine_slices.get("widget/button"),
        Some(&NineSliceBorderDef::uniform(3))
    );
}

#[test]
fn test_gui_style_sheet_merge_and_ron_roundtrip() {
    let mut sheet = GuiStyleSheet::default();
    let mut custom_chest = ContainerLayoutDef::default_chest();
    custom_chest.width = 240;
    sheet.override_container("chest", custom_chest);

    let ron_str = ron::to_string(&sheet).expect("serialize sheet to ron");
    let deserialized = GuiStyleSheet::from_ron_str(&ron_str).expect("deserialize sheet from ron");
    assert_eq!(deserialized.chest.width, 240);
    assert_eq!(deserialized.inventory.width, 176);

    let json_str = serde_json::to_string(&sheet).expect("serialize sheet to json");
    let deserialized_json =
        GuiStyleSheet::from_json_str(&json_str).expect("deserialize sheet from json");
    assert_eq!(deserialized_json.chest.width, 240);
}
