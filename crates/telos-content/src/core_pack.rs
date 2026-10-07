//! Built-in vanilla `core` pack (`telos` namespace) definitions.

use crate::schema::block::{BlockDef, BlockItemPolicy, BlockShapeDef, OpacityDef, RenderLayerDef};
use crate::schema::item::{ArmorSlotDef, ItemDef, ItemTypeDef};

/// Standard core block definition entries: (path, `display_name`, `BlockDef`).
#[allow(clippy::too_many_lines)]
pub fn core_blocks() -> Vec<(&'static str, &'static str, BlockDef)> {
    vec![
        // Note: air is implicitly registered at ID 0 by BlockRegistry
        (
            "stone",
            "Stone",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.5,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(1),
                ..Default::default()
            },
        ),
        (
            "dirt",
            "Dirt",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 0.5,
                blast_resistance: 0.5,
                tool: Some("telos:shovel".into()),
                sound: Some("telos:gravel".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(2),
                ..Default::default()
            },
        ),
        (
            "grass_block",
            "Grass Block",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 0.6,
                blast_resistance: 0.6,
                tool: Some("telos:shovel".into()),
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(3),
                ..Default::default()
            },
        ),
        (
            "bedrock",
            "Bedrock",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: -1.0,
                blast_resistance: 3_600_000.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(4),
                ..Default::default()
            },
        ),
        (
            "sand",
            "Sand",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 0.5,
                blast_resistance: 0.5,
                tool: Some("telos:shovel".into()),
                sound: Some("telos:sand".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(5),
                ..Default::default()
            },
        ),
        (
            "water",
            "Water",
            BlockDef {
                shape: BlockShapeDef::Fluid {
                    level: 0,
                    falling: false,
                },
                render_layer: RenderLayerDef::Translucent,
                opacity: OpacityDef::Filter(3),
                hardness: 100.0,
                blast_resistance: 100.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(6),
                ..Default::default()
            },
        ),
        (
            "oak_planks",
            "Oak Planks",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 3.0,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(7),
                ..Default::default()
            },
        ),
        (
            "oak_leaves",
            "Oak Leaves",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Opaque,
                hardness: 0.2,
                blast_resistance: 0.2,
                tool: Some("telos:shears".into()),
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(8),
                ..Default::default()
            },
        ),
        (
            "glass",
            "Glass",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.3,
                blast_resistance: 0.3,
                tool: None,
                sound: Some("telos:glass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(9),
                ..Default::default()
            },
        ),
        (
            "stone_slab",
            "Stone Slab",
            BlockDef {
                shape: BlockShapeDef::Slab { bottom: true },
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 2.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(10),
                ..Default::default()
            },
        ),
        (
            "oak_stairs",
            "Oak Stairs",
            BlockDef {
                shape: BlockShapeDef::Stairs {
                    facing: "north".into(),
                    half: "bottom".into(),
                },
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 2.0,
                blast_resistance: 3.0,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(11),
                ..Default::default()
            },
        ),
        (
            "poppy",
            "Poppy",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(12),
                ..Default::default()
            },
        ),
        (
            "dandelion",
            "Dandelion",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(13),
                ..Default::default()
            },
        ),
        (
            "torch",
            "Torch",
            BlockDef {
                shape: BlockShapeDef::Torch { wall: None },
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                light_emission: 14,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(14),
                ..Default::default()
            },
        ),
        (
            "flowing_water",
            "Flowing Water",
            BlockDef {
                shape: BlockShapeDef::Fluid {
                    level: 1,
                    falling: false,
                },
                render_layer: RenderLayerDef::Translucent,
                opacity: OpacityDef::Filter(3),
                hardness: 100.0,
                blast_resistance: 100.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(15),
                ..Default::default()
            },
        ),
        (
            "short_grass",
            "Short Grass",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(16),
                ..Default::default()
            },
        ),
        (
            "fern",
            "Fern",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(17),
                ..Default::default()
            },
        ),
        (
            "dead_bush",
            "Dead Bush",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(18),
                ..Default::default()
            },
        ),
        (
            "logic_wire",
            "Logic Wire",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(19),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "logic_wire_powered",
            "Powered Wire",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::None,
                material_texture_index: Some(20),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_power_block",
            "Logic Power Block",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 5.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(21),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_lever",
            "Lever",
            BlockDef {
                shape: BlockShapeDef::Lever { powered: false },
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.5,
                blast_resistance: 0.5,
                tool: None,
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(22),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "logic_lever_on",
            "Lever (On)",
            BlockDef {
                shape: BlockShapeDef::Lever { powered: true },
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.5,
                blast_resistance: 0.5,
                tool: None,
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::None,
                material_texture_index: Some(23),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_lamp",
            "Logic Lamp",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 0.3,
                blast_resistance: 0.3,
                tool: None,
                sound: Some("telos:glass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(24),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "logic_lamp_lit",
            "Logic Lamp (Lit)",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                light_emission: 15,
                hardness: 0.3,
                blast_resistance: 0.3,
                tool: None,
                sound: Some("telos:glass".into()),
                item: BlockItemPolicy::None,
                material_texture_index: Some(25),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_repeater",
            "Logic Repeater",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(26),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "logic_repeater_powered",
            "Logic Repeater (Powered)",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::None,
                material_texture_index: Some(27),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_inverter",
            "Logic Inverter",
            BlockDef {
                shape: BlockShapeDef::Post,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(28),
                logic_component: true,
                logic_powered: true,
                ..Default::default()
            },
        ),
        (
            "logic_inverter_off",
            "Logic Inverter (Off)",
            BlockDef {
                shape: BlockShapeDef::Post,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::None,
                material_texture_index: Some(29),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "logic_diode",
            "Logic Diode",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(30),
                logic_component: true,
                logic_powered: false,
                ..Default::default()
            },
        ),
        (
            "lava",
            "Lava",
            BlockDef {
                shape: BlockShapeDef::Fluid {
                    level: 0,
                    falling: false,
                },
                render_layer: RenderLayerDef::Translucent,
                opacity: OpacityDef::Filter(15),
                light_emission: 15,
                hardness: 100.0,
                blast_resistance: 100.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(31),
                ..Default::default()
            },
        ),
        (
            "flowing_lava",
            "Flowing Lava",
            BlockDef {
                shape: BlockShapeDef::Fluid {
                    level: 1,
                    falling: false,
                },
                render_layer: RenderLayerDef::Translucent,
                opacity: OpacityDef::Filter(15),
                light_emission: 15,
                hardness: 100.0,
                blast_resistance: 100.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(32),
                ..Default::default()
            },
        ),
        (
            "fire",
            "Fire",
            BlockDef {
                shape: BlockShapeDef::Cross,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                light_emission: 15,
                hardness: 0.0,
                blast_resistance: 0.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(33),
                ..Default::default()
            },
        ),
        (
            "nether_portal",
            "Nether Portal",
            BlockDef {
                shape: BlockShapeDef::FlatPlate,
                render_layer: RenderLayerDef::Translucent,
                opacity: OpacityDef::Transparent,
                light_emission: 11,
                hardness: -1.0,
                blast_resistance: 0.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(34),
                ..Default::default()
            },
        ),
        (
            "cobblestone",
            "Cobblestone",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(4),
                ..Default::default()
            },
        ),
        (
            "oak_log",
            "Oak Log",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 2.0,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(5),
                ..Default::default()
            },
        ),
        (
            "crafting_table",
            "Crafting Table",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.5,
                blast_resistance: 2.5,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(14),
                ..Default::default()
            },
        ),
        (
            "missing",
            "Missing Block",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.0,
                blast_resistance: 1.0,
                tool: None,
                sound: None,
                item: BlockItemPolicy::None,
                material_texture_index: Some(1),
                ..Default::default()
            },
        ),
    ]
}

/// Standard core non-block item definition entries: (path, `ItemDef`).
pub fn core_items() -> Vec<(&'static str, ItemDef)> {
    vec![
        (
            "stick",
            ItemDef {
                name: "Stick".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "iron_helmet",
            ItemDef {
                name: "Iron Helmet".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Armor {
                    slot: ArmorSlotDef::Helmet,
                    defense: 2,
                    toughness: 0.0,
                },
            },
        ),
        (
            "iron_chestplate",
            ItemDef {
                name: "Iron Chestplate".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Armor {
                    slot: ArmorSlotDef::Chestplate,
                    defense: 6,
                    toughness: 0.0,
                },
            },
        ),
        (
            "iron_leggings",
            ItemDef {
                name: "Iron Leggings".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Armor {
                    slot: ArmorSlotDef::Leggings,
                    defense: 5,
                    toughness: 0.0,
                },
            },
        ),
        (
            "iron_boots",
            ItemDef {
                name: "Iron Boots".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Armor {
                    slot: ArmorSlotDef::Boots,
                    defense: 2,
                    toughness: 0.0,
                },
            },
        ),
    ]
}
