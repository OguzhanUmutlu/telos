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
            "obsidian",
            "Obsidian",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 50.0,
                blast_resistance: 1200.0,
                tool: Some("telos:diamond_pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(36),
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
        // Procedural ore distribution & large sinuous ore veins (Phase 45)
        (
            "coal_ore",
            "Coal Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(37),
                ..Default::default()
            },
        ),
        (
            "iron_ore",
            "Iron Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(38),
                ..Default::default()
            },
        ),
        (
            "copper_ore",
            "Copper Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(39),
                ..Default::default()
            },
        ),
        (
            "gold_ore",
            "Gold Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(40),
                ..Default::default()
            },
        ),
        (
            "redstone_ore",
            "Redstone Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(41),
                ..Default::default()
            },
        ),
        (
            "lapis_ore",
            "Lapis Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(42),
                ..Default::default()
            },
        ),
        (
            "diamond_ore",
            "Diamond Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(43),
                ..Default::default()
            },
        ),
        (
            "emerald_ore",
            "Emerald Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(44),
                ..Default::default()
            },
        ),
        (
            "deepslate",
            "Deepslate",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 3.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(45),
                ..Default::default()
            },
        ),
        (
            "deepslate_coal_ore",
            "Deepslate Coal Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(46),
                ..Default::default()
            },
        ),
        (
            "deepslate_iron_ore",
            "Deepslate Iron Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(47),
                ..Default::default()
            },
        ),
        (
            "deepslate_copper_ore",
            "Deepslate Copper Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(48),
                ..Default::default()
            },
        ),
        (
            "deepslate_gold_ore",
            "Deepslate Gold Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(49),
                ..Default::default()
            },
        ),
        (
            "deepslate_redstone_ore",
            "Deepslate Redstone Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(50),
                ..Default::default()
            },
        ),
        (
            "deepslate_lapis_ore",
            "Deepslate Lapis Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(51),
                ..Default::default()
            },
        ),
        (
            "deepslate_diamond_ore",
            "Deepslate Diamond Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(52),
                ..Default::default()
            },
        ),
        (
            "deepslate_emerald_ore",
            "Deepslate Emerald Ore",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 4.5,
                blast_resistance: 3.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(53),
                ..Default::default()
            },
        ),
        (
            "granite",
            "Granite",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.5,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(54),
                ..Default::default()
            },
        ),
        (
            "diorite",
            "Diorite",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.5,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(55),
                ..Default::default()
            },
        ),
        (
            "andesite",
            "Andesite",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.5,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(56),
                ..Default::default()
            },
        ),
        (
            "tuff",
            "Tuff",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 1.5,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:deepslate".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(57),
                ..Default::default()
            },
        ),
        (
            "raw_iron_block",
            "Block of Raw Iron",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 5.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(58),
                ..Default::default()
            },
        ),
        (
            "raw_copper_block",
            "Block of Raw Copper",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 5.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(59),
                ..Default::default()
            },
        ),
        // Procedural trees & foliage canopies (Phase 47)
        (
            "birch_log",
            "Birch Log",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 2.0,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(60),
                ..Default::default()
            },
        ),
        (
            "spruce_log",
            "Spruce Log",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 2.0,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(61),
                ..Default::default()
            },
        ),
        (
            "birch_leaves",
            "Birch Leaves",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Opaque,
                hardness: 0.2,
                blast_resistance: 0.2,
                tool: Some("telos:shears".into()),
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(62),
                ..Default::default()
            },
        ),
        (
            "spruce_leaves",
            "Spruce Leaves",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Opaque,
                hardness: 0.2,
                blast_resistance: 0.2,
                tool: Some("telos:shears".into()),
                sound: Some("telos:grass".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(63),
                ..Default::default()
            },
        ),
        // Procedural structures, dungeons & ruins (Phase 49)
        (
            "mossy_cobblestone",
            "Mossy Cobblestone",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Opaque,
                hardness: 2.0,
                blast_resistance: 6.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(64),
                ..Default::default()
            },
        ),
        (
            "monster_spawner",
            "Monster Spawner",
            BlockDef {
                shape: BlockShapeDef::FullCube,
                render_layer: RenderLayerDef::Cutout,
                opacity: OpacityDef::Transparent,
                hardness: 5.0,
                blast_resistance: 5.0,
                tool: Some("telos:pickaxe".into()),
                sound: Some("telos:stone".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(65),
                ..Default::default()
            },
        ),
        (
            "chest",
            "Chest",
            BlockDef {
                shape: BlockShapeDef::Chest,
                render_layer: RenderLayerDef::Opaque,
                opacity: OpacityDef::Transparent,
                hardness: 2.5,
                blast_resistance: 2.5,
                tool: Some("telos:axe".into()),
                sound: Some("telos:wood".into()),
                item: BlockItemPolicy::Auto,
                material_texture_index: Some(66),
                has_block_entity: true,
                ..Default::default()
            },
        ),
    ]
}

/// Standard core non-block item definition entries: (path, `ItemDef`).
#[allow(clippy::too_many_lines)]
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
        (
            "potion",
            ItemDef {
                name: "Potion".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Potion {
                    potion_type: "water".into(),
                },
            },
        ),
        (
            "splash_potion",
            ItemDef {
                name: "Splash Potion".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Potion {
                    potion_type: "water".into(),
                },
            },
        ),
        (
            "glass_bottle",
            ItemDef {
                name: "Glass Bottle".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "brewing_stand",
            ItemDef {
                name: "Brewing Stand".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "nether_wart",
            ItemDef {
                name: "Nether Wart".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "blaze_powder",
            ItemDef {
                name: "Blaze Powder".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "sugar",
            ItemDef {
                name: "Sugar".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "glistering_melon",
            ItemDef {
                name: "Glistering Melon".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "spider_eye",
            ItemDef {
                name: "Spider Eye".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "fermented_spider_eye",
            ItemDef {
                name: "Fermented Spider Eye".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "ghast_tear",
            ItemDef {
                name: "Ghast Tear".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "magma_cream",
            ItemDef {
                name: "Magma Cream".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "rotten_flesh",
            ItemDef {
                name: "Rotten Flesh".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "porkchop",
            ItemDef {
                name: "Raw Porkchop".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "beef",
            ItemDef {
                name: "Raw Beef".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "leather",
            ItemDef {
                name: "Leather".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        // Dungeon & structure loot items (Phase 49)
        (
            "iron_ingot",
            ItemDef {
                name: "Iron Ingot".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "gold_ingot",
            ItemDef {
                name: "Gold Ingot".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "coal",
            ItemDef {
                name: "Coal".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "string",
            ItemDef {
                name: "String".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "gunpowder",
            ItemDef {
                name: "Gunpowder".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "bread",
            ItemDef {
                name: "Bread".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "wheat",
            ItemDef {
                name: "Wheat".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "saddle",
            ItemDef {
                name: "Saddle".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::Generic,
            },
        ),
        (
            "name_tag",
            ItemDef {
                name: "Name Tag".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
        // Ranged combat weapons & projectiles (Phase 51)
        (
            "bow",
            ItemDef {
                name: "Bow".into(),
                max_stack_size: 1,
                item_type: ItemTypeDef::RangedWeapon {
                    weapon_type: "bow".into(),
                },
            },
        ),
        (
            "arrow",
            ItemDef {
                name: "Arrow".into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        ),
    ]
}
