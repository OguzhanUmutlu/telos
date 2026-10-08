//! Data pack schemas and definitions for blocks, items, recipes, and tags.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// Visual render layer classification for meshing and pipeline pass routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RenderLayerDef {
    /// Fully solid, opaque quads.
    #[default]
    Opaque,
    /// Alpha-tested cutout quads (e.g. leaves, glass, flowers).
    Cutout,
    /// Alpha-blended translucent quads (e.g. water, stained glass).
    Translucent,
}

/// Light transmission and occlusion opacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpacityDef {
    /// Fully blocks all light and occludes neighbor faces.
    #[default]
    Opaque,
    /// Fully transparent to light (e.g. air, glass).
    Transparent,
    /// Partially filters light by a fixed level reduction (e.g. water).
    Filter(u8),
}

/// Simplified shape tier for data-driven blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BlockShapeDef {
    /// Tier 0 full 1x1x1 cube.
    #[default]
    FullCube,
    /// Tier 1 axis-aligned half-slab.
    Slab {
        /// Whether slab sits on the bottom half of the block.
        #[serde(default = "default_true")]
        bottom: bool,
    },
    /// Tier 1 stairs.
    Stairs {
        /// Facing direction of stairs ("north", "south", "east", "west").
        #[serde(default = "default_north")]
        facing: String,
        /// Top or bottom half ("bottom", "top").
        #[serde(default = "default_bottom")]
        half: String,
    },
    /// Empty shape without geometry (e.g. Air).
    Empty,
    /// Fluid block shape with height level and falling flag.
    Fluid {
        /// Fluid level (0..=7).
        #[serde(default)]
        level: u8,
        /// Whether fluid is falling from above.
        #[serde(default)]
        falling: bool,
    },
    /// Tier 2 diagonal cross shape (flowers, saplings, tall grass).
    Cross,
    /// Tier 2 upright or wall-mounted torch.
    Torch {
        /// Optional wall facing ("north", "south", "east", "west").
        #[serde(default)]
        wall: Option<String>,
    },
    /// Tier 1 flat horizontal plate (wire, repeater, diode).
    FlatPlate,
    /// Tier 1 small toggleable switch / lever sub-box.
    Lever {
        /// Whether the lever is toggled on.
        #[serde(default)]
        powered: bool,
    },
    /// Tier 1 small vertical post (logic inverter torch).
    Post,
    /// Tier 1 chest container sub-box.
    Chest,
}

const fn default_true() -> bool {
    true
}

fn default_north() -> String {
    "north".to_string()
}

fn default_bottom() -> String {
    "bottom".to_string()
}

/// Policy for generating or linking a corresponding inventory item for this block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BlockItemPolicy {
    /// Automatically register a corresponding block item (`<namespace>:<block_name>`).
    #[default]
    Auto,
    /// Explicitly link an item identifier.
    Explicit(String),
    /// No item exists for this block (e.g. Air, Water).
    None,
}

/// Property definition for multi-state blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyDef {
    /// Boolean state (true/false).
    Bool,
    /// Integer range (min..=max).
    Int {
        /// Minimum value.
        min: i32,
        /// Maximum value.
        max: i32,
    },
    /// Named string enum values.
    Enum(Vec<String>),
}

/// Data-driven definition of a block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Block")]
pub struct BlockDef {
    /// State properties.
    #[serde(default)]
    pub properties: HashMap<String, PropertyDef>,
    /// Geometric shape tier.
    #[serde(default)]
    pub shape: BlockShapeDef,
    /// Rendering layer.
    #[serde(default)]
    pub render_layer: RenderLayerDef,
    /// Light opacity.
    #[serde(default)]
    pub opacity: OpacityDef,
    /// Light emission (0..=15).
    #[serde(default)]
    pub light_emission: u8,
    /// Mining hardness (seconds to break by hand).
    #[serde(default = "default_hardness")]
    pub hardness: f32,
    /// Explosion resistance.
    #[serde(default = "default_blast_resistance")]
    pub blast_resistance: f32,
    /// Required or preferred tool category (e.g. "telos:pickaxe").
    #[serde(default)]
    pub tool: Option<String>,
    /// Sound category (e.g. "telos:stone").
    #[serde(default)]
    pub sound: Option<String>,
    /// Item registration policy.
    #[serde(default)]
    pub item: BlockItemPolicy,
    /// Material texture index in the texture array layer.
    #[serde(default)]
    pub material_texture_index: Option<u16>,
    /// Whether this block is a logic component.
    #[serde(default)]
    pub logic_component: bool,
    /// Whether this block is actively powered.
    #[serde(default)]
    pub logic_powered: bool,
    /// Whether this block has an associated block entity (e.g. chest container).
    #[serde(default)]
    pub has_block_entity: bool,
    /// Optional fallback RGBA color for client procedural texturing when PNG assets are absent.
    #[serde(default)]
    pub base_color: Option<[u8; 4]>,
}

const fn default_hardness() -> f32 {
    1.5
}

const fn default_blast_resistance() -> f32 {
    6.0
}

impl Default for BlockDef {
    fn default() -> Self {
        Self {
            properties: HashMap::new(),
            shape: BlockShapeDef::FullCube,
            render_layer: RenderLayerDef::Opaque,
            opacity: OpacityDef::Opaque,
            light_emission: 0,
            hardness: default_hardness(),
            blast_resistance: default_blast_resistance(),
            tool: None,
            sound: None,
            item: BlockItemPolicy::Auto,
            material_texture_index: None,
            logic_component: false,
            logic_powered: false,
            has_block_entity: false,
            base_color: None,
        }
    }
}

impl BlockDef {
    /// Serializes the definition to pretty RON format.
    pub fn to_ron_string(&self) -> std::result::Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Serializes the definition to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Armor slot classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArmorSlotDef {
    /// Head / Helmet slot.
    Helmet,
    /// Torso / Chestplate slot.
    Chestplate,
    /// Legs / Leggings slot.
    Leggings,
    /// Feet / Boots slot.
    Boots,
}

/// Specialized item behavior and properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ItemTypeDef {
    /// Standard generic item or crafting ingredient.
    #[default]
    Generic,
    /// Item that places a block when used on a block face.
    Block(String),
    /// Equipable armor item.
    Armor {
        /// Target armor equipment slot.
        slot: ArmorSlotDef,
        /// Armor defense rating.
        defense: u32,
        /// Armor toughness rating.
        #[serde(default)]
        toughness: f32,
    },
    /// Mining or combat tool item.
    Tool {
        /// Tool category (e.g. "pickaxe", "axe", "shovel", "sword").
        kind: String,
        /// Material tier (e.g. "wood", "stone", "iron", "diamond").
        #[serde(default)]
        tier: String,
        /// Mining speed multiplier.
        #[serde(default = "default_mining_speed")]
        mining_speed: f32,
        /// Attack damage bonus.
        #[serde(default = "default_attack_damage")]
        attack_damage: f32,
    },
    /// Consumable potion item.
    Potion {
        /// Potion effect descriptor (e.g. "swiftness", "healing").
        potion_type: String,
    },
    /// Ranged projectile weapon (e.g. "bow", "crossbow").
    RangedWeapon {
        /// Weapon archetype (e.g. "bow").
        weapon_type: String,
    },
}

const fn default_mining_speed() -> f32 {
    1.0
}

const fn default_attack_damage() -> f32 {
    1.0
}

/// Data-driven definition of an item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Item")]
pub struct ItemDef {
    /// Human-readable display name.
    #[serde(default)]
    pub name: String,
    /// Maximum stack size (1..=64).
    #[serde(default = "default_max_stack_size")]
    pub max_stack_size: u16,
    /// Specialized item type and attributes.
    #[serde(default)]
    pub item_type: ItemTypeDef,
}

const fn default_max_stack_size() -> u16 {
    64
}

impl Default for ItemDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            max_stack_size: default_max_stack_size(),
            item_type: ItemTypeDef::Generic,
        }
    }
}

impl ItemDef {
    /// Serializes the definition to pretty RON format.
    pub fn to_ron_string(&self) -> std::result::Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Serializes the definition to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

const fn default_count_one() -> u16 {
    1
}

const fn default_cook_duration() -> u16 {
    200
}

const fn default_fuel_burn_ticks() -> u16 {
    300
}

/// Output item stack produced by a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeResultDef {
    /// Item identifier string (e.g. `"sample:ruby_block"` or `"telos:iron_ingot"`).
    pub item: String,
    /// Number of items produced per craft (default 1).
    #[serde(default = "default_count_one")]
    pub count: u16,
}

/// A shaped crafting recipe definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapedRecipeDef {
    /// Width of the pattern in grid columns (1..=3), optional if row pattern strings are provided.
    #[serde(default)]
    pub width: Option<usize>,
    /// Height of the pattern in grid rows (1..=3), optional if row pattern strings are provided.
    #[serde(default)]
    pub height: Option<usize>,
    /// Pattern rows (e.g. `["###", "#R#", "###"]`).
    #[serde(default)]
    pub pattern: Vec<String>,
    /// Key mapping character keys in pattern rows to item identifiers (e.g. `{"#": "telos:stone", "R": "sample:ruby"}`).
    #[serde(default)]
    pub key: HashMap<String, String>,
    /// Flat list of item identifiers (or None/empty string for air), alternative to row pattern.
    #[serde(default)]
    pub flat_pattern: Vec<Option<String>>,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Whether horizontal mirroring is permitted (default true).
    #[serde(default = "default_true")]
    pub mirrored: bool,
    /// Optional remainder item identifier returned in consumed slots (e.g. `"telos:bucket"`).
    #[serde(default)]
    pub remainder: Option<String>,
}

/// A shapeless crafting recipe definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapelessRecipeDef {
    /// Required ingredient item identifiers (multiset).
    pub ingredients: Vec<String>,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Optional remainder item identifier.
    #[serde(default)]
    pub remainder: Option<String>,
}

/// A furnace smelting recipe definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmeltingRecipeDef {
    /// Input ingredient item identifier.
    pub ingredient: String,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Cook duration in simulation ticks (default 200 ticks = 10.0s).
    #[serde(default = "default_cook_duration")]
    pub cook_duration: u16,
    /// Experience reward earned when retrieving smelted items.
    #[serde(default)]
    pub experience: f32,
}

/// A combustible fuel definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuelDef {
    /// Combustible item identifier (e.g. `"sample:blaze_coal"`).
    pub item: String,
    /// Burn duration in simulation ticks (default 300 ticks = 15.0s).
    #[serde(default = "default_fuel_burn_ticks")]
    pub burn_ticks: u16,
}

/// Top-level recipe definition enum parsed from `data/<ns>/recipes/*.ron` or `*.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecipeDef {
    /// Shaped crafting recipe.
    #[serde(alias = "minecraft:crafting_shaped", alias = "crafting_shaped")]
    Shaped(ShapedRecipeDef),
    /// Shapeless crafting recipe.
    #[serde(alias = "minecraft:crafting_shapeless", alias = "crafting_shapeless")]
    Shapeless(ShapelessRecipeDef),
    /// Smelting furnace recipe.
    #[serde(alias = "minecraft:smelting")]
    Smelting(SmeltingRecipeDef),
    /// Combustible fuel item.
    #[serde(alias = "fuel", alias = "furnace_fuel")]
    Fuel(FuelDef),
}

/// An entry in a tag's values array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TagValueEntry {
    /// Identifier or tag reference (prefixed with `#`).
    pub id: String,
    /// Whether this entry is required to exist.
    pub required: bool,
}

impl<'de> Deserialize<'de> for TagValueEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Simple(String),
            Detailed {
                id: String,
                #[serde(default = "default_true")]
                required: bool,
            },
        }

        match Helper::deserialize(deserializer)? {
            Helper::Simple(id) => Ok(Self { id, required: true }),
            Helper::Detailed { id, required } => Ok(Self { id, required }),
        }
    }
}

/// Standard JSON tag schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TagDef {
    /// If `true`, replaces existing values instead of appending.
    #[serde(default)]
    pub replace: bool,
    /// List of identifiers or nested tag references.
    #[serde(default)]
    pub values: Vec<TagValueEntry>,
}

impl RecipeDef {
    /// Serializes the recipe to pretty RON format.
    pub fn to_ron_string(&self) -> std::result::Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Serializes the recipe to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl TagDef {
    /// Serializes the tag to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
