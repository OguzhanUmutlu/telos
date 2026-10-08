//! Fluent data pack definition builders for blocks, items, recipes, and tags.

use std::collections::HashMap;

use crate::data::{
    ArmorSlotDef, BlockDef, BlockItemPolicy, BlockShapeDef, FuelDef, ItemDef, ItemTypeDef,
    OpacityDef, PropertyDef, RecipeDef, RecipeResultDef, RenderLayerDef, ShapedRecipeDef,
    ShapelessRecipeDef, SmeltingRecipeDef, TagDef, TagValueEntry,
};

/// Fluent builder for constructing `BlockDef`.
#[derive(Debug, Default)]
pub struct BlockDefBuilder {
    def: BlockDef,
}

impl BlockDefBuilder {
    /// Creates a new default block definition builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the geometric shape tier of the block.
    #[must_use]
    pub fn shape(mut self, shape: BlockShapeDef) -> Self {
        self.def.shape = shape;
        self
    }

    /// Sets the render layer (Opaque, Cutout, Translucent).
    #[must_use]
    pub fn render_layer(mut self, layer: RenderLayerDef) -> Self {
        self.def.render_layer = layer;
        self
    }

    /// Sets light opacity (Opaque, Transparent, Filter).
    #[must_use]
    pub fn opacity(mut self, opacity: OpacityDef) -> Self {
        self.def.opacity = opacity;
        self
    }

    /// Sets light emission level (0..=15).
    #[must_use]
    pub fn light_emission(mut self, level: u8) -> Self {
        self.def.light_emission = level.min(15);
        self
    }

    /// Sets hand mining hardness in seconds.
    #[must_use]
    pub fn hardness(mut self, hardness: f32) -> Self {
        self.def.hardness = hardness.max(0.0);
        self
    }

    /// Sets explosion blast resistance.
    #[must_use]
    pub fn blast_resistance(mut self, res: f32) -> Self {
        self.def.blast_resistance = res.max(0.0);
        self
    }

    /// Sets preferred mining tool identifier (e.g. `"telos:pickaxe"`).
    #[must_use]
    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        self.def.tool = Some(tool.into());
        self
    }

    /// Sets block footstep / break sound identifier (e.g. `"telos:stone"`).
    #[must_use]
    pub fn sound(mut self, sound: impl Into<String>) -> Self {
        self.def.sound = Some(sound.into());
        self
    }

    /// Sets block item policy.
    #[must_use]
    pub fn item_policy(mut self, policy: BlockItemPolicy) -> Self {
        self.def.item = policy;
        self
    }

    /// Sets procedural fallback base RGBA color.
    #[must_use]
    pub fn base_color(mut self, color: [u8; 4]) -> Self {
        self.def.base_color = Some(color);
        self
    }

    /// Adds a state property to the block.
    #[must_use]
    pub fn property(mut self, name: impl Into<String>, prop: PropertyDef) -> Self {
        self.def.properties.insert(name.into(), prop);
        self
    }

    /// Marks the block as a logic component.
    #[must_use]
    pub fn logic_component(mut self, is_logic: bool) -> Self {
        self.def.logic_component = is_logic;
        self
    }

    /// Marks the block as having a block entity (e.g. container).
    #[must_use]
    pub fn has_block_entity(mut self, has_be: bool) -> Self {
        self.def.has_block_entity = has_be;
        self
    }

    /// Consumes the builder and returns the completed `BlockDef`.
    #[must_use]
    pub fn build(self) -> BlockDef {
        self.def
    }

    /// Serializes the definition to pretty RON format.
    pub fn to_ron_string(&self) -> std::result::Result<String, ron::Error> {
        ron::ser::to_string_pretty(&self.def, ron::ser::PrettyConfig::default())
    }

    /// Serializes the definition to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.def)
    }
}

/// Fluent builder for constructing `ItemDef`.
#[derive(Debug)]
pub struct ItemDefBuilder {
    def: ItemDef,
}

impl ItemDefBuilder {
    /// Creates a new item definition builder with the specified display name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            def: ItemDef {
                name: name.into(),
                max_stack_size: 64,
                item_type: ItemTypeDef::Generic,
            },
        }
    }

    /// Sets maximum stack size (1..=64).
    #[must_use]
    pub fn max_stack_size(mut self, size: u16) -> Self {
        self.def.max_stack_size = size.clamp(1, 64);
        self
    }

    /// Configures the item as a block-placement item.
    #[must_use]
    pub fn block_item(mut self, block_id: impl Into<String>) -> Self {
        self.def.item_type = ItemTypeDef::Block(block_id.into());
        self
    }

    /// Configures the item as equipable armor.
    #[must_use]
    pub fn armor(mut self, slot: ArmorSlotDef, defense: u32, toughness: f32) -> Self {
        self.def.item_type = ItemTypeDef::Armor {
            slot,
            defense,
            toughness,
        };
        self.def.max_stack_size = 1;
        self
    }

    /// Configures the item as a tool or weapon.
    #[must_use]
    pub fn tool(
        mut self,
        kind: impl Into<String>,
        tier: impl Into<String>,
        mining_speed: f32,
        attack_damage: f32,
    ) -> Self {
        self.def.item_type = ItemTypeDef::Tool {
            kind: kind.into(),
            tier: tier.into(),
            mining_speed,
            attack_damage,
        };
        self.def.max_stack_size = 1;
        self
    }

    /// Consumes the builder and returns the completed `ItemDef`.
    #[must_use]
    pub fn build(self) -> ItemDef {
        self.def
    }

    /// Serializes the definition to pretty RON format.
    pub fn to_ron_string(&self) -> std::result::Result<String, ron::Error> {
        ron::ser::to_string_pretty(&self.def, ron::ser::PrettyConfig::default())
    }

    /// Serializes the definition to pretty JSON format.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.def)
    }
}

/// Helper builders for creating recipes.
pub struct RecipeBuilder;

impl RecipeBuilder {
    /// Constructs a shaped crafting recipe.
    pub fn shaped<P, K>(
        pattern: impl IntoIterator<Item = P>,
        keys: impl IntoIterator<Item = (char, K)>,
        result_item: impl Into<String>,
        result_count: u16,
    ) -> RecipeDef
    where
        P: Into<String>,
        K: Into<String>,
    {
        let pat: Vec<String> = pattern.into_iter().map(Into::into).collect();
        let mut key_map = HashMap::new();
        for (ch, item) in keys {
            key_map.insert(ch.to_string(), item.into());
        }

        RecipeDef::Shaped(ShapedRecipeDef {
            width: None,
            height: None,
            pattern: pat,
            key: key_map,
            flat_pattern: Vec::new(),
            result: RecipeResultDef {
                item: result_item.into(),
                count: result_count.max(1),
            },
            mirrored: true,
            remainder: None,
        })
    }

    /// Constructs a shapeless crafting recipe.
    pub fn shapeless<I>(
        ingredients: impl IntoIterator<Item = I>,
        result_item: impl Into<String>,
        result_count: u16,
    ) -> RecipeDef
    where
        I: Into<String>,
    {
        RecipeDef::Shapeless(ShapelessRecipeDef {
            ingredients: ingredients.into_iter().map(Into::into).collect(),
            result: RecipeResultDef {
                item: result_item.into(),
                count: result_count.max(1),
            },
            remainder: None,
        })
    }

    /// Constructs a furnace smelting recipe.
    pub fn smelting(
        ingredient: impl Into<String>,
        result_item: impl Into<String>,
        result_count: u16,
        cook_duration: u16,
        experience: f32,
    ) -> RecipeDef {
        RecipeDef::Smelting(SmeltingRecipeDef {
            ingredient: ingredient.into(),
            result: RecipeResultDef {
                item: result_item.into(),
                count: result_count.max(1),
            },
            cook_duration,
            experience,
        })
    }

    /// Constructs a combustible fuel item definition.
    pub fn fuel(item: impl Into<String>, burn_ticks: u16) -> RecipeDef {
        RecipeDef::Fuel(FuelDef {
            item: item.into(),
            burn_ticks,
        })
    }
}

/// Helper builder for creating tag definitions.
#[derive(Debug, Clone, Default)]
pub struct TagBuilder {
    values: Vec<TagValueEntry>,
    replace: bool,
}

impl TagBuilder {
    /// Creates a new empty tag builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether this tag replaces previous definitions instead of appending.
    #[must_use]
    pub fn replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    /// Adds a required identifier or tag reference to the tag.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, id: impl Into<String>) -> Self {
        self.values.push(TagValueEntry {
            id: id.into(),
            required: true,
        });
        self
    }

    /// Adds an optional identifier or tag reference to the tag.
    #[must_use]
    pub fn add_optional(mut self, id: impl Into<String>) -> Self {
        self.values.push(TagValueEntry {
            id: id.into(),
            required: false,
        });
        self
    }

    /// Consumes the builder and returns the completed `TagDef`.
    #[must_use]
    pub fn build(&self) -> TagDef {
        TagDef {
            replace: self.replace,
            values: self.values.clone(),
        }
    }

    /// Serializes the tag definition to pretty JSON.
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.build())
    }
}
