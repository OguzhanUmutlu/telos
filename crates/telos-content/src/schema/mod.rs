//! Data schemas for blocks, items, and tags.

/// Block schema definitions.
pub mod block;
/// Enchantment schema definitions.
pub mod enchantment;
/// Item schema definitions.
pub mod item;
/// Recipe and combustible fuel schema definitions.
pub mod recipe;
/// Tag schema definitions.
pub mod tag;

pub use block::{
    BlockDef, BlockItemPolicy, BlockShapeDef, OpacityDef, PropertyDef, RenderLayerDef,
};
pub use enchantment::{
    EnchantmentDef, EnchantmentDefError, EnchantmentRarityDef, EnchantmentTargetDef,
};
pub use item::{ArmorSlotDef, ItemDef, ItemTypeDef};
pub use recipe::{
    FuelDef, RecipeDef, RecipeResultDef, ShapedRecipeDef, ShapelessRecipeDef, SmeltingRecipeDef,
};
pub use tag::{TagDef, TagValueEntry};
