//! Data schemas for blocks, items, and tags.

pub mod block;
pub mod item;
pub mod tag;

pub use block::{
    BlockDef, BlockItemPolicy, BlockShapeDef, OpacityDef, PropertyDef, RenderLayerDef,
};
pub use item::{ArmorSlotDef, ItemDef, ItemTypeDef};
pub use tag::{TagDef, TagValueEntry};
