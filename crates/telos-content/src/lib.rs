//! Data-driven registry loading, content packs, and modding architecture.
//!
//! Implements **ADR-20** (data registries + MC-format resource packs; vanilla = built-in "core" pack),
//! **ADR-21** (namespaced IDs), **ADR-22** (namespace `voxel`), and **ADR-06** (paletted chunk storage).

pub mod core_pack;
pub mod error;
pub mod manifest;
pub mod pack;
pub mod registry;
pub mod save_map;
pub mod schema;
pub mod sound;

pub use core_pack::{core_blocks, core_items};
pub use error::{ContentError, Result};
pub use manifest::{DependencySpec, ModInfo, ModManifest, ModOrdering, ModPermissions, ModSide};
pub use pack::{DiscoveredPack, discover_packs, resolve_load_order};
pub use registry::{FrozenRegistries, ItemRegistry, RegistryBuilder, RegistryLifecycle};
pub use save_map::{RegistryRemap, SaveIdEntry, SaveIdStatus, WorldRegistryMap};
pub use schema::{
    ArmorSlotDef, BlockDef, BlockItemPolicy, BlockShapeDef, FuelDef, ItemDef, ItemTypeDef,
    OpacityDef, PropertyDef, RecipeDef, RecipeResultDef, RenderLayerDef, ShapedRecipeDef,
    ShapelessRecipeDef, SmeltingRecipeDef, TagDef, TagValueEntry,
};
pub use sound::{BlockSoundGroup, SoundCategory, SoundEvent};
