//! Telos Mod Development Kit (SDK).
//!
//! A standalone, lightweight guest library for mod authors, providing:
//! - Data pack schema definitions and fluent builders (`BlockDefBuilder`, `ItemDefBuilder`, `RecipeBuilder`, `TagBuilder`).
//! - Mod manifest parsing and validation (`mod.toml`).
//! - Guest event hooks and bitmask filters (`ModEvent`, `EventFilter`).
//! - Safe host API bindings for WASM (`log`, `get_block`, `set_block`, `subscribe_events`, `register_command`).
//! - Export macro `export_mod!` generating unmangled C ABI entrypoints.
//! - TypeScript and JavaScript plugin helpers for server-side `QuickJS` scripts.
//!
//! # Example: Defining a Data-Pack Block
//! ```rust
//! use telos_sdk::builder::BlockDefBuilder;
//! use telos_sdk::data::{BlockShapeDef, RenderLayerDef};
//!
//! let block = BlockDefBuilder::new()
//!     .shape(BlockShapeDef::FullCube)
//!     .render_layer(RenderLayerDef::Opaque)
//!     .light_emission(14)
//!     .hardness(1.5)
//!     .tool("telos:pickaxe")
//!     .build();
//!
//! let ron_data = block.to_ron_string().unwrap();
//! assert!(ron_data.contains("FullCube"));
//! ```
//!
//! # Example: Writing a Rust WASM Mod
//! ```rust,ignore
//! use telos_sdk::export_mod;
//! use telos_sdk::guest::{log, LogLevel, TelosMod};
//! use telos_sdk::events::ModEvent;
//!
//! #[derive(Default)]
//! struct HelloMod;
//!
//! impl TelosMod for HelloMod {
//!     fn init(&mut self) -> Result<(), String> {
//!         log(LogLevel::Info, "HelloMod initialized!");
//!         Ok(())
//!     }
//! }
//!
//! export_mod!(HelloMod);
//! ```

pub mod builder;
pub mod data;
pub mod events;
pub mod guest;
pub mod js;
pub mod macros;
pub mod manifest;

pub use builder::{BlockDefBuilder, ItemDefBuilder, RecipeBuilder, TagBuilder};
pub use data::{
    ArmorSlotDef, BlockDef, BlockItemPolicy, BlockShapeDef, FuelDef, ItemDef, ItemTypeDef,
    OpacityDef, PropertyDef, RecipeDef, RecipeResultDef, RenderLayerDef, ShapedRecipeDef,
    ShapelessRecipeDef, SmeltingRecipeDef, TagDef, TagValueEntry,
};
pub use events::{EventFilter, ModEvent};
pub use guest::{
    LogLevel, TelosMod, get_block, log, register_command, set_block, subscribe_events,
};
pub use manifest::{
    DependencySpec, ManifestError, ModApi, ModInfo, ModManifest, ModOrdering, ModPermissions,
    ModSide,
};
