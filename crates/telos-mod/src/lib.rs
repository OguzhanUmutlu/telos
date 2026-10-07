//! WebAssembly modding runtime and sandbox host for the voxel engine.
//!
//! Provides deterministic, budgeted execution of guest mods with capability-based
//! permissions, event dispatching, and queued world mutations.

pub mod component;
pub mod config;
pub mod engine;
pub mod error;
pub mod host_state;
pub mod js;
pub mod manager;
pub mod permissions;

pub use component::LoadedMod;
pub use config::ModConfig;
pub use engine::WasmEngine;
pub use error::{ModError, ModResult};
pub use host_state::{BlockEdit, HostState, WorldReader};
pub use js::{
    ExecutionController, JsParticleHook, JsParticleParams, JsPlugin, JsPluginEngine, JsSandbox,
    JsSandboxConfig,
};
pub use manager::ModManager;
pub use permissions::ModPermissions;
