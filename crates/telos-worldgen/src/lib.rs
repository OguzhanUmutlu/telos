//! # telos-worldgen
//!
//! Deterministic, server-side procedural world generator for the voxel engine.
//! Evaluates multi-noise continuous climate, coarse 3D density with trilinear
//! interpolation, 3D cheese caves, and top-down surface strata rules to generate
//! 32³ cubic chunks (`telos_voxel::Chunk`).

pub mod aquifer;
pub mod biome;
pub mod climate;
pub mod decoration;
pub mod density;
pub mod error;
pub mod generator;
pub mod math;
pub mod noise;
pub mod ore;
pub mod structure;
pub mod surface;
pub mod tree;

pub use aquifer::{AquiferSample, AquiferSampler, FluidKind};
pub use biome::BiomeId;
pub use climate::ClimatePoint;
pub use decoration::apply_surface_decorations;
pub use error::WorldGenError;
pub use generator::{GeneratorKind, WorldGenerator};
pub use ore::apply_ores;
pub use structure::{
    LootItem, LootTableKind, StructureBoundingBox, apply_dungeons, apply_ruins, apply_structures,
    roll_chest_slots, roll_loot,
};
pub use tree::{TreeSpecies, apply_trees};
