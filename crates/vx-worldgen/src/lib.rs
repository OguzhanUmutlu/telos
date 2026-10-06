//! # vx-worldgen
//!
//! Deterministic, server-side procedural world generator for the voxel engine.
//! Evaluates multi-noise continuous climate, coarse 3D density with trilinear
//! interpolation, 3D cheese caves, and top-down surface strata rules to generate
//! 32³ cubic chunks (`vx_voxel::Chunk`).

pub mod biome;
pub mod climate;
pub mod density;
pub mod error;
pub mod generator;
pub mod math;
pub mod noise;
pub mod surface;

pub use biome::BiomeId;
pub use climate::ClimatePoint;
pub use error::WorldGenError;
pub use generator::{GeneratorKind, WorldGenerator};
