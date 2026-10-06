//! # vx-core
//!
//! Core primitive types, coordinate systems, identifiers, and telemetry for the voxel engine.

pub mod coords;
pub mod ident;
pub mod raycast;
pub mod telemetry;
pub mod time;

pub use coords::{
    BlockPos, CHUNK_EDGE, CHUNK_MASK, CHUNK_SHIFT, CHUNK_VOLUME, ChunkPos, Face, LocalPos,
    REGION_EDGE_CHUNKS, REGION_MASK, REGION_SHIFT, REGION_VOLUME_CHUNKS, RegionPos,
};
pub use ident::{DEFAULT_NAMESPACE, Identifier, ParseIdentError};
pub use raycast::{RaycastHit, raycast_voxels};
pub use telemetry::{TelemetryConfig, init_telemetry};
pub use time::FixedTimestep;
