//! # vx-mesh
//!
//! Pure-CPU voxel meshing engine for the world-class voxel engine.
//!
//! Implements 64-bit binary greedy meshing for full-cube (T0) geometry per ADR-08.
//! Produces 8-byte packed quads (`T0Quad`) grouped into 6 directional draw buckets.

pub mod bitwise;
pub mod greedy;
pub mod light;
pub mod mesh;
pub mod mesher;
pub mod naive;
pub mod quad;

pub use bitwise::NeighborSlices;
pub use light::{
    LightPattern, LightPatternTable, OccupancyLightSampler, VoxelLightSampler, VoxelNeighborhood,
    compute_face_pattern,
};
pub use mesh::{QuadRange, T0Mesh};
pub use mesher::{mesh_blocks_t0, mesh_blocks_with_occupancy, mesh_chunk_t0};
pub use naive::NaiveMesher;
pub use quad::{FaceDir, T0Quad};
