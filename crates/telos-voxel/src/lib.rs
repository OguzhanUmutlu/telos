//! # telos-voxel
//!
//! Canonical in-memory voxel data model for the voxel engine.
//!
//! Features:
//! - 32³ cubic chunk coordinates and indexing (`(y << 10) | (z << 5) | x`).
//! - Power-of-two paletted chunk storage with uniform chunk elision (16 B, 0 B heap).
//! - Mutable `HotBlocks` with refcounted slot reuse and palette growth.
//! - 3-axis occupancy bitmasks (12 KiB) for binary greedy meshing.
//! - Block and block-state registry mapping namespaced IDs to dense numeric IDs.
//! - Generational `ChunkMap` with lock-free `Arc<ChunkSnapshot>` publishing.

pub mod block_entity;
pub mod chunk;
pub mod coords;
pub mod fluid;
pub mod light;
pub mod map;
pub mod occupancy;
pub mod registry;
pub mod shape;
pub mod simd;
pub mod state;
pub mod storage;

pub use block_entity::{
    BlockEntityData, BlockEntityKind, BlockEntitySlot, BlockEntityTable, CHEST_CONTAINER_SLOTS,
    FURNACE_CONTAINER_SLOTS, FURNACE_SLOT_FUEL, FURNACE_SLOT_INPUT, FURNACE_SLOT_OUTPUT,
};
pub use chunk::{Chunk, ChunkSnapshot};
pub use coords::{CHUNK_SIZE, CHUNK_VOLUME, LocalIdx, split_block_pos, to_block_pos};
pub use fluid::{FluidKind, FluidState, calculate_fluid_flow};
pub use light::{ChunkHeightmap, ChunkLight, ColumnHeights, LightBfs, LightLayer};
pub use map::{ChunkHandle, ChunkMap};
pub use occupancy::{Occupancy, transpose_32x32};
pub use registry::{Block, BlockRegistry};
pub use shape::{BlockShape, FaceOcclusionMask, ShapeTier, SubBox};
pub use state::{BlockStateId, StateFlags};
pub use storage::{Blocks, HotBlocks, Packed, fill_box, from_dense};
