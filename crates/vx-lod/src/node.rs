//! In-memory container and source traits for 32³ LOD nodes.

use std::sync::Arc;
use vx_core::coords::CHUNK_VOLUME;
use vx_voxel::chunk::ChunkSnapshot;
use vx_voxel::coords::LocalIdx;
use vx_voxel::occupancy::Occupancy;
use vx_voxel::state::BlockStateId;

use crate::coords::LodNodeKey;

/// Trait abstracting 32³ voxel query access for downsampling and meshing.
pub trait LodVoxelSource: Send + Sync {
    /// Returns `true` if voxel at local coordinate `(x, y, z)` is solid/occupied.
    fn is_solid(&self, x: u32, y: u32, z: u32) -> bool;

    /// Retrieves the `BlockStateId` at local coordinate `(x, y, z)`.
    fn get_state(&self, x: u32, y: u32, z: u32) -> BlockStateId;
}

impl LodVoxelSource for ChunkSnapshot {
    #[inline]
    fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.occupancy().is_solid(x, y, z)
    }

    #[inline]
    fn get_state(&self, x: u32, y: u32, z: u32) -> BlockStateId {
        self.blocks().get(LocalIdx::from_coords_unchecked(x, y, z))
    }
}

/// In-memory representation of a 32³ LOD node.
#[derive(Clone, Debug)]
pub struct LodNode {
    /// Spatial key in the LOD hierarchy.
    pub key: LodNodeKey,
    /// 32³ binary occupancy bitmask.
    pub occupancy: Occupancy,
    /// Per-voxel representative block state IDs (sparse or dense).
    pub states: Vec<BlockStateId>,
}

impl LodNode {
    /// Creates a new empty `LodNode`.
    #[must_use]
    pub fn empty(key: LodNodeKey) -> Self {
        Self {
            key,
            occupancy: Occupancy::empty(),
            states: vec![BlockStateId::AIR; CHUNK_VOLUME],
        }
    }

    /// Creates a solid uniform `LodNode`.
    #[must_use]
    pub fn uniform_solid(key: LodNodeKey, state: BlockStateId) -> Self {
        Self {
            key,
            occupancy: Occupancy::solid(),
            states: vec![state; CHUNK_VOLUME],
        }
    }

    /// Retrieves local 1D index from local coordinates: `(y << 10) | (z << 5) | x`.
    #[inline]
    #[must_use]
    pub const fn local_index(x: u32, y: u32, z: u32) -> usize {
        ((y as usize) << 10) | ((z as usize) << 5) | (x as usize)
    }

    /// Sets the block state and updates occupancy.
    pub fn set_state(&mut self, x: u32, y: u32, z: u32, state: BlockStateId) {
        let idx = Self::local_index(x, y, z);
        self.states[idx] = state;
        self.occupancy.set_solid(x, y, z, !state.is_air());
    }

    /// Retrieves the block state at local coordinate `(x, y, z)`.
    #[inline]
    #[must_use]
    pub fn get_state(&self, x: u32, y: u32, z: u32) -> BlockStateId {
        let idx = Self::local_index(x, y, z);
        self.states[idx]
    }
}

impl LodVoxelSource for LodNode {
    #[inline]
    fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.occupancy.is_solid(x, y, z)
    }

    #[inline]
    fn get_state(&self, x: u32, y: u32, z: u32) -> BlockStateId {
        let idx = Self::local_index(x, y, z);
        self.states[idx]
    }
}

impl<T: LodVoxelSource> LodVoxelSource for Arc<T> {
    #[inline]
    fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        (**self).is_solid(x, y, z)
    }

    #[inline]
    fn get_state(&self, x: u32, y: u32, z: u32) -> BlockStateId {
        (**self).get_state(x, y, z)
    }
}
