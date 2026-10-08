//! Chunk state tracking and versioned immutable snapshots.

use std::sync::Arc;
use telos_core::coords::ChunkPos;

use crate::{
    coords::LocalIdx,
    occupancy::Occupancy,
    state::{BlockStateId, StateFlags},
    storage::{Blocks, HotBlocks},
};

/// An immutable, `Arc`-shareable chunk snapshot consumed by meshers and network serializers.
#[derive(Debug, Clone)]
pub struct ChunkSnapshot {
    position: ChunkPos,
    blocks: Blocks,
    occupancy: Occupancy,
    light: Option<crate::light::ChunkLight>,
    content_version: u64,
    face_version: [u64; 6],
    world_epoch: u64,
    block_entities: Arc<crate::block_entity::BlockEntityTable>,
}

impl ChunkSnapshot {
    /// Creates a snapshot from raw parts.
    #[must_use]
    pub fn from_parts(
        position: ChunkPos,
        blocks: Blocks,
        occupancy: Occupancy,
        light: Option<crate::light::ChunkLight>,
        content_version: u64,
        face_version: [u64; 6],
        world_epoch: u64,
    ) -> Self {
        Self {
            position,
            blocks,
            occupancy,
            light,
            content_version,
            face_version,
            world_epoch,
            block_entities: Arc::new(crate::block_entity::BlockEntityTable::new()),
        }
    }

    /// Sets the block entities for this snapshot.
    #[must_use]
    pub fn with_block_entities(
        mut self,
        block_entities: Arc<crate::block_entity::BlockEntityTable>,
    ) -> Self {
        self.block_entities = block_entities;
        self
    }

    /// Creates a snapshot representing a uniform chunk.
    #[must_use]
    pub fn new_uniform(
        position: ChunkPos,
        state: BlockStateId,
        is_opaque: bool,
        light: Option<crate::light::ChunkLight>,
    ) -> Self {
        let blocks = Blocks::Uniform(state);
        let occupancy = if is_opaque {
            Occupancy::solid()
        } else {
            Occupancy::empty()
        };
        Self {
            position,
            blocks,
            occupancy,
            light,
            content_version: 0,
            face_version: [0; 6],
            world_epoch: 0,
            block_entities: Arc::new(crate::block_entity::BlockEntityTable::new()),
        }
    }

    /// Block entities side table in this chunk.
    #[must_use]
    pub fn block_entities(&self) -> &crate::block_entity::BlockEntityTable {
        &self.block_entities
    }

    /// Position of the chunk in world chunk grid.
    #[must_use]
    pub const fn position(&self) -> ChunkPos {
        self.position
    }

    /// Block storage container.
    #[must_use]
    pub const fn blocks(&self) -> &Blocks {
        &self.blocks
    }

    /// 3-axis occupancy bitmasks for greedy meshing.
    #[must_use]
    pub const fn occupancy(&self) -> &Occupancy {
        &self.occupancy
    }

    /// Monotonically increasing version bumped on any internal block mutation.
    #[must_use]
    pub const fn content_version(&self) -> u64 {
        self.content_version
    }

    /// Version per boundary face (0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z).
    #[must_use]
    pub const fn face_version(&self, face_idx: usize) -> u64 {
        self.face_version[face_idx]
    }

    /// World epoch tick counter when this snapshot was created.
    #[must_use]
    pub const fn world_epoch(&self) -> u64 {
        self.world_epoch
    }

    /// Explicit chunk lighting data if present within simulation radius.
    #[must_use]
    pub const fn light(&self) -> Option<&crate::light::ChunkLight> {
        self.light.as_ref()
    }
}

/// Active mutable chunk with version counters and dirty tracking.
#[derive(Debug)]
pub struct Chunk {
    position: ChunkPos,
    blocks: HotBlocks,
    occupancy: Occupancy,
    light: Option<crate::light::ChunkLight>,
    content_version: u64,
    face_version: [u64; 6],
    world_epoch: u64,
    is_dirty: bool,
    /// Side table containing block entity data for voxels in this chunk.
    pub block_entities: crate::block_entity::BlockEntityTable,
}

impl Chunk {
    /// Creates a new chunk filled uniformly with `state`.
    #[must_use]
    pub fn new_uniform(position: ChunkPos, state: BlockStateId, is_opaque: bool) -> Self {
        let blocks = HotBlocks::new_uniform(state);
        let occupancy = if is_opaque {
            Occupancy::solid()
        } else {
            Occupancy::empty()
        };

        Self {
            position,
            blocks,
            occupancy,
            light: None,
            content_version: 0,
            face_version: [0; 6],
            world_epoch: 0,
            is_dirty: false,
            block_entities: crate::block_entity::BlockEntityTable::new(),
        }
    }

    /// Creates a chunk from an initial `Blocks` representation.
    #[must_use]
    pub fn from_blocks(
        position: ChunkPos,
        blocks: Blocks,
        is_opaque: impl Fn(BlockStateId) -> bool,
    ) -> Self {
        let occupancy = Occupancy::from_blocks(&blocks, &is_opaque);
        let hot_blocks = blocks.into_hot();

        Self {
            position,
            blocks: hot_blocks,
            occupancy,
            light: None,
            content_version: 0,
            face_version: [0; 6],
            world_epoch: 0,
            is_dirty: false,
            block_entities: crate::block_entity::BlockEntityTable::new(),
        }
    }

    /// Chunk position.
    #[must_use]
    pub const fn position(&self) -> ChunkPos {
        self.position
    }

    /// 3-axis occupancy bitmasks for greedy meshing.
    #[must_use]
    pub const fn occupancy(&self) -> &Occupancy {
        &self.occupancy
    }

    /// Returns the frozen immutable `Blocks` representation of this chunk.
    #[must_use]
    pub fn to_blocks(&self) -> Blocks {
        Blocks::from_hot(&self.blocks)
    }

    /// Reads voxel at `idx`.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: LocalIdx) -> BlockStateId {
        self.blocks.get(idx)
    }

    /// Sets voxel at `idx` to `new_state` with given properties and flags.
    ///
    /// Returns `true` if state changed.
    pub fn set(
        &mut self,
        idx: LocalIdx,
        new_state: BlockStateId,
        old_flags: StateFlags,
        new_flags: StateFlags,
        epoch: u64,
    ) -> bool {
        let (_, changed) =
            self.blocks
                .set(idx, new_state, new_flags.contains(StateFlags::TICKABLE));
        if !changed {
            return false;
        }

        self.content_version += 1;
        self.world_epoch = epoch;
        self.is_dirty = true;

        // Toggle occupancy bit if opacity changed
        let was_opaque = old_flags.contains(StateFlags::OPAQUE_FULL);
        let is_opaque = new_flags.contains(StateFlags::OPAQUE_FULL);
        if was_opaque != is_opaque {
            self.occupancy.toggle(idx.x(), idx.y(), idx.z());
        }

        // Bump face versions if on boundary
        if idx.is_on_min_x() {
            self.face_version[0] += 1;
        }
        if idx.is_on_max_x() {
            self.face_version[1] += 1;
        }
        if idx.is_on_min_y() {
            self.face_version[2] += 1;
        }
        if idx.is_on_max_y() {
            self.face_version[3] += 1;
        }
        if idx.is_on_min_z() {
            self.face_version[4] += 1;
        }
        if idx.is_on_max_z() {
            self.face_version[5] += 1;
        }

        // Maintain block entities side table
        let was_be = old_flags.contains(StateFlags::HAS_BLOCK_ENTITY);
        let is_be = new_flags.contains(StateFlags::HAS_BLOCK_ENTITY);
        if was_be && !is_be {
            self.block_entities.remove(idx);
        } else if !was_be && is_be && self.block_entities.get(idx).is_none() {
            let be = if new_state.as_u32() == 82 || new_state.as_u32() == 83 {
                crate::block_entity::BlockEntityData::new_furnace()
            } else {
                crate::block_entity::BlockEntityData::new_chest()
            };
            self.block_entities.insert(idx, be);
        }

        true
    }

    /// Takes an immutable `Arc<ChunkSnapshot>` and clears the dirty flag.
    pub fn publish_snapshot(&mut self) -> Arc<ChunkSnapshot> {
        self.is_dirty = false;
        Arc::new(ChunkSnapshot {
            position: self.position,
            blocks: Blocks::from_hot(&self.blocks),
            occupancy: self.occupancy.clone(),
            light: self.light.clone(),
            content_version: self.content_version,
            face_version: self.face_version,
            world_epoch: self.world_epoch,
            block_entities: Arc::new(self.block_entities.clone()),
        })
    }

    /// Returns a reference to the chunk's block entities side table.
    #[must_use]
    pub fn block_entities(&self) -> &crate::block_entity::BlockEntityTable {
        &self.block_entities
    }

    /// Returns a mutable reference to the chunk's block entities side table.
    pub fn block_entities_mut(&mut self) -> &mut crate::block_entity::BlockEntityTable {
        self.is_dirty = true;
        &mut self.block_entities
    }

    /// Returns a reference to the block entity at `idx`, if present.
    #[must_use]
    pub fn get_block_entity(&self, idx: LocalIdx) -> Option<&crate::block_entity::BlockEntityData> {
        self.block_entities.get(idx)
    }

    /// Returns a mutable reference to the block entity at `idx`, if present.
    pub fn get_block_entity_mut(
        &mut self,
        idx: LocalIdx,
    ) -> Option<&mut crate::block_entity::BlockEntityData> {
        self.is_dirty = true;
        self.block_entities.get_mut(idx)
    }

    /// Inserts or replaces the block entity at `idx`.
    pub fn set_block_entity(
        &mut self,
        idx: LocalIdx,
        data: crate::block_entity::BlockEntityData,
    ) -> Option<crate::block_entity::BlockEntityData> {
        self.is_dirty = true;
        self.block_entities.insert(idx, data)
    }

    /// Removes and returns the block entity at `idx`, if present.
    pub fn remove_block_entity(
        &mut self,
        idx: LocalIdx,
    ) -> Option<crate::block_entity::BlockEntityData> {
        self.is_dirty = true;
        self.block_entities.remove(idx)
    }

    /// Accesses the chunk's lighting data if allocated.
    #[must_use]
    pub const fn light(&self) -> Option<&crate::light::ChunkLight> {
        self.light.as_ref()
    }

    /// Mutably accesses the chunk's lighting data.
    pub fn light_mut(&mut self) -> Option<&mut crate::light::ChunkLight> {
        self.light.as_mut()
    }

    /// Sets or drops the chunk's lighting data.
    pub fn set_light(&mut self, light: Option<crate::light::ChunkLight>) {
        self.light = light;
        self.is_dirty = true;
    }

    /// Returns `true` if chunk has uncommitted edits.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.is_dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_entity::{BlockEntityData, BlockEntityKind, BlockEntitySlot};
    use telos_core::coords::ChunkPos;

    #[test]
    fn test_chunk_set_furnace_lit_transition() {
        let mut chunk = Chunk::new_uniform(ChunkPos::new(0, 0, 0), BlockStateId::AIR, false);
        let idx = LocalIdx::from_coords(4, 5, 6).unwrap();

        let air_flags = StateFlags::AIR;
        let furnace_flags = StateFlags::OPAQUE_CUBE | StateFlags::HAS_BLOCK_ENTITY;
        let lit_flags =
            StateFlags::OPAQUE_CUBE | StateFlags::HAS_BLOCK_ENTITY | StateFlags::EMISSIVE;

        let furnace_state = BlockStateId::new(82);
        let lit_state = BlockStateId::new(83);

        // 1. Place furnace -> creates default Furnace block entity
        chunk.set(idx, furnace_state, air_flags, furnace_flags, 1);
        let be = chunk
            .block_entities()
            .get(idx)
            .expect("furnace entity spawned");
        assert_eq!(be.kind(), BlockEntityKind::Furnace);
        assert_eq!(be.items().len(), 3);

        // 2. Put items and burn time into furnace
        if let Some(BlockEntityData::Furnace {
            items,
            burn_time_remaining,
            ..
        }) = chunk.block_entities_mut().get_mut(idx)
        {
            items[0] = BlockEntitySlot::new(0, 50, 4);
            *burn_time_remaining = 800;
        }

        // 3. Transition to lit furnace -> entity MUST be preserved!
        chunk.set(idx, lit_state, furnace_flags, lit_flags, 2);
        let be_lit = chunk
            .block_entities()
            .get(idx)
            .expect("furnace entity preserved");
        assert_eq!(be_lit.kind(), BlockEntityKind::Furnace);
        assert_eq!(be_lit.items()[0].item, 50);
        assert_eq!(be_lit.items()[0].count, 4);
        if let BlockEntityData::Furnace {
            burn_time_remaining,
            ..
        } = be_lit
        {
            assert_eq!(*burn_time_remaining, 800);
        } else {
            panic!("expected furnace");
        }

        // 4. Transition back to unlit furnace -> entity MUST be preserved!
        chunk.set(idx, furnace_state, lit_flags, furnace_flags, 3);
        let be_unlit = chunk
            .block_entities()
            .get(idx)
            .expect("furnace entity preserved");
        assert_eq!(be_unlit.items()[0].item, 50);

        // 5. Break furnace -> entity removed
        chunk.set(idx, BlockStateId::AIR, furnace_flags, air_flags, 4);
        assert!(chunk.block_entities().get(idx).is_none());
    }
}
