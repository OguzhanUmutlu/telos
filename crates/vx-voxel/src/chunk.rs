//! Chunk state tracking and versioned immutable snapshots.

use std::sync::Arc;
use vx_core::coords::ChunkPos;

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
        }
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
        }
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
        })
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
