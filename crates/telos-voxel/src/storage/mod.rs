//! Paletted chunk storage with uniform chunk elision and power-of-two bitboards.

pub mod bulk;
pub mod hot;
pub mod packed;

pub use bulk::*;
pub use hot::HotBlocks;
pub use packed::Packed;

use crate::{coords::LocalIdx, state::BlockStateId};

/// Canonical in-memory chunk block representation.
///
/// Uniform chunks occupy only 16 bytes on stack/container and allocate **0 bytes of heap memory**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocks {
    /// Chunk filled entirely with a single uniform state (e.g. pure air or solid bedrock).
    Uniform(BlockStateId),
    /// Paletted bitboard with 1, 2, 4, 8, or 16 bits per voxel.
    Packed(Box<Packed>),
}

impl Blocks {
    /// Creates a uniform chunk of air.
    #[must_use]
    pub const fn air() -> Self {
        Self::Uniform(BlockStateId::AIR)
    }

    /// Reads the block state at local index `idx`.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: LocalIdx) -> BlockStateId {
        match self {
            Self::Uniform(state) => *state,
            Self::Packed(packed) => packed.get(idx),
        }
    }

    /// Returns `true` if this chunk is uniform (0 heap bytes).
    #[inline]
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform(_))
    }

    /// Promotes this container to mutable `HotBlocks` for editing.
    #[must_use]
    pub fn to_hot(&self) -> HotBlocks {
        match self {
            Self::Uniform(state) => HotBlocks::new_uniform(*state),
            Self::Packed(packed) => HotBlocks::from_packed((**packed).clone()),
        }
    }

    /// Converts into a mutable `HotBlocks` container, consuming `self` without cloning heap buffers.
    #[must_use]
    pub fn into_hot(self) -> HotBlocks {
        match self {
            Self::Uniform(state) => HotBlocks::new_uniform(state),
            Self::Packed(packed) => HotBlocks::from_packed(*packed),
        }
    }

    /// Creates a `Blocks` instance from an active `HotBlocks` container.
    /// Automatically collapses to `Uniform` if only 1 distinct state exists!
    #[must_use]
    pub fn from_hot(hot: &HotBlocks) -> Self {
        if hot.distinct_live_states() <= 1 {
            let state = hot
                .packed
                .palette
                .first()
                .copied()
                .unwrap_or(BlockStateId::AIR);
            Self::Uniform(state)
        } else {
            Self::Packed(Box::new(hot.packed.clone()))
        }
    }

    /// Total heap memory allocated by this block container in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 0,
            Self::Packed(packed) => packed.heap_bytes(),
        }
    }
}
