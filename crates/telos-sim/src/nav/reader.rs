//! World block query abstraction for navigation and pathfinding.

use telos_core::coords::BlockPos;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

/// Trait abstracting world block queries for navigation graph evaluation.
pub trait NavWorldReader {
    /// Returns the block state at the given world coordinates.
    fn get_block(&self, pos: BlockPos) -> BlockStateId;

    /// Returns a reference to the active block registry.
    fn registry(&self) -> &BlockRegistry;
}

impl<T: NavWorldReader> NavWorldReader for &T {
    #[inline]
    fn get_block(&self, pos: BlockPos) -> BlockStateId {
        (**self).get_block(pos)
    }

    #[inline]
    fn registry(&self) -> &BlockRegistry {
        (**self).registry()
    }
}
