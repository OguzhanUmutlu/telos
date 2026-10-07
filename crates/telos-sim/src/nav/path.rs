//! Planned navigation path and waypoint traversal structures.

use telos_core::coords::BlockPos;

/// A sequence of discrete 3D voxel waypoints guiding an entity toward a destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavPath {
    waypoints: Vec<BlockPos>,
    current_idx: usize,
    target_pos: BlockPos,
}

impl NavPath {
    /// Creates a new `NavPath` from a sequence of waypoints and target destination.
    #[must_use]
    pub fn new(waypoints: Vec<BlockPos>, target_pos: BlockPos) -> Self {
        Self {
            waypoints,
            current_idx: 0,
            target_pos,
        }
    }

    /// Returns the current active waypoint, or `None` if the path is finished.
    #[must_use]
    pub fn current_waypoint(&self) -> Option<BlockPos> {
        self.waypoints.get(self.current_idx).copied()
    }

    /// Advances to the next waypoint along the path.
    pub fn advance(&mut self) {
        if self.current_idx < self.waypoints.len() {
            self.current_idx += 1;
        }
    }

    /// Returns `true` if all waypoints have been completed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.current_idx >= self.waypoints.len()
    }

    /// Total number of waypoints along this path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    /// Returns `true` if this path has no waypoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Full slice of planned waypoints.
    #[must_use]
    pub fn waypoints(&self) -> &[BlockPos] {
        &self.waypoints
    }

    /// Target destination position that generated this path.
    #[must_use]
    pub const fn target_pos(&self) -> BlockPos {
        self.target_pos
    }

    /// Slice of remaining waypoints yet to be reached.
    #[must_use]
    pub fn remaining_waypoints(&self) -> &[BlockPos] {
        if self.current_idx < self.waypoints.len() {
            &self.waypoints[self.current_idx..]
        } else {
            &[]
        }
    }
}
