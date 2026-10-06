//! LOD node coordinates, keys, and spatial hierarchy math.

use glam::{IVec3, Vec3};
use std::fmt;
use vx_core::coords::{BlockPos, ChunkPos};

/// Identifier for a 32³ LOD node in the clipmap pyramid.
///
/// Level 0 corresponds to standard 32³ cubic chunk coordinates.
/// Level $L$ has voxel size $2^L$ blocks and edge length $32 \cdot 2^L$ blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LodNodeKey {
    /// LOD pyramid level ($0 \le L \le 7$).
    pub level: u8,
    /// Discrete node coordinate along X.
    pub x: i32,
    /// Discrete node coordinate along Y.
    pub y: i32,
    /// Discrete node coordinate along Z.
    pub z: i32,
}

impl PartialOrd for LodNodeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LodNodeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.level
            .cmp(&other.level)
            .then_with(|| self.x.cmp(&other.x))
            .then_with(|| self.y.cmp(&other.y))
            .then_with(|| self.z.cmp(&other.z))
    }
}

impl fmt::Display for LodNodeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}({}, {}, {})", self.level, self.x, self.y, self.z)
    }
}

impl LodNodeKey {
    /// Creates a new `LodNodeKey`.
    #[inline]
    #[must_use]
    pub const fn new(level: u8, x: i32, y: i32, z: i32) -> Self {
        Self { level, x, y, z }
    }

    /// Creates a level 0 key from a standard `ChunkPos`.
    #[inline]
    #[must_use]
    pub const fn from_chunk(pos: ChunkPos) -> Self {
        Self {
            level: 0,
            x: pos.0.x,
            y: pos.0.y,
            z: pos.0.z,
        }
    }

    /// Converts this key to a `ChunkPos` (only valid for level 0).
    #[inline]
    #[must_use]
    pub const fn as_chunk(&self) -> ChunkPos {
        ChunkPos(IVec3::new(self.x, self.y, self.z))
    }

    /// Returns the parent key at level $L+1$ via arithmetic shift.
    ///
    /// Uses arithmetic right shift (`>> 1`) which correctly floors negative coordinates.
    #[inline]
    #[must_use]
    pub const fn parent(&self) -> Self {
        Self {
            level: self.level + 1,
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z >> 1,
        }
    }

    /// Child octant index ($0 \le o \le 7$) of this node within its parent.
    #[inline]
    #[must_use]
    pub const fn octant(&self) -> u8 {
        ((self.x & 1) as u8) | (((self.y & 1) as u8) << 1) | (((self.z & 1) as u8) << 2)
    }

    /// Computes the child key for a given octant index ($0 \le \text{octant} \le 7$).
    #[inline]
    #[must_use]
    pub const fn child(&self, octant: u8) -> Self {
        debug_assert!(self.level > 0, "Level 0 nodes have no child nodes");
        let ox = (octant & 1) as i32;
        let oy = ((octant >> 1) & 1) as i32;
        let oz = ((octant >> 2) & 1) as i32;

        Self {
            level: self.level.saturating_sub(1),
            x: (self.x << 1) + ox,
            y: (self.y << 1) + oy,
            z: (self.z << 1) + oz,
        }
    }

    /// Returns the voxel size in world blocks: $2^L$.
    #[inline]
    #[must_use]
    pub const fn voxel_size(&self) -> i32 {
        1 << (self.level as i32)
    }

    /// Returns the total edge length of this node in world blocks: $32 \cdot 2^L$.
    #[inline]
    #[must_use]
    pub const fn node_edge_blocks(&self) -> i32 {
        32 << (self.level as i32)
    }

    /// Minimum corner world block coordinate.
    #[inline]
    #[must_use]
    pub const fn min_block_pos(&self) -> BlockPos {
        let shift = 5 + self.level as i32;
        BlockPos::new(self.x << shift, self.y << shift, self.z << shift)
    }

    /// Maximum corner world block coordinate (exclusive).
    #[inline]
    #[must_use]
    pub const fn max_block_pos(&self) -> BlockPos {
        let shift = 5 + self.level as i32;
        let edge = 32 << (self.level as i32);
        BlockPos::new(
            (self.x << shift) + edge,
            (self.y << shift) + edge,
            (self.z << shift) + edge,
        )
    }

    /// Minimum AABB corner as `Vec3` for GPU frustum culling.
    #[inline]
    #[must_use]
    pub fn min_aabb(&self) -> Vec3 {
        let min_pos = self.min_block_pos();
        Vec3::new(min_pos.x() as f32, min_pos.y() as f32, min_pos.z() as f32)
    }

    /// Maximum AABB corner as `Vec3` for GPU frustum culling.
    #[inline]
    #[must_use]
    pub fn max_aabb(&self) -> Vec3 {
        let max_pos = self.max_block_pos();
        Vec3::new(max_pos.x() as f32, max_pos.y() as f32, max_pos.z() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parent_child_roundtrip() {
        for level in 0..6 {
            for x in -5..=5 {
                for y in -3..=3 {
                    for z in -5..=5 {
                        let node = LodNodeKey::new(level, x, y, z);
                        let parent = node.parent();
                        assert_eq!(parent.level, level + 1);

                        let octant = node.octant();
                        let reconstructed = parent.child(octant);
                        assert_eq!(reconstructed, node, "Mismatch for node {node}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_negative_coordinates_arithmetic_shift() {
        let node_neg1 = LodNodeKey::new(0, -1, -1, -1);
        let parent = node_neg1.parent();
        assert_eq!(parent.x, -1);
        assert_eq!(parent.y, -1);
        assert_eq!(parent.z, -1);
        assert_eq!(node_neg1.octant(), 7);

        let node_neg2 = LodNodeKey::new(0, -2, -2, -2);
        let parent2 = node_neg2.parent();
        assert_eq!(parent2.x, -1);
        assert_eq!(parent2.y, -1);
        assert_eq!(parent2.z, -1);
        assert_eq!(node_neg2.octant(), 0);
    }

    #[test]
    fn test_node_bounds() {
        let node = LodNodeKey::new(1, 2, 0, 3);
        assert_eq!(node.voxel_size(), 2);
        assert_eq!(node.node_edge_blocks(), 64);
        assert_eq!(node.min_block_pos(), BlockPos::new(128, 0, 192));
        assert_eq!(node.max_block_pos(), BlockPos::new(192, 64, 256));
    }
}
