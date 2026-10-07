//! 3D axis-aligned bounding box indexing for procedural structure pieces.

use telos_core::coords::ChunkPos;

/// 3D axis-aligned bounding box with inclusive integer coordinates `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructureBoundingBox {
    /// Minimum X coordinate (inclusive).
    pub min_x: i32,
    /// Minimum Y coordinate (inclusive).
    pub min_y: i32,
    /// Minimum Z coordinate (inclusive).
    pub min_z: i32,
    /// Maximum X coordinate (inclusive).
    pub max_x: i32,
    /// Maximum Y coordinate (inclusive).
    pub max_y: i32,
    /// Maximum Z coordinate (inclusive).
    pub max_z: i32,
}

impl StructureBoundingBox {
    /// Constructs a new `StructureBoundingBox` ensuring `min <= max` on all axes.
    #[must_use]
    pub const fn new(
        min_x: i32,
        min_y: i32,
        min_z: i32,
        max_x: i32,
        max_y: i32,
        max_z: i32,
    ) -> Self {
        Self {
            min_x: if min_x <= max_x { min_x } else { max_x },
            min_y: if min_y <= max_y { min_y } else { max_y },
            min_z: if min_z <= max_z { min_z } else { max_z },
            max_x: if min_x <= max_x { max_x } else { min_x },
            max_y: if min_y <= max_y { max_y } else { min_y },
            max_z: if min_z <= max_z { max_z } else { min_z },
        }
    }

    /// Constructs a bounding box for an entire 32³ cubic chunk.
    #[must_use]
    pub fn from_chunk(pos: ChunkPos) -> Self {
        let min_x = pos.x() * 32;
        let min_y = pos.y() * 32;
        let min_z = pos.z() * 32;
        Self::new(min_x, min_y, min_z, min_x + 31, min_y + 31, min_z + 31)
    }

    /// Returns `true` if this bounding box intersects `other`.
    #[must_use]
    pub const fn intersects(&self, other: &Self) -> bool {
        self.max_x >= other.min_x
            && self.min_x <= other.max_x
            && self.max_y >= other.min_y
            && self.min_y <= other.max_y
            && self.max_z >= other.min_z
            && self.min_z <= other.max_z
    }

    /// Returns `true` if this bounding box intersects a 32³ cubic chunk at `pos`.
    #[must_use]
    pub fn intersects_chunk(&self, pos: ChunkPos) -> bool {
        let chunk_box = Self::from_chunk(pos);
        self.intersects(&chunk_box)
    }

    /// Returns `true` if this bounding box contains the world coordinate `(x, y, z)`.
    #[must_use]
    pub const fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }

    /// Returns the intersection between this bounding box and a 32³ chunk, or `None` if disjoint.
    #[must_use]
    pub fn clamp_to_chunk(&self, pos: ChunkPos) -> Option<Self> {
        let chunk = Self::from_chunk(pos);
        if !self.intersects(&chunk) {
            return None;
        }

        Some(Self::new(
            self.min_x.max(chunk.min_x),
            self.min_y.max(chunk.min_y),
            self.min_z.max(chunk.min_z),
            self.max_x.min(chunk.max_x),
            self.max_y.min(chunk.max_y),
            self.max_z.min(chunk.max_z),
        ))
    }

    /// Horizontal width along the X axis (inclusive count of voxels).
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.max_x - self.min_x + 1
    }

    /// Vertical height along the Y axis (inclusive count of voxels).
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.max_y - self.min_y + 1
    }

    /// Horizontal depth along the Z axis (inclusive count of voxels).
    #[must_use]
    pub const fn depth(&self) -> i32 {
        self.max_z - self.min_z + 1
    }

    /// Total voxel volume enclosed by this bounding box.
    #[must_use]
    pub const fn volume(&self) -> usize {
        (self.width() as usize) * (self.height() as usize) * (self.depth() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_intersection() {
        let b1 = StructureBoundingBox::new(0, 0, 0, 10, 10, 10);
        let b2 = StructureBoundingBox::new(5, 5, 5, 15, 15, 15);
        let b3 = StructureBoundingBox::new(20, 20, 20, 30, 30, 30);

        assert!(b1.intersects(&b2));
        assert!(b2.intersects(&b1));
        assert!(!b1.intersects(&b3));
        assert!(!b3.intersects(&b1));
    }

    #[test]
    fn test_chunk_clamping() {
        let chunk_pos = ChunkPos::new(0, 0, 0);
        let bbox = StructureBoundingBox::new(-5, 10, 10, 15, 20, 20);

        let clamped = bbox
            .clamp_to_chunk(chunk_pos)
            .expect("Must intersect chunk 0,0,0");
        assert_eq!(clamped.min_x, 0);
        assert_eq!(clamped.max_x, 15);
        assert_eq!(clamped.min_y, 10);
        assert_eq!(clamped.max_y, 20);
        assert_eq!(clamped.min_z, 10);
        assert_eq!(clamped.max_z, 20);
    }
}
