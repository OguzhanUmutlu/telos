//! Local chunk coordinate representations and indexing utilities.
//!
//! Enforces the 32³ cubic chunk indexing rule: `(y << 10) | (z << 5) | x`.

use telos_core::coords::{BlockPos, ChunkPos, LocalPos};

/// Linear size of one dimension in a 32³ chunk.
pub const CHUNK_SIZE: usize = 32;

/// Total number of voxels in one 32³ cubic chunk (32 × 32 × 32 = 32,768).
pub const CHUNK_VOLUME: usize = 32_768;

/// A validated linear voxel index inside a 32³ chunk, guaranteed to be in `0..32768`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalIdx(u16);

impl LocalIdx {
    /// Zero index (0, 0, 0).
    pub const ZERO: Self = Self(0);

    /// Creates a `LocalIdx` if the index is strictly less than 32,768.
    #[inline]
    #[must_use]
    pub const fn new(index: u16) -> Option<Self> {
        if (index as usize) < CHUNK_VOLUME {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Creates a `LocalIdx` from a raw `u16` index (`0..32768`).
    /// Returns `None` if `index >= 32768`.
    #[inline]
    #[must_use]
    pub const fn from_u16(index: u16) -> Option<Self> {
        if index < 32768 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Creates a `LocalIdx` without checking the bound.
    ///
    /// # Safety
    /// Caller must guarantee `index < 32768`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_raw_unchecked(index: u16) -> Self {
        Self(index)
    }

    /// Derives a `LocalIdx` from local coordinates `(x, y, z)`.
    /// Returns `None` if any coordinate is $\ge 32$.
    #[inline]
    #[must_use]
    pub const fn from_coords(x: u32, y: u32, z: u32) -> Option<Self> {
        if x < 32 && y < 32 && z < 32 {
            Some(Self(((y << 10) | (z << 5) | x) as u16))
        } else {
            None
        }
    }

    /// Derives a `LocalIdx` from local coordinates `(x, y, z)` without validation.
    ///
    /// # Panics / Invariant
    /// In debug builds, asserts `x < 32 && y < 32 && z < 32`.
    #[inline]
    #[must_use]
    pub const fn from_coords_unchecked(x: u32, y: u32, z: u32) -> Self {
        debug_assert!(
            x < 32 && y < 32 && z < 32,
            "Coordinates out of bounds for 32³ chunk"
        );
        Self(((y << 10) | (z << 5) | x) as u16)
    }

    /// Raw index as `u16`.
    #[inline]
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Raw index as `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Local X coordinate (0..32).
    #[inline]
    #[must_use]
    pub const fn x(self) -> u32 {
        (self.0 as u32) & 31
    }

    /// Local Z coordinate (0..32).
    #[inline]
    #[must_use]
    pub const fn z(self) -> u32 {
        ((self.0 as u32) >> 5) & 31
    }

    /// Local Y coordinate (0..32).
    #[inline]
    #[must_use]
    pub const fn y(self) -> u32 {
        (self.0 as u32) >> 10
    }

    /// Returns `true` if this voxel touches the -X boundary (`x == 0`).
    #[inline]
    #[must_use]
    pub const fn is_on_min_x(self) -> bool {
        self.x() == 0
    }

    /// Returns `true` if this voxel touches the +X boundary (`x == 31`).
    #[inline]
    #[must_use]
    pub const fn is_on_max_x(self) -> bool {
        self.x() == 31
    }

    /// Returns `true` if this voxel touches the -Z boundary (`z == 0`).
    #[inline]
    #[must_use]
    pub const fn is_on_min_z(self) -> bool {
        self.z() == 0
    }

    /// Returns `true` if this voxel touches the +Z boundary (`z == 31`).
    #[inline]
    #[must_use]
    pub const fn is_on_max_z(self) -> bool {
        self.z() == 31
    }

    /// Returns `true` if this voxel touches the -Y boundary (`y == 0`).
    #[inline]
    #[must_use]
    pub const fn is_on_min_y(self) -> bool {
        self.y() == 0
    }

    /// Returns `true` if this voxel touches the +Y boundary (`y == 31`).
    #[inline]
    #[must_use]
    pub const fn is_on_max_y(self) -> bool {
        self.y() == 31
    }

    /// Offsets the index inside the chunk by `(dx, dy, dz)`.
    /// Returns `None` if the offset exits the 32³ chunk boundaries.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Option<Self> {
        let nx = self.x() as i32 + dx;
        let ny = self.y() as i32 + dy;
        let nz = self.z() as i32 + dz;

        if (0..32).contains(&nx) && (0..32).contains(&ny) && (0..32).contains(&nz) {
            #[allow(clippy::cast_sign_loss)]
            Some(Self::from_coords_unchecked(nx as u32, ny as u32, nz as u32))
        } else {
            None
        }
    }
}

impl From<LocalPos> for LocalIdx {
    #[inline]
    fn from(pos: LocalPos) -> Self {
        Self::from_coords_unchecked(u32::from(pos.x()), u32::from(pos.y()), u32::from(pos.z()))
    }
}

impl From<LocalIdx> for LocalPos {
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn from(idx: LocalIdx) -> Self {
        Self::from_xyz(idx.x() as u8, idx.y() as u8, idx.z() as u8)
    }
}

/// Splits a global `BlockPos` into its containing `ChunkPos` and in-chunk `LocalIdx`.
/// Uses negative-safe arithmetic bit-shifts (`>> 5` and `& 31`).
#[inline]
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub const fn split_block_pos(b: BlockPos) -> (ChunkPos, LocalIdx) {
    let cx = b.x() >> 5;
    let cy = b.y() >> 5;
    let cz = b.z() >> 5;
    let lx = (b.x() & 31) as u32;
    let ly = (b.y() & 31) as u32;
    let lz = (b.z() & 31) as u32;
    (
        ChunkPos::new(cx, cy, cz),
        LocalIdx::from_coords_unchecked(lx, ly, lz),
    )
}

/// Computes the world `BlockPos` given a chunk position and a `LocalIdx`.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub const fn to_block_pos(chunk: ChunkPos, local: LocalIdx) -> BlockPos {
    BlockPos::new(
        (chunk.x() << 5) | (local.x() as i32),
        (chunk.y() << 5) | (local.y() as i32),
        (chunk.z() << 5) | (local.z() as i32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_idx_round_trip() {
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    let idx = LocalIdx::from_coords(x, y, z).unwrap();
                    assert_eq!(idx.x(), x);
                    assert_eq!(idx.y(), y);
                    assert_eq!(idx.z(), z);
                    assert!(idx.as_usize() < CHUNK_VOLUME);
                }
            }
        }
    }

    #[test]
    fn test_split_block_pos_negative() {
        let b = BlockPos::new(-1, -33, 31);
        let (c, l) = split_block_pos(b);
        assert_eq!(c, ChunkPos::new(-1, -2, 0));
        assert_eq!(l.x(), 31);
        assert_eq!(l.y(), 31);
        assert_eq!(l.z(), 31);

        assert_eq!(to_block_pos(c, l), b);
    }

    #[test]
    fn test_boundary_predicates() {
        let min_idx = LocalIdx::from_coords(0, 0, 0).unwrap();
        assert!(min_idx.is_on_min_x());
        assert!(min_idx.is_on_min_y());
        assert!(min_idx.is_on_min_z());
        assert!(!min_idx.is_on_max_x());

        let max_idx = LocalIdx::from_coords(31, 31, 31).unwrap();
        assert!(max_idx.is_on_max_x());
        assert!(max_idx.is_on_max_y());
        assert!(max_idx.is_on_max_z());
        assert!(!max_idx.is_on_min_x());
    }
}
