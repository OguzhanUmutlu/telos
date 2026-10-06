//! Cubic region file container (`.vxr`) coordinate math and file management.

pub mod allocator;
pub mod file;

pub use allocator::SectorAllocator;
pub use file::RegionFile;

use telos_core::coords::ChunkPos;

/// Represents coordinates of a cubic region containing 8 × 8 × 8 (512) chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionPos {
    /// Region X coordinate.
    pub x: i32,
    /// Region Y coordinate.
    pub y: i32,
    /// Region Z coordinate.
    pub z: i32,
}

impl RegionPos {
    /// Creates a new `RegionPos`.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Computes the region containing the given `ChunkPos`.
    #[must_use]
    pub const fn from_chunk(chunk: ChunkPos) -> Self {
        Self {
            x: chunk.x() >> 3,
            y: chunk.y() >> 3,
            z: chunk.z() >> 3,
        }
    }

    /// Computes the local chunk offset (0..8) within this region for a `ChunkPos`.
    #[must_use]
    pub const fn local_chunk_coords(chunk: ChunkPos) -> (usize, usize, usize) {
        (
            (chunk.x() & 7) as usize,
            (chunk.y() & 7) as usize,
            (chunk.z() & 7) as usize,
        )
    }

    /// Computes the entry index (0..512) for a given `ChunkPos`.
    ///
    /// Layout is Y-major: `(ly << 6) | (lz << 3) | lx`.
    #[must_use]
    pub const fn entry_index(chunk: ChunkPos) -> usize {
        let (lx, ly, lz) = Self::local_chunk_coords(chunk);
        (ly << 6) | (lz << 3) | lx
    }

    /// Reconstructs the absolute `ChunkPos` from this region position and an entry index.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub const fn chunk_pos_at(&self, entry_index: usize) -> ChunkPos {
        let ly = ((entry_index >> 6) & 7) as i32;
        let lz = ((entry_index >> 3) & 7) as i32;
        let lx = (entry_index & 7) as i32;
        ChunkPos::new((self.x << 3) | lx, (self.y << 3) | ly, (self.z << 3) | lz)
    }

    /// Standard filename for this region file: `r.{x}.{y}.{z}.tlr`.
    #[must_use]
    pub fn filename(&self) -> String {
        format!("r.{}.{}.{}.tlr", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_coords_roundtrip() {
        for cx in -20..20 {
            for cy in -10..10 {
                for cz in -20..20 {
                    let chunk = ChunkPos::new(cx, cy, cz);
                    let region = RegionPos::from_chunk(chunk);
                    let idx = RegionPos::entry_index(chunk);
                    assert!(idx < 512, "Entry index must be < 512, got {idx}");

                    let reconstructed = region.chunk_pos_at(idx);
                    assert_eq!(
                        reconstructed, chunk,
                        "Coordinate roundtrip failed for {chunk:?}"
                    );
                }
            }
        }
    }
}
