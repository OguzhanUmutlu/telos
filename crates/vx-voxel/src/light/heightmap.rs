//! Chunk-local and column-wide heightmaps for sky light propagation across cubic chunks.

use crate::coords::CHUNK_SIZE;

/// Number of 1×1 vertical columns in a 32³ cubic chunk (32 × 32 = 1,024).
pub const COLUMN_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;

/// Chunk-local heightmap storing 1 + highest light-blocking $Y$ coordinate in each column.
///
/// Values:
/// - `0`: No light-blocking voxels in this column throughout the chunk.
/// - `1..=32`: Local $Y$ coordinate is `val - 1` (0..=31).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeightmap {
    /// Heightmap values indexed by `(z << 5) | x`.
    pub top_local: [u8; COLUMN_AREA],
}

impl ChunkHeightmap {
    /// Creates an empty chunk heightmap (no light-blocking blocks).
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            top_local: [0; COLUMN_AREA],
        }
    }

    /// Linear column index from chunk-local $(x, z)$ coordinates.
    #[inline]
    #[must_use]
    pub const fn col_idx(x: u32, z: u32) -> usize {
        ((z & 0x1F) << 5 | (x & 0x1F)) as usize
    }

    /// Gets local height value at $(x, z)$.
    #[inline]
    #[must_use]
    pub fn get(&self, x: u32, z: u32) -> u8 {
        self.top_local[Self::col_idx(x, z)]
    }

    /// Sets local height value at $(x, z)$.
    #[inline]
    pub fn set(&mut self, x: u32, z: u32, val: u8) {
        self.top_local[Self::col_idx(x, z)] = val;
    }

    /// Returns the local $Y$ coordinate of the highest light-blocking block in this column, if any.
    #[inline]
    #[must_use]
    pub fn highest_y(&self, x: u32, z: u32) -> Option<u32> {
        let val = self.get(x, z);
        if val > 0 {
            Some(u32::from(val - 1))
        } else {
            None
        }
    }

    /// Computes heightmap from a predicate checking whether a voxel at `(x, y, z)` blocks light.
    pub fn compute_from_predicate<F>(mut is_blocking: F) -> Self
    where
        F: FnMut(u32, u32, u32) -> bool,
    {
        let mut hm = Self::empty();
        for z in 0..32 {
            for x in 0..32 {
                let mut top = 0u8;
                for y in (0..32).rev() {
                    if is_blocking(x, y, z) {
                        top = (y + 1) as u8;
                        break;
                    }
                }
                hm.set(x, z, top);
            }
        }
        hm
    }

    /// Computes heightmap from 32-bit column occupancy masks where bit $y$ indicates light-blocking.
    ///
    /// Evaluates `32 - mask.leading_zeros()` in $O(1)$ cycles via `lzcnt`.
    #[must_use]
    pub fn from_column_masks(masks: &[u32; COLUMN_AREA]) -> Self {
        let mut top_local = [0u8; COLUMN_AREA];
        for i in 0..COLUMN_AREA {
            let mask = masks[i];
            if mask != 0 {
                top_local[i] = (32 - mask.leading_zeros()) as u8;
            }
        }
        Self { top_local }
    }

    /// Computes heightmap directly from chunk `Occupancy`.
    #[must_use]
    pub fn from_occupancy(occ: &crate::occupancy::Occupancy) -> Self {
        Self::from_column_masks(&occ.col_y)
    }
}

impl Default for ChunkHeightmap {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl std::fmt::Debug for ChunkHeightmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChunkHeightmap(1024 cols)")
    }
}

/// Column-wide heightmap storing world $Y$ coordinates of highest light-blocking blocks across cubic chunks.
#[derive(Clone, PartialEq, Eq)]
pub struct ColumnHeights {
    /// World $Y$ coordinate of highest light-blocking block per $(x, z)$ column.
    pub top: [i16; COLUMN_AREA],
}

impl ColumnHeights {
    /// Creates a column heightmap initialized to `i16::MIN`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self::new_uninitialized()
    }

    /// Creates a column heightmap initialized to `i16::MIN`.
    #[inline]
    #[must_use]
    pub const fn new_uninitialized() -> Self {
        Self {
            top: [i16::MIN; COLUMN_AREA],
        }
    }

    /// Creates a column heightmap with a default base surface level (e.g. 64).
    #[inline]
    #[must_use]
    pub const fn with_base_height(base_y: i16) -> Self {
        Self {
            top: [base_y; COLUMN_AREA],
        }
    }

    /// Returns the world $Y$ coordinate of the highest light-blocking block at local $(x, z)$.
    #[inline]
    #[must_use]
    pub fn get_top(&self, x: u32, z: u32) -> i16 {
        self.top[ChunkHeightmap::col_idx(x, z)]
    }

    /// Sets the highest world $Y$ coordinate at local $(x, z)$.
    #[inline]
    pub fn set_top(&mut self, x: u32, z: u32, world_y: i16) {
        self.top[ChunkHeightmap::col_idx(x, z)] = world_y;
    }

    /// Updates the column heightmap with the contents of a cubic chunk at chunk coordinate $cy$.
    #[allow(clippy::cast_possible_wrap)]
    pub fn update_chunk(&mut self, chunk_cy: i32, chunk_hm: &ChunkHeightmap) {
        let chunk_base_y = chunk_cy << 5;
        for z in 0..32 {
            for x in 0..32 {
                if let Some(local_y) = chunk_hm.highest_y(x, z) {
                    let world_y = (chunk_base_y + local_y as i32) as i16;
                    let idx = ChunkHeightmap::col_idx(x, z);
                    if world_y > self.top[idx] {
                        self.top[idx] = world_y;
                    }
                }
            }
        }
    }

    /// Checks if a voxel at $(x, z)$ and world coordinate $Y$ has unobstructed line-of-sight to the open sky.
    #[inline]
    #[must_use]
    pub fn is_above_top(&self, x: u32, world_y: i32, z: u32) -> bool {
        world_y > i32::from(self.get_top(x, z))
    }
}

impl Default for ColumnHeights {
    #[inline]
    fn default() -> Self {
        Self::new_uninitialized()
    }
}

impl std::fmt::Debug for ColumnHeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnHeights(1024 cols)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightmap_predicate_and_lzcnt() {
        let mut masks = [0u32; COLUMN_AREA];
        // Column (0, 0) has block at y=5
        masks[0] = 1 << 5;
        // Column (1, 0) has block at y=31
        masks[1] = 1 << 31;
        // Column (2, 0) has block at y=0 and y=12
        masks[2] = (1 << 0) | (1 << 12);

        let hm = ChunkHeightmap::from_column_masks(&masks);
        assert_eq!(hm.highest_y(0, 0), Some(5));
        assert_eq!(hm.highest_y(1, 0), Some(31));
        assert_eq!(hm.highest_y(2, 0), Some(12));
        assert_eq!(hm.highest_y(3, 0), None);
    }

    #[test]
    fn test_column_heights_aggregation() {
        let mut col = ColumnHeights::new_uninitialized();

        let mut hm_bottom = ChunkHeightmap::empty();
        hm_bottom.set(0, 0, 10); // local y = 9, chunk cy = 0 -> world y = 9

        let mut hm_top = ChunkHeightmap::empty();
        hm_top.set(0, 0, 4); // local y = 3, chunk cy = 2 -> world y = 2*32 + 3 = 67

        col.update_chunk(0, &hm_bottom);
        assert_eq!(col.get_top(0, 0), 9);

        col.update_chunk(2, &hm_top);
        assert_eq!(col.get_top(0, 0), 67);

        assert!(col.is_above_top(0, 68, 0));
        assert!(!col.is_above_top(0, 67, 0));
        assert!(!col.is_above_top(0, 50, 0));
    }
}
