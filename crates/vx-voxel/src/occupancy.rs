//! 3-axis occupancy bitmasks for binary greedy meshing.
//!
//! Maintains 32-bit column bitboards along X, Y, and Z for fast bitwise face culling.

use crate::{coords::LocalIdx, state::BlockStateId, storage::Blocks};

/// 3-axis occupancy bitmasks (12 KiB per chunk) consumed by binary greedy meshers.
///
/// Bit is set $\iff$ voxel is an opaque full cube (`StateFlags::OPAQUE_FULL`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occupancy {
    /// Columns oriented along Y: index `(z << 5) | x`, bit $y$.
    pub col_y: Box<[u32; 1024]>,
    /// Columns oriented along X: index `(y << 5) | z`, bit $x$.
    pub col_x: Box<[u32; 1024]>,
    /// Columns oriented along Z: index `(y << 5) | x`, bit $z$.
    pub col_z: Box<[u32; 1024]>,
}

impl Default for Occupancy {
    fn default() -> Self {
        Self::empty()
    }
}

impl Occupancy {
    /// Creates an empty occupancy bitmask (all voxels non-occluding).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            col_y: vec![0u32; 1024].into_boxed_slice().try_into().unwrap(),
            col_x: vec![0u32; 1024].into_boxed_slice().try_into().unwrap(),
            col_z: vec![0u32; 1024].into_boxed_slice().try_into().unwrap(),
        }
    }

    /// Creates a fully solid occupancy bitmask (all voxels occluding).
    #[must_use]
    pub fn solid() -> Self {
        Self {
            col_y: vec![u32::MAX; 1024].into_boxed_slice().try_into().unwrap(),
            col_x: vec![u32::MAX; 1024].into_boxed_slice().try_into().unwrap(),
            col_z: vec![u32::MAX; 1024].into_boxed_slice().try_into().unwrap(),
        }
    }

    /// Constructs occupancy bitmasks from a `Blocks` container using the provided flags predicate.
    #[must_use]
    pub fn from_blocks(blocks: &Blocks, is_opaque: impl Fn(BlockStateId) -> bool) -> Self {
        match blocks {
            Blocks::Uniform(state) => {
                if is_opaque(*state) {
                    Self::solid()
                } else {
                    Self::empty()
                }
            }
            Blocks::Packed(packed) => {
                let mut occ = Self::empty();

                // Precompute palette slot opacity to avoid redundant predicate evaluation
                let mut slot_is_opaque = [false; 256];
                for (slot, &state) in packed.palette().iter().enumerate().take(256) {
                    slot_is_opaque[slot] = is_opaque(state);
                }

                for y in 0..32u32 {
                    let mut layer_x = [0u32; 32];
                    let y_bit = 1u32 << y;
                    let y_shift = (y << 5) as usize;

                    for z in 0..32u32 {
                        let mut row_bits = 0u32;
                        let z_shift = (z << 5) as usize;

                        for x in 0..32u32 {
                            let idx = LocalIdx::from_coords_unchecked(x, y, z);
                            let slot = packed.get_slot(idx);
                            let opaque = if slot < 256 {
                                slot_is_opaque[slot]
                            } else {
                                is_opaque(packed.palette()[slot])
                            };

                            if opaque {
                                occ.col_y[z_shift | (x as usize)] |= y_bit;
                                row_bits |= 1 << x;
                            }
                        }
                        occ.col_x[y_shift | (z as usize)] = row_bits;
                        layer_x[z as usize] = row_bits;
                    }

                    // 2. Transpose layer_x (rows z, cols x) to get layer_z (rows x, cols z)
                    transpose_32x32(&mut layer_x);
                    for x in 0..32u32 {
                        occ.col_z[y_shift | (x as usize)] = layer_x[x as usize];
                    }
                }

                occ
            }
        }
    }

    /// Incrementally toggles the occupancy bit for voxel `(x, y, z)`.
    ///
    /// Executes exactly 3 XOR operations (< 2 ns).
    #[inline]
    pub fn toggle(&mut self, x: u32, y: u32, z: u32) {
        debug_assert!(x < 32 && y < 32 && z < 32);
        self.col_y[((z << 5) | x) as usize] ^= 1 << y;
        self.col_x[((y << 5) | z) as usize] ^= 1 << x;
        self.col_z[((y << 5) | x) as usize] ^= 1 << z;
    }

    /// Returns `true` if voxel `(x, y, z)` is solid.
    #[inline]
    #[must_use]
    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        (self.col_y[((z << 5) | x) as usize] & (1 << y)) != 0
    }
}

/// In-place 32×32 bit matrix transpose using Hacker's Delight algorithm (5 rounds of masks).
pub fn transpose_32x32(m: &mut [u32; 32]) {
    let mut k = 16usize;
    let mut mask = 0x0000_FFFF_u32;

    while k > 0 {
        for i in 0..32 {
            if (i & k) == 0 {
                let j = i + k;
                let t = ((m[i] >> k) ^ m[j]) & mask;
                m[i] ^= t << k;
                m[j] ^= t;
            }
        }
        k >>= 1;
        mask ^= mask << k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_transpose_32x32_correctness() {
        let mut matrix = [0u32; 32];
        // Populate test diagonal and corners
        matrix[0] = 0b1000_0000;
        matrix[7] = 0b0000_0001;
        matrix[31] = 0x8000_0000;

        let original = matrix;
        transpose_32x32(&mut matrix);

        for row in 0..32 {
            for col in 0..32 {
                let bit_before = (original[row] >> col) & 1;
                let bit_after = (matrix[col] >> row) & 1;
                assert_eq!(bit_before, bit_after, "Mismatch at row {row}, col {col}");
            }
        }
    }

    #[test]
    fn test_occupancy_toggle() {
        let mut occ = Occupancy::empty();
        assert!(!occ.is_solid(5, 10, 15));

        occ.toggle(5, 10, 15);
        assert!(occ.is_solid(5, 10, 15));

        occ.toggle(5, 10, 15);
        assert!(!occ.is_solid(5, 10, 15));
    }
}
