//! Immutable packed bitboard with power-of-two bit widths {1, 2, 4, 8, 16}.
//!
//! Entries never cross a 64-bit word boundary.

use crate::{coords::LocalIdx, state::BlockStateId};

/// Bitboard holding paletted voxel entries with power-of-two bit width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    pub(crate) log2: u8,
    pub(crate) palette: Box<[BlockStateId]>,
    pub(crate) words: Box<[u64]>,
}

impl Packed {
    /// Creates a new `Packed` bitboard with specified `log2` width (0..=4) and palette.
    ///
    /// # Panics
    /// Panics if `log2 > 4` or palette length exceeds `1 << (1 << log2)`.
    #[must_use]
    pub fn new(log2: u8, palette: Box<[BlockStateId]>) -> Self {
        assert!(log2 <= 4, "log2 bit width must be <= 4 (max 16 bits)");
        let bits = 1usize << log2;
        let max_palette_len = 1usize << bits;
        assert!(
            palette.len() <= max_palette_len,
            "Palette length {} exceeds maximum capacity {} for log2={}",
            palette.len(),
            max_palette_len,
            log2
        );

        let word_count = 512usize << log2;
        let words = vec![0u64; word_count].into_boxed_slice();

        Self {
            log2,
            palette,
            words,
        }
    }

    /// Bit width exponent `log2` (0 => 1 bit, 1 => 2 bits, 2 => 4 bits, 3 => 8 bits, 4 => 16 bits).
    #[inline]
    #[must_use]
    pub const fn log2(&self) -> u8 {
        self.log2
    }

    /// Actual bit width per voxel entry (1, 2, 4, 8, or 16).
    #[inline]
    #[must_use]
    pub const fn bits(&self) -> usize {
        1usize << self.log2
    }

    /// Current palette slice.
    #[inline]
    #[must_use]
    pub fn palette(&self) -> &[BlockStateId] {
        &self.palette
    }

    /// Raw words backing storage.
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Looks up the voxel `BlockStateId` at index `idx`.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: LocalIdx) -> BlockStateId {
        let slot = get_raw(&self.words, self.log2, idx.as_usize());
        // SAFETY: slot is guaranteed to be a valid index into palette by construction
        unsafe { *self.palette.get_unchecked(slot) }
    }

    /// Gets the raw palette slot index at local index `i`.
    #[inline]
    #[must_use]
    pub fn get_slot(&self, idx: LocalIdx) -> usize {
        get_raw(&self.words, self.log2, idx.as_usize())
    }

    /// Total heap memory used by palette and words in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        (self.palette.len() * size_of::<BlockStateId>()) + (self.words.len() * size_of::<u64>())
    }
}

/// Reads a raw palette index from words slice using bitwise arithmetic.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn get_raw(words: &[u64], log2: u8, i: usize) -> usize {
    let pw_log2 = 6 - u32::from(log2);
    let word_idx = i >> pw_log2;
    // SAFETY: i < 32768, words.len() == 512 << log2, so word_idx is always in bounds
    let w = unsafe { *words.get_unchecked(word_idx) };
    let shift = ((i & ((1usize << pw_log2) - 1)) as u32) << log2;
    let mask = (1u64 << (1u32 << log2)) - 1;
    ((w >> shift) & mask) as usize
}

/// Writes a raw palette index into words slice using bitwise arithmetic.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn set_raw(words: &mut [u64], log2: u8, i: usize, v: usize) {
    let pw_log2 = 6 - u32::from(log2);
    let word_idx = i >> pw_log2;
    let shift = ((i & ((1usize << pw_log2) - 1)) as u32) << log2;
    let mask = ((1u64 << (1u32 << log2)) - 1) << shift;
    // SAFETY: i < 32768, words.len() == 512 << log2, so word_idx is always in bounds
    let w = unsafe { words.get_unchecked_mut(word_idx) };
    *w = (*w & !mask) | ((v as u64) << shift);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::CHUNK_VOLUME;

    #[test]
    fn test_packed_round_trip_all_widths() {
        for log2 in 0..=4 {
            let bits = 1 << log2;
            let max_val = (1 << bits) - 1;
            let palette: Vec<BlockStateId> =
                (0..=max_val).map(|v| BlockStateId::new(v as u32)).collect();
            let mut packed = Packed::new(log2, palette.into_boxed_slice());

            for i in 0..CHUNK_VOLUME {
                let val = i & max_val;
                set_raw(&mut packed.words, log2, i, val);
            }

            for i in 0..CHUNK_VOLUME {
                let expected = i & max_val;
                let actual = get_raw(&packed.words, log2, i);
                assert_eq!(actual, expected, "Mismatch at index {i} for log2={log2}");
            }
        }
    }
}
