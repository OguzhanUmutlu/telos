//! Mutable editable chunk storage with slot refcounts and palette growth.

use hashbrown::HashMap;

use crate::{
    coords::{CHUNK_VOLUME, LocalIdx},
    state::BlockStateId,
    storage::packed::{Packed, get_raw, set_raw},
};

/// Mutable, editable chunk storage representation with refcounts and slot reuse.
#[derive(Debug, Clone)]
pub struct HotBlocks {
    pub(crate) packed: Packed,
    pub(crate) refcount: Vec<u16>,
    pub(crate) reverse: HashMap<BlockStateId, u16>,
    pub(crate) free: Vec<u16>,
    pub(crate) tickable: u16,
}

impl HotBlocks {
    /// Creates a `HotBlocks` container from a single uniform block state.
    #[must_use]
    pub fn new_uniform(state: BlockStateId) -> Self {
        let palette = vec![state].into_boxed_slice();
        let packed = Packed::new(0, palette);
        let mut reverse = HashMap::new();
        reverse.insert(state, 0);

        Self {
            packed,
            refcount: vec![CHUNK_VOLUME as u16],
            reverse,
            free: Vec::new(),
            tickable: 0,
        }
    }

    /// Creates a `HotBlocks` container from an existing `Packed` bitboard.
    #[must_use]
    pub fn from_packed(packed: Packed) -> Self {
        let mut refcount = vec![0u16; packed.palette.len()];
        for i in 0..CHUNK_VOLUME {
            let slot = get_raw(&packed.words, packed.log2, i);
            refcount[slot] += 1;
        }

        let mut reverse = HashMap::with_capacity(packed.palette.len());
        let mut free = Vec::new();

        for (slot, (&state, &count)) in packed.palette.iter().zip(refcount.iter()).enumerate() {
            if count > 0 {
                reverse.insert(state, slot as u16);
            } else {
                free.push(slot as u16);
            }
        }

        Self {
            packed,
            refcount,
            reverse,
            free,
            tickable: 0,
        }
    }

    /// Reads the block state at local index `idx`.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: LocalIdx) -> BlockStateId {
        self.packed.get(idx)
    }

    /// Mutates the voxel at `idx` to `new_state`.
    ///
    /// Returns `(old_state, changed)`.
    pub fn set(
        &mut self,
        idx: LocalIdx,
        new_state: BlockStateId,
        is_tickable: bool,
    ) -> (BlockStateId, bool) {
        let i = idx.as_usize();
        let old_slot = get_raw(&self.packed.words, self.packed.log2, i);
        let old_state = self.packed.palette[old_slot];

        if old_state == new_state {
            return (old_state, false);
        }

        // 1. Decrement refcount of old slot
        self.refcount[old_slot] -= 1;
        if self.refcount[old_slot] == 0 {
            self.free.push(old_slot as u16);
            self.reverse.remove(&old_state);
        }

        // 2. Resolve or allocate new slot
        let new_slot = self.resolve_or_allocate_slot(new_state);

        // 3. Write into packed bitboard
        set_raw(
            &mut self.packed.words,
            self.packed.log2,
            i,
            new_slot as usize,
        );
        self.refcount[new_slot as usize] += 1;

        if is_tickable {
            self.tickable = self.tickable.saturating_add(1);
        }

        (old_state, true)
    }

    fn resolve_or_allocate_slot(&mut self, state: BlockStateId) -> u16 {
        if let Some(&slot) = self.reverse.get(&state) {
            return slot;
        }

        // Check freelist first
        if let Some(free_slot) = self.free.pop() {
            let mut pal = Vec::from(std::mem::take(&mut self.packed.palette));
            pal[free_slot as usize] = state;
            self.packed.palette = pal.into_boxed_slice();
            self.reverse.insert(state, free_slot);
            return free_slot;
        }

        let max_palette_cap = 1usize << self.packed.bits();
        if self.packed.palette.len() < max_palette_cap {
            let mut pal = Vec::from(std::mem::take(&mut self.packed.palette));
            let new_slot = pal.len() as u16;
            pal.push(state);
            self.packed.palette = pal.into_boxed_slice();
            self.refcount.push(0);
            self.reverse.insert(state, new_slot);
            return new_slot;
        }

        // Palette is saturated: grow bit width (log2 += 1)
        self.grow();
        self.resolve_or_allocate_slot(state)
    }

    /// Grows the bit width by 1 level (repacks into double bit width).
    fn grow(&mut self) {
        assert!(
            self.packed.log2 < 4,
            "Cannot grow palette beyond 16 bits (32768 states)"
        );

        let old_log2 = self.packed.log2;
        let new_log2 = old_log2 + 1;
        let old_words = &self.packed.words;

        let new_word_count = 512usize << new_log2;
        let mut new_words = vec![0u64; new_word_count].into_boxed_slice();

        // Repack all 32768 entries
        for i in 0..CHUNK_VOLUME {
            let slot = get_raw(old_words, old_log2, i);
            set_raw(&mut new_words, new_log2, i, slot);
        }

        self.packed.log2 = new_log2;
        self.packed.words = new_words;
    }

    /// Compacts palette entries and shrinks bit width if possible.
    ///
    /// Returns `true` if compacted.
    pub fn compact(&mut self) -> bool {
        let live_count = self.refcount.iter().filter(|&&c| c > 0).count();
        if live_count == 0 {
            return false;
        }

        // Determine minimal power-of-two bits needed
        let min_log2 = match live_count {
            0..=2 => 0,
            3..=4 => 1,
            5..=16 => 2,
            17..=256 => 3,
            _ => 4,
        };

        let mut new_palette = Vec::with_capacity(live_count);
        let mut old_to_new = vec![0u16; self.packed.palette.len()];
        let mut new_refcount = Vec::with_capacity(live_count);
        let mut new_reverse = HashMap::with_capacity(live_count);

        for (old_slot, (&state, &count)) in self
            .packed
            .palette
            .iter()
            .zip(self.refcount.iter())
            .enumerate()
        {
            if count > 0 {
                let new_slot = new_palette.len() as u16;
                new_palette.push(state);
                new_refcount.push(count);
                new_reverse.insert(state, new_slot);
                old_to_new[old_slot] = new_slot;
            }
        }

        let new_word_count = 512usize << min_log2;
        let mut new_words = vec![0u64; new_word_count].into_boxed_slice();

        for i in 0..CHUNK_VOLUME {
            let old_slot = get_raw(&self.packed.words, self.packed.log2, i);
            let new_slot = old_to_new[old_slot];
            set_raw(&mut new_words, min_log2, i, new_slot as usize);
        }

        self.packed.log2 = min_log2;
        self.packed.palette = new_palette.into_boxed_slice();
        self.packed.words = new_words;
        self.refcount = new_refcount;
        self.reverse = new_reverse;
        self.free.clear();

        true
    }

    /// Number of distinct live block states in the chunk.
    #[must_use]
    pub fn distinct_live_states(&self) -> usize {
        self.refcount.iter().filter(|&&c| c > 0).count()
    }

    /// Heap memory used by `HotBlocks` in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.packed.heap_bytes()
            + (self.refcount.len() * size_of::<u16>())
            + (self.reverse.capacity() * (size_of::<BlockStateId>() + size_of::<u16>() + 16))
            + (self.free.capacity() * size_of::<u16>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_blocks_set_and_growth() {
        let mut hot = HotBlocks::new_uniform(BlockStateId::AIR);
        assert_eq!(hot.packed.bits(), 1);

        // Fill with distinct states to force growth from 1 -> 2 -> 4 bits
        for i in 1..=16 {
            let state = BlockStateId::new(i);
            let idx = LocalIdx::new(i as u16).unwrap();
            hot.set(idx, state, false);
        }

        assert_eq!(hot.packed.bits(), 8); // 17 states (air + 16 states) require 8 bits (cap 16 exceeded)
        assert_eq!(hot.distinct_live_states(), 17);

        // Verify values intact
        assert_eq!(hot.get(LocalIdx::new(0).unwrap()), BlockStateId::AIR);
        for i in 1..=16 {
            let idx = LocalIdx::new(i as u16).unwrap();
            assert_eq!(hot.get(idx), BlockStateId::new(i));
        }
    }

    #[test]
    fn test_hot_blocks_compact() {
        let mut hot = HotBlocks::new_uniform(BlockStateId::AIR);

        // Add stone then replace back with air
        let stone = BlockStateId::new(1);
        let idx = LocalIdx::new(10).unwrap();
        hot.set(idx, stone, false);
        assert_eq!(hot.distinct_live_states(), 2);

        hot.set(idx, BlockStateId::AIR, false);
        assert_eq!(hot.distinct_live_states(), 1);

        hot.compact();
        assert_eq!(hot.packed.bits(), 1);
        assert_eq!(hot.packed.palette.len(), 1);
    }
}
