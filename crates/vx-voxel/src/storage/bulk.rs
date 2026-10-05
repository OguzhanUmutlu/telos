//! High-throughput bulk operations for worldgen packing and layer fills.

use hashbrown::HashMap;

use crate::{
    coords::{CHUNK_VOLUME, LocalIdx},
    state::BlockStateId,
    storage::{Blocks, HotBlocks, packed::Packed},
};

/// Constructs an optimized `Blocks` representation from a dense array of 32,768 states.
///
/// If all voxels are identical, returns `Blocks::Uniform` without allocating any heap memory.
#[must_use]
pub fn from_dense(dense: &[BlockStateId; CHUNK_VOLUME]) -> Blocks {
    let first = dense[0];
    let mut is_uniform = true;
    for &state in dense.iter().skip(1) {
        if state != first {
            is_uniform = false;
            break;
        }
    }

    if is_uniform {
        return Blocks::Uniform(first);
    }

    // Pass 1: Build palette with state -> slot mapping using run-length caching
    let mut palette = Vec::with_capacity(16);
    let mut state_to_slot = HashMap::with_capacity(16);

    let mut last_state = BlockStateId(u32::MAX);

    for &state in dense {
        if state == last_state {
            continue;
        }
        if !state_to_slot.contains_key(&state) {
            let slot = palette.len();
            palette.push(state);
            state_to_slot.insert(state, slot);
        }
        last_state = state;
    }

    let min_log2 = match palette.len() {
        0..=2 => 0,
        3..=4 => 1,
        5..=16 => 2,
        17..=256 => 3,
        _ => 4,
    };

    let word_count = 512usize << min_log2;
    let mut words = vec![0u64; word_count].into_boxed_slice();

    // Pass 2: Pack entries directly into 64-bit words without intermediate memory passes
    let entries_per_word = 64 >> min_log2;
    let bits = 1usize << min_log2;
    let max_state = palette.iter().map(|s| s.0).max().unwrap_or(0);

    if max_state < 256 {
        let mut lut = [0u8; 256];
        for (slot, &state) in palette.iter().enumerate() {
            lut[state.0 as usize] = slot as u8;
        }

        for (word_idx, chunk) in dense.chunks_exact(entries_per_word).enumerate() {
            let mut word = 0u64;
            for (sub_idx, &state) in chunk.iter().enumerate() {
                let slot = u64::from(lut[state.0 as usize]);
                word |= slot << (sub_idx * bits);
            }
            words[word_idx] = word;
        }
    } else {
        last_state = BlockStateId(u32::MAX);
        let mut last_slot = 0usize;

        for (word_idx, chunk) in dense.chunks_exact(entries_per_word).enumerate() {
            let mut word = 0u64;
            for (sub_idx, &state) in chunk.iter().enumerate() {
                let slot = if state == last_state {
                    last_slot
                } else {
                    let s = state_to_slot[&state];
                    last_state = state;
                    last_slot = s;
                    s
                };
                word |= (slot as u64) << (sub_idx * bits);
            }
            words[word_idx] = word;
        }
    }

    let packed = Packed {
        log2: min_log2,
        palette: palette.into_boxed_slice(),
        words,
    };

    Blocks::Packed(Box::new(packed))
}

/// Fills an axis-aligned box inside a `HotBlocks` container with a given `state`.
pub fn fill_box(
    hot: &mut HotBlocks,
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    state: BlockStateId,
) {
    let min_x = min_x.min(31);
    let max_x = max_x.min(31);
    let min_y = min_y.min(31);
    let max_y = max_y.min(31);
    let min_z = min_z.min(31);
    let max_z = max_z.min(31);

    for y in min_y..=max_y {
        for z in min_z..=max_z {
            for x in min_x..=max_x {
                let idx = LocalIdx::from_coords_unchecked(x, y, z);
                hot.set(idx, state, false);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::large_stack_arrays)]
mod tests {
    use super::*;

    #[test]
    fn test_from_dense_uniform_elision() {
        let dense = [BlockStateId::AIR; CHUNK_VOLUME];
        let blocks = from_dense(&dense);
        assert!(blocks.is_uniform());
        assert_eq!(blocks.heap_bytes(), 0);
    }

    #[test]
    fn test_from_dense_non_uniform() {
        let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
        dense[10] = BlockStateId::new(1); // Stone
        let blocks = from_dense(&dense);
        assert!(!blocks.is_uniform());
        assert_eq!(blocks.get(LocalIdx::new(10).unwrap()), BlockStateId::new(1));
        assert_eq!(blocks.get(LocalIdx::new(0).unwrap()), BlockStateId::AIR);
    }
}
