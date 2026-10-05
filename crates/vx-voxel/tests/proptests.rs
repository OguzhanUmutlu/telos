//! Property-based correctness tests for voxel data storage and occupancy bitmasks.

#![allow(clippy::large_stack_arrays)]

use proptest::prelude::*;
use vx_voxel::{
    coords::{CHUNK_VOLUME, LocalIdx},
    occupancy::Occupancy,
    state::BlockStateId,
    storage::{HotBlocks, from_dense},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_dense_round_trip(
        distinct_count in 1usize..=64usize,
        seed in any::<u64>(),
    ) {
        let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
        let states: Vec<BlockStateId> = (0..distinct_count)
            .map(|id| BlockStateId::new(id as u32))
            .collect();

        let mut rng = seed;
        for voxel in &mut dense {
            // Simple XorShift64
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let choice = (rng as usize) % distinct_count;
            *voxel = states[choice];
        }

        let blocks = from_dense(&dense);

        if distinct_count == 1 {
            prop_assert!(blocks.is_uniform());
            prop_assert_eq!(blocks.heap_bytes(), 0);
        }

        for (i, &expected) in dense.iter().enumerate() {
            let idx = LocalIdx::new(i as u16).unwrap();
            let actual = blocks.get(idx);
            prop_assert_eq!(actual, expected, "Mismatch at local index {}", i);
        }
    }

    #[test]
    fn prop_hot_blocks_random_edits_and_compact(
        edits in prop::collection::vec((0..CHUNK_VOLUME, 0u32..32u32), 1..200),
    ) {
        let mut hot = HotBlocks::new_uniform(BlockStateId::AIR);
        let mut shadow = [BlockStateId::AIR; CHUNK_VOLUME];

        for (idx_raw, state_raw) in edits {
            let idx = LocalIdx::new(idx_raw as u16).unwrap();
            let state = BlockStateId::new(state_raw);
            hot.set(idx, state, false);
            shadow[idx_raw] = state;
        }

        // Verify all values match shadow
        for (i, &expected) in shadow.iter().enumerate() {
            let idx = LocalIdx::new(i as u16).unwrap();
            prop_assert_eq!(hot.get(idx), expected);
        }

        // Run compaction and re-verify
        hot.compact();

        for (i, &expected) in shadow.iter().enumerate() {
            let idx = LocalIdx::new(i as u16).unwrap();
            prop_assert_eq!(hot.get(idx), expected);
        }
    }

    #[test]
    fn prop_occupancy_incremental_matches_scratch(
        edits in prop::collection::vec((0u32..32u32, 0u32..32u32, 0u32..32u32, any::<bool>()), 1..100),
    ) {
        let mut hot = HotBlocks::new_uniform(BlockStateId::AIR);
        let mut occ = Occupancy::empty();
        let solid_state = BlockStateId::new(1); // Solid stone

        for (x, y, z, make_solid) in edits {
            let idx = LocalIdx::from_coords(x, y, z).unwrap();
            let new_state = if make_solid { solid_state } else { BlockStateId::AIR };
            let old_state = hot.get(idx);

            if (old_state == solid_state) != make_solid {
                occ.toggle(x, y, z);
            }
            hot.set(idx, new_state, false);
        }

        // Rebuild from scratch
        let scratch_occ = Occupancy::from_blocks(&vx_voxel::storage::Blocks::from_hot(&hot), |s| s == solid_state);

        // Verify incremental bitmasks match scratch bitmasks exactly
        prop_assert_eq!(occ.col_y, scratch_occ.col_y, "col_y mismatch");
        prop_assert_eq!(occ.col_x, scratch_occ.col_x, "col_x mismatch");
        prop_assert_eq!(occ.col_z, scratch_occ.col_z, "col_z mismatch");
    }
}
