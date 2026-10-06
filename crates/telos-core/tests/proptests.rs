//! Property-based invariants tests for telos-core.

use proptest::prelude::*;
use telos_core::{BlockPos, Identifier, LocalPos};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn prop_block_pos_chunk_local_round_trip(
        x in -1_000_000..=1_000_000,
        y in -1_000..=2_000,
        z in -1_000_000..=1_000_000,
    ) {
        let block = BlockPos::new(x, y, z);
        let (chunk, local) = block.to_chunk_and_local();
        let reconstructed = chunk.block_pos(local);

        prop_assert_eq!(block, reconstructed, "Decomposed and reconstructed BlockPos did not match!");
    }

    #[test]
    fn prop_local_pos_components(
        x in 0u8..32,
        y in 0u8..32,
        z in 0u8..32,
    ) {
        let local = LocalPos::from_xyz(x, y, z);
        prop_assert_eq!(local.x(), x);
        prop_assert_eq!(local.y(), y);
        prop_assert_eq!(local.z(), z);
        prop_assert_eq!(local.to_xyz(), (x, y, z));
    }

    #[test]
    fn prop_valid_identifier_round_trip(
        ns in "[a-z0-9_.-]{1,16}",
        path in "[a-z0-9_./-]{1,32}",
    ) {
        let raw = format!("{ns}:{path}");
        let id: Identifier = raw.parse().expect("Valid pattern must parse");
        prop_assert_eq!(id.namespace(), ns.as_str());
        prop_assert_eq!(id.path(), path.as_str());
        prop_assert_eq!(id.to_string(), raw);
    }
}
