//! Leaf decay simulation and foliage support connectivity checks.
//!
//! Uses a zero-allocation Chebyshev BFS within radius $\le 4$ to verify if
//! leaves are supported by wood logs or have become orphaned and must decay.

use crate::nav::NavWorldReader;
use telos_core::coords::BlockPos;

/// Maximum Chebyshev distance from a leaf to a supporting wood log before decay.
pub const LEAF_DECAY_RADIUS: i32 = 4;
const SEARCH_DIAMETER: usize = (LEAF_DECAY_RADIUS * 2 + 1) as usize; // 9
const BITSET_WORDS: usize = (SEARCH_DIAMETER * SEARCH_DIAMETER * SEARCH_DIAMETER).div_ceil(64); // 12 words

/// Tests whether the leaf block at `pos` is orphaned and should decay.
///
/// Returns `false` (supported) if a wood log (`world.registry().is_log(state)`) is
/// connected to this leaf within Chebyshev distance $\le \text{max\_dist}$ through contiguous leaves.
/// Returns `true` (decaying) if no wood log is reachable.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn is_leaf_decaying(pos: BlockPos, max_dist: u32, world: &impl NavWorldReader) -> bool {
    let limit = (max_dist as i32).clamp(1, LEAF_DECAY_RADIUS);
    let reg = world.registry();

    // If pos itself is a log, it is not decaying
    let root_state = world.get_block(pos);
    if reg.is_log(root_state) {
        return false;
    }
    if !reg.is_leaves(root_state) {
        return false;
    }

    // Fixed-size stack-allocated bitset and queue for BFS
    let mut visited = [0u64; BITSET_WORDS];
    let mut queue = [BlockPos::new(0, 0, 0); 729];
    let mut head = 0usize;
    let mut tail = 0usize;

    // Enqueue root pos
    queue[tail] = pos;
    tail += 1;
    let root_idx = local_bit_idx(0, 0, 0);
    visited[root_idx / 64] |= 1 << (root_idx % 64);

    let cardinals = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    while head < tail {
        let cur = queue[head];
        head += 1;

        for &(dx, dy, dz) in &cardinals {
            let nx = cur.x() + dx;
            let ny = cur.y() + dy;
            let nz = cur.z() + dz;

            let rel_x = nx - pos.x();
            let rel_y = ny - pos.y();
            let rel_z = nz - pos.z();

            if rel_x.abs() > limit || rel_y.abs() > limit || rel_z.abs() > limit {
                continue;
            }

            let neighbor_pos = BlockPos::new(nx, ny, nz);
            let state = world.get_block(neighbor_pos);

            // If neighbor is a wood log, the leaf is supported!
            if reg.is_log(state) {
                return false;
            }

            // If neighbor is leaves, explore it
            if reg.is_leaves(state) {
                let bit_idx = local_bit_idx(rel_x, rel_y, rel_z);
                let word_idx = bit_idx / 64;
                let mask = 1u64 << (bit_idx % 64);
                if (visited[word_idx] & mask) == 0 {
                    visited[word_idx] |= mask;
                    if tail < queue.len() {
                        queue[tail] = neighbor_pos;
                        tail += 1;
                    }
                }
            }
        }
    }

    // No log found within the search envelope: leaf decays
    true
}

#[inline]
const fn local_bit_idx(dx: i32, dy: i32, dz: i32) -> usize {
    let lx = (dx + LEAF_DECAY_RADIUS) as usize;
    let ly = (dy + LEAF_DECAY_RADIUS) as usize;
    let lz = (dz + LEAF_DECAY_RADIUS) as usize;
    lx + ly * SEARCH_DIAMETER + lz * SEARCH_DIAMETER * SEARCH_DIAMETER
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashMap;
    use telos_voxel::registry::BlockRegistry;
    use telos_voxel::state::BlockStateId;

    struct TestWorld {
        blocks: HashMap<BlockPos, BlockStateId>,
        registry: BlockRegistry,
    }

    impl NavWorldReader for TestWorld {
        fn get_block(&self, pos: BlockPos) -> BlockStateId {
            self.blocks.get(&pos).copied().unwrap_or(BlockStateId::AIR)
        }

        fn registry(&self) -> &BlockRegistry {
            &self.registry
        }
    }

    #[test]
    fn test_leaf_supported_directly_by_log() {
        let reg = BlockRegistry::standard();
        let log = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_log").unwrap())
            .unwrap()
            .default_state();
        let leaves = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_leaves").unwrap())
            .unwrap()
            .default_state();

        let mut world = TestWorld {
            blocks: HashMap::new(),
            registry: reg,
        };

        let leaf_pos = BlockPos::new(0, 1, 0);
        let log_pos = BlockPos::new(0, 0, 0);
        world.blocks.insert(leaf_pos, leaves);
        world.blocks.insert(log_pos, log);

        assert!(!is_leaf_decaying(leaf_pos, 4, &world));
    }

    #[test]
    fn test_leaf_supported_through_leaf_chain() {
        let reg = BlockRegistry::standard();
        let log = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_log").unwrap())
            .unwrap()
            .default_state();
        let leaves = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_leaves").unwrap())
            .unwrap()
            .default_state();

        let mut world = TestWorld {
            blocks: HashMap::new(),
            registry: reg,
        };

        // Log at (0, 0, 0), chain of leaves at (0, 1, 0), (0, 2, 0), (0, 3, 0)
        world.blocks.insert(BlockPos::new(0, 0, 0), log);
        world.blocks.insert(BlockPos::new(0, 1, 0), leaves);
        world.blocks.insert(BlockPos::new(0, 2, 0), leaves);
        world.blocks.insert(BlockPos::new(0, 3, 0), leaves);

        assert!(!is_leaf_decaying(BlockPos::new(0, 3, 0), 4, &world));
    }

    #[test]
    fn test_leaf_orphaned_when_log_removed() {
        let reg = BlockRegistry::standard();
        let leaves = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_leaves").unwrap())
            .unwrap()
            .default_state();

        let mut world = TestWorld {
            blocks: HashMap::new(),
            registry: reg,
        };

        world.blocks.insert(BlockPos::new(0, 1, 0), leaves);
        world.blocks.insert(BlockPos::new(0, 2, 0), leaves);

        assert!(is_leaf_decaying(BlockPos::new(0, 1, 0), 4, &world));
        assert!(is_leaf_decaying(BlockPos::new(0, 2, 0), 4, &world));
    }

    #[test]
    fn test_leaf_decay_beyond_distance_4() {
        let reg = BlockRegistry::standard();
        let log = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_log").unwrap())
            .unwrap()
            .default_state();
        let leaves = reg
            .get(&telos_core::ident::Identifier::new("telos", "oak_leaves").unwrap())
            .unwrap()
            .default_state();

        let mut world = TestWorld {
            blocks: HashMap::new(),
            registry: reg,
        };

        world.blocks.insert(BlockPos::new(0, 0, 0), log);
        for y in 1..=5 {
            world.blocks.insert(BlockPos::new(0, y, 0), leaves);
        }

        // Distance 4 is supported
        assert!(!is_leaf_decaying(BlockPos::new(0, 4, 0), 4, &world));
        // Distance 5 is orphaned (> 4)
        assert!(is_leaf_decaying(BlockPos::new(0, 5, 0), 4, &world));
    }
}
