//! Integration tests for procedural tree generation and foliage canopies across cubic chunk seams.

use telos_core::coords::ChunkPos;
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::generator::WorldGenerator;

#[test]
fn test_tree_generation_spawns_wood_and_leaves() {
    let registry = BlockRegistry::standard();
    let seed = 12345;
    let generator = WorldGenerator::new(seed, &registry);

    let mut found_wood = false;
    let mut found_leaves = false;

    // Scan chunks around origin in a temperate forest/plains area
    for cy in 0..=3 {
        for cz in -2..=2 {
            for cx in -2..=2 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for z in 0..32 {
                    for x in 0..32 {
                        for y in 0..32 {
                            let idx = LocalIdx::from_coords(x, y, z).unwrap();
                            let block = chunk.get(idx);
                            if registry.is_log(block) {
                                found_wood = true;
                            }
                            if registry.is_leaves(block) {
                                found_leaves = true;
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(found_wood, "Expected tree wood logs to generate in world");
    assert!(found_leaves, "Expected tree leaves to generate in world");
}

#[test]
fn test_tree_vertical_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let seed = 42;
    let generator = WorldGenerator::new(seed, &registry);

    // Look for chunks where a tree trunk crosses the vertical boundary y=31 -> y=32 (cy -> cy+1)
    let mut verified_boundary_trees = 0;

    for cz in -4..=4 {
        for cx in -4..=4 {
            for cy in 0..=3 {
                let lower_chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                let upper_chunk = generator.generate_chunk(ChunkPos::new(cx, cy + 1, cz));

                for z in 0..32 {
                    for x in 0..32 {
                        let lower_idx = LocalIdx::from_coords(x, 31, z).unwrap();
                        let upper_idx = LocalIdx::from_coords(x, 0, z).unwrap();
                        let lower_top = lower_chunk.get(lower_idx);
                        let upper_bottom = upper_chunk.get(upper_idx);

                        if registry.is_log(lower_top) && registry.is_log(upper_bottom) {
                            verified_boundary_trees += 1;
                        }
                    }
                }
            }
        }
    }

    assert!(
        verified_boundary_trees > 0,
        "Expected at least one tree trunk to cross vertical chunk boundaries seamlessly"
    );
}

#[test]
fn test_tree_horizontal_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let seed = 999;
    let generator = WorldGenerator::new(seed, &registry);

    let mut verified_horizontal_leaves = 0;

    // Verify leaf canopies spanning horizontally across x=31 -> x=0 (cx -> cx+1)
    for cz in -3..=3 {
        for cy in 0..=3 {
            for cx in -3..=3 {
                let left_chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                let right_chunk = generator.generate_chunk(ChunkPos::new(cx + 1, cy, cz));

                for y in 0..32 {
                    for z in 0..32 {
                        let left_idx = LocalIdx::from_coords(31, y, z).unwrap();
                        let right_idx = LocalIdx::from_coords(0, y, z).unwrap();
                        let left_edge = left_chunk.get(left_idx);
                        let right_edge = right_chunk.get(right_idx);

                        if registry.is_leaves(left_edge) && registry.is_leaves(right_edge) {
                            verified_horizontal_leaves += 1;
                        }
                    }
                }
            }
        }
    }

    assert!(
        verified_horizontal_leaves > 0,
        "Expected tree foliage to span seamlessly across horizontal chunk boundaries"
    );
}

#[test]
fn test_tree_generation_determinism() {
    let registry = BlockRegistry::standard();
    let seed = 0xDEAD_BEEF_CAFE;
    let gen1 = WorldGenerator::new(seed, &registry);
    let gen2 = WorldGenerator::new(seed, &registry);

    for cz in -1..=1 {
        for cy in 0..=2 {
            for cx in -1..=1 {
                let pos = ChunkPos::new(cx, cy, cz);
                let c1 = gen1.generate_chunk(pos);
                let c2 = gen2.generate_chunk(pos);

                for y in 0..32 {
                    for z in 0..32 {
                        for x in 0..32 {
                            let idx = LocalIdx::from_coords(x, y, z).unwrap();
                            assert_eq!(
                                c1.get(idx),
                                c2.get(idx),
                                "Mismatch at ({x}, {y}, {z}) in chunk {pos:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
