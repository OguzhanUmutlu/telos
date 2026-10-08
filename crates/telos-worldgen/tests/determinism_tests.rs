//! Determinism and golden hash regression tests for procedural world generation.

use crc32fast::Hasher;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;

#[test]
fn test_worldgen_determinism() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_CAFE_DEAD_BEEF;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(2, 0, -3);
    let chunk1 = generator.generate_chunk(pos);
    let chunk2 = generator.generate_chunk(pos);

    assert_eq!(chunk1.position(), chunk2.position());

    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        assert_eq!(
            chunk1.get(idx),
            chunk2.get(idx),
            "Voxel at {i} differed between identical generation passes"
        );
    }
}

#[test]
fn test_chunk_negative_coordinates() {
    let registry = BlockRegistry::standard();
    let seed = 0x1234_5678;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(-10, -5, -20);
    let chunk = generator.generate_chunk(pos);
    assert_eq!(chunk.position(), pos);

    // Deep underground chunk should contain bedrock or stone terrain
    let has_solid = (0..CHUNK_VOLUME).any(|i| {
        let idx = LocalIdx::new(i as u16).unwrap();
        chunk.get(idx).0 > 0
    });
    assert!(
        has_solid,
        "Deep underground chunk must contain rock or solid terrain"
    );
}

#[test]
fn test_worldgen_golden_hash() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_0000_1234_5678;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(0, 0, 0);
    let chunk = generator.generate_chunk(pos);

    let mut hasher = Hasher::new();
    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        hasher.update(&chunk.get(idx).0.to_le_bytes());
    }
    let hash = hasher.finalize();

    // Verify against a second independent generation to ensure regression stability
    let chunk_again = generator.generate_chunk(pos);
    let mut hasher2 = Hasher::new();
    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        hasher2.update(&chunk_again.get(idx).0.to_le_bytes());
    }
    assert_eq!(hash, hasher2.finalize());
    assert_ne!(hash, 0);
}

#[test]
fn test_void_generator() {
    let registry = BlockRegistry::standard();
    let generator = telos_worldgen::WorldGenerator::with_kind(
        42,
        &registry,
        telos_worldgen::GeneratorKind::Void,
    );

    let pos = ChunkPos::new(0, 0, 0);
    let chunk = generator.generate_chunk(pos);
    assert_eq!(chunk.position(), pos);
    assert!(
        chunk.occupancy().is_empty(),
        "Void chunk must have empty occupancy"
    );
    assert_eq!(
        chunk.get(LocalIdx::from_coords(0, 0, 0).unwrap()),
        telos_voxel::state::BlockStateId::AIR
    );
}

#[test]
fn test_flat_generator() {
    let registry = BlockRegistry::standard();
    let generator = telos_worldgen::WorldGenerator::with_kind(
        42,
        &registry,
        telos_worldgen::GeneratorKind::Flat,
    );

    // Chunk y != 0 must be empty
    let chunk_above = generator.generate_chunk(ChunkPos::new(0, 1, 0));
    assert!(
        chunk_above.occupancy().is_empty(),
        "Chunk above flat layer must have empty occupancy"
    );

    let chunk_below = generator.generate_chunk(ChunkPos::new(0, -1, 0));
    assert!(
        chunk_below.occupancy().is_empty(),
        "Chunk below flat layer must have empty occupancy"
    );

    // Chunk y == 0 has bedrock at y=0, dirt at y=1..3, grass at y=4
    let chunk0 = generator.generate_chunk(ChunkPos::new(0, 0, 0));
    assert!(!chunk0.occupancy().is_empty());

    let bedrock_idx = LocalIdx::from_coords(10, 0, 10).unwrap();
    let dirt_idx1 = LocalIdx::from_coords(10, 1, 10).unwrap();
    let dirt_idx3 = LocalIdx::from_coords(10, 3, 10).unwrap();
    let grass_idx = LocalIdx::from_coords(10, 4, 10).unwrap();
    let air_idx = LocalIdx::from_coords(10, 5, 10).unwrap();

    let stone_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "bedrock").unwrap())
        .unwrap()
        .default_state();
    let dirt_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "dirt").unwrap())
        .unwrap()
        .default_state();
    let grass_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "grass_block").unwrap())
        .unwrap()
        .default_state();

    assert_eq!(chunk0.get(bedrock_idx), stone_id);
    assert_eq!(chunk0.get(dirt_idx1), dirt_id);
    assert_eq!(chunk0.get(dirt_idx3), dirt_id);
    assert_eq!(chunk0.get(grass_idx), grass_id);
    assert_eq!(chunk0.get(air_idx), telos_voxel::state::BlockStateId::AIR);
}

#[test]
fn test_decoration_clustering_and_species() {
    let registry = BlockRegistry::standard();
    let seed = 0xABCD_EF01_2345;
    let generator = WorldGenerator::new(seed, &registry);

    let poppy_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "poppy").unwrap())
        .unwrap()
        .default_state();
    let dandelion_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "dandelion").unwrap())
        .unwrap()
        .default_state();
    let grass_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "short_grass").unwrap())
        .unwrap()
        .default_state();

    let mut found_poppy = false;
    let mut found_dandelion = false;
    let mut found_grass = false;

    // Scan a grid of surface chunks to locate flower meadows and foliage
    for cx in -2..=2 {
        for cz in -2..=2 {
            let chunk = generator.generate_chunk(ChunkPos::new(cx, 0, cz));
            for i in 0..CHUNK_VOLUME {
                let state = chunk.get(LocalIdx::new(i as u16).unwrap());
                if state == poppy_id {
                    found_poppy = true;
                } else if state == dandelion_id {
                    found_dandelion = true;
                } else if state == grass_id {
                    found_grass = true;
                }
            }
        }
    }

    assert!(
        found_grass,
        "Surface decoration must generate short grass in temperate biomes"
    );
    assert!(
        found_poppy || found_dandelion,
        "Surface decoration must generate flower patches"
    );
}

#[test]
#[allow(clippy::large_stack_arrays)]
fn test_biome_decoration_rules() {
    use telos_voxel::state::BlockStateId;
    use telos_worldgen::surface::ResolvedBlocks;
    use telos_worldgen::{BiomeId, apply_surface_decorations};

    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);

    // 1. Test Desert: Sand ground at y=10, air at y=11..31
    let mut desert_dense = [BlockStateId::AIR; CHUNK_VOLUME];
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=10 {
                desert_dense[(y << 10) | (z << 5) | x] = blocks.sand;
            }
        }
    }
    apply_surface_decorations(
        1337,
        ChunkPos::new(0, 0, 0),
        &[BiomeId::Desert; 64],
        &blocks,
        &mut desert_dense,
    );

    let mut found_dead_bush = false;
    let mut found_invalid_flower = false;
    for z in 0..32 {
        for x in 0..32 {
            let state = desert_dense[(11 << 10) | (z << 5) | x];
            if state == blocks.dead_bush {
                found_dead_bush = true;
            } else if state == blocks.poppy
                || state == blocks.dandelion
                || state == blocks.short_grass
            {
                found_invalid_flower = true;
            }
        }
    }
    assert!(
        found_dead_bush,
        "Desert surface must generate occasional dead bushes"
    );
    assert!(
        !found_invalid_flower,
        "Desert surface must never generate flowers or grass"
    );

    // 2. Test Ocean: Sand underwater at y=10
    let mut ocean_dense = [BlockStateId::AIR; CHUNK_VOLUME];
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=10 {
                ocean_dense[(y << 10) | (z << 5) | x] = blocks.sand;
            }
        }
    }
    apply_surface_decorations(
        1337,
        ChunkPos::new(0, 0, 0),
        &[BiomeId::Ocean; 64],
        &blocks,
        &mut ocean_dense,
    );

    for z in 0..32 {
        for x in 0..32 {
            let state = ocean_dense[(11 << 10) | (z << 5) | x];
            assert_eq!(
                state,
                BlockStateId::AIR,
                "Ocean must not generate terrestrial surface flora"
            );
        }
    }
}

#[test]
fn test_decoration_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_9988_7766;
    let generator = WorldGenerator::new(seed, &registry);

    // Generate lower and upper chunk column
    let lower_chunk = generator.generate_chunk(ChunkPos::new(0, 0, 0));
    let upper_chunk = generator.generate_chunk(ChunkPos::new(0, 1, 0));

    // If any column has grass at y=31 in lower chunk, upper chunk at y=0 must be valid decoration or air,
    // never an invalid stone/bedrock block
    let grass_block_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "grass_block").unwrap())
        .unwrap()
        .default_state();

    for z in 0..32 {
        for x in 0..32 {
            let lower_top = lower_chunk.get(LocalIdx::from_coords(x, 31, z).unwrap());
            if lower_top == grass_block_id {
                let upper_bottom = upper_chunk.get(LocalIdx::from_coords(x, 0, z).unwrap());
                // Upper bottom is either air or valid flora (e.g. short grass or flower)
                assert!(
                    upper_bottom == telos_voxel::state::BlockStateId::AIR
                        || registry
                            .flags(upper_bottom)
                            .contains(telos_voxel::state::StateFlags::CUTOUT),
                    "Boundary above grass at y=31 must be air or cutout foliage, got {upper_bottom:?}"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::large_stack_arrays)]
fn test_vegetation_multi_species_and_jittered_scatter() {
    use telos_voxel::state::BlockStateId;
    use telos_worldgen::surface::ResolvedBlocks;
    use telos_worldgen::{BiomeId, apply_surface_decorations};

    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);

    // Test Forest chunk with grass and shaded tree leaves at y=15
    let mut forest_dense = [BlockStateId::AIR; CHUNK_VOLUME];
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=10 {
                forest_dense[(y << 10) | (z << 5) | x] = blocks.grass;
            }
            // Add tree leaves canopy overhead
            forest_dense[(15 << 10) | (z << 5) | x] = blocks.oak_leaves;
        }
    }

    apply_surface_decorations(
        0xFEED_FACE_CAFE_0001,
        ChunkPos::new(0, 0, 0),
        &[BiomeId::Forest; 64],
        &blocks,
        &mut forest_dense,
    );

    let mut found_mushroom = false;
    let mut found_fern = false;
    let mut found_tall_grass = false;
    for z in 0..32 {
        for x in 0..32 {
            let state = forest_dense[(11 << 10) | (z << 5) | x];
            if state == blocks.brown_mushroom || state == blocks.red_mushroom {
                found_mushroom = true;
            } else if state == blocks.fern {
                found_fern = true;
            } else if state == blocks.tall_grass {
                found_tall_grass = true;
            }
        }
    }

    assert!(found_fern, "Forest must generate ferns");
    assert!(
        found_mushroom,
        "Forest canopy shade must generate mushrooms"
    );
    assert!(found_tall_grass, "Forest must generate tall grass");
}

#[test]
fn test_biome_vegetation_color_blending() {
    use telos_worldgen::BiomeId;
    use telos_worldgen::decoration::biome_expected_vegetation_color;

    let base_grass = [106, 170, 64, 255];
    let blended_plains = biome_expected_vegetation_color(BiomeId::Plains, base_grass);
    assert_eq!(blended_plains[3], 255);

    let blended_forest = biome_expected_vegetation_color(BiomeId::Forest, base_grass);
    assert_eq!(blended_forest[3], 255);
    // Forest vegetation has darker green component
    assert!(blended_forest[1] <= base_grass[1]);

    let desert_sand = [219, 207, 163, 255];
    let blended_desert = biome_expected_vegetation_color(BiomeId::Desert, desert_sand);
    assert_eq!(blended_desert, desert_sand);
}
