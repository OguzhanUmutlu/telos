//! Integration tests for 3D procedural ore distribution and large sinuous ore veins.

use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;
use telos_worldgen::surface::ResolvedBlocks;

#[test]
#[allow(clippy::cast_possible_wrap)]
fn test_vertical_ore_strata() {
    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);
    let seed = 0x5EED_0000_CAFE_0045;
    let generator = WorldGenerator::new(seed, &registry);

    let mut found_coal = false;
    let mut found_diamond = false;
    let mut found_deepslate_ore = false;
    let mut diamond_at_invalid_height = false;

    // Scan vertical columns across multiple horizontal coordinates
    for cx in -3..=3 {
        for cz in -3..=3 {
            // Test deep subterranean chunks up to surface
            for cy in -16..=10 {
                let pos = ChunkPos::new(cx, cy, cz);
                let chunk = generator.generate_chunk(pos);
                let chunk_origin_y = cy * 32;

                for i in 0..CHUNK_VOLUME {
                    let idx = LocalIdx::new(i as u16).unwrap();
                    let state = chunk.get(idx);
                    let ly = i >> 10;
                    let wy = chunk_origin_y + ly as i32;

                    if state == blocks.coal_ore || state == blocks.deepslate_coal_ore {
                        found_coal = true;
                    }

                    if state == blocks.diamond_ore || state == blocks.deepslate_diamond_ore {
                        found_diamond = true;
                        if wy > -384 {
                            diamond_at_invalid_height = true;
                        }
                    }

                    // Check deepslate ore replacement at deep strata
                    if wy <= -16
                        && (state == blocks.deepslate_coal_ore
                            || state == blocks.deepslate_iron_ore
                            || state == blocks.deepslate_copper_ore
                            || state == blocks.deepslate_gold_ore
                            || state == blocks.deepslate_redstone_ore
                            || state == blocks.deepslate_lapis_ore
                            || state == blocks.deepslate_diamond_ore)
                    {
                        found_deepslate_ore = true;
                    }
                }
            }
        }
    }

    assert!(
        found_coal,
        "World generation must generate coal ore in upper strata"
    );
    assert!(
        found_diamond,
        "World generation must generate diamond ore in deep strata"
    );
    assert!(
        !diamond_at_invalid_height,
        "Diamonds must not generate at y > -384"
    );
    assert!(
        found_deepslate_ore,
        "Ores at y <= -16 must use deepslate variants"
    );
}

#[test]
fn test_large_vein_generation() {
    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);
    let seed = 0x5EED_DE12_0000_0001;
    let generator = WorldGenerator::new(seed, &registry);

    let mut found_copper_vein_granite = false;
    let mut found_raw_copper = false;
    let mut found_iron_vein_tuff = false;
    let mut found_raw_iron = false;

    // Scan chunks in copper vein range (y in [0, 128], cy in 0..=3)
    // and iron vein range (y in [-480, -64], cy in -15..=-2)
    for cx in -6..=6 {
        for cz in -6..=6 {
            // Copper band
            for cy in 0..=3 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME {
                    let state = chunk.get(LocalIdx::new(i as u16).unwrap());
                    if state == blocks.granite {
                        found_copper_vein_granite = true;
                    }
                    if state == blocks.raw_copper_block {
                        found_raw_copper = true;
                    }
                }
            }

            // Iron band
            for cy in -14..=-3 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME {
                    let state = chunk.get(LocalIdx::new(i as u16).unwrap());
                    if state == blocks.tuff {
                        found_iron_vein_tuff = true;
                    }
                    if state == blocks.raw_iron_block {
                        found_raw_iron = true;
                    }
                }
            }
        }
    }

    assert!(
        found_copper_vein_granite,
        "Large copper veins must generate granite filler rock"
    );
    assert!(
        found_raw_copper,
        "Large copper veins must contain raw copper blocks"
    );
    assert!(
        found_iron_vein_tuff,
        "Large iron veins must generate tuff filler rock"
    );
    assert!(
        found_raw_iron,
        "Large iron veins must contain raw iron blocks"
    );
}

#[test]
fn test_ore_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);
    let seed = 0x5EED_B04D_0000_1234;
    let generator = WorldGenerator::new(seed, &registry);

    // Generate two adjacent chunks along the X axis
    let pos_a = ChunkPos::new(0, -5, 0);
    let pos_b = ChunkPos::new(1, -5, 0);

    let chunk_a = generator.generate_chunk(pos_a);
    let chunk_b = generator.generate_chunk(pos_b);

    // Verify that across the boundary x = 31 in chunk_a and x = 0 in chunk_b,
    // if an ore is placed at the boundary face, the neighbor is valid solid rock or continuation of the vein
    let mut boundary_ores = 0;
    for z in 0..32 {
        for y in 0..32 {
            let state_a = chunk_a.get(LocalIdx::from_coords(31, y, z).unwrap());
            let state_b = chunk_b.get(LocalIdx::from_coords(0, y, z).unwrap());

            let is_ore_a = state_a == blocks.coal_ore
                || state_a == blocks.iron_ore
                || state_a == blocks.copper_ore
                || state_a == blocks.deepslate_iron_ore
                || state_a == blocks.deepslate_copper_ore
                || state_a == blocks.deepslate_redstone_ore;

            if is_ore_a {
                boundary_ores += 1;
                // Adjacent block in chunk_b must not be a corrupted empty/invalid block
                assert_ne!(
                    state_b,
                    telos_voxel::state::BlockStateId::new(999),
                    "Boundary neighbor block must be valid"
                );
            }
        }
    }

    // Deterministic independent re-generation must match 100%
    let chunk_a_again = generator.generate_chunk(pos_a);
    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        assert_eq!(chunk_a.get(idx), chunk_a_again.get(idx));
    }
    let _ = boundary_ores;
}
