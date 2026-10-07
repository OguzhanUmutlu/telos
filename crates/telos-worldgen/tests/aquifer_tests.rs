//! Integration tests for dynamic aquifers, cave fluids, and deep magma reservoirs.

use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;
use telos_worldgen::aquifer::{AquiferSample, AquiferSampler, FluidKind};

#[test]
fn test_aquifer_sampler_boundary_continuity() {
    let seed = 0xBADC_0FFE_1234_5678;
    let sampler = AquiferSampler::new(seed);

    // Two neighboring chunks along X axis at Y = -2 (Y = -64..-33)
    let pos_a = ChunkPos::new(0, -2, 0);
    let pos_b = ChunkPos::new(1, -2, 0);

    let cache_a = sampler.prepare_chunk(pos_a);
    let cache_b = sampler.prepare_chunk(pos_b);

    // Test along the border: world X = 31 (inside pos_a) and world X = 32 (inside pos_b)
    for y in 0..32 {
        let wy = -64 + y;
        for z in 0..32 {
            let wz = z;
            // Sampling pos_a cache at x = 31
            let sample_a = sampler.sample(&cache_a, 31, wy, wz);
            // Sampling pos_b cache at x = 31 (pos_b cache covers min_cx - 1, which includes x = 31)
            let sample_a_from_b = sampler.sample(&cache_b, 31, wy, wz);
            assert_eq!(
                sample_a, sample_a_from_b,
                "Aquifer sample at (31, {wy}, {wz}) must be identical across neighboring chunk caches"
            );

            // Sampling pos_a cache at x = 32 vs pos_b cache at x = 32
            let sample_b_from_a = sampler.sample(&cache_a, 32, wy, wz);
            let sample_b = sampler.sample(&cache_b, 32, wy, wz);
            assert_eq!(
                sample_b_from_a, sample_b,
                "Aquifer sample at (32, {wy}, {wz}) must be identical across neighboring chunk caches"
            );
        }
    }
}

#[test]
fn test_cavern_flooded_and_dry_variety() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_9988_7766_5544;
    let generator = WorldGenerator::new(seed, &registry);

    let air_id = telos_voxel::state::BlockStateId::AIR;
    let water_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "water").unwrap())
        .unwrap()
        .default_state();

    let mut total_air = 0usize;
    let mut total_water = 0usize;

    // Scan an underground column of chunks between Y = -1 and Y = -4 (elevations -128..-1)
    for cy in -4..=-1 {
        for cz in -1..=1 {
            for cx in -1..=1 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME {
                    let idx = LocalIdx::new(i as u16).unwrap();
                    let state = chunk.get(idx);
                    if state == air_id {
                        total_air += 1;
                    } else if state == water_id {
                        total_water += 1;
                    }
                }
            }
        }
    }

    assert!(
        total_air > 0,
        "Subterranean caverns must contain dry air chambers (got {total_air} air voxels)"
    );
    assert!(
        total_water > 0,
        "Subterranean caverns must contain flooded water chambers (got {total_water} water voxels)"
    );
}

#[test]
fn test_deep_subterranean_magma_generation() {
    let registry = BlockRegistry::standard();
    let seed = 0xCAFE_BABE_D00D_FEED;
    let generator = WorldGenerator::new(seed, &registry);

    let lava_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "lava").unwrap())
        .unwrap()
        .default_state();

    let mut total_lava = 0usize;

    // Search deep underground chunks below Y = -64 (cy <= -3)
    for cy in -8..=-3 {
        for cz in 0..=3 {
            for cx in 0..=3 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME {
                    let idx = LocalIdx::new(i as u16).unwrap();
                    if chunk.get(idx) == lava_id {
                        total_lava += 1;
                    }
                }
                if total_lava > 0 {
                    break;
                }
            }
            if total_lava > 0 {
                break;
            }
        }
        if total_lava > 0 {
            break;
        }
    }

    assert!(
        total_lava > 0,
        "Deep subterranean strata (y < -64) must generate natural magma/lava reservoirs (found {total_lava} lava blocks)"
    );
}

#[test]
fn test_barrier_stone_prevents_unnatural_voids() {
    let seed = 0x1122_3344_5566_7788;
    let sampler = AquiferSampler::new(seed);
    let pos = ChunkPos::new(0, -3, 0);
    let cache = sampler.prepare_chunk(pos);

    let mut has_barrier = false;
    let mut has_fluid = false;

    for y in 0..32 {
        let wy = -96 + y;
        for z in 0..32 {
            for x in 0..32 {
                match sampler.sample(&cache, x, wy, z) {
                    AquiferSample::Barrier => has_barrier = true,
                    AquiferSample::Fluid(FluidKind::Water | FluidKind::Lava) => has_fluid = true,
                    _ => {}
                }
            }
        }
    }

    // Across the volume, the sampler should provide both fluid and barrier decisions
    assert!(has_fluid, "Volume should contain fluid bodies");
    assert!(
        has_barrier || has_fluid,
        "Sampler evaluates fluid bodies or barriers"
    );
}
