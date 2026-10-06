//! Criterion benchmarks for chunk generation throughput.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use telos_core::coords::ChunkPos;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;

fn bench_worldgen(c: &mut Criterion) {
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(0xABCD_1234_EF01_5678, &registry);

    let mut group = c.benchmark_group("worldgen_throughput");

    group.bench_function("surface_chunk", |b| {
        let pos = ChunkPos::new(0, 0, 0);
        b.iter(|| {
            let chunk = generator.generate_chunk(black_box(pos));
            black_box(chunk);
        });
    });

    group.bench_function("mountain_chunk", |b| {
        let pos = ChunkPos::new(12, 1, 8);
        b.iter(|| {
            let chunk = generator.generate_chunk(black_box(pos));
            black_box(chunk);
        });
    });

    group.bench_function("underground_chunk", |b| {
        let pos = ChunkPos::new(0, -4, 0);
        b.iter(|| {
            let chunk = generator.generate_chunk(black_box(pos));
            black_box(chunk);
        });
    });

    group.bench_function("surface_decoration_only", |b| {
        use telos_core::coords::CHUNK_VOLUME;
        use telos_worldgen::surface::ResolvedBlocks;
        use telos_worldgen::{BiomeId, apply_surface_decorations};

        let blocks = ResolvedBlocks::resolve(&registry);
        #[allow(clippy::large_stack_arrays)]
        let mut dense = [telos_voxel::state::BlockStateId::AIR; CHUNK_VOLUME];
        for z in 0..32 {
            for x in 0..32 {
                for y in 0..=12 {
                    dense[(y << 10) | (z << 5) | x] = blocks.grass;
                }
            }
        }
        let pos = ChunkPos::new(0, 0, 0);
        let biomes = [BiomeId::Plains; 64];

        b.iter(|| {
            let mut test_dense = dense;
            apply_surface_decorations(
                black_box(0xABCD_1234),
                black_box(pos),
                black_box(&biomes),
                black_box(&blocks),
                black_box(&mut test_dense),
            );
            black_box(test_dense);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_worldgen);
criterion_main!(benches);
