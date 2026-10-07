//! Criterion benchmarks for procedural tree generation and foliage canopies.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;
use telos_worldgen::surface::ResolvedBlocks;
use telos_worldgen::{BiomeId, WorldGenerator, apply_trees};

fn bench_trees(c: &mut Criterion) {
    let registry = BlockRegistry::standard();
    let blocks = ResolvedBlocks::resolve(&registry);
    let seed = 0x1234_5678_9ABC_DEF0;

    let mut group = c.benchmark_group("tree_generation");

    // Benchmark tree_generation_only in forest
    group.bench_function("trees_forest_chunk_only", |b| {
        let pos = ChunkPos::new(0, 0, 0);
        let biomes = [BiomeId::Forest; 64];
        #[allow(clippy::large_stack_arrays)]
        let base_dense = [BlockStateId::AIR; CHUNK_VOLUME];

        b.iter(|| {
            let mut test_dense = base_dense;
            apply_trees(
                black_box(seed),
                black_box(pos),
                black_box(&biomes),
                black_box(&blocks),
                black_box(&mut test_dense),
            );
            black_box(test_dense);
        });
    });

    // Benchmark tree_generation_only in mountains
    group.bench_function("trees_mountain_chunk_only", |b| {
        let pos = ChunkPos::new(12, 1, 8);
        let biomes = [BiomeId::Mountains; 64];
        #[allow(clippy::large_stack_arrays)]
        let base_dense = [BlockStateId::AIR; CHUNK_VOLUME];

        b.iter(|| {
            let mut test_dense = base_dense;
            apply_trees(
                black_box(seed),
                black_box(pos),
                black_box(&biomes),
                black_box(&blocks),
                black_box(&mut test_dense),
            );
            black_box(test_dense);
        });
    });

    // Benchmark full chunk generation with trees enabled
    let generator = WorldGenerator::new(seed, &registry);
    group.bench_function("full_chunk_with_trees", |b| {
        let pos = ChunkPos::new(0, 1, 0);
        b.iter(|| {
            let chunk = generator.generate_chunk(black_box(pos));
            black_box(chunk);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_trees);
criterion_main!(benches);
