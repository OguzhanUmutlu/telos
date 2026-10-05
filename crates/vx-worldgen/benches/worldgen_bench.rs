//! Criterion benchmarks for chunk generation throughput.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vx_core::coords::ChunkPos;
use vx_voxel::registry::BlockRegistry;
use vx_worldgen::WorldGenerator;

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

    group.finish();
}

criterion_group!(benches, bench_worldgen);
criterion_main!(benches);
