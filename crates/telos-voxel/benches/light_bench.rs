//! Performance benchmarks for telos-voxel lighting engine and heightmaps.

#![allow(missing_docs)]
#![allow(clippy::large_stack_arrays)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use telos_voxel::{
    light::{ChunkHeightmap, ColumnHeights, LightBfs, LightLayer},
    occupancy::Occupancy,
};

fn bench_light_propagation(c: &mut Criterion) {
    let mut bfs = LightBfs::new();

    // 1. Point block light add & remove (e.g. torch placed and broken)
    c.bench_function("point_light_add_remove", |b| {
        let mut light = LightLayer::zero();
        let get_opacity = |_idx| 0u8;

        b.iter(|| {
            bfs.add_source(black_box(&mut light), 16, 16, 16, 15);
            bfs.propagate_block_add(black_box(&mut light), get_opacity);
            bfs.remove_source(black_box(&mut light), 16, 16, 16);
            bfs.propagate_block_remove(black_box(&mut light), get_opacity);
        });
    });

    // 2. Initial sky light propagation across a realistic terrain surface chunk
    let mut occ = Occupancy::empty();
    for y in 0..16u32 {
        for z in 0..32u32 {
            for x in 0..32u32 {
                occ.toggle(x, y, z);
            }
        }
    }
    let local_h = ChunkHeightmap::from_occupancy(&occ);
    let mut col_h = ColumnHeights::new();
    col_h.update_chunk(0, &local_h);

    c.bench_function("sky_light_bfs_propagation", |b| {
        let mut light = LightLayer::zero();
        let is_opaque = |idx: usize| -> u8 {
            let y = (idx >> 10) & 0x1F;
            if y < 16 { 15 } else { 0 }
        };

        b.iter(|| {
            bfs.compute_initial_sky_light(
                black_box(0),
                black_box(&col_h),
                black_box(&mut light),
                is_opaque,
            );
        });
    });
}

fn bench_light_storage(c: &mut Criterion) {
    // 3. Fast 64-bit word uniform collapse scan on 16 KiB buffer
    let mut layer = LightLayer::Nibbles(Box::new([0xAA; 16384]));
    c.bench_function("uniform_collapse_scan_16k", |b| {
        b.iter(|| {
            black_box(&mut layer).try_collapse();
        });
    });

    // 4. Branchless lzcnt heightmap from 32-bit column masks
    let mut occ = Occupancy::empty();
    for y in 0..20u32 {
        for z in 0..32u32 {
            for x in 0..32u32 {
                occ.toggle(x, y, z);
            }
        }
    }

    c.bench_function("chunk_heightmap_from_occupancy", |b| {
        b.iter(|| {
            black_box(ChunkHeightmap::from_occupancy(black_box(&occ)));
        });
    });
}

criterion_group!(benches, bench_light_propagation, bench_light_storage);
criterion_main!(benches);
