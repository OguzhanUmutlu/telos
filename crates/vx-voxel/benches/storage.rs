//! Performance benchmarks for vx-voxel storage operations.

#![allow(missing_docs)]
#![allow(clippy::large_stack_arrays)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vx_voxel::{
    coords::{CHUNK_VOLUME, LocalIdx},
    occupancy::Occupancy,
    state::BlockStateId,
    storage::{HotBlocks, from_dense},
};

fn bench_get_set(c: &mut Criterion) {
    let mut hot = HotBlocks::new_uniform(BlockStateId::AIR);
    let stone = BlockStateId::new(1);
    let dirt = BlockStateId::new(2);

    // Warm up palette with stone and dirt
    hot.set(LocalIdx::new(0).unwrap(), stone, false);
    hot.set(LocalIdx::new(1).unwrap(), dirt, false);

    c.bench_function("voxel_get", |b| {
        let idx = LocalIdx::new(0).unwrap();
        b.iter(|| {
            black_box(hot.get(black_box(idx)));
        });
    });

    c.bench_function("voxel_set_existing_state", |b| {
        let idx = LocalIdx::new(0).unwrap();
        let mut toggle = false;
        b.iter(|| {
            let state = if toggle { stone } else { dirt };
            toggle = !toggle;
            black_box(hot.set(black_box(idx), black_box(state), false));
        });
    });
}

fn bench_bulk(c: &mut Criterion) {
    // Typical natural chunk: stone base, dirt layer, air above
    let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
    let stone = BlockStateId::new(1);
    let dirt = BlockStateId::new(2);

    for y in 0..16 {
        for z in 0..32 {
            for x in 0..32 {
                let idx = (y << 10) | (z << 5) | x;
                dense[idx] = stone;
            }
        }
    }
    for y in 16..20 {
        for z in 0..32 {
            for x in 0..32 {
                let idx = (y << 10) | (z << 5) | x;
                dense[idx] = dirt;
            }
        }
    }

    c.bench_function("from_dense_32k", |b| {
        b.iter(|| {
            black_box(from_dense(black_box(&dense)));
        });
    });

    let blocks = from_dense(&dense);
    c.bench_function("occupancy_from_blocks", |b| {
        b.iter(|| {
            black_box(Occupancy::from_blocks(black_box(&blocks), |s| !s.is_air()));
        });
    });
}

criterion_group!(benches, bench_get_set, bench_bulk);
criterion_main!(benches);
