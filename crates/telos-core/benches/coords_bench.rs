//! Criterion benchmarks for coordinate operations.
#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use telos_core::{BlockPos, LocalPos};

fn bench_coords(c: &mut Criterion) {
    let mut group = c.benchmark_group("coords");

    group.bench_function("to_chunk_and_local", |b| {
        let pos = BlockPos::new(-12345, 67, 98765);
        b.iter(|| {
            let (chunk, local) = black_box(pos).to_chunk_and_local();
            black_box((chunk, local));
        });
    });

    group.bench_function("local_pos_to_xyz", |b| {
        let local = LocalPos::from_xyz(15, 7, 31);
        b.iter(|| {
            let xyz = black_box(local).to_xyz();
            black_box(xyz);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_coords);
criterion_main!(benches);
