//! Criterion benchmarks for SIMD-accelerated voxel occupancy and lighting operations.

#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use telos_voxel::simd::light::{
    try_collapse_avx2, try_collapse_scalar, update_column_heights_avx2,
};
use telos_voxel::simd::occupancy::{is_empty_avx2, is_empty_scalar};

fn bench_simd_voxel(c: &mut Criterion) {
    // 1. Occupancy is_empty scan (4096 bytes)
    {
        let mut group = c.benchmark_group("occupancy_is_empty");
        group.throughput(Throughput::Bytes(4096));

        let col_y = [0u32; 1024];

        group.bench_function("scalar", |b| {
            b.iter(|| {
                let res = is_empty_scalar(black_box(&col_y));
                black_box(res);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                // SAFETY: Benchmark runs on CPU with AVX2.
                let res = unsafe { is_empty_avx2(black_box(&col_y)) };
                black_box(res);
            });
        });

        group.finish();
    }

    // 2. Light layer try_collapse scan (16384 bytes)
    {
        let mut group = c.benchmark_group("light_try_collapse_16k");
        group.throughput(Throughput::Bytes(16384));

        let bytes = Box::new([0x55_u8; 16384]);

        group.bench_function("scalar", |b| {
            b.iter(|| {
                let res = try_collapse_scalar(black_box(&bytes), 0x55);
                black_box(res);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                // SAFETY: Benchmark runs on CPU with AVX2.
                let res = unsafe { try_collapse_avx2(black_box(&bytes), 0x55) };
                black_box(res);
            });
        });

        group.finish();
    }

    // 3. Column heights update merge (1024 i16 columns = 2048 bytes)
    {
        let mut group = c.benchmark_group("column_heights_merge_1024");
        group.throughput(Throughput::Elements(1024));

        let mut top_scalar = [10i16; 1024];
        let mut top_avx2 = [10i16; 1024];
        let candidates = [15i16; 1024];

        group.bench_function("scalar", |b| {
            b.iter(|| {
                for i in 0..1024 {
                    if candidates[i] > top_scalar[i] {
                        top_scalar[i] = candidates[i];
                    }
                }
                black_box(&top_scalar);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                // SAFETY: Benchmark runs on CPU with AVX2.
                unsafe {
                    update_column_heights_avx2(black_box(&mut top_avx2), black_box(&candidates));
                }
                black_box(&top_avx2);
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_simd_voxel);
criterion_main!(benches);
