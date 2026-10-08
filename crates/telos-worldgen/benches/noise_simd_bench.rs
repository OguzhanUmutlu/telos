//! Criterion benchmarks measuring SIMD vs scalar noise throughput.

#![allow(missing_docs, clippy::too_many_lines)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use telos_worldgen::noise::simd::scalar;
use telos_worldgen::noise::{fbm3d_slice, noise2_batch_8, noise3_batch_8};

fn bench_noise_simd(c: &mut Criterion) {
    let seed = 0xABCD_1234_EF01_5678;

    // 1. Simplex 2D 8-wide comparison
    {
        let mut group = c.benchmark_group("simplex2d_8wide");
        group.throughput(Throughput::Elements(8));

        let xs = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];
        let ys = [7.8, 6.7, 5.6, 4.5, 3.4, 2.3, 1.2, 0.1];
        let mut out = [0.0f32; 8];

        group.bench_function("scalar", |b| {
            b.iter(|| {
                scalar::noise2_batch_8_scalar(
                    seed,
                    black_box(&xs),
                    black_box(&ys),
                    black_box(&mut out),
                );
                black_box(out);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                noise2_batch_8(seed, black_box(&xs), black_box(&ys), black_box(&mut out));
                black_box(out);
            });
        });

        group.finish();
    }

    // 2. Simplex 3D 8-wide comparison
    {
        let mut group = c.benchmark_group("simplex3d_8wide");
        group.throughput(Throughput::Elements(8));

        let xs = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];
        let ys = [7.8, 6.7, 5.6, 4.5, 3.4, 2.3, 1.2, 0.1];
        let zs = [3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 1.1];
        let mut out = [0.0f32; 8];

        group.bench_function("scalar", |b| {
            b.iter(|| {
                scalar::noise3_batch_8_scalar(
                    seed,
                    black_box(&xs),
                    black_box(&ys),
                    black_box(&zs),
                    black_box(&mut out),
                );
                black_box(out);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                noise3_batch_8(
                    seed,
                    black_box(&xs),
                    black_box(&ys),
                    black_box(&zs),
                    black_box(&mut out),
                );
                black_box(out);
            });
        });

        group.finish();
    }

    // 3. 405-sample chunk corner 3D fBm slice comparison
    {
        let mut group = c.benchmark_group("fbm3d_chunk_corners_405");
        group.throughput(Throughput::Elements(405));

        let xs: Vec<f32> = (0..405).map(|i| (i % 9) as f32 * 4.0).collect();
        let ys: Vec<f32> = (0..405).map(|i| ((i / 9) % 5) as f32 * 8.0).collect();
        let zs: Vec<f32> = (0..405).map(|i| (i / 45) as f32 * 4.0).collect();

        let mut out_scalar = vec![0.0f32; 405];
        let mut out_simd = vec![0.0f32; 405];

        group.bench_function("scalar", |b| {
            b.iter(|| {
                scalar::fbm3d_slice_scalar(
                    seed,
                    black_box(&xs),
                    black_box(&ys),
                    black_box(&zs),
                    2,
                    1.0 / 64.0,
                    2.0,
                    0.5,
                    black_box(&mut out_scalar),
                );
                black_box(&out_scalar);
            });
        });

        group.bench_function("simd_avx2", |b| {
            b.iter(|| {
                fbm3d_slice(
                    seed,
                    black_box(&xs),
                    black_box(&ys),
                    black_box(&zs),
                    2,
                    1.0 / 64.0,
                    2.0,
                    0.5,
                    black_box(&mut out_simd),
                );
                black_box(&out_simd);
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_noise_simd);
criterion_main!(benches);
