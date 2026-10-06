//! Criterion benchmarks for chunk save and load throughput.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use telos_core::coords::ChunkPos;
use telos_storage::compression::{compress, decompress};
use telos_storage::format::header::CodecId;
use telos_storage::format::section::{ChunkPayload, ChunkStatus};
use telos_storage::io::SimFs;
use telos_storage::region::RegionFile;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;

fn bench_storage(c: &mut Criterion) {
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(0xABCD_1234_EF01_5678, &registry);
    let chunk = generator.generate_chunk(ChunkPos::new(0, 0, 0));
    let payload = ChunkPayload::new(chunk.to_blocks(), ChunkStatus::default());
    let encoded = payload.encode();

    let mut group = c.benchmark_group("storage_throughput");

    group.bench_function("encode_chunk", |b| {
        b.iter(|| {
            let bytes = payload.encode();
            black_box(bytes);
        });
    });

    group.bench_function("decode_chunk", |b| {
        b.iter(|| {
            let decoded = ChunkPayload::decode(black_box(&encoded)).unwrap();
            black_box(decoded);
        });
    });

    group.bench_function("compress_zstd_lvl3", |b| {
        b.iter(|| {
            let compressed = compress(CodecId::Zstd, black_box(&encoded)).unwrap();
            black_box(compressed);
        });
    });

    let compressed = compress(CodecId::Zstd, &encoded).unwrap();
    group.bench_function("decompress_zstd", |b| {
        b.iter(|| {
            let decompressed =
                decompress(CodecId::Zstd, black_box(&compressed), encoded.len() as u32).unwrap();
            black_box(decompressed);
        });
    });

    // Benchmark full region chunk read and commit
    let sim_fs = SimFs::new();
    let mut region = RegionFile::open(sim_fs, 0, 0, 0).unwrap();
    let pos = ChunkPos::new(0, 0, 0);
    region
        .commit_chunks(&[(pos, Some(payload.clone()))], CodecId::Zstd, 100)
        .unwrap();

    group.bench_function("region_read_chunk", |b| {
        b.iter(|| {
            let read = region.read_chunk(black_box(pos)).unwrap();
            black_box(read);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_storage);
criterion_main!(benches);
