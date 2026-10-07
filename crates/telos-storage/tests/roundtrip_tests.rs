//! Integration tests for `.tlr` round-trip storage, INLINE elision, and natural chunk size budgets.

use telos_core::coords::ChunkPos;
use telos_storage::compression::compress;
use telos_storage::format::header::{CodecId, EntryFlags};
use telos_storage::format::section::{ChunkPayload, ChunkStatus};
use telos_storage::io::SimFs;
use telos_storage::region::{RegionFile, RegionPos};
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;
use telos_voxel::storage::Blocks;
use telos_worldgen::WorldGenerator;

#[test]
fn test_uniform_chunk_inline_elision() {
    let sim_fs = SimFs::new();
    let rx = 0;
    let ry = 0;
    let rz = 0;

    let mut region = RegionFile::open(sim_fs.clone(), rx, ry, rz).expect("failed to open region");
    assert_eq!(region.generation(), 1);

    // Create 3 uniform chunks (air, stone, bedrock)
    let pos_air = ChunkPos::new(0, 5, 0);
    let pos_stone = ChunkPos::new(1, 5, 0);
    let pos_bedrock = ChunkPos::new(2, 5, 0);

    let payload_air = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::AIR),
        ChunkStatus {
            gen_stage: 6,
            inhabited_ticks: 0,
            flags: 0,
        },
    );
    let payload_stone = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(1)),
        ChunkStatus {
            gen_stage: 6,
            inhabited_ticks: 0,
            flags: 0,
        },
    );
    let payload_bedrock = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(5)),
        ChunkStatus {
            gen_stage: 6,
            inhabited_ticks: 0,
            flags: 0,
        },
    );

    let batch = vec![
        (pos_air, Some(payload_air.clone())),
        (pos_stone, Some(payload_stone.clone())),
        (pos_bedrock, Some(payload_bedrock.clone())),
    ];

    region
        .commit_chunks(&batch, CodecId::Zstd, 12345)
        .expect("commit failed");

    assert_eq!(region.generation(), 2);

    // Verify header entry flags: must be INLINE, and take 0 data sectors
    let entry_idx_air = RegionPos::entry_index(pos_air);
    let entry_air = &region.active_slot().entries[entry_idx_air];
    assert!(entry_air.flags.contains(EntryFlags::INLINE));
    assert_eq!(entry_air.sector, 0);
    assert_eq!(entry_air.sector_span, 0);

    // Reopen from disk to verify persistence
    let reloaded_region = RegionFile::open(sim_fs, rx, ry, rz).expect("reload failed");
    assert_eq!(reloaded_region.generation(), 2);

    // Read back chunks
    let read_air = reloaded_region.read_chunk(pos_air).unwrap().unwrap();
    assert_eq!(read_air.blocks, Blocks::Uniform(BlockStateId::AIR));

    let read_stone = reloaded_region.read_chunk(pos_stone).unwrap().unwrap();
    assert_eq!(read_stone.blocks, Blocks::Uniform(BlockStateId::new(1)));

    let read_bedrock = reloaded_region.read_chunk(pos_bedrock).unwrap().unwrap();
    assert_eq!(read_bedrock.blocks, Blocks::Uniform(BlockStateId::new(5)));
}

#[test]
fn test_natural_chunks_roundtrip_and_size_budget() {
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(0xCAFE_BABE_1234_5678, &registry);
    let sim_fs = SimFs::new();

    let rx = 0;
    let ry = 0;
    let rz = 0;
    let mut region = RegionFile::open(sim_fs.clone(), rx, ry, rz).expect("failed to open");

    // Generate a diverse sample of 16 natural chunks in this region
    let mut chunks_to_save = Vec::new();
    let mut original_payloads = Vec::new();
    let mut total_stored_bytes = 0usize;

    for ly in 0..2 {
        for lz in 0..2 {
            for lx in 0..4 {
                let pos = ChunkPos::new(lx, ly, lz);
                let chunk_snapshot = generator.generate_chunk(pos);
                let payload = ChunkPayload::new(
                    chunk_snapshot.to_blocks(),
                    ChunkStatus {
                        gen_stage: 6,
                        inhabited_ticks: 100,
                        flags: 1,
                    },
                );
                original_payloads.push((pos, payload.clone()));
                chunks_to_save.push((pos, Some(payload)));
            }
        }
    }

    region
        .commit_chunks(&chunks_to_save, CodecId::Zstd, 1000)
        .expect("commit failed");

    // Reopen container
    let reloaded = RegionFile::open(sim_fs.clone(), rx, ry, rz).expect("reopen failed");
    assert_eq!(reloaded.generation(), 2);

    // Read every chunk back and assert voxel-for-voxel identity across all 32,768 voxels
    for (pos, orig) in &original_payloads {
        let loaded = reloaded
            .read_chunk(*pos)
            .expect("read chunk failed")
            .expect("chunk must exist");

        assert_eq!(loaded.status, orig.status);

        // Compare all 32,768 voxels
        for i in 0..32768 {
            let local = LocalIdx::new(i).unwrap();
            assert_eq!(
                loaded.blocks.get(local),
                orig.blocks.get(local),
                "Voxel mismatch at local index {i} in chunk {pos:?}"
            );
        }

        // Measure compressed size
        let encoded = orig.encode();
        let compressed = compress(CodecId::Zstd, &encoded).unwrap();
        total_stored_bytes += compressed.len();
    }

    let avg_size = total_stored_bytes as f64 / original_payloads.len() as f64;
    println!("Average compressed natural chunk payload size: {avg_size:.1} B (budget: <= 1536 B)");
    // Budget check: average natural chunk must be <= 1.5 KiB (1536 B)
    assert!(
        avg_size <= 1536.0,
        "Compressed natural chunk size {avg_size:.1} B exceeded 1536 B budget"
    );

    // Check disk sector consumption: pack sectors should have kept total size small
    let total_file_len = sim_fs.dump_current_bytes().len();
    println!("Total region container file length: {total_file_len} bytes");
    // 10 header sectors = 40,960 bytes. Data sectors should be well below uncompressed size
    assert!(total_file_len < 100_000);
}

#[test]
fn test_chunk_update_and_delete() {
    let sim_fs = SimFs::new();
    let mut region = RegionFile::open(sim_fs.clone(), 0, 0, 0).expect("open failed");

    let pos = ChunkPos::new(0, 0, 0);

    // Initial commit: uniform air
    let payload1 = ChunkPayload::new(Blocks::air(), ChunkStatus::default());
    region
        .commit_chunks(&[(pos, Some(payload1))], CodecId::Zstd, 10)
        .expect("commit 1 failed");
    assert_eq!(region.generation(), 2);
    let loaded1 = region.read_chunk(pos).unwrap().unwrap();
    assert_eq!(loaded1.blocks, Blocks::air());

    // Update commit: non-uniform chunk
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(42, &registry);
    let natural_chunk = generator.generate_chunk(pos);
    let payload2 = ChunkPayload::new(natural_chunk.to_blocks(), ChunkStatus::default());
    region
        .commit_chunks(&[(pos, Some(payload2.clone()))], CodecId::Zstd, 20)
        .expect("commit 2 failed");
    assert_eq!(region.generation(), 3);

    let loaded2 = region.read_chunk(pos).unwrap().unwrap();
    assert_eq!(
        loaded2.blocks.get(LocalIdx::ZERO),
        payload2.blocks.get(LocalIdx::ZERO)
    );

    // Delete commit
    region
        .commit_chunks(&[(pos, None)], CodecId::Zstd, 30)
        .expect("commit 3 failed");
    assert_eq!(region.generation(), 4);

    let loaded3 = region.read_chunk(pos).unwrap();
    assert!(loaded3.is_none(), "Deleted chunk must return None");
}

#[test]
fn test_configurable_region_compression() {
    let sim_fs = SimFs::new();
    let rx = 1;
    let ry = 2;
    let rz = 3;

    // 1. Initialize region with Zstd level 9 compression
    let mut region = RegionFile::open_with_compression(sim_fs.clone(), rx, ry, rz, true, 9)
        .expect("failed to open with compression level 9");
    assert!(region.is_compressed());
    assert_eq!(region.compression_level(), 9);

    let pos = ChunkPos::new(8, 16, 24);
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(12345, &registry);
    let natural_chunk = generator.generate_chunk(pos);
    let payload = ChunkPayload::new(natural_chunk.to_blocks(), ChunkStatus::default());

    region
        .commit_chunks(&[(pos, Some(payload.clone()))], CodecId::Zstd, 100)
        .expect("commit chunks level 9 failed");

    // 2. Reopen and verify header preserved flags and compression level
    let reloaded = RegionFile::open(sim_fs.clone(), rx, ry, rz).expect("reopen failed");
    assert!(reloaded.is_compressed());
    assert_eq!(reloaded.compression_level(), 9);

    let loaded_chunk = reloaded.read_chunk(pos).unwrap().unwrap();
    assert_eq!(
        loaded_chunk.blocks.get(LocalIdx::ZERO),
        payload.blocks.get(LocalIdx::ZERO)
    );
}

#[test]
fn test_uncompressed_container_roundtrip() {
    let sim_fs = SimFs::new();
    let rx = 4;
    let ry = 0;
    let rz = 4;

    // Initialize region with compression explicitly disabled
    let mut region = RegionFile::open_with_compression(sim_fs.clone(), rx, ry, rz, false, 0)
        .expect("failed to open uncompressed");
    assert!(!region.is_compressed());
    assert_eq!(region.compression_level(), 0);

    let pos = ChunkPos::new(32, 0, 32);
    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(999, &registry);
    let natural_chunk = generator.generate_chunk(pos);
    let payload = ChunkPayload::new(natural_chunk.to_blocks(), ChunkStatus::default());

    region
        .commit_chunks(&[(pos, Some(payload.clone()))], CodecId::Zstd, 200)
        .expect("commit chunks uncompressed failed");

    let reloaded = RegionFile::open(sim_fs, rx, ry, rz).expect("reopen uncompressed failed");
    assert!(!reloaded.is_compressed());
    assert_eq!(reloaded.compression_level(), 0);

    let loaded_chunk = reloaded.read_chunk(pos).unwrap().unwrap();
    assert_eq!(
        loaded_chunk.blocks.get(LocalIdx::ZERO),
        payload.blocks.get(LocalIdx::ZERO)
    );
}
