//! Crash resilience, power-loss simulation, and double-buffered recovery tests.

use vx_core::coords::ChunkPos;
use vx_storage::format::header::CodecId;
use vx_storage::format::section::{ChunkPayload, ChunkStatus};
use vx_storage::io::{RegionIo, SimFs};
use vx_storage::region::RegionFile;
use vx_voxel::state::BlockStateId;
use vx_voxel::storage::Blocks;

#[test]
fn test_crash_recovery_before_header_commit() {
    let sim_fs = SimFs::new();
    let pos = ChunkPos::new(0, 0, 0);

    // 1. Initial commit (generation 1 -> 2)
    let mut region = RegionFile::open(sim_fs.clone(), 0, 0, 0).expect("initial open");
    assert_eq!(region.generation(), 1);

    let payload_v1 = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(42)),
        ChunkStatus::default(),
    );
    region
        .commit_chunks(&[(pos, Some(payload_v1))], CodecId::Zstd, 100)
        .expect("commit 1");
    assert_eq!(region.generation(), 2);

    // 2. Prepare next commit (would be generation 3)
    // Inject simulated crash: unsynced writes to data sectors disappear
    sim_fs.write_at(40960, &[0xFF; 4096]).unwrap(); // write to data sector without sync
    sim_fs.simulate_crash(); // simulate power cut

    // 3. Re-open container after crash
    let recovered = RegionFile::open(sim_fs, 0, 0, 0).expect("recovery open must succeed");
    assert_eq!(recovered.generation(), 2);

    // Data from generation 2 must be fully intact
    let loaded = recovered.read_chunk(pos).unwrap().unwrap();
    assert_eq!(loaded.blocks, Blocks::Uniform(BlockStateId::new(42)));
}

#[test]
fn test_crash_recovery_torn_header_slot() {
    let sim_fs = SimFs::new();
    let pos = ChunkPos::new(0, 0, 0);

    // 1. Initial commit (generation 1 -> 2, written to Slot B at offset 20480)
    let mut region = RegionFile::open(sim_fs.clone(), 0, 0, 0).expect("initial open");
    let payload_v1 = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(10)),
        ChunkStatus::default(),
    );
    region
        .commit_chunks(&[(pos, Some(payload_v1))], CodecId::Zstd, 100)
        .expect("commit 1");
    assert_eq!(region.generation(), 2);

    // 2. Commit generation 3 (written to Slot A at offset 0)
    let payload_v2 = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(20)),
        ChunkStatus::default(),
    );
    region
        .commit_chunks(&[(pos, Some(payload_v2))], CodecId::Zstd, 200)
        .expect("commit 2");
    assert_eq!(region.generation(), 3);

    // Now simulate that Slot A (offset 0, generation 3) suffered a torn write on disk
    let corrupt_preamble = [0xEE; 64];
    sim_fs.corrupt_bytes(0, &corrupt_preamble);

    // 3. Re-open container: Slot A checksum will fail, container must cleanly fall back to Slot B (generation 2)
    let recovered = RegionFile::open(sim_fs, 0, 0, 0).expect("must fall back to valid slot B");
    assert_eq!(recovered.generation(), 2);

    let loaded = recovered.read_chunk(pos).unwrap().unwrap();
    assert_eq!(loaded.blocks, Blocks::Uniform(BlockStateId::new(10)));
}

#[test]
fn test_crash_recovery_truncated_file() {
    let sim_fs = SimFs::new();
    let pos = ChunkPos::new(0, 0, 0);

    let mut region = RegionFile::open(sim_fs.clone(), 0, 0, 0).expect("initial open");
    let payload = ChunkPayload::new(
        Blocks::Uniform(BlockStateId::new(1)),
        ChunkStatus::default(),
    );
    region
        .commit_chunks(&[(pos, Some(payload))], CodecId::Zstd, 100)
        .expect("commit");
    assert_eq!(region.generation(), 2);

    // Truncate file shorter than the header claimed file_sectors
    let len = sim_fs.len().unwrap();
    sim_fs.set_len(len - 100).unwrap();

    // Opening should fail or fall back if neither slot fits the truncated file
    let res = RegionFile::open(sim_fs, 0, 0, 0);
    // Since both slots claim at least 10 sectors (40,960 B), truncating below that causes both slots to be rejected
    assert!(
        res.is_err(),
        "Truncated file below claimed sectors must be rejected"
    );
}

#[test]
fn test_commit_cycle_alternating_slots() {
    let sim_fs = SimFs::new();
    let mut region = RegionFile::open(sim_fs.clone(), 0, 0, 0).expect("initial open");
    assert_eq!(region.generation(), 1);

    // Perform 10 consecutive commits within region (0, 0, 0)
    for i in 1..=10 {
        let pos = ChunkPos::new((i - 1) % 8, (i - 1) / 8, 0);
        let payload = ChunkPayload::new(
            Blocks::Uniform(BlockStateId::new(i as u32)),
            ChunkStatus::default(),
        );
        region
            .commit_chunks(&[(pos, Some(payload))], CodecId::Zstd, i as u32 * 10)
            .expect("commit must succeed");
        assert_eq!(region.generation(), 1 + i as u64);
    }

    assert_eq!(region.generation(), 11);

    // Re-open from disk and verify all 10 chunks exist with their exact states
    let reloaded = RegionFile::open(sim_fs, 0, 0, 0).expect("reopen after 10 commits");
    assert_eq!(reloaded.generation(), 11);

    for i in 1..=10 {
        let pos = ChunkPos::new((i - 1) % 8, (i - 1) / 8, 0);
        let chunk = reloaded.read_chunk(pos).unwrap().unwrap();
        assert_eq!(chunk.blocks, Blocks::Uniform(BlockStateId::new(i as u32)));
    }
}
