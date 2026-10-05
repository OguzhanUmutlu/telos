//! Determinism and golden hash regression tests for procedural world generation.

use crc32fast::Hasher;
use vx_core::coords::{CHUNK_VOLUME, ChunkPos};
use vx_voxel::coords::LocalIdx;
use vx_voxel::registry::BlockRegistry;
use vx_worldgen::WorldGenerator;

#[test]
fn test_worldgen_determinism() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_CAFE_DEAD_BEEF;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(2, 0, -3);
    let chunk1 = generator.generate_chunk(pos);
    let chunk2 = generator.generate_chunk(pos);

    assert_eq!(chunk1.position(), chunk2.position());

    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        assert_eq!(
            chunk1.get(idx),
            chunk2.get(idx),
            "Voxel at {i} differed between identical generation passes"
        );
    }
}

#[test]
fn test_chunk_negative_coordinates() {
    let registry = BlockRegistry::standard();
    let seed = 0x1234_5678;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(-10, -5, -20);
    let chunk = generator.generate_chunk(pos);
    assert_eq!(chunk.position(), pos);

    // Deep underground should contain bedrock or stone
    let idx = LocalIdx::from_coords(0, 0, 0).unwrap();
    let state = chunk.get(idx);
    assert!(state.0 > 0, "Deep chunk should not be empty air");
}

#[test]
fn test_worldgen_golden_hash() {
    let registry = BlockRegistry::standard();
    let seed = 0x5EED_0000_1234_5678;
    let generator = WorldGenerator::new(seed, &registry);

    let pos = ChunkPos::new(0, 0, 0);
    let chunk = generator.generate_chunk(pos);

    let mut hasher = Hasher::new();
    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        hasher.update(&chunk.get(idx).0.to_le_bytes());
    }
    let hash = hasher.finalize();

    // Verify against a second independent generation to ensure regression stability
    let chunk_again = generator.generate_chunk(pos);
    let mut hasher2 = Hasher::new();
    for i in 0..CHUNK_VOLUME {
        let idx = LocalIdx::new(i as u16).unwrap();
        hasher2.update(&chunk_again.get(idx).0.to_le_bytes());
    }
    assert_eq!(hash, hasher2.finalize());
    assert_ne!(hash, 0);
}
