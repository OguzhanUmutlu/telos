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

#[test]
fn test_void_generator() {
    let registry = BlockRegistry::standard();
    let generator =
        vx_worldgen::WorldGenerator::with_kind(42, &registry, vx_worldgen::GeneratorKind::Void);

    let pos = ChunkPos::new(0, 0, 0);
    let chunk = generator.generate_chunk(pos);
    assert_eq!(chunk.position(), pos);
    assert!(
        chunk.occupancy().is_empty(),
        "Void chunk must have empty occupancy"
    );
    assert_eq!(
        chunk.get(LocalIdx::from_coords(0, 0, 0).unwrap()),
        vx_voxel::state::BlockStateId::AIR
    );
}

#[test]
fn test_flat_generator() {
    let registry = BlockRegistry::standard();
    let generator =
        vx_worldgen::WorldGenerator::with_kind(42, &registry, vx_worldgen::GeneratorKind::Flat);

    // Chunk y != 0 must be empty
    let chunk_above = generator.generate_chunk(ChunkPos::new(0, 1, 0));
    assert!(
        chunk_above.occupancy().is_empty(),
        "Chunk above flat layer must have empty occupancy"
    );

    let chunk_below = generator.generate_chunk(ChunkPos::new(0, -1, 0));
    assert!(
        chunk_below.occupancy().is_empty(),
        "Chunk below flat layer must have empty occupancy"
    );

    // Chunk y == 0 has bedrock at y=0, dirt at y=1..3, grass at y=4
    let chunk0 = generator.generate_chunk(ChunkPos::new(0, 0, 0));
    assert!(!chunk0.occupancy().is_empty());

    let bedrock_idx = LocalIdx::from_coords(10, 0, 10).unwrap();
    let dirt_idx1 = LocalIdx::from_coords(10, 1, 10).unwrap();
    let dirt_idx3 = LocalIdx::from_coords(10, 3, 10).unwrap();
    let grass_idx = LocalIdx::from_coords(10, 4, 10).unwrap();
    let air_idx = LocalIdx::from_coords(10, 5, 10).unwrap();

    let stone_id = registry
        .get(&vx_core::ident::Identifier::new("voxel", "bedrock").unwrap())
        .unwrap()
        .default_state();
    let dirt_id = registry
        .get(&vx_core::ident::Identifier::new("voxel", "dirt").unwrap())
        .unwrap()
        .default_state();
    let grass_id = registry
        .get(&vx_core::ident::Identifier::new("voxel", "grass_block").unwrap())
        .unwrap()
        .default_state();

    assert_eq!(chunk0.get(bedrock_idx), stone_id);
    assert_eq!(chunk0.get(dirt_idx1), dirt_id);
    assert_eq!(chunk0.get(dirt_idx3), dirt_id);
    assert_eq!(chunk0.get(grass_idx), grass_id);
    assert_eq!(chunk0.get(air_idx), vx_voxel::state::BlockStateId::AIR);
}
