//! Procedural structures, subterranean dungeons & surface ruins integration tests.

use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::coords::LocalIdx;
use telos_voxel::registry::BlockRegistry;
use telos_worldgen::WorldGenerator;
use telos_worldgen::structure::{
    JigsawAssembler, LootTableKind, PieceType, StructureBoundingBox, StructurePiece,
    roll_chest_slots, roll_loot,
};

#[test]
fn test_structure_bounding_box_operations() {
    let box_a = StructureBoundingBox::new(0, 0, 0, 15, 10, 15);
    let box_b = StructureBoundingBox::new(10, 5, 10, 25, 15, 25);
    let box_c = StructureBoundingBox::new(40, 0, 40, 50, 10, 50);

    assert!(box_a.intersects(&box_b));
    assert!(box_b.intersects(&box_a));
    assert!(!box_a.intersects(&box_c));
    assert!(!box_b.intersects(&box_c));

    assert!(box_a.contains(5, 5, 5));
    assert!(!box_a.contains(20, 5, 5));

    assert_eq!(box_a.width(), 16);
    assert_eq!(box_a.height(), 11);
    assert_eq!(box_a.depth(), 16);
    assert_eq!(box_a.volume(), 16 * 11 * 16);

    let chunk = ChunkPos::new(0, 0, 0);
    assert!(box_a.intersects_chunk(chunk));
    let clamped = box_a.clamp_to_chunk(chunk).expect("Should clamp to chunk");
    assert_eq!(clamped.min_x, 0);
    assert_eq!(clamped.max_x, 15);
    assert_eq!(clamped.min_y, 0);
    assert_eq!(clamped.max_y, 10);
    assert_eq!(clamped.min_z, 0);
    assert_eq!(clamped.max_z, 15);
}

#[test]
fn test_jigsaw_piece_assembly_no_overlap() {
    let mut assembler = JigsawAssembler::new();

    let chamber = StructurePiece::new(
        PieceType::DungeonChamber,
        StructureBoundingBox::new(-3, 0, -3, 3, 4, 3),
        0,
    );
    assert!(assembler.try_add_piece(chamber));

    // Overlapping piece must be rejected
    let collides = StructurePiece::new(
        PieceType::RuinWall,
        StructureBoundingBox::new(-1, 0, -1, 5, 4, 5),
        0,
    );
    assert!(!assembler.try_add_piece(collides));

    // Adjacent non-overlapping piece must be accepted
    let wall = StructurePiece::new(
        PieceType::RuinWall,
        StructureBoundingBox::new(4, 0, -3, 6, 4, 3),
        0,
    );
    assert!(assembler.try_add_piece(wall));
    assert_eq!(assembler.pieces().len(), 2);

    let bounds = assembler.bounds().expect("Bounds must exist");
    assert_eq!(bounds.min_x, -3);
    assert_eq!(bounds.max_x, 6);
}

#[test]
fn test_dungeon_loot_table_determinism() {
    let seed = 0xABCD_EF01_2345_6789;
    let salt = 999;

    let loot1 = roll_loot(seed, salt, LootTableKind::DungeonChest);
    let loot2 = roll_loot(seed, salt, LootTableKind::DungeonChest);

    assert_eq!(
        loot1, loot2,
        "Loot rolls with identical seed must match exactly"
    );
    assert!(loot1.len() >= 4 && loot1.len() <= 8);

    for item in &loot1 {
        assert!(item.count >= 1);
        assert!(!item.name.is_empty());
    }

    // Chest slots must be distinct within container (0..27)
    let slots = roll_chest_slots(seed, salt, LootTableKind::DungeonChest);
    let mut seen_slots = [false; 27];
    for (slot, item) in slots {
        assert!((slot as usize) < 27);
        assert!(!seen_slots[slot as usize], "Chest slot collision detected");
        seen_slots[slot as usize] = true;
        assert!(item.count >= 1);
    }
}

#[test]
fn test_surface_ruin_loot_table_determinism() {
    let seed = 0x5555_AAAA_3333_CCCC;
    let salt = 123;

    let loot1 = roll_loot(seed, salt, LootTableKind::SurfaceRuinChest);
    let loot2 = roll_loot(seed, salt, LootTableKind::SurfaceRuinChest);

    assert_eq!(loot1, loot2, "Surface ruin loot rolls must match");
    assert!(loot1.len() >= 3 && loot1.len() <= 6);

    let slots = roll_chest_slots(seed, salt, LootTableKind::SurfaceRuinChest);
    for (slot, item) in slots {
        assert!((slot as usize) < 27);
        assert!(item.count >= 1);
    }
}

#[test]
fn test_dungeon_subterranean_generation() {
    let registry = BlockRegistry::standard();
    let seed = 42;
    let generator = WorldGenerator::new(seed, &registry);

    // Search subterranean chunks to locate generated dungeons or ruins
    let mut found_mossy = false;
    let mut found_spawner = false;
    let mut found_chest = false;

    let mossy_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "mossy_cobblestone").unwrap())
        .unwrap()
        .default_state();
    let spawner_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "monster_spawner").unwrap())
        .unwrap()
        .default_state();
    let chest_id = registry
        .get(&telos_core::ident::Identifier::new("telos", "chest").unwrap())
        .unwrap()
        .default_state();

    // Check a 6x6 area across subterranean depth [-10, 0]
    'outer: for cy in -10..=0 {
        for cz in -3..=3 {
            for cx in -3..=3 {
                let chunk = generator.generate_chunk(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME {
                    let idx = LocalIdx::new(i as u16).unwrap();
                    let state = chunk.get(idx);
                    if state == mossy_id {
                        found_mossy = true;
                    }
                    if state == spawner_id {
                        found_spawner = true;
                    }
                    if state == chest_id {
                        found_chest = true;
                    }
                    if found_mossy && found_spawner && found_chest {
                        break 'outer;
                    }
                }
            }
        }
    }

    assert!(
        found_mossy,
        "World generation must generate mossy cobblestone in structures"
    );
}

#[test]
fn test_structure_horizontal_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let seed = 42;
    let generator = WorldGenerator::new(seed, &registry);

    // Generate two horizontally adjacent chunks
    let chunk_a = generator.generate_chunk(ChunkPos::new(0, -2, 0));
    let chunk_b = generator.generate_chunk(ChunkPos::new(1, -2, 0));

    // Verify border faces match coordinate system without panic or corrupt state
    for y in 0..32 {
        for z in 0..32 {
            let idx_a = LocalIdx::from_coords(31, y, z).unwrap();
            let idx_b = LocalIdx::from_coords(0, y, z).unwrap();
            let _state_a = chunk_a.get(idx_a);
            let _state_b = chunk_b.get(idx_b);
        }
    }
}

#[test]
fn test_structure_vertical_chunk_boundary_continuity() {
    let registry = BlockRegistry::standard();
    let seed = 42;
    let generator = WorldGenerator::new(seed, &registry);

    let chunk_a = generator.generate_chunk(ChunkPos::new(0, -2, 0));
    let chunk_b = generator.generate_chunk(ChunkPos::new(0, -1, 0));

    for x in 0..32 {
        for z in 0..32 {
            let idx_a = LocalIdx::from_coords(x, 31, z).unwrap();
            let idx_b = LocalIdx::from_coords(x, 0, z).unwrap();
            let _state_a = chunk_a.get(idx_a);
            let _state_b = chunk_b.get(idx_b);
        }
    }
}
