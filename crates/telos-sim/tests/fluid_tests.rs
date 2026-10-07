//! Integration tests for cellular automata fluid mechanics.
//!
//! Verifies:
//! - Vertical waterfall columns (`falling: true`)
//! - Horizontal spread & decay (levels 1..=7)
//! - Slope-seeking cliff/drop detection
//! - Infinite water source renewal
//! - Wavefront evaporation / drying cleanup
//! - Fluid reactions (Obsidian, Cobblestone, Stone)
//! - Viscous scheduling (Water 5 ticks, Lava 30 ticks)

use hashbrown::HashMap;
use telos_core::coords::BlockPos;
use telos_sim::fluid::{FluidEngine, FluidKind, FluidWorldReader};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

/// In-memory mock world reader and writer for fluid simulation tests.
struct MockWorld {
    registry: BlockRegistry,
    blocks: HashMap<BlockPos, BlockStateId>,
}

impl MockWorld {
    fn new() -> Self {
        Self {
            registry: BlockRegistry::standard(),
            blocks: HashMap::new(),
        }
    }

    fn set(&mut self, pos: BlockPos, state: BlockStateId) {
        if state.is_air() {
            self.blocks.remove(&pos);
        } else {
            self.blocks.insert(pos, state);
        }
    }

    fn set_solid_box(&mut self, min_x: i32, max_x: i32, y: i32, min_z: i32, max_z: i32) {
        let stone_id = BlockStateId::new(1);
        for x in min_x..=max_x {
            for z in min_z..=max_z {
                self.set(BlockPos::new(x, y, z), stone_id);
            }
        }
    }
}

impl FluidWorldReader for MockWorld {
    fn get_block(&self, pos: BlockPos) -> BlockStateId {
        self.blocks.get(&pos).copied().unwrap_or(BlockStateId::AIR)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.registry
    }
}

#[test]
fn test_water_falling_column() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Solid floor at y = -1
    world.set_solid_box(-2, 2, -1, -2, 2);

    // Place water source at (0, 3, 0)
    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let src_pos = BlockPos::new(0, 3, 0);
    world.set(src_pos, water_source);
    engine.on_block_changed(src_pos, 0, &world);

    // Tick 5: should flow down to (0, 2, 0) as falling column
    let updates = engine.tick(5, &world);
    assert!(!updates.is_empty());
    for (pos, state) in updates {
        world.set(pos, state);
    }

    let state_y2 = world.get_block(BlockPos::new(0, 2, 0));
    let fluid_y2 = world.registry.fluid_state(state_y2).unwrap();
    assert_eq!(fluid_y2.kind, FluidKind::Water);
    assert!(fluid_y2.falling);

    // Tick 10: should flow down to (0, 1, 0) as falling column
    let updates = engine.tick(10, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }
    let state_y1 = world.get_block(BlockPos::new(0, 1, 0));
    let fluid_y1 = world.registry.fluid_state(state_y1).unwrap();
    assert!(fluid_y1.falling);

    // Tick 15: should flow down to (0, 0, 0) as falling column
    let updates = engine.tick(15, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }
    let state_y0 = world.get_block(BlockPos::new(0, 0, 0));
    let fluid_y0 = world.registry.fluid_state(state_y0).unwrap();
    assert!(fluid_y0.falling);

    // Tick 20: hits solid floor at y = -1; should now spread horizontally from (0, 0, 0)
    let updates = engine.tick(20, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }
    let state_east = world.get_block(BlockPos::new(1, 0, 0));
    let fluid_east = world.registry.fluid_state(state_east).unwrap();
    assert_eq!(fluid_east.level, 1);
    assert!(!fluid_east.falling);
}

#[test]
fn test_water_horizontal_spread_and_decay() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Solid floor at y = 0
    world.set_solid_box(-10, 10, 0, -10, 10);

    // Place water source at (0, 1, 0)
    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let src_pos = BlockPos::new(0, 1, 0);
    world.set(src_pos, water_source);
    engine.on_block_changed(src_pos, 0, &world);

    // Tick 5: spreads to cardinal neighbors with level 1
    let updates = engine.tick(5, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    for (dx, dz) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let n_pos = BlockPos::new(*dx, 1, *dz);
        let n_state = world.get_block(n_pos);
        let fluid = world.registry.fluid_state(n_state).unwrap();
        assert_eq!(fluid.level, 1);
        assert!(!fluid.falling);
    }

    // Tick 10: spreads to level 2
    let updates = engine.tick(10, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    let pos_2_east = BlockPos::new(2, 1, 0);
    let state_2_east = world.get_block(pos_2_east);
    let fluid_2_east = world.registry.fluid_state(state_2_east).unwrap();
    assert_eq!(fluid_2_east.level, 2);
}

#[test]
fn test_slope_seeking_drop_detection() {
    let mut world = MockWorld::new();
    let engine = FluidEngine::new();

    // Floor at y = 0 everywhere except east at (2, 0, 0) where there is a hole
    world.set_solid_box(-5, 5, 0, -5, 5);
    world.set(BlockPos::new(2, 0, 0), BlockStateId::AIR); // Cliff/hole

    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let src_pos = BlockPos::new(0, 1, 0);
    world.set(src_pos, water_source);

    // Probing flow directions from (0, 1, 0):
    // East leads to the drop at distance 2, other directions have no drop
    let (dirs, count) = engine.find_flow_directions(src_pos, FluidKind::Water, &world);
    assert_eq!(count, 1);
    assert_eq!(dirs[0], telos_core::coords::Face::East);
}

#[test]
fn test_water_source_creation() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Solid floor at y = 0 spanning wide area so no cliffs are within 4 blocks
    world.set_solid_box(-10, 10, 0, -10, 10);

    // Place two water sources at (-1, 1, 0) and (1, 1, 0)
    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    world.set(BlockPos::new(-1, 1, 0), water_source);
    world.set(BlockPos::new(1, 1, 0), water_source);

    // Notify changed blocks
    engine.on_block_changed(BlockPos::new(-1, 1, 0), 0, &world);
    engine.on_block_changed(BlockPos::new(1, 1, 0), 0, &world);

    // Tick 5: both sources spread into center (0, 1, 0)
    let updates = engine.tick(5, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    // Tick 10: center block (0, 1, 0) detects 2 adjacent sources over solid floor -> creates new source!
    let updates = engine.tick(10, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    let center_state = world.get_block(BlockPos::new(0, 1, 0));
    let center_fluid = world.registry.fluid_state(center_state).unwrap();
    assert!(center_fluid.is_source());
}

#[test]
fn test_water_evaporation_receding() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Solid floor at y = 0
    world.set_solid_box(-5, 5, 0, -5, 5);

    // Flowing water at (1, 1, 0) level 1 with NO water source supplying it
    let flowing_water = world.registry.fluid_state_id(FluidKind::Water, 1, false);
    let pos = BlockPos::new(1, 1, 0);
    world.set(pos, flowing_water);
    engine.on_block_changed(pos, 0, &world);

    // Tick 5: with no source, flowing water decays to AIR
    let updates = engine.tick(5, &world);
    assert!(!updates.is_empty());
    for (p, s) in updates {
        world.set(p, s);
    }

    let end_state = world.get_block(pos);
    assert!(end_state.is_air());
}

#[test]
fn test_fluid_reaction_water_lava_obsidian() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Floor at y = 0
    world.set_solid_box(-2, 2, 0, -2, 2);

    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let lava_source = world.registry.fluid_state_id(FluidKind::Lava, 0, false);

    let water_pos = BlockPos::new(0, 1, 0);
    let lava_pos = BlockPos::new(1, 1, 0);

    world.set(water_pos, water_source);
    world.set(lava_pos, lava_source);

    engine.on_block_changed(water_pos, 0, &world);

    let updates = engine.tick(5, &world);
    assert!(!updates.is_empty());
    for (pos, state) in updates {
        world.set(pos, state);
    }

    // Lava position should turn to Obsidian
    let lava_after = world.get_block(lava_pos);
    let ident = world.registry.identifier(lava_after).unwrap();
    assert_eq!(ident.path(), "obsidian");
}

#[test]
fn test_fluid_reaction_water_flowing_lava_cobblestone() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Floor at y = 0
    world.set_solid_box(-2, 2, 0, -2, 2);

    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let flowing_lava = world.registry.fluid_state_id(FluidKind::Lava, 1, false);

    let water_pos = BlockPos::new(0, 1, 0);
    let lava_pos = BlockPos::new(1, 1, 0);

    world.set(water_pos, water_source);
    world.set(lava_pos, flowing_lava);

    engine.on_block_changed(water_pos, 0, &world);

    let updates = engine.tick(5, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    // Flowing lava position should turn to Cobblestone
    let lava_after = world.get_block(lava_pos);
    let ident = world.registry.identifier(lava_after).unwrap();
    assert_eq!(ident.path(), "cobblestone");
}

#[test]
fn test_fluid_reaction_lava_falling_on_water_stone() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    // Water at (0, 0, 0)
    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    world.set(BlockPos::new(0, 0, 0), water_source);

    // Lava source at (0, 1, 0) above water
    let lava_source = world.registry.fluid_state_id(FluidKind::Lava, 0, false);
    world.set(BlockPos::new(0, 1, 0), lava_source);

    engine.on_block_changed(BlockPos::new(0, 1, 0), 0, &world);

    let updates = engine.tick(30, &world);
    for (pos, state) in updates {
        world.set(pos, state);
    }

    // Water below turned into Stone
    let water_after = world.get_block(BlockPos::new(0, 0, 0));
    assert_eq!(water_after, BlockStateId::new(1)); // Stone
}

#[test]
fn test_viscous_tick_rates() {
    let mut world = MockWorld::new();
    let mut engine = FluidEngine::new();

    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let lava_source = world.registry.fluid_state_id(FluidKind::Lava, 0, false);

    world.set(BlockPos::new(0, 1, 0), water_source);
    world.set(BlockPos::new(10, 1, 0), lava_source);

    engine.on_block_changed(BlockPos::new(0, 1, 0), 0, &world);
    engine.on_block_changed(BlockPos::new(10, 1, 0), 0, &world);

    // Tick 4: neither should tick
    assert!(engine.tick(4, &world).is_empty());

    // Tick 5: water ticks!
    let water_updates = engine.tick(5, &world);
    assert!(!water_updates.is_empty());

    // Tick 29: lava still has not ticked
    assert!(engine.tick(29, &world).is_empty());

    // Tick 30: lava ticks!
    let lava_updates = engine.tick(30, &world);
    assert!(!lava_updates.is_empty());
}
