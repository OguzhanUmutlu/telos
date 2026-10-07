//! Unit and integration tests for 3D voxel navigation, walkability evaluation, and `LocalAStar` pathfinding.

use hashbrown::HashMap;
use telos_core::coords::{BlockPos, ChunkPos};
use telos_core::ident::Identifier;
use telos_sim::nav::{
    ChunkPortalGraph, LocalAStar, NavWorldReader, PathProfile, is_hazard, is_walkable_node,
};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

/// Simple in-memory mock world for navigation testing.
struct MockNavWorld {
    registry: BlockRegistry,
    blocks: HashMap<BlockPos, BlockStateId>,
}

impl MockNavWorld {
    fn new() -> Self {
        let registry = BlockRegistry::standard();
        Self {
            registry,
            blocks: HashMap::new(),
        }
    }

    fn set_block(&mut self, pos: BlockPos, name: &str) {
        let ident = Identifier::new("telos", name).expect("Valid identifier");
        let block = self.registry.get(&ident).expect("Block registered");
        self.blocks.insert(pos, block.default_state());
    }

    fn fill_box(&mut self, min: BlockPos, max: BlockPos, name: &str) {
        for x in min.x()..=max.x() {
            for y in min.y()..=max.y() {
                for z in min.z()..=max.z() {
                    self.set_block(BlockPos::new(x, y, z), name);
                }
            }
        }
    }
}

impl NavWorldReader for MockNavWorld {
    fn get_block(&self, pos: BlockPos) -> BlockStateId {
        self.blocks.get(&pos).copied().unwrap_or(BlockStateId::AIR)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.registry
    }
}

#[test]
fn test_walkability_primitives() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Solid stone ground at y = 63
    world.set_block(BlockPos::new(0, 63, 0), "stone");
    let standing_node = BlockPos::new(0, 64, 0);

    assert!(is_walkable_node(standing_node, &profile, &world));

    // Obstruct headroom with stone at y = 65
    world.set_block(BlockPos::new(0, 65, 0), "stone");
    assert!(!is_walkable_node(standing_node, &profile, &world));

    // Clear headroom, place lava hazard under feet
    world.set_block(BlockPos::new(0, 65, 0), "air");
    world.set_block(BlockPos::new(0, 63, 0), "lava");
    assert!(!is_walkable_node(standing_node, &profile, &world));
}

#[test]
fn test_flat_straight_path() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Construct flat stone floor at y = 63
    world.fill_box(
        BlockPos::new(-10, 63, -10),
        BlockPos::new(20, 63, 20),
        "stone",
    );

    let start = BlockPos::new(0, 64, 0);
    let goal = BlockPos::new(10, 64, 0);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Path must be found");

    assert!(!path.is_empty());
    assert_eq!(path.target_pos(), goal);
    // Waypoints must progress toward goal along X axis
    let last_wp = path.waypoints().last().copied().unwrap();
    assert_eq!(last_wp, goal);
}

#[test]
fn test_u_shaped_obstacle_avoidance() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Ground floor at y = 63
    world.fill_box(
        BlockPos::new(-10, 63, -10),
        BlockPos::new(20, 63, 20),
        "stone",
    );

    // Build a 2-block tall U-shaped obstacle blocking direct X progression at x = 5
    // Wall from z = -3 to z = 3 at x = 5
    world.fill_box(BlockPos::new(5, 64, -3), BlockPos::new(5, 65, 3), "stone");

    let start = BlockPos::new(0, 64, 0);
    let goal = BlockPos::new(10, 64, 0);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Path around obstacle must be found");

    // Verify none of the waypoints intersect the wall
    for wp in path.waypoints() {
        assert!(
            !(wp.x() == 5 && wp.z() >= -3 && wp.z() <= 3),
            "Waypoint {wp:?} intersects obstacle wall!"
        );
    }

    assert_eq!(path.waypoints().last().copied().unwrap(), goal);
}

#[test]
fn test_jump_up_stairs() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Construct stepped ascending terrain:
    // x = 0..3: floor y = 63 (stand at y = 64)
    // x = 4: step to floor y = 64 (stand at y = 65)
    // x = 5: step to floor y = 65 (stand at y = 66)
    world.fill_box(BlockPos::new(0, 63, 0), BlockPos::new(3, 63, 0), "stone");
    world.set_block(BlockPos::new(4, 64, 0), "stone");
    world.set_block(BlockPos::new(5, 65, 0), "stone");

    let start = BlockPos::new(0, 64, 0);
    let goal = BlockPos::new(5, 66, 0);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Path up steps must be found");

    let last_wp = path.waypoints().last().copied().unwrap();
    assert_eq!(last_wp, goal);

    // Verify that the path contains a jump transition where y increases
    let has_jump = path.waypoints().windows(2).any(|w| w[1].y() > w[0].y());
    assert!(has_jump, "Path must contain step-up jump transitions");
}

#[test]
fn test_safe_drop_down_and_cliff_avoidance() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // High platform at x = 0..3, y = 65 (stand at 66)
    world.fill_box(BlockPos::new(0, 65, 0), BlockPos::new(3, 65, 0), "stone");

    // Low platform at x = 4..8, y = 63 (stand at 64) -> a 2-block drop (safe <= 3)
    world.fill_box(BlockPos::new(4, 63, 0), BlockPos::new(8, 63, 0), "stone");

    let start = BlockPos::new(1, 66, 0);
    let goal = BlockPos::new(6, 64, 0);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Safe 2-block drop path must be found");

    let last_wp = path.waypoints().last().copied().unwrap();
    assert_eq!(last_wp, goal);

    // Lethal drop test: floor at y = 50 (drop of 16 blocks > max_drop 3)
    let mut lethal_world = MockNavWorld::new();
    lethal_world.fill_box(BlockPos::new(0, 65, 0), BlockPos::new(3, 65, 0), "stone");
    lethal_world.fill_box(BlockPos::new(4, 50, 0), BlockPos::new(8, 50, 0), "stone");

    let lethal_goal = BlockPos::new(6, 51, 0);
    let lethal_path = astar.find_path(start, lethal_goal, 0.5, &profile, &lethal_world);
    // Entity should NOT jump off a 15-block cliff to reach the goal!
    if let Some(p) = lethal_path {
        // If a partial path is returned, verify it stopped on the high ledge
        assert!(
            p.waypoints().iter().all(|wp| wp.y() >= 66),
            "Mob should not walk off lethal cliff!"
        );
    }
}

#[test]
fn test_hazard_avoidance() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Stone floor
    world.fill_box(BlockPos::new(-5, 63, -5), BlockPos::new(15, 63, 5), "stone");

    // Lava pit at x = 5, z = -1..=1 at floor level
    world.fill_box(BlockPos::new(5, 63, -1), BlockPos::new(5, 63, 1), "lava");

    let start = BlockPos::new(0, 64, 0);
    let goal = BlockPos::new(10, 64, 0);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Path around lava hazard must be found");

    // Verify waypoints steer around the lava pit
    for wp in path.waypoints() {
        let block_below = world.get_block(wp.down(1));
        assert!(
            !is_hazard(block_below, world.registry()),
            "Waypoint {wp:?} steps into or on hazard!"
        );
    }
}

#[test]
fn test_diagonal_corner_cutting_avoidance() {
    let mut world = MockNavWorld::new();
    let profile = PathProfile::humanoid();

    // Floor at y = 63
    world.fill_box(BlockPos::new(0, 63, 0), BlockPos::new(5, 63, 5), "stone");

    // Place a solid pillar at (1, 64, 0)
    world.set_block(BlockPos::new(1, 64, 0), "stone");
    world.set_block(BlockPos::new(1, 65, 0), "stone");

    let start = BlockPos::new(0, 64, 0);
    // Goal at (1, 64, 1): diagonal neighbor
    let goal = BlockPos::new(1, 64, 1);

    let astar = LocalAStar::default();
    let path = astar
        .find_path(start, goal, 0.5, &profile, &world)
        .expect("Path must be found");

    // Path must NOT cut through the corner if (1, 64, 0) is solid!
    // Instead it must step via (0, 64, 1) to (1, 64, 1)
    let waypoints = path.waypoints();
    assert!(waypoints.contains(&BlockPos::new(0, 64, 1)));
}

#[test]
fn test_chunk_portal_corridor() {
    let start_chunk = ChunkPos::new(0, 0, 0);
    let goal_chunk = ChunkPos::new(3, 0, 0);

    let corridor = ChunkPortalGraph::find_chunk_corridor(start_chunk, goal_chunk);
    assert_eq!(corridor.len(), 4);
    assert_eq!(corridor.first(), Some(&start_chunk));
    assert_eq!(corridor.last(), Some(&goal_chunk));
}
