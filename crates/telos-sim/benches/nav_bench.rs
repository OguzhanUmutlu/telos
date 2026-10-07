//! Micro-benchmarks for 3D voxel A* navigation and hierarchical portal routing.
//!
//! Validates the <= 50 µs local pathfinding budget.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap;
use telos_core::coords::{BlockPos, ChunkPos};
use telos_sim::nav::{ChunkPortalGraph, ChunkPortals, LocalAStar, NavWorldReader, PathProfile};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

struct BenchWorld {
    registry: BlockRegistry,
    blocks: HashMap<BlockPos, BlockStateId>,
}

impl BenchWorld {
    fn new() -> Self {
        Self {
            registry: BlockRegistry::standard(),
            blocks: HashMap::new(),
        }
    }

    fn set(&mut self, pos: BlockPos, state: BlockStateId) {
        self.blocks.insert(pos, state);
    }
}

impl NavWorldReader for BenchWorld {
    fn get_block(&self, pos: BlockPos) -> BlockStateId {
        self.blocks.get(&pos).copied().unwrap_or(BlockStateId::AIR)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.registry
    }
}

fn bench_flat_straight_path(c: &mut Criterion) {
    let mut world = BenchWorld::new();
    let stone = BlockStateId::new(1);

    for x in -5..=30 {
        for z in -5..=5 {
            world.set(BlockPos::new(x, 0, z), stone);
        }
    }

    let astar = LocalAStar::default();
    let profile = PathProfile::humanoid();
    let start = BlockPos::new(0, 1, 0);
    let target = BlockPos::new(20, 1, 0);

    c.bench_function("nav_flat_20_blocks", |b| {
        b.iter(|| astar.find_path(start, target, 1.0, &profile, &world));
    });
}

fn bench_obstacle_wall_avoidance(c: &mut Criterion) {
    let mut world = BenchWorld::new();
    let stone = BlockStateId::new(1);

    for x in -5..=30 {
        for z in -10..=10 {
            world.set(BlockPos::new(x, 0, z), stone);
        }
    }

    // Build a 2-block high wall at x = 10, z in -6..=6
    for z in -6..=6 {
        world.set(BlockPos::new(10, 1, z), stone);
        world.set(BlockPos::new(10, 2, z), stone);
    }

    let astar = LocalAStar::default();
    let profile = PathProfile::humanoid();
    let start = BlockPos::new(0, 1, 0);
    let target = BlockPos::new(20, 1, 0);

    c.bench_function("nav_obstacle_wall_avoidance", |b| {
        b.iter(|| astar.find_path(start, target, 1.0, &profile, &world));
    });
}

fn bench_slope_jump_up_stairs(c: &mut Criterion) {
    let mut world = BenchWorld::new();
    let stone = BlockStateId::new(1);

    // Build a staircase climbing from y=0 to y=10
    for i in 0..=10 {
        for z in -2..=2 {
            for y in 0..=i {
                world.set(BlockPos::new(i * 2, y, z), stone);
                world.set(BlockPos::new(i * 2 + 1, y, z), stone);
            }
        }
    }

    let astar = LocalAStar::default();
    let profile = PathProfile::humanoid();
    let start = BlockPos::new(0, 1, 0);
    let target = BlockPos::new(20, 11, 0);

    c.bench_function("nav_slope_jump_up_stairs", |b| {
        b.iter(|| astar.find_path(start, target, 1.0, &profile, &world));
    });
}

fn bench_chunk_portal_graph(c: &mut Criterion) {
    let mut world = BenchWorld::new();
    let stone = BlockStateId::new(1);
    for x in -2..=33 {
        for z in -2..=33 {
            world.set(BlockPos::new(x, 0, z), stone);
        }
    }
    let profile = PathProfile::humanoid();

    c.bench_function("nav_chunk_portal_extract_32x32", |b| {
        b.iter(|| ChunkPortals::extract(ChunkPos::new(0, 0, 0), &profile, &world));
    });

    c.bench_function("nav_chunk_portal_corridor_8x8", |b| {
        b.iter(|| {
            ChunkPortalGraph::find_chunk_corridor(ChunkPos::new(0, 0, 0), ChunkPos::new(8, 0, 8))
        });
    });
}

criterion_group!(
    benches,
    bench_flat_straight_path,
    bench_obstacle_wall_avoidance,
    bench_slope_jump_up_stairs,
    bench_chunk_portal_graph,
);
criterion_main!(benches);
