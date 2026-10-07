//! Micro-benchmarks for cellular automata fluid flow engine.
//!
//! Validates the < 15 µs per active fluid chunk evaluation budget.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap;
use telos_core::coords::BlockPos;
use telos_sim::fluid::{FluidEngine, FluidKind, FluidWorldReader};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

struct BenchWorld {
    registry: BlockRegistry,
    blocks: HashMap<BlockPos, BlockStateId>,
}

impl BenchWorld {
    fn new() -> Self {
        let mut world = Self {
            registry: BlockRegistry::standard(),
            blocks: HashMap::new(),
        };
        let stone = BlockStateId::new(1);
        for x in -16..=16 {
            for z in -16..=16 {
                world.blocks.insert(BlockPos::new(x, 0, z), stone);
            }
        }
        world
    }
}

impl FluidWorldReader for BenchWorld {
    fn get_block(&self, pos: BlockPos) -> BlockStateId {
        self.blocks.get(&pos).copied().unwrap_or(BlockStateId::AIR)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.registry
    }
}

fn bench_water_spread_step(c: &mut Criterion) {
    let world = BenchWorld::new();
    let water_source = world.registry.fluid_state_id(FluidKind::Water, 0, false);
    let mut setup_world = BenchWorld::new();
    let src = BlockPos::new(0, 1, 0);
    setup_world.blocks.insert(src, water_source);

    c.bench_function("fluid_water_spread_step", |b| {
        b.iter_batched(
            || {
                let mut engine = FluidEngine::new();
                engine.on_block_changed(src, 0, &setup_world);
                engine
            },
            |mut engine| engine.tick(5, &setup_world),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_waterfall_column_step(c: &mut Criterion) {
    let mut setup_world = BenchWorld::new();
    let water_source = setup_world
        .registry
        .fluid_state_id(FluidKind::Water, 0, false);
    let src = BlockPos::new(0, 20, 0);
    setup_world.blocks.insert(src, water_source);

    c.bench_function("fluid_waterfall_column_step", |b| {
        b.iter_batched(
            || {
                let mut engine = FluidEngine::new();
                engine.on_block_changed(src, 0, &setup_world);
                engine
            },
            |mut engine| engine.tick(5, &setup_world),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_reaction_step(c: &mut Criterion) {
    let mut setup_world = BenchWorld::new();
    let water_source = setup_world
        .registry
        .fluid_state_id(FluidKind::Water, 0, false);
    let lava_source = setup_world
        .registry
        .fluid_state_id(FluidKind::Lava, 0, false);
    let water_pos = BlockPos::new(0, 1, 0);
    let lava_pos = BlockPos::new(1, 1, 0);
    setup_world.blocks.insert(water_pos, water_source);
    setup_world.blocks.insert(lava_pos, lava_source);

    c.bench_function("fluid_reaction_obsidian_step", |b| {
        b.iter_batched(
            || {
                let mut engine = FluidEngine::new();
                engine.on_block_changed(water_pos, 0, &setup_world);
                engine
            },
            |mut engine| engine.tick(5, &setup_world),
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_water_spread_step,
    bench_waterfall_column_step,
    bench_reaction_step,
);
criterion_main!(benches);
