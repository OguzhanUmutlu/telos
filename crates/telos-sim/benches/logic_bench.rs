//! Micro-benchmarks for deterministic logic & signal propagation engine.
//!
//! Validates the < 5 µs per chunk evaluation budget.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use telos_core::coords::BlockPos;
use telos_sim::logic::{LogicEngine, LogicKind};

fn bench_empty_chunk_lookup(c: &mut Criterion) {
    let engine = LogicEngine::new();
    let query_pos = BlockPos::new(15, 15, 15);

    c.bench_function("logic_empty_chunk_lookup", |b| {
        b.iter(|| engine.get_power(query_pos));
    });
}

fn bench_15_wire_propagation(c: &mut Criterion) {
    c.bench_function("logic_15_wire_propagation", |b| {
        b.iter_batched(
            || {
                let mut engine = LogicEngine::new();
                for x in 1..=15 {
                    engine.set_component(BlockPos::new(x, 0, 0), LogicKind::Wire);
                }
                engine
            },
            |mut engine| {
                let src_pos = BlockPos::new(0, 0, 0);
                engine.set_component(src_pos, LogicKind::PowerBlock);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_100_wire_dense_chunk_evaluation(c: &mut Criterion) {
    c.bench_function("logic_100_wire_dense_chunk_eval", |b| {
        b.iter_batched(
            || {
                let mut engine = LogicEngine::new();
                // Create a 10x10 grid of wires inside chunk (0, 0, 0)
                for z in 0..10 {
                    for x in 0..10 {
                        engine.set_component(BlockPos::new(x, 0, z), LogicKind::Wire);
                    }
                }
                engine
            },
            |mut engine| {
                // Toggle power source at (0, 0, 0)
                let src_pos = BlockPos::new(0, 0, -1);
                engine.set_component(src_pos, LogicKind::PowerBlock);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_empty_chunk_lookup,
    bench_15_wire_propagation,
    bench_100_wire_dense_chunk_evaluation,
);
criterion_main!(benches);
