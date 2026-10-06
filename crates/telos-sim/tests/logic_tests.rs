//! Deterministic logic simulation integration tests.

use telos_core::coords::{BlockPos, Face};
use telos_sim::logic::{LogicEngine, LogicKind, step_in_dir};

#[test]
fn test_complex_circuit_multi_branch() {
    let mut engine = LogicEngine::new();
    let src = BlockPos::new(0, 0, 0);
    engine.set_component(src, LogicKind::PowerBlock);

    // Branch 1: North for 5 blocks
    for d in 1..=5 {
        engine.set_component(step_in_dir(src, Face::North, d), LogicKind::Wire);
    }

    // Branch 2: East for 5 blocks
    for d in 1..=5 {
        engine.set_component(step_in_dir(src, Face::East, d), LogicKind::Wire);
    }

    // Branch 3: South for 5 blocks
    for d in 1..=5 {
        engine.set_component(step_in_dir(src, Face::South, d), LogicKind::Wire);
    }

    // Branch 4: West for 5 blocks
    for d in 1..=5 {
        engine.set_component(step_in_dir(src, Face::West, d), LogicKind::Wire);
    }

    // All 4 branches must have bit-exact identical power at distance 5 (15 - 5 = 10)
    assert_eq!(engine.get_power(step_in_dir(src, Face::North, 5)), 10);
    assert_eq!(engine.get_power(step_in_dir(src, Face::East, 5)), 10);
    assert_eq!(engine.get_power(step_in_dir(src, Face::South, 5)), 10);
    assert_eq!(engine.get_power(step_in_dir(src, Face::West, 5)), 10);
}

#[test]
fn test_loop_determinism_and_no_infinite_cycle() {
    let mut engine = LogicEngine::new();

    // Create a 4x4 wire ring:
    // (0,0) - (1,0) - (2,0)
    //   |               |
    // (0,1)           (2,1)
    //   |               |
    // (0,2) - (1,2) - (2,2)
    let ring_coords = [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ];

    for &(x, z) in &ring_coords {
        engine.set_component(BlockPos::new(x, 0, z), LogicKind::Wire);
    }

    // Initially all 0
    for &(x, z) in &ring_coords {
        assert_eq!(engine.get_power(BlockPos::new(x, 0, z)), 0);
    }

    // Introduce power at (0, 0)
    let src = BlockPos::new(0, 0, -1);
    engine.set_component(src, LogicKind::PowerBlock);

    // Ring power values must settle deterministically:
    // (0,0) is dist 1 -> 14
    // (1,0) and (0,1) are dist 2 -> 13
    // (2,0) and (0,2) are dist 3 -> 12
    // (2,1) and (1,2) are dist 4 -> 11
    // (2,2) is dist 5 -> 10
    assert_eq!(engine.get_power(BlockPos::new(0, 0, 0)), 14);
    assert_eq!(engine.get_power(BlockPos::new(1, 0, 0)), 13);
    assert_eq!(engine.get_power(BlockPos::new(0, 0, 1)), 13);
    assert_eq!(engine.get_power(BlockPos::new(2, 0, 0)), 12);
    assert_eq!(engine.get_power(BlockPos::new(0, 0, 2)), 12);
    assert_eq!(engine.get_power(BlockPos::new(2, 0, 1)), 11);
    assert_eq!(engine.get_power(BlockPos::new(1, 0, 2)), 11);
    assert_eq!(engine.get_power(BlockPos::new(2, 0, 2)), 10);
}

#[test]
fn test_inverter_oscillation_clock_at_20tps() {
    let mut engine = LogicEngine::new();

    // Rapid NOT-gate oscillator: Inverter facing East pointing directly back into its own wire input:
    // Inverter at (0, 0, 0) facing East. Input is West (-1, 0, 0).
    // Wire from (1, 0, 0) around to (-1, 0, 0).
    let inv_pos = BlockPos::new(0, 0, 0);
    let wire_out = BlockPos::new(1, 0, 0);
    let wire_corner1 = BlockPos::new(1, 0, 1);
    let wire_mid = BlockPos::new(0, 0, 1);
    let wire_corner2 = BlockPos::new(-1, 0, 1);
    let wire_in = BlockPos::new(-1, 0, 0);

    engine.set_component(
        inv_pos,
        LogicKind::Inverter {
            facing: Face::East,
            powered: true,
        },
    );
    engine.set_component(wire_out, LogicKind::Wire);
    engine.set_component(wire_corner1, LogicKind::Wire);
    engine.set_component(wire_mid, LogicKind::Wire);
    engine.set_component(wire_corner2, LogicKind::Wire);
    engine.set_component(wire_in, LogicKind::Wire);

    // Initial state: inverter output is powered, wire_in receives power
    assert_eq!(engine.get_power(wire_out), 14);
    assert_eq!(engine.get_power(wire_in), 10);

    // Tick 1: inverter input has power 10 >= 1, inverter scheduled to turn OFF
    engine.tick(1);
    assert_eq!(engine.get_power(wire_out), 0);
    assert_eq!(engine.get_power(wire_in), 0);

    // Tick 2: inverter input is now 0, inverter turns ON!
    engine.tick(2);
    assert_eq!(engine.get_power(wire_out), 14);
    assert_eq!(engine.get_power(wire_in), 10);

    // Tick 3: inverter turns OFF again (clean 1-tick 10Hz square wave clock)!
    engine.tick(3);
    assert_eq!(engine.get_power(wire_out), 0);
    assert_eq!(engine.get_power(wire_in), 0);
}
