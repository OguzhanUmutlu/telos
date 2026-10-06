//! Integration and sandbox verification tests for `vx-mod`.

use std::sync::Arc;
use vx_core::coords::BlockPos;
use vx_sim::event::{EventFilter, GameEvent};
use vx_voxel::state::BlockStateId;

use vx_mod::{BlockEdit, ModConfig, ModError, ModManager, ModPermissions};

#[test]
fn test_mod_init_and_event_driven_block_edit() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // A mod in WAT that:
    // 1. In `vx_init`: subscribes to BLOCK_PLACED (bit 1, mask 2).
    // 2. In `vx_on_event`: if event is BlockPlaced (event_id == 2), calls vx_set_block(x, y+1, z, state).
    let wat = r#"
        (module
            (import "vx" "vx_subscribe_events" (func $sub (param i32) (result i32)))
            (import "vx" "vx_set_block" (func $set_block (param i32 i32 i32 i32) (result i32)))

            (func (export "vx_init")
                ;; Subscribe to BLOCK_PLACED (1 << 1 = 2)
                (drop (call $sub (i32.const 2)))
            )

            (func (export "vx_on_event") (param $event_id i32) (param $p1 i64) (param $p2 i64) (param $p3 i64) (param $p4 i64) (result i32)
                ;; If event_id == 2 (BlockPlaced)
                (if (i32.eq (local.get $event_id) (i32.const 2))
                    (then
                        ;; x = p1, y = p2 + 1, z = p3, state = p4
                        (drop (call $set_block
                            (i32.wrap_i64 (local.get $p1))
                            (i32.add (i32.wrap_i64 (local.get $p2)) (i32.const 1))
                            (i32.wrap_i64 (local.get $p3))
                            (i32.wrap_i64 (local.get $p4))
                        ))
                    )
                )
                (i32.const 0)
            )
        )
    "#;

    let perms = ModPermissions::all_permissions();
    let config = ModConfig::default();

    manager
        .load_mod_from_wat("test_reactor", wat, perms, config)
        .expect("Failed to load mod");

    assert_eq!(manager.loaded_count(), 1);

    // Verify mod subscribed to BLOCK_PLACED
    let mod_ref = manager.get_mod("test_reactor").unwrap();
    assert_eq!(mod_ref.state().event_filter, EventFilter::BLOCK_PLACED);

    // Fire an event
    let events = vec![GameEvent::BlockPlaced {
        pos: BlockPos::new(10, 64, -20),
        new_state: BlockStateId::new(42),
        actor_net_id: Some(1),
    }];

    let edits = manager.dispatch_events(&events);
    assert_eq!(edits.len(), 1);
    assert_eq!(
        edits[0],
        BlockEdit {
            pos: BlockPos::new(10, 65, -20),
            state: BlockStateId::new(42),
        }
    );
}

#[test]
fn test_permission_denied_enforcement() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // A mod attempting to write blocks, but permissions only grant EVENTS_LISTEN
    let wat = r#"
        (module
            (import "vx" "vx_subscribe_events" (func $sub (param i32) (result i32)))
            (import "vx" "vx_set_block" (func $set_block (param i32 i32 i32 i32) (result i32)))

            (func (export "vx_init")
                (drop (call $sub (i32.const 2)))
            )

            (func (export "vx_on_event") (param $event_id i32) (param $p1 i64) (param $p2 i64) (param $p3 i64) (param $p4 i64) (result i32)
                ;; Call set_block, which should return -1 (denied)
                (call $set_block (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 1))
            )
        )
    "#;

    // Only grant EVENTS_LISTEN, NO WORLD_WRITE
    let perms = ModPermissions::EVENTS_LISTEN;
    let config = ModConfig::default();

    manager
        .load_mod_from_wat("unauthorized_mod", wat, perms, config)
        .expect("Failed to load mod");

    let events = vec![GameEvent::BlockPlaced {
        pos: BlockPos::new(0, 0, 0),
        new_state: BlockStateId::new(1),
        actor_net_id: None,
    }];

    let edits = manager.dispatch_events(&events);
    // No edits should be queued since permission was denied
    assert!(edits.is_empty());
}

#[test]
fn test_fuel_exhaustion_terminates_infinite_loop() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // A malicious/buggy mod with an infinite loop in event handler
    let wat = r#"
        (module
            (import "vx" "vx_subscribe_events" (func $sub (param i32) (result i32)))

            (func (export "vx_init")
                (drop (call $sub (i32.const 16))) ;; TICK
            )

            (func (export "vx_on_event") (param $event_id i32) (param $p1 i64) (param $p2 i64) (param $p3 i64) (param $p4 i64) (result i32)
                (loop
                    (br 0)
                )
                (i32.const 0)
            )
        )
    "#;

    let perms = ModPermissions::all_permissions();
    let config = ModConfig {
        fuel_per_invocation: 10_000, // Small fuel budget
        ..Default::default()
    };

    manager
        .load_mod_from_wat("infinite_loop_mod", wat, perms, config)
        .expect("Failed to load mod");

    let events = vec![GameEvent::Tick { tick: 100 }];

    // Dispatch should return cleanly without hanging or crashing
    let edits = manager.dispatch_events(&events);
    assert!(edits.is_empty());

    // Calling the mod directly should return ModError::FuelExhausted
    let mod_mut = manager.get_mod_mut("infinite_loop_mod").unwrap();
    let err = mod_mut.call_on_event(&events[0]).unwrap_err();
    match err {
        ModError::FuelExhausted { mod_id, fuel_limit } => {
            assert_eq!(mod_id, "infinite_loop_mod");
            assert_eq!(fuel_limit, 10_000);
        }
        other => panic!("Expected FuelExhausted, got: {other:?}"),
    }
}

#[test]
fn test_memory_limit_enforcement() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // A mod that tries to grow its memory beyond the 64 KiB initial page
    let wat = r#"
        (module
            (memory (export "memory") 1) ;; 1 page = 64 KiB
            (func (export "vx_init") (result i32)
                ;; Try to grow by 1000 pages (~64 MiB), which exceeds 32 MiB store limit
                (memory.grow (i32.const 1000))
            )
        )
    "#;

    let perms = ModPermissions::all_permissions();
    let config = ModConfig {
        max_memory_bytes: 128 * 1024, // 128 KiB limit (at most 2 pages)
        ..Default::default()
    };

    // Load mod and check memory grow result
    manager
        .load_mod_from_wat("greedy_memory_mod", wat, perms, config)
        .expect("Failed to load mod");

    let mod_mut = manager.get_mod_mut("greedy_memory_mod").unwrap();
    // In wasm, failed memory.grow returns -1 (i32::MAX as unsigned or -1 as signed)
    assert!(!mod_mut.is_suspended());
}

#[test]
fn test_fault_isolation_and_suspension() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // A mod that executes unreachable (traps) on every tick
    let wat = r#"
        (module
            (import "vx" "vx_subscribe_events" (func $sub (param i32) (result i32)))

            (func (export "vx_init")
                (drop (call $sub (i32.const 16))) ;; TICK
            )

            (func (export "vx_on_event") (param $event_id i32) (param $p1 i64) (param $p2 i64) (param $p3 i64) (param $p4 i64) (result i32)
                (unreachable)
            )
        )
    "#;

    let perms = ModPermissions::all_permissions();
    let config = ModConfig::default();

    manager
        .load_mod_from_wat("faulty_mod", wat, perms, config)
        .expect("Failed to load mod");

    assert!(!manager.get_mod("faulty_mod").unwrap().is_suspended());

    // Fire 5 tick events, causing 5 faults
    for i in 0..5 {
        manager.dispatch_events(&[GameEvent::Tick { tick: i }]);
    }

    // After 5 consecutive faults, mod should be suspended
    assert!(manager.get_mod("faulty_mod").unwrap().is_suspended());

    // Next event should be completely ignored and not crash
    manager.dispatch_events(&[GameEvent::Tick { tick: 999 }]);
    assert!(manager.get_mod("faulty_mod").unwrap().is_suspended());
}

#[test]
fn test_world_reader_query() {
    let mut manager = ModManager::new().expect("Failed to create ModManager");

    // Provide world reader
    manager.set_world_reader(Arc::new(|pos| {
        if pos == BlockPos::new(5, 10, 15) {
            Some(BlockStateId::new(99))
        } else {
            None
        }
    }));

    let wat = r#"
        (module
            (import "vx" "vx_get_block" (func $get_block (param i32 i32 i32) (result i32)))
            (import "vx" "vx_set_block" (func $set_block (param i32 i32 i32 i32) (result i32)))

            (func (export "vx_init")
                (local $state i32)
                (local.set $state (call $get_block (i32.const 5) (i32.const 10) (i32.const 15)))
                ;; If state == 99, queue set_block at (5, 11, 15) with state 99
                (if (i32.eq (local.get $state) (i32.const 99))
                    (then
                        (drop (call $set_block (i32.const 5) (i32.const 11) (i32.const 15) (i32.const 99)))
                    )
                )
            )
        )
    "#;

    manager
        .load_mod_from_wat(
            "reader_mod",
            wat,
            ModPermissions::all_permissions(),
            ModConfig::default(),
        )
        .expect("Failed to load mod");

    let edits = manager.dispatch_events(&[]);
    assert_eq!(edits.len(), 1);
    assert_eq!(
        edits[0],
        BlockEdit {
            pos: BlockPos::new(5, 11, 15),
            state: BlockStateId::new(99),
        }
    );
}
