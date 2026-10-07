//! Integration tests for the sandboxed JavaScript runtime, security limits, and dynamic hooks.

use std::time::{Duration, Instant};
use telos_mod::{
    JsParticleHook, JsParticleParams, JsPlugin, JsPluginEngine, JsSandbox, JsSandboxConfig,
    ModError,
};

#[test]
fn test_js_sandbox_math_and_globals() {
    let sandbox = JsSandbox::new(JsSandboxConfig::default()).expect("Create JS sandbox");
    let result: f64 = sandbox
        .eval("Math.sin(Math.PI / 2) * 10 + Math.sqrt(16)")
        .expect("Evaluate math expression");
    assert!((result - 14.0).abs() < 1e-6);

    let text: String = sandbox
        .eval("'Telos ' + 'Voxel Engine'")
        .expect("String concatenation");
    assert_eq!(text, "Telos Voxel Engine");
}

#[test]
fn test_js_infinite_loop_timeout() {
    let config = JsSandboxConfig {
        timeout_duration: Duration::from_millis(10),
        max_instructions: 50_000,
        ..Default::default()
    };

    let sandbox = JsSandbox::new(config).expect("Create JS sandbox");
    let start = Instant::now();

    let result: Result<i32, ModError> = sandbox.eval("let i = 0; while (true) { i++; } i;");
    let elapsed = start.elapsed();

    assert!(result.is_err(), "Infinite loop must fail");
    match result.unwrap_err() {
        ModError::JsTimeout { message, .. } => {
            println!("Timeout caught successfully in {elapsed:?}: {message}");
        }
        other => panic!("Expected JsTimeout, got: {other:?}"),
    }
    assert!(
        elapsed < Duration::from_millis(500),
        "Timeout must trigger quickly without locking thread"
    );
}

#[test]
fn test_js_memory_quota_exceeded() {
    let config = JsSandboxConfig {
        memory_limit: 1024 * 1024,
        timeout_duration: Duration::from_millis(500),
        max_instructions: 10_000_000,
        ..Default::default()
    };

    let sandbox = JsSandbox::new(config).expect("Create JS sandbox");
    // Attempt unbounded array allocation
    let code = r#"
        const list = [];
        for (let i = 0; i < 500000; i++) {
            list.push(new Array(1000).fill("large_string_allocation_chunk"));
        }
    "#;

    let result: Result<(), ModError> = sandbox.eval(code);
    assert!(result.is_err(), "Memory quota breach must return an error");
    match result.unwrap_err() {
        ModError::JsMemoryLimitExceeded { .. } | ModError::JsExecution { .. } => {
            // Memory quota was enforced without crashing or panicking
        }
        other => panic!("Expected memory limit error, got: {other:?}"),
    }
}

#[test]
fn test_js_stack_overflow_protection() {
    let config = JsSandboxConfig {
        max_stack_size: 64 * 1024, // 64 KiB stack
        ..Default::default()
    };

    let sandbox = JsSandbox::new(config).expect("Create JS sandbox");
    let code = r"
        function recurse(n) {
            return recurse(n + 1);
        }
        recurse(0);
    ";

    let result: Result<(), ModError> = sandbox.eval(code);
    assert!(
        result.is_err(),
        "Deep recursion must be stopped by stack limit"
    );
}

#[test]
fn test_js_zero_ambient_capabilities() {
    let sandbox = JsSandbox::new(JsSandboxConfig::default()).expect("Create JS sandbox");
    let check: bool = sandbox
        .eval("typeof fetch === 'undefined' && typeof process === 'undefined' && typeof require === 'undefined'")
        .expect("Security check");
    assert!(check, "Ambient I/O capabilities must be undefined");
}

#[test]
fn test_js_particle_hook() {
    let script = r"
        function onParticleSpawn(info) {
            return {
                vx: info.vx * 2.0,
                vy: info.vy + 0.5,
                color: 0x00FF_2200,
                scale: 1.5,
                lifetime: 35.0
            };
        }

        function onSoundPlay(sound) {
            return {
                volume: sound.volume * 0.8,
                pitch: sound.pitch * 1.2
            };
        }
    ";

    let hook = JsParticleHook::new(script).expect("Compile particle hook");
    assert!(hook.has_particle_hook());
    assert!(hook.has_sound_hook());

    let initial = JsParticleParams {
        pos: [10.0, 64.0, 10.0],
        velocity: [1.0, 0.2, -1.0],
        color_tint: 0xFFFF_FFFF,
        scale: 1.0,
        lifetime: 20.0,
    };

    let modified = hook.on_particle_spawn("block_debris", initial);
    assert!((modified.velocity[0] - 2.0).abs() < 1e-5);
    assert!((modified.velocity[1] - 0.7).abs() < 1e-5);
    assert_eq!(modified.color_tint, 0x00FF_2200);
    assert!((modified.scale - 1.5).abs() < 1e-5);
    assert!((modified.lifetime - 35.0).abs() < 1e-5);

    let (vol, pitch) = hook.on_sound_play("block.stone.break", 1.0, 1.0);
    assert!((vol - 0.8).abs() < 1e-5);
    assert!((pitch - 1.2).abs() < 1e-5);
}

#[test]
fn test_js_plugin_chat_and_break() {
    let script = r#"
        function onPlayerChat(player, message) {
            if (message.startsWith("!badword")) {
                return null; // Censor / suppress
            }
            if (message.startsWith("!shout ")) {
                return message.substring(7).toUpperCase() + "!";
            }
            return message;
        }

        function onBlockBreak(event) {
            // Protect spawn area [-10..10, -10..10]
            if (event.x >= -10 && event.x <= 10 && event.z >= -10 && event.z <= 10) {
                return false; // Forbidden
            }
            return true;
        }
    "#;

    let plugin = JsPlugin::new("spawn_protect", script).expect("Instantiate JS plugin");
    let mut engine = JsPluginEngine::new();
    engine.add_plugin(plugin);

    // Test chat censorship
    assert_eq!(
        engine.dispatch_player_chat("Alice", "!badword test".into()),
        None
    );
    // Test chat transformation
    assert_eq!(
        engine.dispatch_player_chat("Bob", "!shout hello world".into()),
        Some("HELLO WORLD!".into())
    );
    // Test normal chat
    assert_eq!(
        engine.dispatch_player_chat("Charlie", "Hello everyone".into()),
        Some("Hello everyone".into())
    );

    // Test spawn protection
    assert!(!engine.dispatch_block_break(1, 1, 0, 64, 0)); // Inside spawn -> rejected
    assert!(engine.dispatch_block_break(1, 1, 100, 64, 100)); // Outside spawn -> allowed
}
