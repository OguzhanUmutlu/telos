//! WebAssembly engine and linker setup for sandboxed mods.

use wasmtime::{Config, Engine, Linker};

use crate::error::{ModError, ModResult};
use crate::host_state::HostState;

/// Shared Wasmtime engine configured for deterministic execution and budgeting.
pub struct WasmEngine {
    engine: Engine,
}

impl WasmEngine {
    /// Creates a new `WasmEngine` with fuel consumption and component model enabled.
    pub fn new() -> ModResult<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);

        let engine = Engine::new(&config).map_err(|e| ModError::EngineInit(e.to_string()))?;

        Ok(Self { engine })
    }

    /// Returns a reference to the underlying Wasmtime `Engine`.
    #[must_use]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Creates a new linker pre-populated with standard host API functions.
    pub fn create_linker(&self) -> ModResult<Linker<HostState>> {
        let mut linker = Linker::new(&self.engine);
        register_host_functions(&mut linker)?;
        Ok(linker)
    }
}

#[allow(clippy::too_many_lines)]
fn register_host_functions(linker: &mut Linker<HostState>) -> ModResult<()> {
    // Register host functions under "vx" and "env"
    for module in &["vx", "env"] {
        linker
            .func_wrap(
                module,
                "telos_log",
                |mut caller: wasmtime::Caller<'_, HostState>, level: i32, ptr: i32, len: i32| {
                    if ptr < 0 || !(0..=1024).contains(&len) {
                        return;
                    }
                    let Some(wasmtime::Extern::Memory(memory)) = caller.get_export("memory") else {
                        return;
                    };
                    let (data, store) = memory.data_and_store_mut(&mut caller);
                    let start = ptr as usize;
                    let end = start.saturating_add(len as usize);
                    if end > data.len() {
                        return;
                    }
                    if let Ok(msg) = std::str::from_utf8(&data[start..end]) {
                        let tracing_level = match level {
                            0 => tracing::Level::TRACE,
                            1 => tracing::Level::DEBUG,
                            2 => tracing::Level::INFO,
                            3 => tracing::Level::WARN,
                            _ => tracing::Level::ERROR,
                        };
                        store.record_log(tracing_level, msg.to_string());
                    }
                },
            )
            .map_err(|e| ModError::EngineInit(e.to_string()))?;

        linker
            .func_wrap(
                module,
                "telos_get_block",
                |caller: wasmtime::Caller<'_, HostState>, x: i32, y: i32, z: i32| -> i32 {
                    let pos = telos_core::coords::BlockPos::new(x, y, z);
                    match caller.data().read_block(pos) {
                        Ok(Some(state)) => i32::try_from(state.0).unwrap_or(0),
                        Ok(None) => 0,
                        Err(_) => -1, // Permission denied or error
                    }
                },
            )
            .map_err(|e| ModError::EngineInit(e.to_string()))?;

        linker
            .func_wrap(
                module,
                "telos_set_block",
                |mut caller: wasmtime::Caller<'_, HostState>,
                 x: i32,
                 y: i32,
                 z: i32,
                 state: i32|
                 -> i32 {
                    let pos = telos_core::coords::BlockPos::new(x, y, z);
                    let state_id = telos_voxel::state::BlockStateId::new(state.unsigned_abs());
                    match caller.data_mut().queue_block_edit(pos, state_id) {
                        Ok(()) => 0,
                        Err(ModError::Denied { .. }) => -1,
                        Err(_) => -2, // Quota exceeded or invalid
                    }
                },
            )
            .map_err(|e| ModError::EngineInit(e.to_string()))?;

        linker
            .func_wrap(
                module,
                "telos_subscribe_events",
                |mut caller: wasmtime::Caller<'_, HostState>, filter_bits: i32| -> i32 {
                    let filter = telos_sim::event::EventFilter::from_bits_truncate(
                        filter_bits.unsigned_abs(),
                    );
                    match caller.data_mut().subscribe_events(filter) {
                        Ok(()) => 0,
                        Err(_) => -1,
                    }
                },
            )
            .map_err(|e| ModError::EngineInit(e.to_string()))?;

        linker
            .func_wrap(
                module,
                "telos_register_command",
                |mut caller: wasmtime::Caller<'_, HostState>, ptr: i32, len: i32| -> i32 {
                    if ptr < 0 || !(0..=64).contains(&len) {
                        return -2;
                    }
                    let Some(wasmtime::Extern::Memory(memory)) = caller.get_export("memory") else {
                        return -3;
                    };
                    let (data, store) = memory.data_and_store_mut(&mut caller);
                    let start = ptr as usize;
                    let end = start.saturating_add(len as usize);
                    if end > data.len() {
                        return -2;
                    }
                    if let Ok(name) = std::str::from_utf8(&data[start..end]) {
                        match store.register_command(name.to_string()) {
                            Ok(()) => 0,
                            Err(_) => -1,
                        }
                    } else {
                        -2
                    }
                },
            )
            .map_err(|e| ModError::EngineInit(e.to_string()))?;
    }

    Ok(())
}
