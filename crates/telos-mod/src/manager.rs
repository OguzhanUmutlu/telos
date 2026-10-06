//! Mod manager orchestrating loaded mods and event dispatching.

use telos_sim::event::GameEvent;
use wasmtime::{Linker, Module, Store};

use crate::component::LoadedMod;
use crate::config::ModConfig;
use crate::engine::WasmEngine;
use crate::error::{ModError, ModResult};
use crate::host_state::{BlockEdit, HostState, WorldReader};
use crate::permissions::ModPermissions;

/// Central manager orchestrating sandboxed mod loading, execution, and event dispatch.
pub struct ModManager {
    engine: WasmEngine,
    linker: Linker<HostState>,
    mods: Vec<LoadedMod>,
    world_reader: Option<WorldReader>,
}

impl ModManager {
    /// Creates a new `ModManager` instance.
    pub fn new() -> ModResult<Self> {
        let engine = WasmEngine::new()?;
        let linker = engine.create_linker()?;

        Ok(Self {
            engine,
            linker,
            mods: Vec::new(),
            world_reader: None,
        })
    }

    /// Sets the world reader callback used by sandboxed mods to query blocks.
    pub fn set_world_reader(&mut self, reader: WorldReader) {
        self.world_reader = Some(reader);
    }

    /// Returns the number of loaded mods.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.mods.len()
    }

    /// Returns a slice of all loaded mods.
    #[must_use]
    pub fn mods(&self) -> &[LoadedMod] {
        &self.mods
    }

    /// Returns a mutable slice of all loaded mods.
    pub fn mods_mut(&mut self) -> &mut [LoadedMod] {
        &mut self.mods
    }

    /// Finds a loaded mod by its identifier.
    #[must_use]
    pub fn get_mod(&self, mod_id: &str) -> Option<&LoadedMod> {
        self.mods.iter().find(|m| m.mod_id() == mod_id)
    }

    /// Finds a loaded mod by its identifier (mutable).
    pub fn get_mod_mut(&mut self, mod_id: &str) -> Option<&mut LoadedMod> {
        self.mods.iter_mut().find(|m| m.mod_id() == mod_id)
    }

    /// Returns all command names registered across all loaded mods.
    #[must_use]
    pub fn registered_commands(&self) -> Vec<String> {
        let mut commands = Vec::new();
        for m in &self.mods {
            for cmd in &m.state().registered_commands {
                if !commands.contains(cmd) {
                    commands.push(cmd.clone());
                }
            }
        }
        commands
    }

    /// Loads and instantiates a mod from WebAssembly binary bytes (`.wasm`).
    pub fn load_mod_from_bytes(
        &mut self,
        mod_id: &str,
        wasm_bytes: &[u8],
        permissions: ModPermissions,
        config: ModConfig,
    ) -> ModResult<()> {
        let module =
            Module::new(self.engine.engine(), wasm_bytes).map_err(|e| ModError::Compile {
                mod_id: mod_id.to_string(),
                message: e.to_string(),
            })?;

        self.instantiate_module(mod_id, &module, permissions, config)
    }

    /// Loads and instantiates a mod from WebAssembly text format (`.wat`).
    pub fn load_mod_from_wat(
        &mut self,
        mod_id: &str,
        wat_source: &str,
        permissions: ModPermissions,
        config: ModConfig,
    ) -> ModResult<()> {
        let module = Module::new(self.engine.engine(), wat_source.as_bytes()).map_err(|e| {
            ModError::Compile {
                mod_id: mod_id.to_string(),
                message: e.to_string(),
            }
        })?;

        self.instantiate_module(mod_id, &module, permissions, config)
    }

    fn instantiate_module(
        &mut self,
        mod_id: &str,
        module: &Module,
        permissions: ModPermissions,
        config: ModConfig,
    ) -> ModResult<()> {
        let host_state = HostState::new(
            mod_id.to_string(),
            permissions,
            config,
            self.world_reader.clone(),
        );

        let mut store = Store::new(self.engine.engine(), host_state);
        let instance =
            self.linker
                .instantiate(&mut store, module)
                .map_err(|e| ModError::Instantiation {
                    mod_id: mod_id.to_string(),
                    message: e.to_string(),
                })?;

        let mut loaded_mod = LoadedMod::new(mod_id.to_string(), store, instance);
        loaded_mod.call_init()?;

        self.mods.push(loaded_mod);
        tracing::info!("Loaded sandboxed WASM mod '{}'", mod_id);

        Ok(())
    }

    /// Dispatches a batch of game events to all subscribed mods and collects queued block edits.
    pub fn dispatch_events(&mut self, events: &[GameEvent]) -> Vec<BlockEdit> {
        let mut all_edits = Vec::new();

        for m in &mut self.mods {
            if m.is_suspended() {
                continue;
            }

            for event in events {
                if let Err(err) = m.call_on_event(event) {
                    tracing::warn!(
                        "Mod '{}' error handling event {:?}: {}",
                        m.mod_id(),
                        event,
                        err
                    );
                }
            }

            let edits = m.drain_edits();
            all_edits.extend(edits);

            for (level, msg) in m.drain_logs() {
                match level {
                    tracing::Level::TRACE => tracing::trace!(mod_id = m.mod_id(), "{}", msg),
                    tracing::Level::DEBUG => tracing::debug!(mod_id = m.mod_id(), "{}", msg),
                    tracing::Level::INFO => tracing::info!(mod_id = m.mod_id(), "{}", msg),
                    tracing::Level::WARN => tracing::warn!(mod_id = m.mod_id(), "{}", msg),
                    tracing::Level::ERROR => tracing::error!(mod_id = m.mod_id(), "{}", msg),
                }
            }
        }

        all_edits
    }

    /// Dispatches a registered command to any mod handling it.
    pub fn dispatch_command(&mut self, cmd: &str, args: &str) -> bool {
        let mut handled = false;

        for m in &mut self.mods {
            if m.is_suspended() {
                continue;
            }

            match m.call_on_command(cmd, args) {
                Ok(true) => handled = true,
                Ok(false) => {}
                Err(err) => {
                    tracing::warn!(
                        "Mod '{}' error executing command '{}': {}",
                        m.mod_id(),
                        cmd,
                        err
                    );
                }
            }
        }

        handled
    }
}
