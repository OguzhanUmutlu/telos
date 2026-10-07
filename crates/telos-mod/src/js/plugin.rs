//! Server-side JavaScript plugin engine for gameplay logic, chat filtering, and event handling.

use rquickjs::{Function, Object, Value};

use super::sandbox::{JsSandbox, JsSandboxConfig};
use crate::error::ModResult;

/// A loaded server-side JavaScript plugin.
pub struct JsPlugin {
    id: String,
    sandbox: JsSandbox,
    has_chat_hook: bool,
    has_block_break_hook: bool,
    has_tick_hook: bool,
}

impl JsPlugin {
    /// Compiles and instantiates a new JavaScript plugin from source.
    pub fn new(id: impl Into<String>, source: &str) -> ModResult<Self> {
        let id = id.into();
        let sandbox = JsSandbox::new(JsSandboxConfig::default())?;
        sandbox.load_script(&id, source)?;

        let (has_chat, has_break, has_tick) = sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let has_chat = globals
                .get::<_, Value>("onPlayerChat")
                .is_ok_and(|v| v.is_function());
            let has_break = globals
                .get::<_, Value>("onBlockBreak")
                .is_ok_and(|v| v.is_function());
            let has_tick = globals
                .get::<_, Value>("onTick")
                .is_ok_and(|v| v.is_function());

            Ok((has_chat, has_break, has_tick))
        })?;

        Ok(Self {
            id,
            sandbox,
            has_chat_hook: has_chat,
            has_block_break_hook: has_break,
            has_tick_hook: has_tick,
        })
    }

    /// Plugin identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Invokes `onPlayerChat(player, message)`:
    /// - Returns `None` if the message is suppressed/cancelled.
    /// - Returns `Some(modified_message)` if allowed or transformed.
    pub fn on_player_chat(&self, player: &str, message: &str) -> Option<String> {
        if !self.has_chat_hook {
            return Some(message.to_string());
        }

        let res: Result<Option<String>, _> = self.sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let func: Function = globals.get("onPlayerChat")?;

            let ret: Value = func.call((player, message))?;
            if ret.is_null() || ret.is_undefined() {
                Ok(None)
            } else if let Some(b) = ret.as_bool() {
                if b {
                    Ok(Some(message.to_string()))
                } else {
                    Ok(None)
                }
            } else if let Some(s) = ret.as_string() {
                Ok(s.to_string().ok())
            } else {
                Ok(Some(message.to_string()))
            }
        });

        res.unwrap_or(Some(message.to_string()))
    }

    /// Invokes `onBlockBreak(player_id, block_id, x, y, z)`:
    /// - Returns `false` if block break is forbidden (e.g. protected region).
    /// - Returns `true` if allowed.
    pub fn on_block_break(&self, player_id: u32, block_id: u32, x: i32, y: i32, z: i32) -> bool {
        if !self.has_block_break_hook {
            return true;
        }

        let res: Result<bool, _> = self.sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let func: Function = globals.get("onBlockBreak")?;

            let arg = Object::new(ctx)?;
            arg.set("playerId", player_id)?;
            arg.set("blockId", block_id)?;
            arg.set("x", x)?;
            arg.set("y", y)?;
            arg.set("z", z)?;

            let ret: Value = func.call((arg,))?;
            if let Some(b) = ret.as_bool() {
                Ok(b)
            } else {
                Ok(true)
            }
        });

        res.unwrap_or(true)
    }

    /// Invokes periodic `onTick(tick_count)`.
    pub fn on_tick(&self, tick_count: u64) {
        if !self.has_tick_hook {
            return;
        }

        let _ = self.sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let func: Function = globals.get("onTick")?;
            let _: Value = func.call((tick_count,))?;
            Ok(())
        });
    }
}

/// Central manager orchestrating server-side JavaScript plugins.
#[derive(Default)]
pub struct JsPluginEngine {
    plugins: Vec<JsPlugin>,
}

impl JsPluginEngine {
    /// Creates a new empty `JsPluginEngine`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Registers and activates a new JavaScript plugin.
    pub fn add_plugin(&mut self, plugin: JsPlugin) {
        self.plugins.push(plugin);
    }

    /// Returns the number of loaded plugins.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Dispatches player chat across all loaded plugins in order.
    pub fn dispatch_player_chat(&self, player: &str, mut message: String) -> Option<String> {
        for plugin in &self.plugins {
            message = plugin.on_player_chat(player, &message)?;
        }
        Some(message)
    }

    /// Dispatches block break verification across all loaded plugins.
    #[must_use]
    pub fn dispatch_block_break(
        &self,
        player_id: u32,
        block_id: u32,
        x: i32,
        y: i32,
        z: i32,
    ) -> bool {
        for plugin in &self.plugins {
            if !plugin.on_block_break(player_id, block_id, x, y, z) {
                return false; // Cancelled by plugin
            }
        }
        true
    }

    /// Advances the simulation tick for all loaded plugins.
    pub fn dispatch_tick(&self, tick_count: u64) {
        for plugin in &self.plugins {
            plugin.on_tick(tick_count);
        }
    }
}
