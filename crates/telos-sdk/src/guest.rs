//! Guest-side host API bindings, logging, world access, and `TelosMod` lifecycle trait.

use crate::events::{EventFilter, ModEvent};

/// Logging levels for mod log messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace level.
    Trace = 0,
    /// Debug level.
    Debug = 1,
    /// Information level.
    Info = 2,
    /// Warning level.
    Warn = 3,
    /// Error level.
    Error = 4,
}

/// The core trait implemented by Telos guest mods.
pub trait TelosMod {
    /// Called when the mod is loaded and initialized by the host engine.
    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Called when a subscribed game event occurs.
    fn on_event(&mut self, _event: &ModEvent) {}

    /// Called when a player executes a command registered by this mod.
    fn on_command(&mut self, _cmd: &str, _args: &str) -> Result<String, String> {
        Ok(String::new())
    }
}

// Low-level extern FFI bindings
#[cfg(target_arch = "wasm32")]
mod ffi {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        pub fn telos_log(level: i32, ptr: *const u8, len: i32);
        pub fn telos_get_block(x: i32, y: i32, z: i32) -> i32;
        pub fn telos_set_block(x: i32, y: i32, z: i32, state: i32) -> i32;
        pub fn telos_subscribe_events(filter_bits: i32) -> i32;
        pub fn telos_register_command(ptr: *const u8, len: i32) -> i32;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod ffi {
    pub unsafe fn telos_log(_level: i32, _ptr: *const u8, _len: i32) {}
    pub unsafe fn telos_get_block(_x: i32, _y: i32, _z: i32) -> i32 {
        0
    }
    pub unsafe fn telos_set_block(_x: i32, _y: i32, _z: i32, _state: i32) -> i32 {
        0
    }
    pub unsafe fn telos_subscribe_events(_filter_bits: i32) -> i32 {
        0
    }
    pub unsafe fn telos_register_command(_ptr: *const u8, _len: i32) -> i32 {
        0
    }
}

/// Emits a log message to the host engine console.
pub fn log(level: LogLevel, message: &str) {
    let bytes = message.as_bytes();
    let len = i32::try_from(bytes.len()).unwrap_or(i32::MAX);
    // SAFETY: bytes.as_ptr() is a valid pointer for len bytes.
    unsafe {
        ffi::telos_log(level as i32, bytes.as_ptr(), len);
    }
}

/// Reads the block state ID at the given world coordinates.
/// Returns `None` if the chunk is not loaded or permission is denied.
#[must_use]
pub fn get_block(x: i32, y: i32, z: i32) -> Option<u32> {
    // SAFETY: Foreign host call passing primitive coordinates.
    let res = unsafe { ffi::telos_get_block(x, y, z) };
    if res >= 0 {
        Some(res.unsigned_abs())
    } else {
        None
    }
}

/// Queues a block change at the given world coordinates.
pub fn set_block(x: i32, y: i32, z: i32, state_id: u32) -> Result<(), String> {
    let state = i32::try_from(state_id).unwrap_or(i32::MAX);
    // SAFETY: Foreign host call passing primitive values.
    let res = unsafe { ffi::telos_set_block(x, y, z, state) };
    match res {
        0 => Ok(()),
        -1 => Err("Permission denied (world.write not granted)".to_string()),
        -2 => Err("Queued block edits quota exceeded".to_string()),
        _ => Err(format!("Host set_block failed with error code {res}")),
    }
}

/// Subscribes to events matching the specified filter mask.
pub fn subscribe_events(filter: EventFilter) -> Result<(), String> {
    let bits = i32::try_from(filter.0).unwrap_or(i32::MAX);
    // SAFETY: Foreign host call passing bitmask.
    let res = unsafe { ffi::telos_subscribe_events(bits) };
    if res == 0 {
        Ok(())
    } else {
        Err("Permission denied (events.listen not granted)".to_string())
    }
}

/// Registers a command with the host command router.
pub fn register_command(name: &str) -> Result<(), String> {
    let bytes = name.as_bytes();
    let len = i32::try_from(bytes.len()).unwrap_or(i32::MAX);
    // SAFETY: name.as_ptr() is valid for len bytes.
    let res = unsafe { ffi::telos_register_command(bytes.as_ptr(), len) };
    match res {
        0 => Ok(()),
        -1 => Err("Permission denied (command.register not granted)".to_string()),
        _ => Err(format!("Host register_command failed with code {res}")),
    }
}
