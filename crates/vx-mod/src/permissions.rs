//! Capability-based permission system for sandboxed mods.

bitflags::bitflags! {
    /// Capability flags granted to a sandboxed mod.
    ///
    /// Every capability corresponds to a specific host API interface.
    /// Ungranted capabilities will be rejected immediately with `ModError::Denied`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct ModPermissions: u32 {
        /// Allows reading world voxel blocks and properties (`vx:world/read`).
        const WORLD_READ = 1 << 0;
        /// Allows queuing block modifications and fills (`vx:world/write`).
        const WORLD_WRITE = 1 << 1;
        /// Allows subscribing to and receiving game simulation events (`vx:server/events`).
        const EVENTS_LISTEN = 1 << 2;
        /// Allows registering custom server/chat commands (`vx:server/commands`).
        const COMMANDS_REGISTER = 1 << 3;
    }
}

impl ModPermissions {
    /// Returns a full permissions set granting all capabilities.
    #[must_use]
    pub const fn all_permissions() -> Self {
        Self::all()
    }

    /// Returns a safe default set for standard gameplay mods (world read + events).
    #[must_use]
    pub const fn standard() -> Self {
        Self::from_bits_truncate(Self::WORLD_READ.bits() | Self::EVENTS_LISTEN.bits())
    }
}
