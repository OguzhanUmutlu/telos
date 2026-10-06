//! Resource limits and execution budgeting configuration for sandboxed mods.

/// Configuration parameters governing mod sandbox execution limits and budgets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModConfig {
    /// Maximum CPU instruction fuel granted per tick or export invocation.
    /// Default: 500,000 instructions (~0.5 ms CPU budget).
    pub fuel_per_invocation: u64,

    /// Maximum linear memory table size in bytes allocated per mod store.
    /// Default: 32 MiB (33,554,432 bytes).
    pub max_memory_bytes: usize,

    /// Maximum number of queued block edits permitted from a mod per tick.
    /// Default: 512 edits.
    pub max_queued_edits_per_tick: usize,

    /// Maximum number of host log messages allowed per mod per tick.
    /// Default: 64 messages.
    pub max_log_messages_per_tick: usize,

    /// Maximum volume in blocks that can be queried in a single box call.
    /// Default: 32,768 voxels (one 32³ cubic chunk equivalent).
    pub max_box_query_volume: usize,
}

impl Default for ModConfig {
    fn default() -> Self {
        Self {
            fuel_per_invocation: 500_000,
            max_memory_bytes: 32 * 1024 * 1024,
            max_queued_edits_per_tick: 512,
            max_log_messages_per_tick: 64,
            max_box_query_volume: 32_768,
        }
    }
}
