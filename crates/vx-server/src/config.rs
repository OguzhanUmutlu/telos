//! Server configuration parameters.

/// Configuration parameters for server tick rate, view distance, and quotas.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Target simulation ticks per second (default: 20).
    pub tps: u32,
    /// Default horizontal view distance in chunks (radius, default: 8).
    pub view_distance: u32,
    /// Vertical chunk radius (default: 2 chunks above and below).
    pub vertical_view_distance: u32,
    /// Maximum chunks delivered to a player per server tick (default: 16).
    pub chunks_per_tick_per_player: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tps: 20,
            view_distance: 8,
            vertical_view_distance: 2,
            chunks_per_tick_per_player: 16,
        }
    }
}
