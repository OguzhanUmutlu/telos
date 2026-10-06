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
    /// Maximum LOD level to generate and stream (default: 2).
    pub max_lod_level: u8,
    /// Maximum LOD nodes delivered to a player per tick (default: 4).
    pub lod_nodes_per_tick_per_player: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tps: 20,
            view_distance: 8,
            vertical_view_distance: 2,
            chunks_per_tick_per_player: 16,
            max_lod_level: 2,
            lod_nodes_per_tick_per_player: 4,
        }
    }
}
