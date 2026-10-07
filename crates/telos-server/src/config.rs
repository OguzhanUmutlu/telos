//! Server configuration parameters and file I/O.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::multi_world::WorldConfig;

/// Errors arising from configuration loading and serialization.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File I/O error reading or writing configuration.
    #[error("Failed to read/write config file '{0}': {1}")]
    Io(PathBuf, std::io::Error),
    /// Failed to parse TOML document.
    #[error("Failed to parse TOML configuration from '{0}': {1}")]
    Parse(PathBuf, toml::de::Error),
    /// Failed to serialize TOML document.
    #[error("Failed to serialize TOML configuration: {0}")]
    Serialize(#[from] toml::ser::Error),
}

fn default_bind_address() -> String {
    "0.0.0.0:25565".to_string()
}

fn default_motd() -> String {
    "A Telos Voxel Server".to_string()
}

const fn default_max_players() -> usize {
    64
}

const fn default_true() -> bool {
    true
}

fn default_log_level() -> String {
    "info".to_string()
}

const fn default_tps() -> u32 {
    20
}

const fn default_view_distance() -> u32 {
    8
}

const fn default_simulation_distance() -> u32 {
    8
}

const fn default_vertical_view_distance() -> u32 {
    12
}

const fn default_chunks_per_tick_per_player() -> usize {
    16
}

const fn default_max_lod_level() -> u8 {
    2
}

const fn default_lod_nodes_per_tick_per_player() -> usize {
    4
}

const fn default_autosave_interval_ticks() -> u32 {
    600
}

fn default_worlds() -> Vec<WorldConfig> {
    vec![WorldConfig::default()]
}

/// Configuration parameters for server tick rate, view distance, network, and dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address and port for client connections (default: "0.0.0.0:25565").
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    /// Message of the day shown in server listings.
    #[serde(default = "default_motd")]
    pub motd: String,
    /// Maximum concurrent player connections (default: 64).
    #[serde(default = "default_max_players")]
    pub max_players: usize,
    /// Enable LAN discovery UDP broadcast beacons (default: true).
    #[serde(default = "default_true")]
    pub lan_broadcast: bool,
    /// Logging filter level (default: "info").
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Target simulation ticks per second (default: 20).
    #[serde(default = "default_tps")]
    pub tps: u32,
    /// Default horizontal view distance in chunks (radius, default: 8).
    #[serde(default = "default_view_distance")]
    pub view_distance: u32,
    /// Default horizontal simulation distance in chunks (radius, default: 8, range 2..=32).
    /// Chunks outside this radius from all players freeze entity logic and scheduled ticks.
    #[serde(default = "default_simulation_distance")]
    pub simulation_distance: u32,
    /// Vertical chunk radius (default: 12 chunks above and below = 384 blocks).
    #[serde(default = "default_vertical_view_distance")]
    pub vertical_view_distance: u32,
    /// Maximum chunks delivered to a player per server tick (default: 16).
    #[serde(default = "default_chunks_per_tick_per_player")]
    pub chunks_per_tick_per_player: usize,
    /// Maximum LOD level to generate and stream (default: 2).
    #[serde(default = "default_max_lod_level")]
    pub max_lod_level: u8,
    /// Maximum LOD nodes delivered to a player per tick (default: 4).
    #[serde(default = "default_lod_nodes_per_tick_per_player")]
    pub lod_nodes_per_tick_per_player: usize,
    /// Optional world save directory path (e.g. `saves/world`). If `None`, persistence is in-memory only.
    #[serde(default)]
    pub save_directory: Option<PathBuf>,
    /// Autosave interval in server ticks (default: 600 = 30 seconds at 20 TPS).
    #[serde(default = "default_autosave_interval_ticks")]
    pub autosave_interval_ticks: u32,
    /// Data pack directory paths to scan and load on server startup.
    #[serde(default)]
    pub data_pack_directories: Vec<PathBuf>,
    /// Configured dimensions / worlds (default: one standard world "overworld").
    #[serde(default = "default_worlds")]
    pub worlds: Vec<WorldConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: default_bind_address(),
            motd: default_motd(),
            max_players: default_max_players(),
            lan_broadcast: default_true(),
            log_level: default_log_level(),
            tps: default_tps(),
            view_distance: default_view_distance(),
            simulation_distance: default_simulation_distance(),
            vertical_view_distance: default_vertical_view_distance(),
            chunks_per_tick_per_player: default_chunks_per_tick_per_player(),
            max_lod_level: default_max_lod_level(),
            lod_nodes_per_tick_per_player: default_lod_nodes_per_tick_per_player(),
            save_directory: None,
            autosave_interval_ticks: default_autosave_interval_ticks(),
            data_pack_directories: Vec::new(),
            worlds: default_worlds(),
        }
    }
}

impl ServerConfig {
    /// Loads server configuration from a TOML file.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let p = path.as_ref();
        let content =
            std::fs::read_to_string(p).map_err(|err| ConfigError::Io(p.to_path_buf(), err))?;
        toml::from_str(&content).map_err(|err| ConfigError::Parse(p.to_path_buf(), err))
    }

    /// Serializes and saves server configuration to a TOML file.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let serialized = toml::to_string_pretty(self)?;
        std::fs::write(p, serialized).map_err(|err| ConfigError::Io(p.to_path_buf(), err))?;
        Ok(())
    }

    /// Generates a well-documented default `server.toml` template.
    #[must_use]
    pub fn default_toml_template() -> &'static str {
        r#"# ==============================================================================
# Telos Voxel Server Configuration
# ==============================================================================

# Network bind address and port (default: "0.0.0.0:25565")
bind_address = "0.0.0.0:25565"

# Server Message of the Day shown in connection listings
motd = "A Telos Voxel Server"

# Maximum concurrent player connections
max_players = 64

# Enable LAN discovery UDP broadcast beacons
lan_broadcast = true

# Logging level filter (trace, debug, info, warn, error)
log_level = "info"

# Simulation ticks per second (default: 20)
tps = 20

# Horizontal view distance in chunks radius (default: 8)
view_distance = 8

# Vertical view distance in chunks radius (default: 12)
vertical_view_distance = 12

# Maximum chunk packets sent to a single player per tick
chunks_per_tick_per_player = 16

# Maximum far-field LOD clipmap level (default: 2)
max_lod_level = 2

# Maximum far-field LOD nodes delivered to a single player per tick
lod_nodes_per_tick_per_player = 4

# Save directory for world persistence (leave empty or omit for in-memory ephemeral)
save_directory = "saves/world"

# Frequency of automatic world flushing to disk in ticks (600 = 30 seconds)
autosave_interval_ticks = 600

# Additional directories to scan for content data packs
data_pack_directories = []

# Configured dimensions / worlds
[[worlds]]
name = "overworld"
seed = 1337
generator = "Standard"

[[worlds]]
name = "flat"
seed = 42
generator = "Flat"

[[worlds]]
name = "void"
seed = 99
generator = "Void"
"#
    }
}
