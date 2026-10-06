//! Fluent builder for configuring and launching embeddable game servers.

use std::net::SocketAddr;
use std::path::PathBuf;
use vx_worldgen::GeneratorKind;

use crate::config::ServerConfig;
use crate::error::ServerError;
use crate::multi_world::WorldConfig;
use crate::server::Server;

/// Fluent builder for constructing and starting a `Server` instance.
#[derive(Debug, Clone)]
pub struct ServerBuilder {
    config: ServerConfig,
    seed: u64,
    custom_worlds: Vec<WorldConfig>,
    bind_target: Option<String>,
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerBuilder {
    /// Creates a new `ServerBuilder` with default configuration and seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
            seed: 1337,
            custom_worlds: Vec::new(),
            bind_target: None,
        }
    }

    /// Sets the base procedural world generation seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Configures the network bind address and port (e.g. `"0.0.0.0:25565"`).
    ///
    /// Validates that the address can be parsed as a `SocketAddr`.
    pub fn bind(mut self, addr: impl AsRef<str>) -> Result<Self, ServerError> {
        let addr_str = addr.as_ref();
        let _parsed: SocketAddr = addr_str
            .parse()
            .map_err(|e| ServerError::InvalidAddress(addr_str.to_string(), e))?;
        self.config.bind_address = addr_str.to_string();
        self.bind_target = Some(addr_str.to_string());
        Ok(self)
    }

    /// Sets the server Message of the Day (MOTD) displayed in server listings.
    #[must_use]
    pub fn motd(mut self, motd: impl Into<String>) -> Self {
        self.config.motd = motd.into();
        self
    }

    /// Sets the maximum concurrent player connection limit.
    #[must_use]
    pub fn max_players(mut self, max: usize) -> Self {
        self.config.max_players = max;
        self
    }

    /// Enables or disables UDP LAN discovery beacon broadcasting.
    #[must_use]
    pub fn lan_broadcast(mut self, enabled: bool) -> Self {
        self.config.lan_broadcast = enabled;
        self
    }

    /// Sets the target simulation ticks per second (default: 20).
    #[must_use]
    pub fn tps(mut self, tps: u32) -> Self {
        self.config.tps = tps;
        self
    }

    /// Sets the horizontal view distance in chunks.
    #[must_use]
    pub fn view_distance(mut self, dist: u32) -> Self {
        self.config.view_distance = dist;
        self
    }

    /// Sets the vertical chunk radius.
    #[must_use]
    pub fn vertical_view_distance(mut self, dist: u32) -> Self {
        self.config.vertical_view_distance = dist;
        self
    }

    /// Sets the world save directory path for persistent chunk storage.
    #[must_use]
    pub fn save_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.save_directory = Some(dir.into());
        self
    }

    /// Sets the frequency in server ticks for automatic disk flushing.
    #[must_use]
    pub fn autosave_interval_ticks(mut self, ticks: u32) -> Self {
        self.config.autosave_interval_ticks = ticks;
        self
    }

    /// Registers a custom dimension/world with name, seed, and generator kind.
    #[must_use]
    pub fn world(mut self, name: impl Into<String>, seed: u64, generator: GeneratorKind) -> Self {
        self.custom_worlds.push(WorldConfig {
            name: name.into(),
            seed,
            generator,
            save_directory: None,
        });
        self
    }

    /// Adds a fully specified `WorldConfig` dimension.
    #[must_use]
    pub fn world_config(mut self, config: WorldConfig) -> Self {
        self.custom_worlds.push(config);
        self
    }

    /// Overrides the server configuration completely with a `ServerConfig`.
    #[must_use]
    pub fn config(mut self, config: ServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds and initializes the authoritative `Server` instance.
    ///
    /// If `.bind(...)` was called on this builder, the QUIC network listener is bound immediately.
    pub fn build(mut self) -> Result<Server, ServerError> {
        if !self.custom_worlds.is_empty() {
            self.config.worlds = self.custom_worlds;
        }

        let mut server = Server::new(self.seed, self.config);

        if let Some(bind_str) = self.bind_target {
            let addr: SocketAddr = bind_str
                .parse()
                .map_err(|e| ServerError::InvalidAddress(bind_str, e))?;
            server.listen_addr(addr)?;
        }

        Ok(server)
    }
}
