//! Multi-world dimension manager and world configuration.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use telos_worldgen::GeneratorKind;
use thiserror::Error;

use crate::world::ServerWorld;

/// Errors arising from world management operations.
#[derive(Debug, Error)]
pub enum WorldError {
    /// World name already registered.
    #[error("World '{0}' already exists")]
    AlreadyExists(String),
    /// World name not found.
    #[error("World '{0}' not found")]
    NotFound(String),
    /// Default world cannot be removed.
    #[error("Cannot remove default world '{0}'")]
    CannotRemoveDefault(String),
}

/// Configuration for an individual world or dimension on the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Unique identifier for the world (e.g. "overworld", "mining", "lobby").
    pub name: String,
    /// Procedural generation seed.
    pub seed: u64,
    /// Generation algorithm kind (Standard, Flat, Void).
    #[serde(default)]
    pub generator: GeneratorKind,
    /// Optional custom world save directory.
    #[serde(default)]
    pub save_directory: Option<PathBuf>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            name: "overworld".to_string(),
            seed: 1337,
            generator: GeneratorKind::Standard,
            save_directory: None,
        }
    }
}

/// Manages multiple independent concurrent worlds / dimensions on the authoritative server.
pub struct MultiWorldManager {
    worlds: HashMap<String, ServerWorld>,
    default_world_name: String,
}

impl MultiWorldManager {
    /// Creates a new `MultiWorldManager` initialized with a primary default world.
    pub fn new(default_world_name: impl Into<String>, default_world: ServerWorld) -> Self {
        let name = default_world_name.into();
        let mut worlds = HashMap::new();
        worlds.insert(name.clone(), default_world);
        Self {
            worlds,
            default_world_name: name,
        }
    }

    /// Registers an additional world with the manager.
    pub fn add_world(
        &mut self,
        name: impl Into<String>,
        world: ServerWorld,
    ) -> Result<(), WorldError> {
        let name = name.into();
        if self.worlds.contains_key(&name) {
            return Err(WorldError::AlreadyExists(name));
        }
        self.worlds.insert(name, world);
        Ok(())
    }

    /// Unregisters a world by name, returning the removed `ServerWorld`.
    pub fn remove_world(&mut self, name: &str) -> Result<ServerWorld, WorldError> {
        if name == self.default_world_name {
            return Err(WorldError::CannotRemoveDefault(name.to_string()));
        }
        self.worlds
            .remove(name)
            .ok_or_else(|| WorldError::NotFound(name.to_string()))
    }

    /// Returns a reference to the primary default world.
    #[must_use]
    pub fn default_world(&self) -> &ServerWorld {
        self.worlds
            .get(&self.default_world_name)
            .expect("Default world must always exist")
    }

    /// Returns a mutable reference to the primary default world.
    pub fn default_world_mut(&mut self) -> &mut ServerWorld {
        self.worlds
            .get_mut(&self.default_world_name)
            .expect("Default world must always exist")
    }

    /// Returns the name of the primary default world.
    #[must_use]
    pub fn default_world_name(&self) -> &str {
        &self.default_world_name
    }

    /// Looks up a world by name immutably.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ServerWorld> {
        self.worlds.get(name)
    }

    /// Looks up a world by name mutably.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ServerWorld> {
        self.worlds.get_mut(name)
    }

    /// Looks up a world by name immutably, falling back to the default world if not found.
    #[must_use]
    pub fn get_or_default(&self, name: &str) -> &ServerWorld {
        if self.worlds.contains_key(name) {
            self.worlds.get(name).unwrap()
        } else {
            self.default_world()
        }
    }

    /// Looks up a world by name mutably, falling back to the default world if not found.
    pub fn get_or_default_mut(&mut self, name: &str) -> &mut ServerWorld {
        if self.worlds.contains_key(name) {
            self.worlds.get_mut(name).unwrap()
        } else {
            self.default_world_mut()
        }
    }

    /// Checks if a world with the given name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.worlds.contains_key(name)
    }

    /// Returns sorted list of all active world names.
    #[must_use]
    pub fn world_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.worlds.keys().cloned().collect();
        names.sort();
        names
    }

    /// Total number of loaded worlds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Returns true if no worlds are loaded (never in practice).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Iterates over all active worlds.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ServerWorld)> {
        self.worlds.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Mutably iterates over all active worlds.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut ServerWorld)> {
        self.worlds.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Saves dirty chunks and flushes persistence storage across all worlds.
    pub fn save_all(&mut self) -> Result<usize, telos_storage::StorageError> {
        let mut total_saved = 0;
        for world in self.worlds.values_mut() {
            total_saved += world.save_dirty_chunks()?;
            world.flush_storage()?;
        }
        Ok(total_saved)
    }
}
