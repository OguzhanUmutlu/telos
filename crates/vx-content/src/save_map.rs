//! Persistent world-local save-ID mapping and O(palette) chunk palette remapping.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::debug;
use vx_core::ident::Identifier;
use vx_voxel::state::BlockStateId;
use vx_voxel::storage::{Blocks, Packed};

use crate::error::{ContentError, Result};
use crate::registry::FrozenRegistries;

/// Status of an entry in the world-local save-ID table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SaveIdStatus {
    /// Actively mapped to a live registered block state.
    #[default]
    Live,
    /// Previously saved block whose mod/pack is currently removed.
    Missing,
}

/// An entry in the world's save-ID mapping table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaveIdEntry {
    /// World-local numeric save ID.
    pub save_id: u32,
    /// Canonical namespaced identifier (e.g. "voxel:stone").
    pub identifier: String,
    /// Entry status.
    #[serde(default)]
    pub status: SaveIdStatus,
}

/// World-local registry mapping table persisted on disk (e.g. `world/registry.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WorldRegistryMap {
    /// Sequential list of save ID entries (index == `save_id`).
    pub entries: Vec<SaveIdEntry>,
    /// Reverse mapping from identifier string to numeric save ID.
    #[serde(skip)]
    ident_to_save: HashMap<String, u32>,
}

impl WorldRegistryMap {
    /// Creates a new empty `WorldRegistryMap`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads or initializes a `WorldRegistryMap` for a given world save directory.
    pub fn open_or_create(world_dir: &Path, registries: &FrozenRegistries) -> Result<Self> {
        let registry_file = world_dir.join("registry.json");
        let mut map = if registry_file.is_file() {
            let content =
                std::fs::read_to_string(&registry_file).map_err(|e| ContentError::Io {
                    path: registry_file.clone(),
                    source: e,
                })?;
            let mut parsed: Self =
                serde_json::from_str(&content).map_err(|e| ContentError::DataParse {
                    path: registry_file.clone(),
                    reason: format!("JSON error: {e}"),
                })?;
            parsed.rebuild_index();
            parsed
        } else {
            Self::new()
        };

        // Ensure Air is strictly save ID 0
        if map.entries.is_empty() {
            map.entries.push(SaveIdEntry {
                save_id: 0,
                identifier: "voxel:air".to_string(),
                status: SaveIdStatus::Live,
            });
            map.ident_to_save.insert("voxel:air".to_string(), 0);
        }

        // Register any new active blocks from registries that aren't in the save map yet
        let mut added_new = false;

        // Standard blocks
        for ident in registries.block_ident_to_state.keys() {
            let ident_str = ident.to_string();
            if !map.ident_to_save.contains_key(&ident_str) {
                let next_id = map.entries.len() as u32;
                map.entries.push(SaveIdEntry {
                    save_id: next_id,
                    identifier: ident_str.clone(),
                    status: SaveIdStatus::Live,
                });
                map.ident_to_save.insert(ident_str, next_id);
                added_new = true;
            }
        }

        if added_new || !registry_file.is_file() {
            map.save_to_disk(&registry_file)?;
        }

        Ok(map)
    }

    /// Rebuilds internal `HashMap` index from the sequential entries list.
    pub fn rebuild_index(&mut self) {
        self.ident_to_save.clear();
        for entry in &self.entries {
            self.ident_to_save
                .insert(entry.identifier.clone(), entry.save_id);
        }
    }

    /// Saves the registry mapping table to disk as JSON.
    pub fn save_to_disk(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ContentError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        let json = serde_json::to_string_pretty(self).map_err(|e| ContentError::DataParse {
            path: path.to_path_buf(),
            reason: format!("JSON serialization error: {e}"),
        })?;

        std::fs::write(path, json).map_err(|e| ContentError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        debug!(path = %path.display(), total_entries = self.entries.len(), "Saved world registry map");
        Ok(())
    }

    /// Builds bidirectional remap tables between disk save IDs and session runtime `BlockStateId`s.
    pub fn build_remap(&self, registries: &FrozenRegistries) -> RegistryRemap {
        let missing_ident = Identifier::new("voxel", "missing").unwrap();
        let missing_state = registries
            .get_block_state(&missing_ident)
            .unwrap_or(BlockStateId::new(1)); // Fallback to stone if missing placeholder not registered

        // 1. save_to_runtime: maps disk save ID -> session runtime BlockStateId
        let mut save_to_runtime: Vec<BlockStateId> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let state = if entry.status == SaveIdStatus::Missing {
                missing_state
            } else if let Ok(ident) = Identifier::new(
                entry
                    .identifier
                    .split_once(':')
                    .map_or("voxel", |(ns, _)| ns),
                entry
                    .identifier
                    .split_once(':')
                    .map_or(entry.identifier.as_str(), |(_, p)| p),
            ) {
                registries.get_block_state(&ident).unwrap_or(missing_state)
            } else {
                missing_state
            };
            save_to_runtime.push(state);
        }

        // 2. runtime_to_save: maps session runtime BlockStateId -> disk save ID
        let total_runtime_states = registries.total_block_states();
        let mut runtime_to_save: Vec<u32> = vec![0; total_runtime_states];

        for (ident, state) in &registries.block_ident_to_state {
            let ident_str = ident.to_string();
            let save_id = self.ident_to_save.get(&ident_str).copied().unwrap_or(0);
            let idx = state.as_usize();
            if idx < runtime_to_save.len() {
                runtime_to_save[idx] = save_id;
            }
        }

        RegistryRemap {
            save_to_runtime,
            runtime_to_save,
            missing_state,
        }
    }
}

/// Bidirectional $O(\text{palette})$ remap tables between disk save IDs and session runtime IDs.
#[derive(Debug, Clone)]
pub struct RegistryRemap {
    save_to_runtime: Vec<BlockStateId>,
    runtime_to_save: Vec<u32>,
    missing_state: BlockStateId,
}

impl RegistryRemap {
    /// Remaps a disk save ID to a session runtime `BlockStateId`.
    #[inline]
    #[must_use]
    pub fn save_to_runtime(&self, save_id: u32) -> BlockStateId {
        self.save_to_runtime
            .get(save_id as usize)
            .copied()
            .unwrap_or(self.missing_state)
    }

    /// Remaps a session runtime `BlockStateId` to a disk save ID.
    #[inline]
    #[must_use]
    pub fn runtime_to_save(&self, state: BlockStateId) -> u32 {
        self.runtime_to_save
            .get(state.as_usize())
            .copied()
            .unwrap_or(0)
    }

    /// Remaps a `Blocks` container decoded from disk (with save IDs) to runtime `BlockStateId`s.
    ///
    /// Runs in $O(\text{palette})$ time because only the palette entries are transformed!
    #[must_use]
    pub fn remap_blocks_save_to_runtime(&self, blocks: Blocks) -> Blocks {
        match blocks {
            Blocks::Uniform(state) => {
                let runtime_state = self.save_to_runtime(state.as_u32());
                Blocks::Uniform(runtime_state)
            }
            Blocks::Packed(packed) => {
                let old_palette = packed.palette();
                let mut new_palette = Vec::with_capacity(old_palette.len());
                for &save_state in old_palette {
                    new_palette.push(self.save_to_runtime(save_state.as_u32()));
                }
                let log2 = packed.log2();
                let words = packed.words().to_vec().into_boxed_slice();
                Blocks::Packed(Box::new(Packed::from_raw_parts(
                    log2,
                    new_palette.into_boxed_slice(),
                    words,
                )))
            }
        }
    }

    /// Remaps an in-memory `Blocks` container (with runtime `BlockStateId`s) to disk save IDs for saving.
    ///
    /// Runs in $O(\text{palette})$ time because only the palette entries are transformed!
    #[must_use]
    pub fn remap_blocks_runtime_to_save(&self, blocks: &Blocks) -> Blocks {
        match blocks {
            Blocks::Uniform(state) => {
                let save_id = self.runtime_to_save(*state);
                Blocks::Uniform(BlockStateId::new(save_id))
            }
            Blocks::Packed(packed) => {
                let old_palette = packed.palette();
                let mut new_palette = Vec::with_capacity(old_palette.len());
                for &runtime_state in old_palette {
                    let save_id = self.runtime_to_save(runtime_state);
                    new_palette.push(BlockStateId::new(save_id));
                }
                let log2 = packed.log2();
                let words = packed.words().to_vec().into_boxed_slice();
                Blocks::Packed(Box::new(Packed::from_raw_parts(
                    log2,
                    new_palette.into_boxed_slice(),
                    words,
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::RegistryBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_save_id_mapping_roundtrip() {
        let mut builder = RegistryBuilder::new();
        builder.load_core_pack().unwrap();
        let frozen = builder.freeze().unwrap();

        let dir = tempdir().unwrap();
        let save_map = WorldRegistryMap::open_or_create(dir.path(), &frozen).unwrap();
        let remap = save_map.build_remap(&frozen);

        let stone_ident = Identifier::new("voxel", "stone").unwrap();
        let stone_runtime = frozen.get_block_state(&stone_ident).unwrap();
        let stone_save_id = remap.runtime_to_save(stone_runtime);
        assert_ne!(stone_save_id, 0);
        assert_eq!(remap.save_to_runtime(stone_save_id), stone_runtime);

        // Test O(palette) remap on uniform block
        let uniform = Blocks::Uniform(stone_runtime);
        let disk_uniform = remap.remap_blocks_runtime_to_save(&uniform);
        let restored = remap.remap_blocks_save_to_runtime(disk_uniform);
        assert_eq!(restored, uniform);
    }
}
