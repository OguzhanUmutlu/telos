//! Registry state machine, builder, and immutable frozen content registries.

pub mod item_registry;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use telos_core::ident::Identifier;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;
use tracing::{debug, info};

use crate::core_pack::{core_blocks, core_items};
use crate::error::{ContentError, Result};
use crate::pack::DiscoveredPack;
use crate::schema::block::{BlockDef, BlockItemPolicy};
use crate::schema::item::ItemDef;
use crate::schema::tag::TagDef;

pub use item_registry::ItemRegistry;

/// State machine tracking the one-way lifecycle of registries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegistryLifecycle {
    /// Bootstrapping primitives and core codecs.
    #[default]
    Bootstrap,
    /// Discovering packs on disk and embedded core.
    Discover,
    /// Resolving manifest dependencies and deterministic load order.
    Resolve,
    /// Loading data definitions from packs into builder.
    LoadData,
    /// Resolving tags and validating identifier references.
    Tags,
    /// Assigning runtime numeric IDs and computing integrity hashes.
    Frozen,
    /// Active runtime state sharing immutable `Arc<FrozenRegistries>`.
    Running,
}

/// Builder used during the `LoadData` and `Tags` phases to assemble all content.
#[derive(Debug)]
pub struct RegistryBuilder {
    lifecycle: RegistryLifecycle,
    block_registry: BlockRegistry,
    block_defs: Vec<(Identifier, BlockDef)>,
    block_ident_to_state: HashMap<Identifier, BlockStateId>,
    item_registry: ItemRegistry,
    unresolved_tags: Vec<(Identifier, TagDef)>,
}

impl Default for RegistryBuilder {
    fn default() -> Self {
        let mut builder = Self {
            lifecycle: RegistryLifecycle::Bootstrap,
            block_registry: BlockRegistry::new(),
            block_defs: Vec::new(),
            block_ident_to_state: HashMap::new(),
            item_registry: ItemRegistry::new(),
            unresolved_tags: Vec::new(),
        };

        // Standard air at ID 0 is already registered by BlockRegistry::new() and ItemRegistry::new()
        let air_id = Identifier::new("telos", "air").expect("Valid air identifier");
        builder
            .block_ident_to_state
            .insert(air_id, BlockStateId::AIR);

        builder
    }
}

impl RegistryBuilder {
    /// Creates a new `RegistryBuilder`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the built-in vanilla `core` pack content.
    pub fn load_core_pack(&mut self) -> Result<()> {
        debug!("Loading built-in core content pack ('telos')");

        // 1. Register core blocks
        for (path, display_name, block_def) in core_blocks() {
            let ident = Identifier::new("telos", path)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;

            let flags = block_def.compute_flags();
            let shape = block_def.compute_shape();
            let state_id = self
                .block_registry
                .register_with_shape(ident.clone(), flags, shape);

            self.block_ident_to_state.insert(ident.clone(), state_id);

            // Register block item if policy is Auto
            if matches!(block_def.item, BlockItemPolicy::Auto) {
                let item_ident = ident.clone();
                let item_def = ItemDef::new_block_item(display_name, ident.to_string());
                self.item_registry.register(item_ident, item_def);
            }

            self.block_defs.push((ident, block_def));
        }

        // 2. Register core items
        for (path, item_def) in core_items() {
            let ident = Identifier::new("telos", path)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
            self.item_registry.register(ident, item_def);
        }

        Ok(())
    }

    /// Loads definitions from a discovered pack directory.
    pub fn load_pack(&mut self, pack: &DiscoveredPack) -> Result<()> {
        self.lifecycle = RegistryLifecycle::LoadData;
        let root = &pack.root;
        let mod_id = pack.id();

        debug!(mod_id, path = %root.display(), "Loading content pack data");

        // Collect all declared namespaces (primary mod id + extra declared)
        let mut namespaces = vec![mod_id.to_string()];
        namespaces.extend(pack.manifest.info.namespaces.iter().cloned());

        let data_dir = root.join("data");
        if !data_dir.exists() || !data_dir.is_dir() {
            return Ok(());
        }

        for ns in namespaces {
            let ns_dir = data_dir.join(&ns);
            if !ns_dir.exists() || !ns_dir.is_dir() {
                continue;
            }

            // 1. Load blocks: data/<ns>/block/*.ron or *.json
            let block_dir = ns_dir.join("block");
            if block_dir.is_dir() {
                let files = collect_sorted_files(&block_dir)?;
                for path in files {
                    self.load_block_file(&ns, &path)?;
                }
            }

            // 2. Load items: data/<ns>/item/*.ron or *.json
            let item_dir = ns_dir.join("item");
            if item_dir.is_dir() {
                let files = collect_sorted_files(&item_dir)?;
                for path in files {
                    self.load_item_file(&ns, &path)?;
                }
            }

            // 3. Load tags: data/<ns>/tags/**/*.json
            let tags_dir = ns_dir.join("tags");
            if tags_dir.is_dir() {
                let files = collect_sorted_files(&tags_dir)?;
                for path in files {
                    self.load_tag_file(&ns, &tags_dir, &path)?;
                }
            }
        }

        Ok(())
    }

    fn load_block_file(&mut self, ns: &str, path: &Path) -> Result<()> {
        let stem =
            path.file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| ContentError::DataParse {
                    path: path.to_path_buf(),
                    reason: "Invalid file name".into(),
                })?;

        let ident = Identifier::new(ns, stem)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;

        let content = std::fs::read_to_string(path).map_err(|e| ContentError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let block_def: BlockDef = if path.extension().is_some_and(|ext| ext == "json") {
            serde_json::from_str(&content).map_err(|e| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: format!("JSON error: {e}"),
            })?
        } else {
            ron::from_str(&content).map_err(|e| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: format!("RON error: {e}"),
            })?
        };

        let flags = block_def.compute_flags();
        let shape = block_def.compute_shape();
        let state_id = self
            .block_registry
            .register_with_shape(ident.clone(), flags, shape);

        self.block_ident_to_state.insert(ident.clone(), state_id);

        if matches!(block_def.item, BlockItemPolicy::Auto) {
            let item_ident = ident.clone();
            let item_name = format_title_case(stem);
            let item_def = ItemDef::new_block_item(item_name, ident.to_string());
            self.item_registry.register(item_ident, item_def);
        } else if let BlockItemPolicy::Explicit(ref explicit_id) = block_def.item {
            let item_ident = Identifier::new(ns, explicit_id)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
            let item_name = format_title_case(explicit_id);
            let item_def = ItemDef::new_block_item(item_name, ident.to_string());
            self.item_registry.register(item_ident, item_def);
        }

        self.block_defs.push((ident, block_def));
        Ok(())
    }

    fn load_item_file(&mut self, ns: &str, path: &Path) -> Result<()> {
        let stem =
            path.file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| ContentError::DataParse {
                    path: path.to_path_buf(),
                    reason: "Invalid file name".into(),
                })?;

        let ident = Identifier::new(ns, stem)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;

        // If block item was already auto-registered, don't double-register
        if self.item_registry.get_by_ident(&ident).is_some() {
            return Ok(());
        }

        let content = std::fs::read_to_string(path).map_err(|e| ContentError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut item_def: ItemDef = if path.extension().is_some_and(|ext| ext == "json") {
            serde_json::from_str(&content).map_err(|e| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: format!("JSON error: {e}"),
            })?
        } else {
            ron::from_str(&content).map_err(|e| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: format!("RON error: {e}"),
            })?
        };

        if item_def.name.is_empty() {
            item_def.name = format_title_case(stem);
        }

        self.item_registry.register(ident, item_def);
        Ok(())
    }

    fn load_tag_file(&mut self, ns: &str, tags_root: &Path, path: &Path) -> Result<()> {
        let rel = path
            .strip_prefix(tags_root)
            .map_err(|_| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: "Path outside tags root".into(),
            })?;

        let rel_str = rel.with_extension("").to_string_lossy().to_string();
        let tag_ident = Identifier::new(ns, rel_str)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;

        let content = std::fs::read_to_string(path).map_err(|e| ContentError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let tag_def: TagDef =
            serde_json::from_str(&content).map_err(|e| ContentError::DataParse {
                path: path.to_path_buf(),
                reason: format!("Tag JSON error: {e}"),
            })?;

        self.unresolved_tags.push((tag_ident, tag_def));
        Ok(())
    }

    /// Freezes registries, validates all references, and produces `FrozenRegistries`.
    pub fn freeze(mut self) -> Result<FrozenRegistries> {
        self.lifecycle = RegistryLifecycle::Tags;

        // Resolve tags into merged map
        let mut resolved_tags: HashMap<Identifier, HashSet<Identifier>> = HashMap::new();
        for (tag_ident, tag_def) in self.unresolved_tags {
            let set = resolved_tags.entry(tag_ident.clone()).or_default();
            if tag_def.replace {
                set.clear();
            }

            for entry in tag_def.values {
                // If it's a direct identifier (no # prefix)
                if !entry.id.starts_with('#') {
                    match Identifier::new(
                        entry.id.split_once(':').map_or("telos", |(ns, _)| ns),
                        entry
                            .id
                            .split_once(':')
                            .map_or(entry.id.as_str(), |(_, p)| p),
                    ) {
                        Ok(target_ident) => {
                            let exists_as_block =
                                self.block_ident_to_state.contains_key(&target_ident);
                            let exists_as_item =
                                self.item_registry.get_by_ident(&target_ident).is_some();

                            if !exists_as_block && !exists_as_item && entry.required {
                                return Err(ContentError::UnknownReference {
                                    referrer: tag_ident.to_string(),
                                    target: target_ident.to_string(),
                                });
                            }
                            set.insert(target_ident);
                        }
                        Err(e) => {
                            if entry.required {
                                return Err(ContentError::InvalidIdentifier(format!("{e}")));
                            }
                        }
                    }
                }
            }
        }

        self.lifecycle = RegistryLifecycle::Frozen;
        self.block_registry.freeze();
        self.item_registry.freeze();

        // Compute 32-byte blake3 content hash over canonical lists
        let mut hasher = blake3::Hasher::new();

        // 1. Hash block identifiers and properties
        for (ident, def) in &self.block_defs {
            hasher.update(ident.to_string().as_bytes());
            hasher.update(&[def.light_emission]);
            hasher.update(&def.hardness.to_le_bytes());
            hasher.update(&def.blast_resistance.to_le_bytes());
        }

        // 2. Hash item identifiers
        for (id, ident, def) in self.item_registry.iter() {
            hasher.update(&id.to_le_bytes());
            hasher.update(ident.to_string().as_bytes());
            hasher.update(&def.max_stack_size.to_le_bytes());
        }

        let content_hash = *hasher.finalize().as_bytes();

        info!(
            total_blocks = self.block_registry.total_states(),
            total_items = self.item_registry.total_items(),
            content_hash = %hasher.finalize().to_hex(),
            "Frozen content registries initialized"
        );

        Ok(FrozenRegistries {
            lifecycle: RegistryLifecycle::Running,
            block_registry: self.block_registry,
            block_defs: self.block_defs,
            block_ident_to_state: self.block_ident_to_state,
            item_registry: self.item_registry,
            tags: resolved_tags,
            content_hash,
        })
    }
}

/// Immutable, thread-safe, Arc-sharable frozen content registries.
#[derive(Debug, Clone)]
pub struct FrozenRegistries {
    lifecycle: RegistryLifecycle,
    block_registry: BlockRegistry,
    block_defs: Vec<(Identifier, BlockDef)>,
    pub(crate) block_ident_to_state: HashMap<Identifier, BlockStateId>,
    item_registry: ItemRegistry,
    tags: HashMap<Identifier, HashSet<Identifier>>,
    content_hash: [u8; 32],
}

impl FrozenRegistries {
    /// Returns the current lifecycle state (always `Running` for frozen registries).
    #[must_use]
    pub const fn lifecycle(&self) -> RegistryLifecycle {
        self.lifecycle
    }

    /// Returns the underlying `telos_voxel::BlockRegistry`.
    #[must_use]
    pub fn block_registry(&self) -> &BlockRegistry {
        &self.block_registry
    }

    /// Returns a map of registered namespaced identifiers to runtime block state IDs.
    #[must_use]
    pub fn block_states(&self) -> &HashMap<Identifier, BlockStateId> {
        &self.block_ident_to_state
    }

    /// Returns the underlying `ItemRegistry`.
    #[must_use]
    pub fn item_registry(&self) -> &ItemRegistry {
        &self.item_registry
    }

    /// Looks up a `BlockStateId` by its namespaced identifier.
    #[must_use]
    pub fn get_block_state(&self, ident: &Identifier) -> Option<BlockStateId> {
        self.block_ident_to_state.get(ident).copied()
    }

    /// Looks up a `BlockDef` by namespaced identifier.
    #[must_use]
    pub fn get_block_def(&self, ident: &Identifier) -> Option<&BlockDef> {
        self.block_defs
            .iter()
            .find(|(id, _)| id == ident)
            .map(|(_, def)| def)
    }

    /// 32-byte blake3 content hash for registry integrity verification.
    #[must_use]
    pub fn content_hash(&self) -> &[u8; 32] {
        &self.content_hash
    }

    /// Formats the blake3 content hash as a hex string.
    #[must_use]
    pub fn content_hash_hex(&self) -> String {
        blake3::Hash::from_bytes(self.content_hash)
            .to_hex()
            .to_string()
    }

    /// Checks if a given entry identifier is included in a tag.
    #[must_use]
    pub fn is_in_tag(&self, tag: &Identifier, entry: &Identifier) -> bool {
        self.tags.get(tag).is_some_and(|set| set.contains(entry))
    }

    /// Returns the total number of registered block states.
    #[must_use]
    pub fn total_block_states(&self) -> usize {
        self.block_registry.total_states()
    }

    /// Returns the total number of registered items.
    #[must_use]
    pub fn total_items(&self) -> usize {
        self.item_registry.total_items()
    }

    /// Creates a `FrozenRegistries` instance containing only the built-in core engine pack (`telos`).
    #[must_use]
    pub fn new_default() -> Self {
        let mut builder = RegistryBuilder::new();
        builder
            .load_core_pack()
            .expect("Core pack must load successfully");
        builder
            .freeze()
            .expect("Core pack must freeze successfully")
    }
}

impl Default for FrozenRegistries {
    fn default() -> Self {
        Self::new_default()
    }
}

fn collect_sorted_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_recursive(dir, &mut files)?;
    // Sort strictly by byte order of relative path for cross-platform determinism
    files.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    Ok(files)
}

fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| ContentError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| ContentError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn format_title_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_pack_loading() {
        let mut builder = RegistryBuilder::new();
        builder.load_core_pack().expect("Loads core pack");
        let frozen = builder.freeze().expect("Freezes successfully");

        let stone_id = Identifier::new("telos", "stone").unwrap();
        let state = frozen.get_block_state(&stone_id).expect("Stone registered");
        assert_eq!(state.as_u32(), 1);

        let dirt_id = Identifier::new("telos", "dirt").unwrap();
        let dirt_state = frozen.get_block_state(&dirt_id).expect("Dirt registered");
        assert_eq!(dirt_state.as_u32(), 2);

        // Check item auto-generation
        let item_id = frozen.item_registry().get_by_ident(&stone_id).unwrap();
        assert_eq!(frozen.item_registry().item_name(item_id), "Stone");
        assert_eq!(
            frozen.item_registry().placed_block(item_id),
            Some("telos:stone")
        );
    }
}
