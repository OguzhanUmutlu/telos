//! Pack discovery and deterministic Kahn's topological load-order resolution.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::PathBuf;
use tracing::{debug, info};

use crate::error::{ContentError, Result};
use crate::manifest::{ModManifest, ModSide};

/// A discovered pack with its manifest and root location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredPack {
    /// Parsed mod manifest.
    pub manifest: ModManifest,
    /// Absolute or relative path to the pack's root directory.
    pub root: PathBuf,
    /// Whether this is the built-in core engine pack (`telos`).
    pub is_core: bool,
}

impl DiscoveredPack {
    /// Creates a new `DiscoveredPack`.
    pub fn new(manifest: ModManifest, root: PathBuf, is_core: bool) -> Self {
        Self {
            manifest,
            root,
            is_core,
        }
    }

    /// Mod ID shortcut.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.manifest.info.id
    }
}

/// Discovers packs across the provided directory paths.
pub fn discover_packs(dirs: &[PathBuf]) -> Result<Vec<DiscoveredPack>> {
    let mut discovered = Vec::new();
    let mut by_id: HashMap<String, PathBuf> = HashMap::new();

    for dir in dirs {
        if !dir.exists() || !dir.is_dir() {
            continue;
        }

        let entries = std::fs::read_dir(dir).map_err(|e| ContentError::Io {
            path: dir.clone(),
            source: e,
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| ContentError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("mod.toml");
            if manifest_path.is_file() {
                let manifest = ModManifest::from_file(&manifest_path)?;
                manifest.validate()?;

                let id = manifest.info.id.clone();
                let is_core = id == "telos";

                if let Some(existing) = by_id.insert(id.clone(), path.clone()) {
                    return Err(ContentError::DuplicateModId {
                        id,
                        first: existing,
                        second: path,
                    });
                }

                debug!(id = id.as_str(), path = %path.display(), "Discovered content pack");
                discovered.push(DiscoveredPack::new(manifest, path, is_core));
            }
        }
    }

    Ok(discovered)
}

/// Resolves the deterministic load order of discovered packs using Kahn's algorithm with a min-heap.
///
/// Ensures:
/// 1. The built-in `voxel` pack is always first.
/// 2. All hard dependencies exist and satisfy semver requirements.
/// 3. Incompatible mod versions cause an error.
/// 4. Ties in dependency ordering are broken lexicographically by mod ID for 100% determinism.
#[allow(clippy::too_many_lines)]
pub fn resolve_load_order(
    packs: Vec<DiscoveredPack>,
    target_side: ModSide,
) -> Result<Vec<DiscoveredPack>> {
    let mut pack_map: HashMap<String, DiscoveredPack> = HashMap::new();
    for pack in packs {
        // Filter by target side
        let side = pack.manifest.info.side;
        if side != ModSide::Both && target_side != ModSide::Both && side != target_side {
            debug!(id = pack.id(), "Skipping pack not matching target side");
            continue;
        }
        pack_map.insert(pack.id().to_string(), pack);
    }

    // Verify dependencies and conflicts
    for (id, pack) in &pack_map {
        for (dep_id, dep_spec) in &pack.manifest.dependencies {
            let dep_pack = pack_map.get(dep_id);
            match dep_pack {
                Some(dep) => {
                    if !dep_spec.version.matches(&dep.manifest.info.version) {
                        return Err(ContentError::MissingDependency {
                            mod_id: id.clone(),
                            dep_id: dep_id.clone(),
                            req: dep_spec.version.to_string(),
                        });
                    }
                }
                None => {
                    if !dep_spec.optional {
                        return Err(ContentError::MissingDependency {
                            mod_id: id.clone(),
                            dep_id: dep_id.clone(),
                            req: dep_spec.version.to_string(),
                        });
                    }
                }
            }
        }

        for (conflict_id, rule) in &pack.manifest.ordering.incompatible {
            if let Some(conflict) = pack_map.get(conflict_id) {
                if let Ok(req) = semver::VersionReq::parse(rule) {
                    if req.matches(&conflict.manifest.info.version) {
                        return Err(ContentError::IncompatibleMod {
                            mod_id: id.clone(),
                            conflict_id: conflict_id.clone(),
                            rule: rule.clone(),
                        });
                    }
                } else {
                    return Err(ContentError::IncompatibleMod {
                        mod_id: id.clone(),
                        conflict_id: conflict_id.clone(),
                        rule: rule.clone(),
                    });
                }
            }
        }
    }

    // Build DAG: edge u -> v means u must load before v
    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();
    let mut in_degree: HashMap<String, usize> = HashMap::new();

    for id in pack_map.keys() {
        adj.entry(id.clone()).or_default();
        in_degree.insert(id.clone(), 0);
    }

    // Core pack `voxel` must load before all other mods
    if pack_map.contains_key("telos") {
        for id in pack_map.keys() {
            if id != "telos" {
                adj.get_mut("telos").unwrap().insert(id.clone());
            }
        }
    }

    for (id, pack) in &pack_map {
        // Hard dependencies: dep loads before id
        for dep_id in pack.manifest.dependencies.keys() {
            if pack_map.contains_key(dep_id) && dep_id != id {
                adj.get_mut(dep_id).unwrap().insert(id.clone());
            }
        }

        // `after`: item loads before id
        for after_id in &pack.manifest.ordering.after {
            if pack_map.contains_key(after_id) && after_id != id {
                adj.get_mut(after_id).unwrap().insert(id.clone());
            }
        }

        // `before`: id loads before item
        for before_id in &pack.manifest.ordering.before {
            if pack_map.contains_key(before_id) && before_id != id {
                adj.get_mut(id).unwrap().insert(before_id.clone());
            }
        }
    }

    // Compute in-degrees
    for edges in adj.values() {
        for target in edges {
            *in_degree.get_mut(target).unwrap() += 1;
        }
    }

    // Kahn's algorithm with min-heap for deterministic tie-breaking
    let mut heap: BinaryHeap<Reverse<String>> = BinaryHeap::new();
    for (id, &deg) in &in_degree {
        if deg == 0 {
            heap.push(Reverse(id.clone()));
        }
    }

    let mut sorted_ids: Vec<String> = Vec::new();
    while let Some(Reverse(curr)) = heap.pop() {
        sorted_ids.push(curr.clone());

        if let Some(neighbors) = adj.get(&curr) {
            for neighbor in neighbors {
                let deg = in_degree.get_mut(neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    heap.push(Reverse(neighbor.clone()));
                }
            }
        }
    }

    if sorted_ids.len() < pack_map.len() {
        // Find remaining nodes involved in cycle
        let remaining: Vec<String> = in_degree
            .into_iter()
            .filter(|(_, deg)| *deg > 0)
            .map(|(id, _)| id)
            .collect();
        return Err(ContentError::DependencyCycle { cycle: remaining });
    }

    info!(
        load_order = ?sorted_ids,
        "Resolved deterministic pack load order"
    );

    let mut result = Vec::with_capacity(sorted_ids.len());
    for id in sorted_ids {
        result.push(pack_map.remove(&id).unwrap());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    fn make_test_pack(id: &str, deps: &[&str], after: &[&str], before: &[&str]) -> DiscoveredPack {
        let mut manifest_str = format!(
            r#"
            [mod]
            id = "{id}"
            version = "1.0.0"
            name = "{id}"
            "#
        );

        if !deps.is_empty() {
            manifest_str.push_str("\n[dependencies]\n");
            for dep in deps {
                let _ = writeln!(manifest_str, "{dep} = \"*\"");
            }
        }

        if !after.is_empty() || !before.is_empty() {
            manifest_str.push_str("\n[ordering]\n");
            if !after.is_empty() {
                let _ = writeln!(manifest_str, "after = {after:?}");
            }
            if !before.is_empty() {
                let _ = writeln!(manifest_str, "before = {before:?}");
            }
        }

        let manifest = ModManifest::from_toml_str(&manifest_str).unwrap();
        DiscoveredPack::new(
            manifest,
            PathBuf::from(format!("packs/{id}")),
            id == "telos",
        )
    }

    #[test]
    fn test_topological_sort_deterministic() {
        let packs = vec![
            make_test_pack("c_mod", &[], &[], &[]),
            make_test_pack("a_mod", &[], &[], &[]),
            make_test_pack("b_mod", &[], &[], &[]),
            make_test_pack("telos", &[], &[], &[]),
        ];

        let sorted = resolve_load_order(packs, ModSide::Both).unwrap();
        let ids: Vec<&str> = sorted.iter().map(DiscoveredPack::id).collect();
        // Core pack always first, ties broken lexicographically (a, b, c)
        assert_eq!(ids, vec!["telos", "a_mod", "b_mod", "c_mod"]);
    }

    #[test]
    fn test_dependency_ordering() {
        let packs = vec![
            make_test_pack("addon", &["lib"], &[], &[]),
            make_test_pack("telos", &[], &[], &[]),
            make_test_pack("lib", &["telos"], &[], &[]),
        ];

        let sorted = resolve_load_order(packs, ModSide::Both).unwrap();
        let ids: Vec<&str> = sorted.iter().map(DiscoveredPack::id).collect();
        assert_eq!(ids, vec!["telos", "lib", "addon"]);
    }

    #[test]
    fn test_cycle_detection() {
        let packs = vec![
            make_test_pack("mod_a", &["mod_b"], &[], &[]),
            make_test_pack("mod_b", &["mod_a"], &[], &[]),
        ];

        let err = resolve_load_order(packs, ModSide::Both).unwrap_err();
        match err {
            ContentError::DependencyCycle { cycle } => {
                assert!(cycle.contains(&"mod_a".to_string()));
                assert!(cycle.contains(&"mod_b".to_string()));
            }
            other => panic!("Expected DependencyCycle, got {other:?}"),
        }
    }
}
