//! Persistent player advancements storage in the world directory.
//!
//! Stores per-player advancement progress as JSON documents in
//! `<world_dir>/advancements/<player_id>.json` using atomic write
//! semantics (`.tmp` + rename) to guarantee crash safety.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Result, StorageError};

/// Resolves the canonical file path for a player's advancement JSON document.
#[must_use]
pub fn player_advancements_path(world_dir: &Path, player_id: &str) -> PathBuf {
    // Sanitize player_id to avoid path traversal vulnerabilities
    let sanitized: String = player_id
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();

    let safe_id = if sanitized.is_empty() {
        "default"
    } else {
        &sanitized
    };

    world_dir
        .join("advancements")
        .join(format!("{safe_id}.json"))
}

/// Loads a player's advancement progress from `<world_dir>/advancements/<player_id>.json`.
///
/// Returns `Ok(None)` if the advancement file does not exist yet.
pub fn load_player_advancements<T: serde::de::DeserializeOwned>(
    world_dir: &Path,
    player_id: &str,
) -> Result<Option<T>> {
    let path = player_advancements_path(world_dir, player_id);
    if !path.exists() {
        return Ok(None);
    }

    let bytes = fs::read(&path).map_err(StorageError::Io)?;
    let data: T = serde_json::from_slice(&bytes).map_err(|e| {
        StorageError::CorruptPayload(format!("Failed to parse advancements JSON: {e}"))
    })?;

    Ok(Some(data))
}

/// Atomically saves a player's advancement progress to `<world_dir>/advancements/<player_id>.json`.
///
/// Writes to a temporary `.tmp` file and flushes before performing an atomic filesystem rename.
pub fn save_player_advancements<T: serde::Serialize>(
    world_dir: &Path,
    player_id: &str,
    advancements: &T,
) -> Result<()> {
    let final_path = player_advancements_path(world_dir, player_id);
    let dir = final_path.parent().expect("advancements dir has parent");

    fs::create_dir_all(dir).map_err(StorageError::Io)?;

    let tmp_path = final_path.with_extension("json.tmp");

    let json_bytes = serde_json::to_vec_pretty(advancements).map_err(|e| {
        StorageError::CorruptPayload(format!("Failed to serialize advancements JSON: {e}"))
    })?;

    {
        let mut file = fs::File::create(&tmp_path).map_err(StorageError::Io)?;
        file.write_all(&json_bytes).map_err(StorageError::Io)?;
        file.sync_data().map_err(StorageError::Io)?;
    }

    fs::rename(&tmp_path, &final_path).map_err(StorageError::Io)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct MockProgress {
        completed: BTreeMap<String, u64>,
    }

    #[test]
    fn test_save_and_load_player_advancements_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let world_path = dir.path();

        let mut progress = MockProgress {
            completed: BTreeMap::new(),
        };
        progress
            .completed
            .insert("telos:story/root".to_string(), 12345);
        progress
            .completed
            .insert("telos:story/mine_wood".to_string(), 12350);

        // Non-existent load returns Ok(None)
        let loaded = load_player_advancements::<MockProgress>(world_path, "PlayerOne")
            .expect("load non-existent");
        assert_eq!(loaded, None);

        // Save progress
        save_player_advancements(world_path, "PlayerOne", &progress).expect("save advancements");

        // Load progress and verify exact equality
        let loaded = load_player_advancements::<MockProgress>(world_path, "PlayerOne")
            .expect("load saved")
            .expect("progress exists");
        assert_eq!(loaded, progress);

        // Path traversal sanitization check
        let path = player_advancements_path(world_path, "../../etc/passwd");
        assert_eq!(path, world_path.join("advancements").join("etcpasswd.json"));
    }
}
