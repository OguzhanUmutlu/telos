//! Persistent player identity registry and Trust-On-First-Use (TOFU) key binding cache.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use telos_core::profile::PlayerUuid;

/// Error returned when player authentication encounters identity conflict or hijacking attempt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdentityConflictError {
    /// Username is already bound to a different Ed25519 public key.
    #[error("Username '{username}' is already bound to a different public key")]
    KeyMismatch {
        /// Conflicting username.
        username: String,
        /// Existing public key registered for this username.
        bound_key: [u8; 32],
        /// Incoming public key attempting to claim the username.
        incoming_key: [u8; 32],
    },
    /// Unauthenticated offline player attempting to use a username already bound to a cryptographic key.
    #[error("Username '{username}' is reserved for authenticated cryptographic players")]
    OfflineLoginForRegisteredKey {
        /// Conflicting username.
        username: String,
    },
}

/// A registered player record in the server identity cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserCacheEntry {
    /// Canonical username.
    pub username: String,
    /// Persistent 128-bit player UUID.
    pub player_uuid: PlayerUuid,
    /// Bound Ed25519 public key (if authenticated via Keyed mode).
    pub public_key: Option<[u8; 32]>,
    /// First time this identity connected to the server (Unix seconds).
    pub first_seen_secs: u64,
    /// Last time this identity connected to the server (Unix seconds).
    pub last_seen_secs: u64,
}

/// In-memory and on-disk player identity cache enforcing TOFU key pinning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserCache {
    /// Map from lowercase username to `UserCacheEntry`.
    entries: HashMap<String, UserCacheEntry>,
}

impl UserCache {
    /// Creates a new empty `UserCache`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Loads the user cache from a `usercache.json` file.
    ///
    /// # Errors
    /// Returns `std::io::Error` on read or parse failure.
    pub fn load_from_path(path: &Path) -> Result<Self, std::io::Error> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = std::fs::read_to_string(path)?;
        let cache: Self = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(cache)
    }

    /// Saves the user cache to a `usercache.json` file.
    ///
    /// # Errors
    /// Returns `std::io::Error` on write or serialization failure.
    pub fn save_to_path(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Looks up a player record by case-insensitive username.
    #[must_use]
    pub fn lookup_by_username(&self, username: &str) -> Option<&UserCacheEntry> {
        let key = username.trim().to_ascii_lowercase();
        self.entries.get(&key)
    }

    /// Looks up a player record by persistent UUID.
    #[must_use]
    pub fn lookup_by_uuid(&self, uuid: &PlayerUuid) -> Option<&UserCacheEntry> {
        self.entries.values().find(|e| e.player_uuid == *uuid)
    }

    /// Validates an incoming login against the TOFU cache or registers the player on first join.
    ///
    /// # Errors
    /// Returns `IdentityConflictError::KeyMismatch` if public keys conflict.
    /// Returns `IdentityConflictError::OfflineLoginForRegisteredKey` if unauthenticated offline login uses a registered key name.
    pub fn verify_or_bind(
        &mut self,
        username: &str,
        uuid: PlayerUuid,
        public_key: Option<[u8; 32]>,
        now_secs: u64,
    ) -> Result<(), IdentityConflictError> {
        let key = username.trim().to_ascii_lowercase();

        if let Some(entry) = self.entries.get_mut(&key) {
            match (entry.public_key, public_key) {
                (Some(bound), Some(incoming)) => {
                    if bound != incoming {
                        return Err(IdentityConflictError::KeyMismatch {
                            username: username.to_string(),
                            bound_key: bound,
                            incoming_key: incoming,
                        });
                    }
                    entry.last_seen_secs = now_secs;
                    entry.username = username.to_string();
                    Ok(())
                }
                (Some(_bound), None) => Err(IdentityConflictError::OfflineLoginForRegisteredKey {
                    username: username.to_string(),
                }),
                (None, Some(incoming)) => {
                    // Upgrade previously offline name to keyed identity
                    entry.public_key = Some(incoming);
                    entry.player_uuid = uuid;
                    entry.last_seen_secs = now_secs;
                    entry.username = username.to_string();
                    Ok(())
                }
                (None, None) => {
                    entry.last_seen_secs = now_secs;
                    entry.username = username.to_string();
                    Ok(())
                }
            }
        } else {
            // First time seeing this username: bind via TOFU
            self.entries.insert(
                key,
                UserCacheEntry {
                    username: username.to_string(),
                    player_uuid: uuid,
                    public_key,
                    first_seen_secs: now_secs,
                    last_seen_secs: now_secs,
                },
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tofu_binding_and_hijacking_prevention() {
        let mut cache = UserCache::new();
        let key_alice = [0x11u8; 32];
        let uuid_alice = PlayerUuid::from_public_key(&key_alice);

        let now = 1_000_000;
        cache
            .verify_or_bind("Alice", uuid_alice, Some(key_alice), now)
            .expect("bind alice");

        // Legitimate return with same key succeeds
        cache
            .verify_or_bind("alice", uuid_alice, Some(key_alice), now + 50)
            .expect("return alice");

        // Attacker attempting to use Alice's name with Bob's key fails
        let key_bob = [0x22u8; 32];
        let uuid_bob = PlayerUuid::from_public_key(&key_bob);
        let hijack = cache.verify_or_bind("Alice", uuid_bob, Some(key_bob), now + 100);
        assert!(matches!(
            hijack,
            Err(IdentityConflictError::KeyMismatch { .. })
        ));

        // Offline client attempting to use Alice's name fails
        let offline_uuid = PlayerUuid::from_offline_name("Alice");
        let offline_attempt = cache.verify_or_bind("Alice", offline_uuid, None, now + 100);
        assert!(matches!(
            offline_attempt,
            Err(IdentityConflictError::OfflineLoginForRegisteredKey { .. })
        ));
    }

    #[test]
    fn test_usercache_save_and_reload() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("usercache.json");

        let mut cache = UserCache::new();
        let key = [0x77u8; 32];
        let uuid = PlayerUuid::from_public_key(&key);
        cache
            .verify_or_bind("Miner", uuid, Some(key), 12345)
            .expect("bind");
        cache.save_to_path(&path).expect("save");

        let loaded = UserCache::load_from_path(&path).expect("load");
        let entry = loaded.lookup_by_username("miner").expect("lookup");
        assert_eq!(entry.username, "Miner");
        assert_eq!(entry.player_uuid, uuid);
        assert_eq!(entry.public_key, Some(key));
    }
}
