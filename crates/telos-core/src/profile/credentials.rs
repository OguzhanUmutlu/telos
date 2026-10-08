//! Local player credentials storage, keypair persistence, and profile management.

use super::cert::IdentityCert;
use super::keypair::{IdentityKeypair, KeypairError};
use super::skin::PlayerSkinData;
use super::uuid::PlayerUuid;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Error types related to player credentials operations.
#[derive(Debug, thiserror::Error)]
pub enum CredentialsError {
    /// IO error reading or writing credentials file.
    #[error("Credentials IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization failure.
    #[error("Credentials JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Cryptographic key derivation failure.
    #[error("Keypair error: {0}")]
    Keypair(#[from] KeypairError),
}

/// Persistent player profile and credentials stored on the local client.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlayerCredentials {
    /// Active player username.
    pub username: String,
    /// 32-byte Ed25519 private seed.
    seed: [u8; 32],
    /// Optional active player skin configuration.
    pub skin: Option<PlayerSkinData>,
    /// Optional certified account certificate from an Account Authority.
    pub certificate: Option<IdentityCert>,
}

impl std::fmt::Debug for PlayerCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlayerCredentials")
            .field("username", &self.username)
            .field("uuid", &self.uuid())
            .field("skin", &self.skin)
            .field("certificate", &self.certificate)
            .finish_non_exhaustive()
    }
}

impl PlayerCredentials {
    /// Creates new random credentials for a username.
    ///
    /// # Errors
    /// Returns `KeypairError::EntropyFailure` if random seed generation fails.
    pub fn new(username: &str) -> Result<Self, KeypairError> {
        let keypair = IdentityKeypair::generate()?;
        Ok(Self {
            username: username.to_string(),
            seed: keypair.seed(),
            skin: None,
            certificate: None,
        })
    }

    /// Creates credentials from an explicit 32-byte seed.
    #[must_use]
    pub fn from_seed(username: &str, seed: [u8; 32]) -> Self {
        Self {
            username: username.to_string(),
            seed,
            skin: None,
            certificate: None,
        }
    }

    /// Derives the active Ed25519 `IdentityKeypair`.
    ///
    /// # Errors
    /// Returns `KeypairError` if the seed fails validation.
    pub fn keypair(&self) -> Result<IdentityKeypair, KeypairError> {
        IdentityKeypair::from_seed(self.seed)
    }

    /// Derives the 32-byte public key.
    #[must_use]
    pub fn public_key(&self) -> [u8; 32] {
        self.keypair().map_or([0u8; 32], |kp| kp.public_key())
    }

    /// Derives the cryptographic `PlayerUuid` from the player's public key.
    #[must_use]
    pub fn uuid(&self) -> PlayerUuid {
        PlayerUuid::from_public_key(&self.public_key())
    }

    /// Returns the default credentials storage path (`~/.config/telos/credentials.json`).
    #[must_use]
    pub fn default_credentials_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("telos").join("credentials.json"))
    }

    /// Loads credentials from a JSON file path.
    ///
    /// # Errors
    /// Returns `CredentialsError` on IO or JSON parse error.
    pub fn load_from_path(path: &Path) -> Result<Self, CredentialsError> {
        let content = std::fs::read_to_string(path)?;
        let creds: Self = serde_json::from_str(&content)?;
        Ok(creds)
    }

    /// Saves credentials to a JSON file path with restricted permissions on Unix.
    ///
    /// # Errors
    /// Returns `CredentialsError` on IO or serialization failure.
    pub fn save_to_path(&self, path: &Path) -> Result<(), CredentialsError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(path, permissions);
        }

        Ok(())
    }

    /// Loads existing credentials from a custom path or generates and saves a new identity.
    ///
    /// # Errors
    /// Returns `CredentialsError` on failure to generate, read, or write credentials.
    pub fn load_or_create_at(path: &Path, username: &str) -> Result<Self, CredentialsError> {
        if path.exists()
            && let Ok(mut creds) = Self::load_from_path(path)
        {
            if !username.trim().is_empty() && creds.username != username {
                creds.username = username.to_string();
                let _ = creds.save_to_path(path);
            }
            return Ok(creds);
        }
        let new_creds = Self::new(username)?;
        let _ = new_creds.save_to_path(path);
        Ok(new_creds)
    }

    /// Loads existing default credentials or generates and saves a new identity.
    ///
    /// # Errors
    /// Returns `CredentialsError` on failure to generate, read, or write credentials.
    pub fn load_or_create(username: &str) -> Result<Self, CredentialsError> {
        if let Some(path) = Self::default_credentials_path() {
            return Self::load_or_create_at(&path, username);
        }

        Ok(Self::new(username)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_save_and_load_roundtrip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("creds.json");

        let creds = PlayerCredentials::new("Commander").expect("new creds");
        let uuid = creds.uuid();
        creds.save_to_path(&path).expect("save");

        let loaded = PlayerCredentials::load_from_path(&path).expect("load");
        assert_eq!(loaded.username, "Commander");
        assert_eq!(loaded.uuid(), uuid);
        assert_eq!(loaded.public_key(), creds.public_key());
    }
}
