//! Cryptographic player profiles, Ed25519 identities, and Account Service.

pub mod account_service;
pub mod cert;
pub mod credentials;
pub mod keypair;
pub mod skin;
pub mod uuid;

pub use account_service::{AccountAuthority, AccountError, RegisteredAccountRecord};
pub use cert::IdentityCert;
pub use credentials::{CredentialsError, PlayerCredentials};
pub use keypair::{IdentityKeypair, KeypairError, Signature};
pub use skin::PlayerSkinData;
pub use uuid::{ParseUuidError, PlayerUuid};

use serde::{Deserialize, Serialize};

/// High-level player profile containing identity, skin, and cryptographic proofs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerProfile {
    /// Persistent 128-bit player UUID.
    pub uuid: PlayerUuid,
    /// Canonical display username.
    pub username: String,
    /// Optional Ed25519 public key for cryptographic authentication.
    pub public_key: Option<[u8; 32]>,
    /// Optional certified player skin and cosmetic metadata.
    pub skin: Option<PlayerSkinData>,
    /// Optional signed account certificate from an Account Authority.
    pub certificate: Option<IdentityCert>,
}

impl PlayerProfile {
    /// Creates an offline mode player profile with a deterministically generated UUID.
    #[must_use]
    pub fn from_offline(username: &str) -> Self {
        Self {
            uuid: PlayerUuid::from_offline_name(username),
            username: username.to_string(),
            public_key: None,
            skin: None,
            certificate: None,
        }
    }

    /// Creates a profile from local player credentials.
    #[must_use]
    pub fn from_credentials(creds: &PlayerCredentials) -> Self {
        Self {
            uuid: creds.uuid(),
            username: creds.username.clone(),
            public_key: Some(creds.public_key()),
            skin: creds.skin.clone(),
            certificate: creds.certificate.clone(),
        }
    }

    /// Returns `true` if this profile is backed by an Ed25519 cryptographic public key.
    #[must_use]
    pub const fn is_authenticated(&self) -> bool {
        self.public_key.is_some()
    }
}
