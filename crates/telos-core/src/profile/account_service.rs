//! Dedicated Account Authority and authentication service for player identities.

use super::cert::IdentityCert;
use super::keypair::{IdentityKeypair, KeypairError};
use super::uuid::PlayerUuid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error types related to account service operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AccountError {
    /// Account username is already claimed by another public key.
    #[error("Username '{0}' is already registered with a different public key")]
    UsernameAlreadyClaimed(String),
    /// Account not found in registry.
    #[error("Account '{0}' not found")]
    AccountNotFound(String),
    /// Certificate expired or has invalid signature.
    #[error("Certificate validation failed: {0}")]
    InvalidCertificate(&'static str),
    /// Keypair error.
    #[error("Keypair error: {0}")]
    Keypair(#[from] KeypairError),
}

/// Persistent record of an account registered with an authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredAccountRecord {
    /// Canonical username.
    pub username: String,
    /// Ed25519 public key.
    pub public_key: [u8; 32],
    /// Persistent player UUID.
    pub player_uuid: PlayerUuid,
    /// Registration timestamp in seconds.
    pub registered_at: u64,
}

/// Standalone account service / certificate authority for Telos networks.
#[derive(Debug, Clone)]
pub struct AccountAuthority {
    /// Name or domain of the authority (e.g. `auth.telos.local`).
    pub name: String,
    /// Authority's Ed25519 root keypair.
    keypair: IdentityKeypair,
    /// Registered player database.
    accounts: HashMap<String, RegisteredAccountRecord>,
}

impl AccountAuthority {
    /// Creates a new random account authority.
    ///
    /// # Errors
    /// Returns `KeypairError` if random seed generation fails.
    pub fn new(name: &str) -> Result<Self, KeypairError> {
        let keypair = IdentityKeypair::generate()?;
        Ok(Self {
            name: name.to_string(),
            keypair,
            accounts: HashMap::new(),
        })
    }

    /// Creates an authority from an existing root keypair.
    #[must_use]
    pub fn from_keypair(name: &str, keypair: IdentityKeypair) -> Self {
        Self {
            name: name.to_string(),
            keypair,
            accounts: HashMap::new(),
        }
    }

    /// Returns the authority's 32-byte public key.
    #[must_use]
    pub fn public_key(&self) -> [u8; 32] {
        self.keypair.public_key()
    }

    /// Registers a player account or checks existing registration.
    ///
    /// # Errors
    /// Returns `AccountError::UsernameAlreadyClaimed` if the username is already registered with another key.
    pub fn register_player(
        &mut self,
        username: &str,
        public_key: [u8; 32],
        now_secs: u64,
    ) -> Result<PlayerUuid, AccountError> {
        let normalized = username.trim().to_ascii_lowercase();
        if let Some(existing) = self.accounts.get(&normalized) {
            if existing.public_key != public_key {
                return Err(AccountError::UsernameAlreadyClaimed(username.to_string()));
            }
            return Ok(existing.player_uuid);
        }

        let player_uuid = PlayerUuid::from_public_key(&public_key);
        self.accounts.insert(
            normalized,
            RegisteredAccountRecord {
                username: username.to_string(),
                public_key,
                player_uuid,
                registered_at: now_secs,
            },
        );

        Ok(player_uuid)
    }

    /// Issues an `IdentityCert` for an authorized player.
    ///
    /// # Errors
    /// Returns `AccountError::UsernameAlreadyClaimed` if the key does not match existing registration.
    pub fn issue_player_cert(
        &mut self,
        username: &str,
        public_key: &[u8; 32],
        validity_secs: u64,
        now_secs: u64,
    ) -> Result<IdentityCert, AccountError> {
        let _ = self.register_player(username, *public_key, now_secs)?;
        Ok(IdentityCert::issue(
            &self.keypair,
            username,
            public_key,
            validity_secs,
            now_secs,
        ))
    }

    /// Verifies that an `IdentityCert` was signed by this authority and is currently valid.
    #[must_use]
    pub fn verify_cert(&self, cert: &IdentityCert, now_secs: u64) -> bool {
        if cert.is_expired(now_secs) {
            return false;
        }
        cert.verify(&self.public_key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_authority_registration_and_cert() {
        let mut authority = AccountAuthority::new("AuthService").expect("new authority");
        let player_kp = IdentityKeypair::generate().expect("player kp");

        let now = 1_700_000_000;
        let uuid = authority
            .register_player("Alice", player_kp.public_key(), now)
            .expect("register alice");

        assert_eq!(uuid, PlayerUuid::from_public_key(&player_kp.public_key()));

        // Second registration with same key succeeds
        let uuid2 = authority
            .register_player("alice", player_kp.public_key(), now)
            .expect("re-register alice");
        assert_eq!(uuid, uuid2);

        // Hijack attempt with another key must fail
        let attacker_kp = IdentityKeypair::generate().expect("attacker kp");
        let hijack = authority.register_player("alice", attacker_kp.public_key(), now);
        assert!(matches!(
            hijack,
            Err(AccountError::UsernameAlreadyClaimed(_))
        ));

        // Issue certificate and verify
        let cert = authority
            .issue_player_cert("Alice", &player_kp.public_key(), 3600, now)
            .expect("issue cert");
        assert!(authority.verify_cert(&cert, now + 100));
        assert!(!authority.verify_cert(&cert, now + 4000));
    }
}
