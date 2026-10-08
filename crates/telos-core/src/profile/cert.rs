//! Cryptographic identity certificates issued by federated or central account authorities.

use super::keypair::{IdentityKeypair, Signature};
use super::uuid::PlayerUuid;
use serde::{Deserialize, Serialize};

/// Account Authority issued cryptographic player identity certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityCert {
    /// Certified player public key.
    pub public_key: [u8; 32],
    /// Certified player username.
    pub username: String,
    /// Certified persistent player UUID.
    pub player_uuid: PlayerUuid,
    /// Certificate issuance timestamp (Unix seconds).
    pub issued_at: u64,
    /// Certificate expiration timestamp (Unix seconds).
    pub expires_at: u64,
    /// Public key of the issuing Account Authority.
    pub issuer_public_key: [u8; 32],
    /// Ed25519 signature by the authority's private key.
    pub issuer_signature: Signature,
}

impl IdentityCert {
    /// Computes canonical binary representation for signing and verification.
    #[must_use]
    pub fn canonical_payload(
        public_key: &[u8; 32],
        username: &str,
        player_uuid: &PlayerUuid,
        issued_at: u64,
        expires_at: u64,
        issuer_public_key: &[u8; 32],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(b"telos-cert-v1:");
        buf.extend_from_slice(public_key);
        buf.extend_from_slice(username.as_bytes());
        buf.extend_from_slice(player_uuid.as_bytes());
        buf.extend_from_slice(&issued_at.to_be_bytes());
        buf.extend_from_slice(&expires_at.to_be_bytes());
        buf.extend_from_slice(issuer_public_key);
        buf
    }

    /// Issues a new signed identity certificate using an Account Authority's keypair.
    #[must_use]
    pub fn issue(
        authority_keypair: &IdentityKeypair,
        username: &str,
        player_pubkey: &[u8; 32],
        validity_secs: u64,
        now_secs: u64,
    ) -> Self {
        let player_uuid = PlayerUuid::from_public_key(player_pubkey);
        let issued_at = now_secs;
        let expires_at = now_secs.saturating_add(validity_secs);
        let issuer_public_key = authority_keypair.public_key();

        let payload = Self::canonical_payload(
            player_pubkey,
            username,
            &player_uuid,
            issued_at,
            expires_at,
            &issuer_public_key,
        );
        let issuer_signature = authority_keypair.sign(&payload);

        Self {
            public_key: *player_pubkey,
            username: username.to_string(),
            player_uuid,
            issued_at,
            expires_at,
            issuer_public_key,
            issuer_signature,
        }
    }

    /// Verifies the certificate's cryptographic signature against an expected authority public key.
    #[must_use]
    pub fn verify(&self, expected_issuer_key: &[u8; 32]) -> bool {
        if self.issuer_public_key != *expected_issuer_key {
            return false;
        }
        let payload = Self::canonical_payload(
            &self.public_key,
            &self.username,
            &self.player_uuid,
            self.issued_at,
            self.expires_at,
            &self.issuer_public_key,
        );
        IdentityKeypair::verify(expected_issuer_key, &payload, &self.issuer_signature)
    }

    /// Checks if the certificate has expired given the current Unix timestamp in seconds.
    #[must_use]
    pub fn is_expired(&self, current_time: u64) -> bool {
        current_time > self.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_cert_issue_and_verify() {
        let authority_kp = IdentityKeypair::generate().expect("authority kp");
        let player_kp = IdentityKeypair::generate().expect("player kp");

        let now = 1_700_000_000;
        let cert = IdentityCert::issue(
            &authority_kp,
            "MinerBob",
            &player_kp.public_key(),
            86400,
            now,
        );

        assert!(cert.verify(&authority_kp.public_key()));
        assert!(!cert.is_expired(now + 100));
        assert!(cert.is_expired(now + 90000));

        let imposter_kp = IdentityKeypair::generate().expect("imposter kp");
        assert!(!cert.verify(&imposter_kp.public_key()));
    }
}
