//! Cryptographically signed player skin and cape metadata.

use super::keypair::{IdentityKeypair, Signature};
use super::uuid::PlayerUuid;
use serde::{Deserialize, Serialize};

/// Cryptographically signed skin and cosmetic profile data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerSkinData {
    /// Blake3 hash of the 64x64 or 128x128 skin PNG texture.
    pub texture_hash: [u8; 32],
    /// Whether the model uses slim arms (3px width Alex model) or classic (4px Steve).
    pub is_slim: bool,
    /// Optional Blake3 hash of the cape texture.
    pub cape_hash: Option<[u8; 32]>,
    /// Creation Unix timestamp in seconds.
    pub timestamp: u64,
    /// Cryptographic Ed25519 signature over `(uuid, texture_hash, is_slim, cape_hash, timestamp)`.
    pub signature: Option<Signature>,
}

impl PlayerSkinData {
    /// Constructs skin metadata from raw skin PNG bytes.
    #[must_use]
    pub fn from_raw_texture(
        png_bytes: &[u8],
        is_slim: bool,
        cape_bytes: Option<&[u8]>,
        timestamp: u64,
    ) -> Self {
        let texture_hash = *blake3::hash(png_bytes).as_bytes();
        let cape_hash = cape_bytes.map(|b| *blake3::hash(b).as_bytes());
        Self {
            texture_hash,
            is_slim,
            cape_hash,
            timestamp,
            signature: None,
        }
    }

    /// Computes the canonical byte sequence for cryptographic signing and verification.
    #[must_use]
    pub fn signing_payload(&self, uuid: &PlayerUuid) -> Vec<u8> {
        let mut buf = Vec::with_capacity(96);
        buf.extend_from_slice(b"telos-skin-v1:");
        buf.extend_from_slice(uuid.as_bytes());
        buf.extend_from_slice(&self.texture_hash);
        buf.push(u8::from(self.is_slim));
        if let Some(cape) = self.cape_hash {
            buf.push(1);
            buf.extend_from_slice(&cape);
        } else {
            buf.push(0);
        }
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        buf
    }

    /// Signs the skin metadata with player's Ed25519 identity keypair.
    pub fn sign(&mut self, keypair: &IdentityKeypair, uuid: &PlayerUuid) {
        let payload = self.signing_payload(uuid);
        let sig = keypair.sign(&payload);
        self.signature = Some(sig);
    }

    /// Verifies the cryptographic signature against the player's public key.
    #[must_use]
    pub fn verify(&self, public_key: &[u8; 32], uuid: &PlayerUuid) -> bool {
        let Some(sig) = self.signature else {
            return false;
        };
        let payload = self.signing_payload(uuid);
        IdentityKeypair::verify(public_key, &payload, &sig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skin_signing_and_verification() {
        let keypair = IdentityKeypair::generate().expect("keypair");
        let uuid = PlayerUuid::from_public_key(&keypair.public_key());

        let mut skin =
            PlayerSkinData::from_raw_texture(b"fake_skin_png_data", true, None, 1_700_000_000);
        assert!(!skin.verify(&keypair.public_key(), &uuid));

        skin.sign(&keypair, &uuid);
        assert!(skin.verify(&keypair.public_key(), &uuid));

        // Tampering with is_slim must invalidate signature
        skin.is_slim = false;
        assert!(!skin.verify(&keypair.public_key(), &uuid));
    }
}
