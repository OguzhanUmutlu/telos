//! Ed25519 cryptographic keypair management, challenge signing, and verification.

use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Deref;

/// Error types related to cryptographic keypair operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeypairError {
    /// OS secure random number generator failure.
    #[error("Failed to generate secure random bytes from OS entropy")]
    EntropyFailure,
    /// Invalid seed or rejected private key material.
    #[error("Ed25519 key rejected from seed")]
    KeyRejected,
}

/// 64-byte Ed25519 cryptographic signature.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Signature(pub [u8; 64]);

impl Signature {
    /// Returns the raw 64-byte signature array.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 64] {
        self.0
    }

    /// Returns a reference to the raw 64-byte signature array.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

impl Default for Signature {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

impl Deref for Signature {
    type Target = [u8; 64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[u8; 64]> for Signature {
    fn as_ref(&self) -> &[u8; 64] {
        &self.0
    }
}

impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; 64]> for Signature {
    fn from(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }
}

impl From<Signature> for [u8; 64] {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}

impl std::fmt::Debug for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Signature({})", hex_encode(&self.0))
    }
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex_encode(&self.0))
    }
}

impl Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(&hex_encode(&self.0))
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let s = s.trim();
            if s.len() != 128 {
                return Err(serde::de::Error::custom(
                    "Signature hex must be 128 characters",
                ));
            }
            let mut bytes = [0u8; 64];
            let raw = s.as_bytes();
            for i in 0..64 {
                let h = hex_nibble(raw[i * 2])
                    .ok_or_else(|| serde::de::Error::custom("invalid hex"))?;
                let l = hex_nibble(raw[i * 2 + 1])
                    .ok_or_else(|| serde::de::Error::custom("invalid hex"))?;
                bytes[i] = (h << 4) | l;
            }
            Ok(Self(bytes))
        } else {
            let bytes = <Vec<u8>>::deserialize(deserializer)?;
            if bytes.len() != 64 {
                return Err(serde::de::Error::custom("Signature bytes must be 64 bytes"));
            }
            let mut out = [0u8; 64];
            out.copy_from_slice(&bytes);
            Ok(Self(out))
        }
    }
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Cryptographic Ed25519 identity keypair for player authentication.
#[derive(Clone, Serialize, Deserialize)]
pub struct IdentityKeypair {
    /// 32-byte private key seed.
    seed: [u8; 32],
    /// 32-byte compressed public key.
    public_key: [u8; 32],
}

impl std::fmt::Debug for IdentityKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityKeypair")
            .field("public_key", &hex_encode(&self.public_key))
            .field("seed", &"[REDACTED]")
            .finish()
    }
}

impl IdentityKeypair {
    /// Generates a new cryptographically secure random Ed25519 keypair.
    ///
    /// # Errors
    /// Returns `KeypairError::EntropyFailure` if OS random generation fails.
    pub fn generate() -> Result<Self, KeypairError> {
        let rng = SystemRandom::new();
        let mut seed = [0u8; 32];
        rng.fill(&mut seed)
            .map_err(|_| KeypairError::EntropyFailure)?;
        Self::from_seed(seed)
    }

    /// Derives an Ed25519 keypair deterministically from a 32-byte seed.
    ///
    /// # Errors
    /// Returns `KeypairError::KeyRejected` if the seed fails Ed25519 validation.
    pub fn from_seed(seed: [u8; 32]) -> Result<Self, KeypairError> {
        let pair =
            Ed25519KeyPair::from_seed_unchecked(&seed).map_err(|_| KeypairError::KeyRejected)?;
        let mut public_key = [0u8; 32];
        public_key.copy_from_slice(pair.public_key().as_ref());
        Ok(Self { seed, public_key })
    }

    /// Returns the 32-byte public key.
    #[must_use]
    pub const fn public_key(&self) -> [u8; 32] {
        self.public_key
    }

    /// Returns a reference to the 32-byte public key.
    #[must_use]
    pub const fn public_key_bytes(&self) -> &[u8; 32] {
        &self.public_key
    }

    /// Returns the 32-byte private seed.
    #[must_use]
    pub const fn seed(&self) -> [u8; 32] {
        self.seed
    }

    /// Signs an arbitrary message using this private key, returning an Ed25519 signature.
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> Signature {
        let pair = Ed25519KeyPair::from_seed_unchecked(&self.seed)
            .expect("Valid seed was validated at construction");
        let sig = pair.sign(message);
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(sig.as_ref());
        Signature(sig_bytes)
    }

    /// Verifies an Ed25519 signature against a public key and message.
    #[must_use]
    pub fn verify(public_key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
        let peer = UnparsedPublicKey::new(&ED25519, public_key);
        peer.verify(message, signature).is_ok()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation_and_sign_verify() {
        let keypair = IdentityKeypair::generate().expect("generate keypair");
        let message = b"telos-login-v1:test-challenge-nonce:PlayerOne";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(
            IdentityKeypair::verify(&pubkey, message, &signature),
            "Valid signature must verify"
        );

        let tampered_msg = b"telos-login-v1:test-challenge-nonce:Attacker";
        assert!(
            !IdentityKeypair::verify(&pubkey, tampered_msg, &signature),
            "Tampered message must be rejected"
        );

        let mut tampered_sig = signature;
        tampered_sig.0[0] ^= 0x01;
        assert!(
            !IdentityKeypair::verify(&pubkey, message, &tampered_sig),
            "Tampered signature must be rejected"
        );
    }

    #[test]
    fn test_keypair_deterministic_seed() {
        let seed = [0x42u8; 32];
        let kp1 = IdentityKeypair::from_seed(seed).expect("from_seed 1");
        let kp2 = IdentityKeypair::from_seed(seed).expect("from_seed 2");

        assert_eq!(kp1.public_key(), kp2.public_key());

        let sig1 = kp1.sign(b"hello");
        let sig2 = kp2.sign(b"hello");
        assert_eq!(sig1, sig2);
    }
}
