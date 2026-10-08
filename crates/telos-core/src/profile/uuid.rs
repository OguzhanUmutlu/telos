//! Persistent cryptographic player UUID definitions.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// 128-bit persistent player identifier conforming to RFC 4122 formatting.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PlayerUuid(pub [u8; 16]);

impl PlayerUuid {
    /// Nil UUID with all zeroes.
    pub const NIL: Self = Self([0u8; 16]);

    /// Creates a `PlayerUuid` from raw 16 bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Returns a reference to the raw 16 bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Consumes self and returns the raw 16 bytes.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Generates a deterministic offline UUID from a username.
    ///
    /// Computes `Blake3("telos:offline:<lowercase_username>")`, truncated to 16 bytes,
    /// with RFC 4122 version 3 (namespace) and variant bits applied.
    #[must_use]
    pub fn from_offline_name(username: &str) -> Self {
        let normalized = username.trim().to_ascii_lowercase();
        let payload = format!("telos:offline:{normalized}");
        let hash = blake3::hash(payload.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&hash.as_bytes()[..16]);

        // RFC 4122 version 3 (bits 4..7 of octet 6 = 0011)
        bytes[6] = (bytes[6] & 0x0f) | 0x30;
        // RFC 4122 variant (bits 6..7 of octet 8 = 10)
        bytes[8] = (bytes[8] & 0x3f) | 0x80;

        Self(bytes)
    }

    /// Generates a deterministic UUID from an Ed25519 public key.
    ///
    /// Computes `Blake3("telos:player:" || pubkey)`, truncated to 16 bytes,
    /// with RFC 4122 version 4 (cryptographic/random) and variant bits applied.
    #[must_use]
    pub fn from_public_key(pubkey: &[u8; 32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"telos:player:");
        hasher.update(pubkey);
        let hash = hasher.finalize();

        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&hash.as_bytes()[..16]);

        // RFC 4122 version 4 (bits 4..7 of octet 6 = 0100)
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        // RFC 4122 variant (bits 6..7 of octet 8 = 10)
        bytes[8] = (bytes[8] & 0x3f) | 0x80;

        Self(bytes)
    }

    /// Formats the UUID with standard 8-4-4-4-12 hyphens.
    #[must_use]
    pub fn to_string_hyphenated(&self) -> String {
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.0[0],
            self.0[1],
            self.0[2],
            self.0[3],
            self.0[4],
            self.0[5],
            self.0[6],
            self.0[7],
            self.0[8],
            self.0[9],
            self.0[10],
            self.0[11],
            self.0[12],
            self.0[13],
            self.0[14],
            self.0[15]
        )
    }

    /// Formats the UUID as a compact 32-character hexadecimal string without hyphens.
    #[must_use]
    pub fn to_string_simple(&self) -> String {
        let mut s = String::with_capacity(32);
        for byte in self.0 {
            use std::fmt::Write;
            let _ = write!(s, "{byte:02x}");
        }
        s
    }
}

impl fmt::Debug for PlayerUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlayerUuid({})", self.to_string_hyphenated())
    }
}

impl fmt::Display for PlayerUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_hyphenated())
    }
}

/// Error returned when parsing an invalid UUID string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseUuidError {
    /// Invalid string length (must be 32 without hyphens or 36 with hyphens).
    #[error("Invalid UUID string length: expected 32 or 36 characters, got {0}")]
    InvalidLength(usize),
    /// Invalid character in hexadecimal representation.
    #[error("Invalid hexadecimal character in UUID: {0}")]
    InvalidChar(char),
}

impl FromStr for PlayerUuid {
    type Err = ParseUuidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let cleaned: String = if s.len() == 36 {
            // Validate and strip hyphens at positions 8, 13, 18, 23
            let chars: Vec<char> = s.chars().collect();
            if chars[8] != '-' || chars[13] != '-' || chars[18] != '-' || chars[23] != '-' {
                return Err(ParseUuidError::InvalidLength(s.len()));
            }
            chars.into_iter().filter(|&c| c != '-').collect()
        } else if s.len() == 32 {
            s.to_string()
        } else {
            return Err(ParseUuidError::InvalidLength(s.len()));
        };

        if cleaned.len() != 32 {
            return Err(ParseUuidError::InvalidLength(cleaned.len()));
        }

        let mut bytes = [0u8; 16];
        let raw = cleaned.as_bytes();
        for i in 0..16 {
            let high = hex_val(raw[i * 2])?;
            let low = hex_val(raw[i * 2 + 1])?;
            bytes[i] = (high << 4) | low;
        }

        Ok(Self(bytes))
    }
}

fn hex_val(b: u8) -> Result<u8, ParseUuidError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(ParseUuidError::InvalidChar(other as char)),
    }
}

impl From<[u8; 16]> for PlayerUuid {
    fn from(b: [u8; 16]) -> Self {
        Self(b)
    }
}

impl From<PlayerUuid> for [u8; 16] {
    fn from(id: PlayerUuid) -> Self {
        id.0
    }
}

impl Serialize for PlayerUuid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(&self.to_string_hyphenated())
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for PlayerUuid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            Self::from_str(&s).map_err(serde::de::Error::custom)
        } else {
            let bytes = <[u8; 16]>::deserialize(deserializer)?;
            Ok(Self(bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_offline_name_determinism() {
        let u1 = PlayerUuid::from_offline_name("Steve");
        let u2 = PlayerUuid::from_offline_name("steve");
        let u3 = PlayerUuid::from_offline_name("Alex");

        assert_eq!(u1, u2, "Offline names should be case-insensitive");
        assert_ne!(u1, u3, "Distinct names must yield distinct UUIDs");

        // Verify version 3 and variant
        assert_eq!(u1.0[6] >> 4, 3, "Must have RFC 4122 version 3");
        assert_eq!(u1.0[8] >> 6, 2, "Must have RFC 4122 variant 1");
    }

    #[test]
    fn test_uuid_from_public_key() {
        let pubkey = [0x55u8; 32];
        let uuid = PlayerUuid::from_public_key(&pubkey);

        assert_eq!(uuid.0[6] >> 4, 4, "Must have RFC 4122 version 4");
        assert_eq!(uuid.0[8] >> 6, 2, "Must have RFC 4122 variant 1");
    }

    #[test]
    fn test_uuid_formatting_and_parsing_roundtrip() {
        let original = PlayerUuid::from_offline_name("Alice");
        let hyphenated = original.to_string_hyphenated();
        let simple = original.to_string_simple();

        assert_eq!(hyphenated.len(), 36);
        assert_eq!(simple.len(), 32);

        let parsed1 = PlayerUuid::from_str(&hyphenated).expect("parse hyphenated");
        let parsed2 = PlayerUuid::from_str(&simple).expect("parse simple");

        assert_eq!(original, parsed1);
        assert_eq!(original, parsed2);
    }

    #[test]
    fn test_uuid_json_serde() {
        let original = PlayerUuid::from_offline_name("Bob");
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: PlayerUuid = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, deserialized);
    }
}
