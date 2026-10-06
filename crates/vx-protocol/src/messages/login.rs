//! Login phase messages for identity and player initialization.

use crate::bounded::BoundedString;
use crate::error::{ProtocolError, Result};
use crate::varint::{decode_varint, encode_varint};

/// Authentication mode used for session login.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AuthMode {
    /// Local / offline identity mode.
    Offline = 0,
    /// Cryptographic Ed25519 public key authentication.
    Keyed = 1,
}

impl AuthMode {
    /// Converts numeric code to `AuthMode`.
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::Offline),
            1 => Ok(Self::Keyed),
            other => Err(ProtocolError::InvalidDiscriminant {
                enum_name: "AuthMode",
                value: other,
            }),
        }
    }
}

/// Client initiates authentication by providing username and requested mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2sLoginStart {
    /// Desired player username.
    pub username: BoundedString<32>,
    /// Authentication mode.
    pub mode: AuthMode,
}

impl C2sLoginStart {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.username.encode(buf);
        encode_varint(self.mode as u32, buf);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let username = BoundedString::<32>::decode(cursor)?;
        if username.is_empty() {
            return Err(ProtocolError::InvalidValue {
                field: "username",
                reason: "Username cannot be empty".to_string(),
            });
        }
        if !username
            .as_str()
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(ProtocolError::InvalidValue {
                field: "username",
                reason:
                    "Username contains illegal characters (must be ASCII alphanumeric, '_', or '-')"
                        .to_string(),
            });
        }
        let mode_code = decode_varint(cursor)?;
        let mode = AuthMode::from_u32(mode_code)?;
        Ok(Self { username, mode })
    }
}

/// Server signals successful login and assigns authenticated UUID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S2cLoginSuccess {
    /// Assigned 16-byte UUID for the player.
    pub player_uuid: [u8; 16],
    /// Canonical username confirmed by server.
    pub username: BoundedString<32>,
}

impl S2cLoginSuccess {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.player_uuid);
        self.username.encode(buf);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        if cursor.len() < 16 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut player_uuid = [0u8; 16];
        player_uuid.copy_from_slice(&cursor[..16]);
        *cursor = &cursor[16..];

        let username = BoundedString::<32>::decode(cursor)?;
        Ok(Self {
            player_uuid,
            username,
        })
    }
}
