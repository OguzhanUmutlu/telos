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

/// Server issues cryptographic challenge nonce to client requesting Keyed authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S2cLoginChallenge {
    /// 32-byte cryptographically random challenge nonce.
    pub challenge_nonce: [u8; 32],
    /// Server identifier string.
    pub server_id: BoundedString<64>,
}

impl S2cLoginChallenge {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.challenge_nonce);
        self.server_id.encode(buf);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        if cursor.len() < 32 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut challenge_nonce = [0u8; 32];
        challenge_nonce.copy_from_slice(&cursor[..32]);
        *cursor = &cursor[32..];

        let server_id = BoundedString::<64>::decode(cursor)?;
        Ok(Self {
            challenge_nonce,
            server_id,
        })
    }
}

/// Client responds to login challenge with Ed25519 public key and signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2sLoginProof {
    /// Client's 32-byte Ed25519 public key.
    pub public_key: [u8; 32],
    /// 64-byte Ed25519 signature over challenge nonce and username.
    pub signature: [u8; 64],
    /// Optional serialized Account Authority certificate data.
    pub certificate_data: Option<Vec<u8>>,
}

impl C2sLoginProof {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.public_key);
        buf.extend_from_slice(&self.signature);
        match &self.certificate_data {
            Some(cert) => {
                encode_varint(cert.len() as u32 + 1, buf);
                buf.extend_from_slice(cert);
            }
            None => {
                encode_varint(0, buf);
            }
        }
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        if cursor.len() < 96 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut public_key = [0u8; 32];
        public_key.copy_from_slice(&cursor[..32]);
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&cursor[32..96]);
        *cursor = &cursor[96..];

        let cert_code = decode_varint(cursor)?;
        let certificate_data = if cert_code == 0 {
            None
        } else {
            let len = (cert_code - 1) as usize;
            if len > 2048 {
                return Err(ProtocolError::InvalidValue {
                    field: "certificate_data",
                    reason: "Certificate data exceeds 2048 bytes".to_string(),
                });
            }
            if cursor.len() < len {
                return Err(ProtocolError::UnexpectedEof);
            }
            let data = cursor[..len].to_vec();
            *cursor = &cursor[len..];
            Some(data)
        };

        Ok(Self {
            public_key,
            signature,
            certificate_data,
        })
    }
}
