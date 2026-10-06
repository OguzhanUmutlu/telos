//! Hello phase messages for protocol version and feature negotiation.

use crate::bounded::BoundedString;
use crate::error::{ProtocolError, Result};
use crate::varint::{decode_varint, decode_varlong, encode_varint, encode_varlong};

/// Client initiates connection by announcing supported protocol version and build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2sHello {
    /// Integer protocol version requested (must match `PROTOCOL_VERSION`).
    pub protocol: u32,
    /// Build identifier string.
    pub build: BoundedString<32>,
    /// Bitflags of optional client capabilities/features.
    pub features: u64,
}

impl C2sHello {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.protocol, buf);
        self.build.encode(buf);
        encode_varlong(self.features, buf);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let protocol = decode_varint(cursor)?;
        let build = BoundedString::<32>::decode(cursor)?;
        if build.is_empty() {
            return Err(ProtocolError::InvalidValue {
                field: "build",
                reason: "Build identifier cannot be empty".to_string(),
            });
        }
        let features = decode_varlong(cursor)?;
        Ok(Self {
            protocol,
            build,
            features,
        })
    }
}

/// Server accepts protocol version and confirms agreed features and server identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S2cHelloReply {
    /// Agreed protocol version.
    pub protocol: u32,
    /// Bitmask of agreed features (bitwise intersection of client and server features).
    pub features: u64,
    /// Unique 16-byte server instance identifier.
    pub server_id: [u8; 16],
}

impl S2cHelloReply {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.protocol, buf);
        encode_varlong(self.features, buf);
        buf.extend_from_slice(&self.server_id);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let protocol = decode_varint(cursor)?;
        let features = decode_varlong(cursor)?;
        if cursor.len() < 16 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut server_id = [0u8; 16];
        server_id.copy_from_slice(&cursor[..16]);
        *cursor = &cursor[16..];
        Ok(Self {
            protocol,
            features,
            server_id,
        })
    }
}
