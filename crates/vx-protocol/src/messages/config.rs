//! Configuration phase messages for registry synchronization and client settings.

use crate::bounded::{BoundedString, BoundedVec};
use crate::error::{ProtocolError, Result};
use crate::varint::{decode_varint, encode_varint};

/// Client informs server of the content/registry hashes it has cached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2sKnownRegistries {
    /// Hashes (16-byte MD5/XXH3-128) of registry tables known by the client.
    pub known_hashes: BoundedVec<[u8; 16], 64>,
}

impl C2sKnownRegistries {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.known_hashes.encode_with(buf, |hash, b| {
            b.extend_from_slice(hash);
        });
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let known_hashes = BoundedVec::decode_with(cursor, |cur| {
            if cur.len() < 16 {
                return Err(ProtocolError::UnexpectedEof);
            }
            let mut h = [0u8; 16];
            h.copy_from_slice(&cur[..16]);
            *cur = &cur[16..];
            Ok(h)
        })?;
        Ok(Self { known_hashes })
    }
}

/// Server synchronizes a specific registry and its identifier entries with the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S2cRegistryData {
    /// Registry namespace and path, e.g. "voxel:block".
    pub registry_id: BoundedString<64>,
    /// Sequential list of entry names in this registry.
    pub entries: BoundedVec<BoundedString<64>, 512>,
}

impl S2cRegistryData {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.registry_id.encode(buf);
        self.entries.encode_with(buf, |entry, b| {
            entry.encode(b);
        });
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let registry_id = BoundedString::<64>::decode(cursor)?;
        let entries = BoundedVec::decode_with(cursor, BoundedString::<64>::decode)?;
        Ok(Self {
            registry_id,
            entries,
        })
    }
}

/// Client transmits local rendering and simulation preferences to the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2sClientSettings {
    /// Client-side render distance in chunks.
    pub view_distance: u16,
    /// Client-preferred simulation distance in chunks.
    pub simulation_distance: u16,
    /// Client UI locale, e.g. "`en_US`".
    pub locale: BoundedString<16>,
}

impl C2sClientSettings {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(u32::from(self.view_distance), buf);
        encode_varint(u32::from(self.simulation_distance), buf);
        self.locale.encode(buf);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let view_distance = decode_varint(cursor)? as u16;
        let simulation_distance = decode_varint(cursor)? as u16;
        let locale = BoundedString::<16>::decode(cursor)?;
        Ok(Self {
            view_distance,
            simulation_distance,
            locale,
        })
    }
}

/// Server signals that all registry entries and configuration packets have been sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct S2cConfigDone;

impl S2cConfigDone {
    /// Encodes into wire buffer (zero payload).
    pub fn encode(&self, _buf: &mut Vec<u8>) {}

    /// Decodes from wire buffer (zero payload).
    pub fn decode(_cursor: &mut &[u8]) -> Result<Self> {
        Ok(Self)
    }
}

/// Client acknowledges configuration completion, transitioning both ends to Play phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct C2sConfigAck;

impl C2sConfigAck {
    /// Encodes into wire buffer (zero payload).
    pub fn encode(&self, _buf: &mut Vec<u8>) {}

    /// Decodes from wire buffer (zero payload).
    pub fn decode(_cursor: &mut &[u8]) -> Result<Self> {
        Ok(Self)
    }
}
