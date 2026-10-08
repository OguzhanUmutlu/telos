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
    /// Registry namespace and path, e.g. "telos:block".
    pub registry_id: BoundedString<64>,
    /// 32-byte content integrity hash (blake3).
    pub content_hash: [u8; 32],
    /// Sequential list of entry names in this registry.
    pub entries: BoundedVec<BoundedString<64>, 1024>,
}

impl S2cRegistryData {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.registry_id.encode(buf);
        buf.extend_from_slice(&self.content_hash);
        self.entries.encode_with(buf, |entry, b| {
            entry.encode(b);
        });
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let registry_id = BoundedString::<64>::decode(cursor)?;
        if registry_id.is_empty() {
            return Err(ProtocolError::InvalidValue {
                field: "registry_id",
                reason: "Registry identifier cannot be empty".to_string(),
            });
        }
        if cursor.len() < 32 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&cursor[..32]);
        *cursor = &cursor[32..];

        let entries = BoundedVec::decode_with(cursor, BoundedString::<64>::decode)?;
        Ok(Self {
            registry_id,
            content_hash,
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
        if !(1..=128).contains(&view_distance) {
            return Err(ProtocolError::InvalidValue {
                field: "view_distance",
                reason: format!("View distance {view_distance} outside allowable range 1..=128"),
            });
        }
        let simulation_distance = decode_varint(cursor)? as u16;
        if !(1..=128).contains(&simulation_distance) {
            return Err(ProtocolError::InvalidValue {
                field: "simulation_distance",
                reason: format!(
                    "Simulation distance {simulation_distance} outside allowable range 1..=128"
                ),
            });
        }
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

/// Wire representation of a custom block definition synchronized during the configuration phase.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomBlockDefWire {
    /// Namespaced block identifier (e.g. `sample:ruby_block`).
    pub identifier: BoundedString<64>,
    /// Dense runtime block state ID.
    pub state_id: u32,
    /// Fast block state properties packed into bitflags.
    pub flags: u32,
    /// Shape geometry kind (0: Cube, 1: Slab, 2: Stairs, 3: Cross, 4: Torch, 5: `FlatPlate`, 6: Empty, 7: Lever, 8: Post, 9: Chest, 10: Fluid).
    pub shape_kind: u8,
    /// Light emission level (0..=15).
    pub light_emission: u8,
    /// Mining hardness (seconds to break by hand).
    pub hardness: f32,
    /// Explosion blast resistance.
    pub blast_resistance: f32,
    /// Texture name or resource path (e.g. `ruby_block`).
    pub texture_name: BoundedString<64>,
    /// Fallback RGBA color for client procedural texturing when PNG assets are absent.
    pub base_color: [u8; 4],
}

impl CustomBlockDefWire {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.identifier.encode(buf);
        encode_varint(self.state_id, buf);
        encode_varint(self.flags, buf);
        buf.push(self.shape_kind);
        buf.push(self.light_emission);
        buf.extend_from_slice(&self.hardness.to_le_bytes());
        buf.extend_from_slice(&self.blast_resistance.to_le_bytes());
        self.texture_name.encode(buf);
        buf.extend_from_slice(&self.base_color);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let identifier = BoundedString::<64>::decode(cursor)?;
        let state_id = decode_varint(cursor)?;
        let flags = decode_varint(cursor)?;
        if cursor.is_empty() {
            return Err(ProtocolError::UnexpectedEof);
        }
        let shape_kind = cursor[0];
        *cursor = &cursor[1..];

        if cursor.is_empty() {
            return Err(ProtocolError::UnexpectedEof);
        }
        let light_emission = cursor[0];
        *cursor = &cursor[1..];

        if cursor.len() < 8 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let hardness = f32::from_le_bytes(cursor[..4].try_into().unwrap());
        let blast_resistance = f32::from_le_bytes(cursor[4..8].try_into().unwrap());
        *cursor = &cursor[8..];

        let texture_name = BoundedString::<64>::decode(cursor)?;

        if cursor.len() < 4 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let mut base_color = [0u8; 4];
        base_color.copy_from_slice(&cursor[..4]);
        *cursor = &cursor[4..];

        Ok(Self {
            identifier,
            state_id,
            flags,
            shape_kind,
            light_emission,
            hardness,
            blast_resistance,
            texture_name,
            base_color,
        })
    }
}

/// Wire representation of a custom item definition synchronized during the configuration phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomItemDefWire {
    /// Namespaced item identifier (e.g. "sample:ruby").
    pub identifier: BoundedString<64>,
    /// Dense runtime numeric item ID.
    pub item_id: u32,
    /// User-facing display name.
    pub name: BoundedString<64>,
    /// Maximum stack size (1..=64).
    pub max_stack_size: u16,
    /// Item behavior kind (0: Generic, 1: Block, 2: Tool, 3: Armor, 4: Potion, 5: `RangedWeapon`).
    pub item_type_kind: u8,
}

impl CustomItemDefWire {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.identifier.encode(buf);
        encode_varint(self.item_id, buf);
        self.name.encode(buf);
        encode_varint(u32::from(self.max_stack_size), buf);
        buf.push(self.item_type_kind);
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let identifier = BoundedString::<64>::decode(cursor)?;
        let item_id = decode_varint(cursor)?;
        let name = BoundedString::<64>::decode(cursor)?;
        let max_stack_size = decode_varint(cursor)? as u16;
        if cursor.is_empty() {
            return Err(ProtocolError::UnexpectedEof);
        }
        let item_type_kind = cursor[0];
        *cursor = &cursor[1..];

        Ok(Self {
            identifier,
            item_id,
            name,
            max_stack_size,
            item_type_kind,
        })
    }
}

/// Server synchronizes full custom block and item definitions with the client.
#[derive(Debug, Clone, PartialEq)]
pub struct S2cContentManifest {
    /// List of server-defined custom blocks.
    pub custom_blocks: BoundedVec<CustomBlockDefWire, 256>,
    /// List of server-defined custom items.
    pub custom_items: BoundedVec<CustomItemDefWire, 256>,
}

impl S2cContentManifest {
    /// Encodes into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        self.custom_blocks.encode_with(buf, |block, b| {
            block.encode(b);
        });
        self.custom_items.encode_with(buf, |item, b| {
            item.encode(b);
        });
    }

    /// Decodes from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let custom_blocks = BoundedVec::decode_with(cursor, CustomBlockDefWire::decode)?;
        let custom_items = BoundedVec::decode_with(cursor, CustomItemDefWire::decode)?;
        Ok(Self {
            custom_blocks,
            custom_items,
        })
    }
}
