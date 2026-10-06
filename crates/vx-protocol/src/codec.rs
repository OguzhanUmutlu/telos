//! Frame-level encoding, decoding, and phase-aware message dispatch with structured packet headers.

use crate::error::{ProtocolError, Result};
use crate::messages::config::{
    C2sClientSettings, C2sConfigAck, C2sKnownRegistries, S2cConfigDone, S2cRegistryData,
};
use crate::messages::disconnect::Disconnect;
use crate::messages::hello::{C2sHello, S2cHelloReply};
use crate::messages::login::{C2sLoginStart, S2cLoginSuccess};
use crate::messages::play::{
    C2sBlockAction, C2sChatMessage, C2sCommandSuggest, C2sInteractEntity, C2sInventoryClick,
    C2sKeepAlive, C2sPlayerCommand, C2sPlayerPosition, S2cBlockActionAck, S2cBlockUpdate,
    S2cChatMessage, S2cChunkData, S2cChunkUnload, S2cCommandSuggestions, S2cDespawnEntity,
    S2cEntityMove, S2cEntityStatus, S2cInventoryBulk, S2cInventorySlot, S2cJoinGame, S2cKeepAlive,
    S2cLodNodeData, S2cLodNodeUnload, S2cSpawnEntity, S2cUniformChunk, S2cUpdateStats,
    S2cUpdateTime, S2cUpdateWeather,
};
use crate::messages::{C2sMessage, ConnectionPhase, MSG_ID_DISCONNECT, S2cMessage};
use crate::varint::{decode_varint, encode_varint};

/// Magic identifier bytes identifying Voxel Protocol frames (`"VXPR"`).
pub const PROTOCOL_MAGIC: [u8; 4] = *b"VXPR";

/// Current wire protocol revision.
pub const PROTOCOL_VERSION: u16 = 1;

/// Fixed byte length of the wire `PacketHeader` (16 bytes).
pub const PACKET_HEADER_SIZE: usize = 16;

/// Maximum allowable frame payload size (2 MiB) to prevent memory exhaustion `DoS`.
pub const MAX_FRAME_SIZE: usize = 2 * 1024 * 1024;

/// Payload byte threshold above which automatic LZ4 compression is evaluated.
pub const COMPRESSION_THRESHOLD: usize = 512;

/// Wire phase code for Hello phase.
pub const PHASE_HELLO: u8 = 0;
/// Wire phase code for Login phase.
pub const PHASE_LOGIN: u8 = 1;
/// Wire phase code for Config phase.
pub const PHASE_CONFIG: u8 = 2;
/// Wire phase code for Play phase.
pub const PHASE_PLAY: u8 = 3;
/// Wire phase code indicating a phase-agnostic packet (e.g. Disconnect).
pub const PHASE_ANY: u8 = 0xFF;

bitflags::bitflags! {
    /// Packet envelope header flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct PacketFlags: u8 {
        /// Standard uncompressed payload.
        const NONE = 0x00;
        /// Payload is compressed with LZ4 block format.
        const COMPRESSED = 0x01;
        /// High priority packet.
        const PRIORITY = 0x02;
    }
}

/// Fixed-size 16-byte envelope header prefixed to every network packet frame.
///
/// Layout (16 bytes, little-endian):
/// - `[0..4]`:   Magic bytes `b"VXPR"` (`0x52505856` in LE)
/// - `[4..6]`:   Protocol version `u16` (`PROTOCOL_VERSION`)
/// - `[6]`:      Header flags `u8` (Bit 0: compressed with LZ4, Bits 1..7: reserved)
/// - `[7]`:      Phase indicator `u8` (0: Hello, 1: Login, 2: Config, 3: Play, 255: Any/Disconnect)
/// - `[8..12]`:  Payload length `u32` (excluding this 16-byte header; max 2 MiB)
/// - `[12..16]`: CRC32 checksum `u32` (IEEE 802.3 standard checksum of the wire payload bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PacketHeader {
    /// Protocol magic bytes, must equal `PROTOCOL_MAGIC` (`b"VXPR"`).
    pub magic: [u8; 4],
    /// Wire protocol version, must equal `PROTOCOL_VERSION`.
    pub version: u16,
    /// Packet flags.
    pub flags: PacketFlags,
    /// Connection phase discriminant (0..=3, or 255 for any-phase messages).
    pub phase: u8,
    /// Byte length of the following payload (must be <= `MAX_FRAME_SIZE`).
    pub payload_len: u32,
    /// CRC32-IEEE checksum of the wire payload bytes.
    pub checksum: u32,
}

impl PacketHeader {
    /// Creates a new `PacketHeader` for `payload` with computed CRC32 checksum.
    pub fn new(flags: PacketFlags, phase: u8, payload: &[u8]) -> Result<Self> {
        if payload.len() > MAX_FRAME_SIZE {
            return Err(ProtocolError::FrameTooLarge {
                actual: payload.len(),
                max: MAX_FRAME_SIZE,
            });
        }
        let checksum = crc32fast::hash(payload);
        Ok(Self {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION,
            flags,
            phase,
            payload_len: payload.len() as u32,
            checksum,
        })
    }

    /// Serializes this 16-byte header into `buf`.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.push(self.flags.bits());
        buf.push(self.phase);
        buf.extend_from_slice(&self.payload_len.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
    }

    /// Decodes and rigorously validates a 16-byte packet header from cursor.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        if cursor.len() < PACKET_HEADER_SIZE {
            return Err(ProtocolError::UnexpectedEof);
        }
        let magic: [u8; 4] = cursor[..4].try_into().unwrap();
        if magic != PROTOCOL_MAGIC {
            return Err(ProtocolError::InvalidMagic(magic));
        }

        let version = u16::from_le_bytes(cursor[4..6].try_into().unwrap());
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::ProtocolVersionMismatch {
                expected: PROTOCOL_VERSION,
                actual: version,
            });
        }

        let flags = PacketFlags::from_bits_truncate(cursor[6]);
        let phase = cursor[7];
        if phase > 3 && phase != PHASE_ANY {
            return Err(ProtocolError::InvalidDiscriminant {
                enum_name: "ConnectionPhase",
                value: u32::from(phase),
            });
        }

        let payload_len = u32::from_le_bytes(cursor[8..12].try_into().unwrap());
        if payload_len as usize > MAX_FRAME_SIZE {
            return Err(ProtocolError::FrameTooLarge {
                actual: payload_len as usize,
                max: MAX_FRAME_SIZE,
            });
        }

        let checksum = u32::from_le_bytes(cursor[12..16].try_into().unwrap());
        *cursor = &cursor[PACKET_HEADER_SIZE..];

        Ok(Self {
            magic,
            version,
            flags,
            phase,
            payload_len,
            checksum,
        })
    }
}

fn prepare_wire_payload(raw_payload: Vec<u8>) -> (Vec<u8>, PacketFlags) {
    if raw_payload.len() >= COMPRESSION_THRESHOLD {
        let compressed = lz4_flex::block::compress_prepend_size(&raw_payload);
        if compressed.len() < raw_payload.len() {
            return (compressed, PacketFlags::COMPRESSED);
        }
    }
    (raw_payload, PacketFlags::NONE)
}

fn unpack_wire_payload(header: &PacketHeader, raw_payload: &[u8]) -> Result<Vec<u8>> {
    if header.flags.contains(PacketFlags::COMPRESSED) {
        if raw_payload.len() < 4 {
            return Err(ProtocolError::UnexpectedEof);
        }
        let uncompressed_len = u32::from_le_bytes(raw_payload[..4].try_into().unwrap()) as usize;
        if uncompressed_len > MAX_FRAME_SIZE {
            return Err(ProtocolError::FrameTooLarge {
                actual: uncompressed_len,
                max: MAX_FRAME_SIZE,
            });
        }
        lz4_flex::block::decompress_size_prepended(raw_payload)
            .map_err(|e| ProtocolError::CompressionError(e.to_string()))
    } else {
        Ok(raw_payload.to_vec())
    }
}

/// Encodes a C2S message with 16-byte `PacketHeader` envelope framing.
pub fn encode_c2s(msg: &C2sMessage, buf: &mut Vec<u8>) {
    let mut raw_payload = Vec::new();
    encode_varint(msg.message_id(), &mut raw_payload);
    msg.encode_body(&mut raw_payload);

    let phase_code = msg.phase().map_or(PHASE_ANY, ConnectionPhase::to_wire);
    let (wire_payload, flags) = prepare_wire_payload(raw_payload);
    let header = PacketHeader::new(flags, phase_code, &wire_payload)
        .expect("Wire payload exceeded MAX_FRAME_SIZE");

    header.encode(buf);
    buf.extend_from_slice(&wire_payload);
}

/// Decodes a C2S message from cursor according to the active `phase` after rigorous header,
/// phase isolation, and CRC32 checksum verification.
pub fn decode_c2s(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<C2sMessage> {
    let header = PacketHeader::decode(cursor)?;

    // Rigorous phase isolation check
    if header.phase != PHASE_ANY && header.phase != phase.to_wire() {
        return Err(ProtocolError::PhaseMismatch {
            expected: phase,
            actual: header.phase,
        });
    }

    let payload_len = header.payload_len as usize;
    if cursor.len() < payload_len {
        return Err(ProtocolError::UnexpectedEof);
    }

    let raw_payload = &cursor[..payload_len];
    *cursor = &cursor[payload_len..];

    // Rigorous CRC32 checksum verification
    let actual_checksum = crc32fast::hash(raw_payload);
    if actual_checksum != header.checksum {
        return Err(ProtocolError::ChecksumMismatch {
            expected: header.checksum,
            actual: actual_checksum,
        });
    }

    let payload_bytes = unpack_wire_payload(&header, raw_payload)?;
    let mut frame_cur = &payload_bytes[..];
    let msg_id = decode_varint(&mut frame_cur)?;

    let msg = if msg_id == MSG_ID_DISCONNECT {
        C2sMessage::Disconnect(Disconnect::decode(&mut frame_cur)?)
    } else {
        match phase {
            ConnectionPhase::Hello => match msg_id {
                0 => C2sMessage::Hello(C2sHello::decode(&mut frame_cur)?),
                id => return Err(ProtocolError::UnknownMessageId { phase: "Hello", id }),
            },
            ConnectionPhase::Login => match msg_id {
                0 => C2sMessage::LoginStart(C2sLoginStart::decode(&mut frame_cur)?),
                id => return Err(ProtocolError::UnknownMessageId { phase: "Login", id }),
            },
            ConnectionPhase::Config => match msg_id {
                0 => C2sMessage::KnownRegistries(C2sKnownRegistries::decode(&mut frame_cur)?),
                1 => C2sMessage::ClientSettings(C2sClientSettings::decode(&mut frame_cur)?),
                2 => C2sMessage::ConfigAck(C2sConfigAck::decode(&mut frame_cur)?),
                id => {
                    return Err(ProtocolError::UnknownMessageId {
                        phase: "Config",
                        id,
                    });
                }
            },
            ConnectionPhase::Play => match msg_id {
                0 => C2sMessage::KeepAlive(C2sKeepAlive::decode(&mut frame_cur)?),
                1 => C2sMessage::ChatMessage(C2sChatMessage::decode(&mut frame_cur)?),
                2 => C2sMessage::PlayerPosition(C2sPlayerPosition::decode(&mut frame_cur)?),
                3 => C2sMessage::BlockAction(C2sBlockAction::decode(&mut frame_cur)?),
                4 => C2sMessage::InventoryClick(C2sInventoryClick::decode(&mut frame_cur)?),
                5 => C2sMessage::PlayerCommand(C2sPlayerCommand::decode(&mut frame_cur)?),
                6 => C2sMessage::InteractEntity(C2sInteractEntity::decode(&mut frame_cur)?),
                7 => C2sMessage::CommandSuggest(C2sCommandSuggest::decode(&mut frame_cur)?),
                id => return Err(ProtocolError::UnknownMessageId { phase: "Play", id }),
            },
        }
    };

    if !frame_cur.is_empty() {
        return Err(ProtocolError::TrailingBytes {
            count: frame_cur.len(),
        });
    }

    Ok(msg)
}

/// Encodes an S2C message with 16-byte `PacketHeader` envelope framing.
pub fn encode_s2c(msg: &S2cMessage, buf: &mut Vec<u8>) {
    let mut raw_payload = Vec::new();
    encode_varint(msg.message_id(), &mut raw_payload);
    msg.encode_body(&mut raw_payload);

    let phase_code = msg.phase().map_or(PHASE_ANY, ConnectionPhase::to_wire);
    let (wire_payload, flags) = prepare_wire_payload(raw_payload);
    let header = PacketHeader::new(flags, phase_code, &wire_payload)
        .expect("Wire payload exceeded MAX_FRAME_SIZE");

    header.encode(buf);
    buf.extend_from_slice(&wire_payload);
}

/// Decodes an S2C message from cursor according to the active `phase` after rigorous header,
/// phase isolation, and CRC32 checksum verification.
pub fn decode_s2c(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<S2cMessage> {
    let header = PacketHeader::decode(cursor)?;

    // Rigorous phase isolation check
    if header.phase != PHASE_ANY && header.phase != phase.to_wire() {
        return Err(ProtocolError::PhaseMismatch {
            expected: phase,
            actual: header.phase,
        });
    }

    let payload_len = header.payload_len as usize;
    if cursor.len() < payload_len {
        return Err(ProtocolError::UnexpectedEof);
    }

    let raw_payload = &cursor[..payload_len];
    *cursor = &cursor[payload_len..];

    // Rigorous CRC32 checksum verification
    let actual_checksum = crc32fast::hash(raw_payload);
    if actual_checksum != header.checksum {
        return Err(ProtocolError::ChecksumMismatch {
            expected: header.checksum,
            actual: actual_checksum,
        });
    }

    let payload_bytes = unpack_wire_payload(&header, raw_payload)?;
    let mut frame_cur = &payload_bytes[..];
    let msg_id = decode_varint(&mut frame_cur)?;

    let msg = if msg_id == MSG_ID_DISCONNECT {
        S2cMessage::Disconnect(Disconnect::decode(&mut frame_cur)?)
    } else {
        match phase {
            ConnectionPhase::Hello => match msg_id {
                0 => S2cMessage::HelloReply(S2cHelloReply::decode(&mut frame_cur)?),
                id => return Err(ProtocolError::UnknownMessageId { phase: "Hello", id }),
            },
            ConnectionPhase::Login => match msg_id {
                0 => S2cMessage::LoginSuccess(S2cLoginSuccess::decode(&mut frame_cur)?),
                id => return Err(ProtocolError::UnknownMessageId { phase: "Login", id }),
            },
            ConnectionPhase::Config => match msg_id {
                0 => S2cMessage::RegistryData(S2cRegistryData::decode(&mut frame_cur)?),
                1 => S2cMessage::ConfigDone(S2cConfigDone::decode(&mut frame_cur)?),
                id => {
                    return Err(ProtocolError::UnknownMessageId {
                        phase: "Config",
                        id,
                    });
                }
            },
            ConnectionPhase::Play => match msg_id {
                0 => S2cMessage::KeepAlive(S2cKeepAlive::decode(&mut frame_cur)?),
                1 => S2cMessage::ChatMessage(S2cChatMessage::decode(&mut frame_cur)?),
                2 => S2cMessage::JoinGame(S2cJoinGame::decode(&mut frame_cur)?),
                3 => S2cMessage::ChunkData(S2cChunkData::decode(&mut frame_cur)?),
                4 => S2cMessage::UniformChunk(S2cUniformChunk::decode(&mut frame_cur)?),
                5 => S2cMessage::ChunkUnload(S2cChunkUnload::decode(&mut frame_cur)?),
                6 => S2cMessage::LodNodeData(S2cLodNodeData::decode(&mut frame_cur)?),
                7 => S2cMessage::LodNodeUnload(S2cLodNodeUnload::decode(&mut frame_cur)?),
                8 => S2cMessage::BlockUpdate(S2cBlockUpdate::decode(&mut frame_cur)?),
                9 => S2cMessage::BlockActionAck(S2cBlockActionAck::decode(&mut frame_cur)?),
                10 => S2cMessage::UpdateTime(S2cUpdateTime::decode(&mut frame_cur)?),
                11 => S2cMessage::UpdateStats(S2cUpdateStats::decode(&mut frame_cur)?),
                12 => S2cMessage::InventorySlot(S2cInventorySlot::decode(&mut frame_cur)?),
                13 => S2cMessage::InventoryBulk(S2cInventoryBulk::decode(&mut frame_cur)?),
                14 => S2cMessage::UpdateWeather(S2cUpdateWeather::decode(&mut frame_cur)?),
                15 => S2cMessage::SpawnEntity(S2cSpawnEntity::decode(&mut frame_cur)?),
                16 => S2cMessage::DespawnEntity(S2cDespawnEntity::decode(&mut frame_cur)?),
                17 => S2cMessage::EntityMove(S2cEntityMove::decode(&mut frame_cur)?),
                18 => S2cMessage::EntityStatus(S2cEntityStatus::decode(&mut frame_cur)?),
                19 => {
                    S2cMessage::CommandSuggestions(S2cCommandSuggestions::decode(&mut frame_cur)?)
                }
                id => return Err(ProtocolError::UnknownMessageId { phase: "Play", id }),
            },
        }
    };

    if !frame_cur.is_empty() {
        return Err(ProtocolError::TrailingBytes {
            count: frame_cur.len(),
        });
    }

    Ok(msg)
}

/// Checks a streaming buffer for a complete frame.
/// Returns `Ok(Some((header_bytes, payload_bytes)))` if full frame is ready,
/// `Ok(None)` if more data is needed, or `Err` if invalid.
pub fn peek_frame(buf: &[u8]) -> Result<Option<(usize, usize)>> {
    if buf.len() < PACKET_HEADER_SIZE {
        return Ok(None);
    }
    let mut cursor = buf;
    let header = PacketHeader::decode(&mut cursor)?;
    let payload_len = header.payload_len as usize;

    if buf.len() < PACKET_HEADER_SIZE + payload_len {
        Ok(None)
    } else {
        Ok(Some((PACKET_HEADER_SIZE, payload_len)))
    }
}
