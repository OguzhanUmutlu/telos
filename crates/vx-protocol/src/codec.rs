//! Frame-level encoding, decoding, and phase-aware message dispatch.

use crate::error::{ProtocolError, Result};
use crate::messages::config::{
    C2sClientSettings, C2sConfigAck, C2sKnownRegistries, S2cConfigDone, S2cRegistryData,
};
use crate::messages::disconnect::Disconnect;
use crate::messages::hello::{C2sHello, S2cHelloReply};
use crate::messages::login::{C2sLoginStart, S2cLoginSuccess};
use crate::messages::play::{
    C2sChatMessage, C2sKeepAlive, C2sPlayerPosition, S2cChatMessage, S2cChunkData, S2cChunkUnload,
    S2cJoinGame, S2cKeepAlive, S2cUniformChunk,
};
use crate::messages::{C2sMessage, ConnectionPhase, MSG_ID_DISCONNECT, S2cMessage};
use crate::varint::{decode_varint, encode_varint, varint_size};

/// Maximum allowable frame payload size (2 MiB) to prevent memory exhaustion `DoS`.
pub const MAX_FRAME_SIZE: usize = 2 * 1024 * 1024;

/// Encodes a C2S message with length-prefixed framing: `[VarInt length][VarInt msg_id][body]`.
pub fn encode_c2s(msg: &C2sMessage, buf: &mut Vec<u8>) {
    let mut payload = Vec::new();
    encode_varint(msg.message_id(), &mut payload);
    msg.encode_body(&mut payload);

    encode_varint(payload.len() as u32, buf);
    buf.extend_from_slice(&payload);
}

/// Decodes a C2S message from cursor according to the active `phase`.
pub fn decode_c2s(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<C2sMessage> {
    let frame_len = decode_varint(cursor)? as usize;
    if frame_len > MAX_FRAME_SIZE {
        return Err(ProtocolError::FrameTooLarge {
            actual: frame_len,
            max: MAX_FRAME_SIZE,
        });
    }
    if cursor.len() < frame_len {
        return Err(ProtocolError::UnexpectedEof);
    }

    let frame = &cursor[..frame_len];
    *cursor = &cursor[frame_len..];

    let mut frame_cur = frame;
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
                id => return Err(ProtocolError::UnknownMessageId { phase: "Play", id }),
            },
        }
    };

    if !frame_cur.is_empty() {
        return Err(ProtocolError::Malformed(format!(
            "{} unconsumed trailing bytes in C2S frame",
            frame_cur.len()
        )));
    }

    Ok(msg)
}

/// Encodes an S2C message with length-prefixed framing: `[VarInt length][VarInt msg_id][body]`.
pub fn encode_s2c(msg: &S2cMessage, buf: &mut Vec<u8>) {
    let mut payload = Vec::new();
    encode_varint(msg.message_id(), &mut payload);
    msg.encode_body(&mut payload);

    encode_varint(payload.len() as u32, buf);
    buf.extend_from_slice(&payload);
}

/// Decodes an S2C message from cursor according to the active `phase`.
pub fn decode_s2c(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<S2cMessage> {
    let frame_len = decode_varint(cursor)? as usize;
    if frame_len > MAX_FRAME_SIZE {
        return Err(ProtocolError::FrameTooLarge {
            actual: frame_len,
            max: MAX_FRAME_SIZE,
        });
    }
    if cursor.len() < frame_len {
        return Err(ProtocolError::UnexpectedEof);
    }

    let frame = &cursor[..frame_len];
    *cursor = &cursor[frame_len..];

    let mut frame_cur = frame;
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
                id => return Err(ProtocolError::UnknownMessageId { phase: "Play", id }),
            },
        }
    };

    if !frame_cur.is_empty() {
        return Err(ProtocolError::Malformed(format!(
            "{} unconsumed trailing bytes in S2C frame",
            frame_cur.len()
        )));
    }

    Ok(msg)
}

/// Checks a streaming buffer for a complete frame.
/// Returns `Ok(Some((header_bytes, payload_bytes)))` if full frame is ready,
/// `Ok(None)` if more data is needed, or `Err` if invalid.
pub fn peek_frame(buf: &[u8]) -> Result<Option<(usize, usize)>> {
    let mut cursor = buf;
    let payload_len = match decode_varint(&mut cursor) {
        Ok(len) => len as usize,
        Err(ProtocolError::UnexpectedEof) => return Ok(None),
        Err(e) => return Err(e),
    };

    if payload_len > MAX_FRAME_SIZE {
        return Err(ProtocolError::FrameTooLarge {
            actual: payload_len,
            max: MAX_FRAME_SIZE,
        });
    }

    let header_len = varint_size(payload_len as u32);
    if buf.len() < header_len + payload_len {
        Ok(None)
    } else {
        Ok(Some((header_len, payload_len)))
    }
}
