//! Disconnect reason codes and termination messages.

use crate::bounded::BoundedString;
use crate::error::{ProtocolError, Result};
use crate::varint::{decode_varint, encode_varint};

/// Reason codes sent when closing a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DisconnectReason {
    /// Normal voluntary client disconnect or graceful server shutdown.
    Normal = 0,
    /// Player was kicked by an operator.
    Kicked = 1,
    /// Player is banned from the server.
    Banned = 2,
    /// Protocol version mismatch.
    ProtocolMismatch = 3,
    /// Authentication or cryptographic proof failed.
    AuthFailed = 4,
    /// Server has reached player capacity.
    ServerFull = 5,
    /// Connection timed out waiting for heartbeat or expected phase packet.
    Timeout = 6,
    /// Malformed packet, unknown ID, or protocol violation.
    ProtocolViolation = 7,
    /// Server is stopping cleanly.
    ServerShutdown = 8,
    /// Another session with the same player credentials logged in.
    DuplicateLogin = 9,
    /// Required data pack or mod content could not be resolved.
    ContentMismatch = 10,
    /// Client backpressure exceeded safe threshold.
    TooSlow = 11,
}

impl DisconnectReason {
    /// Converts a raw numeric code to `DisconnectReason`.
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::Normal),
            1 => Ok(Self::Kicked),
            2 => Ok(Self::Banned),
            3 => Ok(Self::ProtocolMismatch),
            4 => Ok(Self::AuthFailed),
            5 => Ok(Self::ServerFull),
            6 => Ok(Self::Timeout),
            7 => Ok(Self::ProtocolViolation),
            8 => Ok(Self::ServerShutdown),
            9 => Ok(Self::DuplicateLogin),
            10 => Ok(Self::ContentMismatch),
            11 => Ok(Self::TooSlow),
            other => Err(ProtocolError::InvalidDiscriminant {
                enum_name: "DisconnectReason",
                value: other,
            }),
        }
    }
}

/// Message payload transmitted when terminating a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    /// Disconnect category reason.
    pub reason: DisconnectReason,
    /// Human-readable explanation or error message.
    pub message: BoundedString<256>,
}

impl Disconnect {
    /// Encodes disconnect message into wire buffer.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.reason as u32, buf);
        self.message.encode(buf);
    }

    /// Decodes disconnect message from wire buffer.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let reason_code = decode_varint(cursor)?;
        let reason = DisconnectReason::from_u32(reason_code)?;
        let message = BoundedString::<256>::decode(cursor)?;
        Ok(Self { reason, message })
    }
}
