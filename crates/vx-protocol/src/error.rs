//! Error types for protocol encoding, decoding, and bounded limit violations.

use thiserror::Error;

/// Protocol errors encountered during message serialization or deserialization.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ProtocolError {
    /// Buffer reached EOF unexpectedly.
    #[error("Unexpected end of buffer while decoding")]
    UnexpectedEof,

    /// `VarInt` encoding exceeded 5 bytes or overflowed `u32`/`i32`.
    #[error("VarInt exceeds 5 bytes or overflowed")]
    VarIntOverflow,

    /// `VarLong` encoding exceeded 10 bytes or overflowed `u64`/`i64`.
    #[error("VarLong exceeds 10 bytes or overflowed")]
    VarLongOverflow,

    /// String length exceeds bounded type limit.
    #[error("String byte length {actual} exceeds limit {limit}")]
    StringTooLong {
        /// Actual string byte length.
        actual: usize,
        /// Maximum allowable limit.
        limit: usize,
    },

    /// Vector element count exceeds bounded collection limit.
    #[error("Vector length {actual} exceeds limit {limit}")]
    VecTooLong {
        /// Actual vector length.
        actual: usize,
        /// Maximum allowable limit.
        limit: usize,
    },

    /// Decoded bytes were not valid UTF-8.
    #[error("Invalid UTF-8 in string: {0}")]
    InvalidUtf8(String),

    /// Unrecognized message ID in the current connection phase.
    #[error("Unknown message ID {id} in phase {phase}")]
    UnknownMessageId {
        /// Connection phase name.
        phase: &'static str,
        /// Unknown message ID.
        id: u32,
    },

    /// Enum discriminant value out of range.
    #[error("Invalid enum discriminant {value} for {enum_name}")]
    InvalidDiscriminant {
        /// Enum name.
        enum_name: &'static str,
        /// Discriminant value found.
        value: u32,
    },

    /// Frame byte length exceeds maximum envelope limit.
    #[error("Frame length {actual} exceeds max allowable limit {max}")]
    FrameTooLarge {
        /// Actual frame byte length.
        actual: usize,
        /// Maximum allowable limit.
        max: usize,
    },

    /// Malformed message contents.
    #[error("Malformed message payload: {0}")]
    Malformed(String),
}

/// Specialized result type for protocol operations.
pub type Result<T> = std::result::Result<T, ProtocolError>;
