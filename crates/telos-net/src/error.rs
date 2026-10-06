//! Error types for networking runtime, transports, and connection failures.

use telos_protocol::ProtocolError;
use telos_protocol::messages::DisconnectReason;
use thiserror::Error;

/// Networking errors occurring across transport implementations.
#[derive(Debug, Error)]
pub enum NetError {
    /// Transport connection has been closed.
    #[error("Connection closed")]
    Closed,

    /// Channel or buffer reached maximum capacity under backpressure.
    #[error("Internal channel full")]
    ChannelFull,

    /// Underlying I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Protocol encoding, decoding, or bounded limit error.
    #[error("Protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    /// Invalid packet or beacon data received.
    #[error("Invalid packet: {0}")]
    InvalidPacket(String),

    /// QUIC transport failure from Quinn.
    #[error("QUIC error: {0}")]
    Quic(String),

    /// TLS certificate or handshake error.
    #[error("TLS configuration error: {0}")]
    Tls(String),

    /// Operation timed out waiting for peer.
    #[error("Operation timed out")]
    Timeout,

    /// Remote party terminated connection with explicit disconnect reason.
    #[error("Disconnected by peer ({reason:?}): {message}")]
    Disconnected {
        /// Categorized disconnect reason.
        reason: DisconnectReason,
        /// Explanation provided by the peer.
        message: String,
    },
}

/// Specialized result type for networking operations.
pub type Result<T> = std::result::Result<T, NetError>;
