//! Audio error types.

use thiserror::Error;

/// Errors produced during audio initialization, decoding, or playback.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Audio output device failed to open or is unavailable.
    #[error("Audio device error: {0}")]
    Device(String),

    /// Audio file decoding failed.
    #[error("Failed to decode audio: {0}")]
    Decode(String),

    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
