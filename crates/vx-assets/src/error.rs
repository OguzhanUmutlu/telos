//! Error types for asset loading and texture processing.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during asset ingestion and texture baking.
#[derive(Debug, Error)]
pub enum AssetError {
    /// An I/O error occurred while reading from the filesystem.
    #[error("I/O error accessing {path:?}: {source}")]
    Io {
        /// File path where the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to decode image file bytes.
    #[error("Failed to decode image from {path:?}: {message}")]
    Decode {
        /// File path of the image.
        path: PathBuf,
        /// Description of the decode error.
        message: String,
    },

    /// Requested texture could not be found in any resource pack in the stack.
    #[error("Texture '{0}' not found in any mounted resource pack")]
    MissingTexture(String),

    /// Image dimensions were invalid or unsupported.
    #[error("Invalid image dimensions {width}x{height} for {name}")]
    InvalidDimensions {
        /// Resource name.
        name: String,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}
