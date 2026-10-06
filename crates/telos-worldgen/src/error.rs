//! Error types for terrain generation.

use thiserror::Error;

/// Errors arising during terrain generation.
#[derive(Debug, Error)]
pub enum WorldGenError {
    /// An unknown or unmapped block was encountered during world generation.
    #[error("Unknown block identifier: {0}")]
    UnknownBlock(String),

    /// Missing block state in registry.
    #[error("Missing block state: {0}")]
    MissingBlockState(String),
}
