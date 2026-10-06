//! Error types for persistent voxel storage operations.

use std::io;
use thiserror::Error;

/// Storage errors encountered during region IO, compression, or decoding.
#[derive(Debug, Error)]
pub enum StorageError {
    /// IO failure during file read, write, or sync.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// File size is too small or truncated to contain a valid container.
    #[error("Truncated region container: length {actual} < expected {expected}")]
    Truncated {
        /// Actual file byte length.
        actual: u64,
        /// Expected minimum byte length.
        expected: u64,
    },

    /// Magic bytes did not match the expected `.vxr` or `VXP` container magic.
    #[error("Invalid magic: expected {expected:?}, got {actual:?}")]
    InvalidMagic {
        /// Expected magic bytes.
        expected: &'static [u8],
        /// Actual magic bytes found.
        actual: Vec<u8>,
    },

    /// Container version is unsupported.
    #[error("Unsupported container version {version} (maximum supported: {max_supported})")]
    UnsupportedContainerVersion {
        /// Found container version.
        version: u16,
        /// Maximum supported version.
        max_supported: u16,
    },

    /// Data schema version is unsupported.
    #[error("Unsupported data version {version} (maximum supported: {max_supported})")]
    UnsupportedDataVersion {
        /// Found data version.
        version: u32,
        /// Maximum supported version.
        max_supported: u32,
    },

    /// Header slot validation failed for both dual-slots.
    #[error(
        "Corrupt header: both slots A and B failed validation: slot A: {slot_a}, slot B: {slot_b}"
    )]
    BothHeaderSlotsCorrupt {
        /// Reason slot A failed.
        slot_a: String,
        /// Reason slot B failed.
        slot_b: String,
    },

    /// Checksum mismatch during payload or header validation.
    #[error("Checksum mismatch: expected {expected:#018x}, computed {computed:#018x}")]
    ChecksumMismatch {
        /// Expected checksum from header or entry.
        expected: u64,
        /// Computed checksum of content.
        computed: u64,
    },

    /// Payload size exceeds safety limits (decompression bomb guard).
    #[error("Payload raw length {raw_len} exceeds safety limit {limit}")]
    DecompressionBomb {
        /// Raw uncompressed length claimed.
        raw_len: u32,
        /// Maximum allowable safety limit.
        limit: u32,
    },

    /// Compression or decompression error.
    #[error("Compression error: {0}")]
    Compression(String),

    /// Format or section decoding error.
    #[error("Corrupt payload: {0}")]
    CorruptPayload(String),

    /// Sector allocation out of bounds or capacity exceeded.
    #[error("Allocation error: {0}")]
    Allocation(String),
}

/// Specialized result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;
