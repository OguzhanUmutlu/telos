//! 48-byte uncompressed payload frame preceding stored chunks.

use crate::error::{Result, StorageError};
use crate::format::header::CodecId;
use xxhash_rust::xxh3::xxh3_64;

/// Byte length of the uncompressed payload frame.
pub const FRAME_BYTES: usize = 48;

/// Magic identifier for payload frames: `VXP\x01`.
pub const FRAME_MAGIC: [u8; 4] = *b"VXP\x01";

/// Uncompressed header preceding each chunk payload on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadFrame {
    /// Payload kind (1 = Chunks).
    pub kind: u8,
    /// Codec used to compress the stored bytes.
    pub codec: CodecId,
    /// Pre-trained dictionary ID (0 = none).
    pub dict_id: u16,
    /// Absolute chunk coordinates (cx, cy, cz).
    pub chunk_pos: (i32, i32, i32),
    /// Stored byte length of the compressed payload.
    pub stored_len: u32,
    /// Raw uncompressed byte length of the payload.
    pub raw_len: u32,
    /// Schema data version.
    pub data_version: u32,
    /// Region commit generation that wrote this frame.
    pub generation: u64,
    /// XXH3 checksum of the stored (compressed) bytes.
    pub checksum: u64,
}

impl PayloadFrame {
    /// Serializes the 48-byte frame into a byte array.
    #[must_use]
    pub fn encode(&self) -> [u8; FRAME_BYTES] {
        let mut buf = [0u8; FRAME_BYTES];
        buf[0..4].copy_from_slice(&FRAME_MAGIC);
        buf[4] = self.kind;
        buf[5] = self.codec as u8;
        buf[6..8].copy_from_slice(&self.dict_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.chunk_pos.0.to_le_bytes());
        buf[12..16].copy_from_slice(&self.chunk_pos.1.to_le_bytes());
        buf[16..20].copy_from_slice(&self.chunk_pos.2.to_le_bytes());
        buf[20..24].copy_from_slice(&self.stored_len.to_le_bytes());
        buf[24..28].copy_from_slice(&self.raw_len.to_le_bytes());
        buf[28..32].copy_from_slice(&self.data_version.to_le_bytes());
        buf[32..40].copy_from_slice(&self.generation.to_le_bytes());
        buf[40..48].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    /// Deserializes and validates a 48-byte frame.
    pub fn decode(buf: &[u8; FRAME_BYTES]) -> Result<Self> {
        if buf[0..4] != FRAME_MAGIC {
            return Err(StorageError::InvalidMagic {
                expected: &FRAME_MAGIC,
                actual: buf[0..4].to_vec(),
            });
        }

        let kind = buf[4];
        let codec = CodecId::try_from(buf[5])?;
        let dict_id = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let cx = i32::from_le_bytes(buf[8..12].try_into().unwrap());
        let cy = i32::from_le_bytes(buf[12..16].try_into().unwrap());
        let cz = i32::from_le_bytes(buf[16..20].try_into().unwrap());
        let stored_len = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let raw_len = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let data_version = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let generation = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let checksum = u64::from_le_bytes(buf[40..48].try_into().unwrap());

        Ok(Self {
            kind,
            codec,
            dict_id,
            chunk_pos: (cx, cy, cz),
            stored_len,
            raw_len,
            data_version,
            generation,
            checksum,
        })
    }

    /// Verifies that `stored_bytes` matches `stored_len` and `checksum`.
    pub fn verify_stored(&self, stored_bytes: &[u8]) -> Result<()> {
        if stored_bytes.len() as u32 != self.stored_len {
            return Err(StorageError::CorruptPayload(format!(
                "Stored length mismatch: frame has {}, got {}",
                self.stored_len,
                stored_bytes.len()
            )));
        }

        let computed = xxh3_64(stored_bytes);
        if computed != self.checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: self.checksum,
                computed,
            });
        }

        Ok(())
    }
}
