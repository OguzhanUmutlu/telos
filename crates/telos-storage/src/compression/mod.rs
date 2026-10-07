//! Compression codecs and decompression bomb guards.

use crate::error::{Result, StorageError};
use crate::format::header::CodecId;

/// Maximum allowable uncompressed chunk payload size (4 MiB decompression bomb guard).
pub const MAX_RAW_PAYLOAD_SIZE: u32 = 4 * 1024 * 1024;

/// Compresses `data` using the specified codec and default compression level (level 3 for Zstd).
pub fn compress(codec: CodecId, data: &[u8]) -> Result<Vec<u8>> {
    compress_with_level(codec, data, 3)
}

/// Compresses `data` using the specified codec and explicit compression level.
///
/// For `CodecId::Zstd`, `level` is clamped to valid Zstandard levels `1..=22`.
pub fn compress_with_level(codec: CodecId, data: &[u8], level: i32) -> Result<Vec<u8>> {
    match codec {
        CodecId::Raw => Ok(data.to_vec()),
        CodecId::Lz4 => Ok(lz4_flex::block::compress(data)),
        CodecId::Zstd => {
            let clamped_level = level.clamp(1, 22);
            zstd::bulk::compress(data, clamped_level)
                .map_err(|e| StorageError::Compression(format!("Zstd compress failed: {e}")))
        }
        CodecId::ZstdDict => Err(StorageError::Compression(
            "Dictionary zstd compression not implemented yet".into(),
        )),
    }
}

/// Decompresses `compressed` using the specified codec into an uncompressed buffer.
pub fn decompress(codec: CodecId, compressed: &[u8], raw_len: u32) -> Result<Vec<u8>> {
    if raw_len > MAX_RAW_PAYLOAD_SIZE {
        return Err(StorageError::DecompressionBomb {
            raw_len,
            limit: MAX_RAW_PAYLOAD_SIZE,
        });
    }

    let raw_len_usize = raw_len as usize;

    match codec {
        CodecId::Raw => {
            if compressed.len() != raw_len_usize {
                return Err(StorageError::CorruptPayload(format!(
                    "Raw payload size mismatch: expected {raw_len}, got {}",
                    compressed.len()
                )));
            }
            Ok(compressed.to_vec())
        }
        CodecId::Lz4 => lz4_flex::block::decompress(compressed, raw_len_usize)
            .map_err(|e| StorageError::Compression(format!("LZ4 decompress failed: {e}"))),
        CodecId::Zstd => zstd::bulk::decompress(compressed, raw_len_usize)
            .map_err(|e| StorageError::Compression(format!("Zstd decompress failed: {e}"))),
        CodecId::ZstdDict => Err(StorageError::Compression(
            "Dictionary zstd decompression not implemented yet".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_round_trip() {
        let sample =
            b"The quick brown fox jumps over the lazy dog. Repeat repeat repeat 1234567890!";

        for codec in [CodecId::Raw, CodecId::Lz4, CodecId::Zstd] {
            let compressed = compress(codec, sample).expect("compression should succeed");
            let decompressed = decompress(codec, &compressed, sample.len() as u32)
                .expect("decompression should succeed");
            assert_eq!(decompressed, sample, "Codec {codec:?} failed round trip");
        }
    }

    #[test]
    fn test_decompression_bomb_guard() {
        let fake = [0u8; 10];
        let err = decompress(CodecId::Raw, &fake, MAX_RAW_PAYLOAD_SIZE + 1);
        assert!(matches!(err, Err(StorageError::DecompressionBomb { .. })));
    }
}
