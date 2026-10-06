//! Tagged section encoder/decoder and payload sections (BLOCKS, STATUS).

use crate::error::{Result, StorageError};
use telos_voxel::{
    state::BlockStateId,
    storage::{Blocks, Packed},
};

/// Tag identifying the `BLOCKS` section.
pub const TAG_BLOCKS: u8 = 0x01;
/// Tag identifying the `SKY_LIGHT` section.
pub const TAG_SKY_LIGHT: u8 = 0x02;
/// Tag identifying the `BLOCK_LIGHT` section.
pub const TAG_BLOCK_LIGHT: u8 = 0x03;
/// Tag identifying the `BIOMES` section.
pub const TAG_BIOMES: u8 = 0x04;
/// Tag identifying the `STATUS` section.
pub const TAG_STATUS: u8 = 0x08;

/// Section flag bit indicating a critical section that unknown readers must reject.
pub const SFLAG_CRITICAL: u8 = 1 << 7;

/// Encodes an unsigned 32-bit integer as LEB128 varint.
pub fn encode_varint_u32(mut val: u32, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push(((val & 0x7f) as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

/// Decodes an unsigned 32-bit integer from LEB128 varint.
pub fn decode_varint_u32(cursor: &mut &[u8]) -> Result<u32> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;

    for _ in 0..5 {
        if cursor.is_empty() {
            return Err(StorageError::CorruptPayload(
                "Unexpected EOF in varint u32".into(),
            ));
        }
        let byte = cursor[0];
        *cursor = &cursor[1..];

        result |= u32::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }

    Err(StorageError::CorruptPayload(
        "Overflow in varint u32".into(),
    ))
}

/// Encodes an unsigned 64-bit integer as LEB128 varint.
pub fn encode_varint_u64(mut val: u64, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push(((val & 0x7f) as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

/// Decodes an unsigned 64-bit integer from LEB128 varint.
pub fn decode_varint_u64(cursor: &mut &[u8]) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for _ in 0..10 {
        if cursor.is_empty() {
            return Err(StorageError::CorruptPayload(
                "Unexpected EOF in varint u64".into(),
            ));
        }
        let byte = cursor[0];
        *cursor = &cursor[1..];

        result |= u64::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }

    Err(StorageError::CorruptPayload(
        "Overflow in varint u64".into(),
    ))
}

/// Chunk status metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkStatus {
    /// Generation stage: 0=Empty, 1=Noise, 2=Surface, 3=Carvers, 4=Features, 5=Light, 6=Full.
    pub gen_stage: u8,
    /// Inhabited ticks by players.
    pub inhabited_ticks: u64,
    /// Bitflags for light validity, column estimation, etc.
    pub flags: u16,
}

impl ChunkStatus {
    /// Encodes status into binary payload.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(self.gen_stage);
        encode_varint_u64(self.inhabited_ticks, buf);
        buf.extend_from_slice(&self.flags.to_le_bytes());
    }

    /// Decodes status from binary payload.
    pub fn decode(mut cursor: &[u8]) -> Result<Self> {
        if cursor.is_empty() {
            return Err(StorageError::CorruptPayload("Empty status payload".into()));
        }
        let gen_stage = cursor[0];
        cursor = &cursor[1..];
        let inhabited_ticks = decode_varint_u64(&mut cursor)?;
        if cursor.len() < 2 {
            return Err(StorageError::CorruptPayload(
                "Truncated status flags".into(),
            ));
        }
        let flags = u16::from_le_bytes(cursor[0..2].try_into().unwrap());
        Ok(Self {
            gen_stage,
            inhabited_ticks,
            flags,
        })
    }
}

/// An unknown or unparsed section preserved for forward compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSection {
    /// Section tag identifier.
    pub tag: u8,
    /// Section flags (bit 7 = CRITICAL).
    pub sflags: u8,
    /// Raw unparsed byte payload.
    pub data: Vec<u8>,
}

/// Decoded chunk payload data containing blocks, status, and raw unknown sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPayload {
    /// Voxel block storage.
    pub blocks: Blocks,
    /// Generation status and metadata.
    pub status: ChunkStatus,
    /// Forward-compatible preserved sections.
    pub unknown_sections: Vec<RawSection>,
}

impl ChunkPayload {
    /// Creates a new `ChunkPayload` with the given blocks and status.
    #[must_use]
    pub fn new(blocks: Blocks, status: ChunkStatus) -> Self {
        Self {
            blocks,
            status,
            unknown_sections: Vec::new(),
        }
    }

    /// Encodes all sections into uncompressed payload bytes with canonical tag ordering.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut sections: Vec<(u8, u8, Vec<u8>)> = Vec::new();

        // 1. BLOCKS section (0x01)
        let mut blocks_buf = Vec::new();
        encode_blocks(&self.blocks, &mut blocks_buf);
        sections.push((TAG_BLOCKS, 0, blocks_buf));

        // 2. STATUS section (0x08)
        let mut status_buf = Vec::new();
        self.status.encode(&mut status_buf);
        sections.push((TAG_STATUS, 0, status_buf));

        // 3. Unknown sections
        for s in &self.unknown_sections {
            sections.push((s.tag, s.sflags, s.data.clone()));
        }

        // Sort by tag ascending for canonical output
        sections.sort_by_key(|&(tag, _, _)| tag);

        let mut out = Vec::new();
        encode_varint_u32(sections.len() as u32, &mut out);

        for (tag, sflags, data) in sections {
            out.push(tag);
            out.push(sflags);
            encode_varint_u32(data.len() as u32, &mut out);
            out.extend_from_slice(&data);
        }

        out
    }

    /// Decodes chunk payload from uncompressed bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cursor = bytes;
        let section_count = decode_varint_u32(&mut cursor)?;

        let mut blocks: Option<Blocks> = None;
        let mut status: Option<ChunkStatus> = None;
        let mut unknown_sections = Vec::new();

        for _ in 0..section_count {
            if cursor.len() < 2 {
                return Err(StorageError::CorruptPayload(
                    "Truncated section header".into(),
                ));
            }
            let tag = cursor[0];
            let sflags = cursor[1];
            cursor = &cursor[2..];

            let len = decode_varint_u32(&mut cursor)? as usize;
            if cursor.len() < len {
                return Err(StorageError::Truncated {
                    actual: cursor.len() as u64,
                    expected: len as u64,
                });
            }

            let section_data = &cursor[..len];
            cursor = &cursor[len..];

            match tag {
                TAG_BLOCKS => {
                    blocks = Some(decode_blocks(section_data)?);
                }
                TAG_STATUS => {
                    status = Some(ChunkStatus::decode(section_data)?);
                }
                _ => {
                    if (sflags & SFLAG_CRITICAL) != 0 {
                        return Err(StorageError::CorruptPayload(format!(
                            "Unrecognized critical section tag {tag:#04x}"
                        )));
                    }
                    unknown_sections.push(RawSection {
                        tag,
                        sflags,
                        data: section_data.to_vec(),
                    });
                }
            }
        }

        let blocks = blocks.ok_or_else(|| {
            StorageError::CorruptPayload("Missing required BLOCKS section".into())
        })?;
        let status = status.unwrap_or_default();

        Ok(Self {
            blocks,
            status,
            unknown_sections,
        })
    }
}

/// Encodes `Blocks` container into the binary BLOCKS section format.
pub fn encode_blocks(blocks: &Blocks, buf: &mut Vec<u8>) {
    match blocks {
        Blocks::Uniform(state) => {
            buf.push(0); // 0 bits
            encode_varint_u32(1, buf); // palette_len = 1
            encode_varint_u32(state.as_u32(), buf);
        }
        Blocks::Packed(packed) => {
            let bits = packed.bits() as u8;
            buf.push(bits);

            let palette = packed.palette();
            encode_varint_u32(palette.len() as u32, buf);
            for state in palette {
                encode_varint_u32(state.as_u32(), buf);
            }

            for &word in packed.words() {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
}

/// Decodes binary BLOCKS section into `Blocks`.
pub fn decode_blocks(mut cursor: &[u8]) -> Result<Blocks> {
    if cursor.is_empty() {
        return Err(StorageError::CorruptPayload("Empty BLOCKS section".into()));
    }

    let bits = cursor[0];
    cursor = &cursor[1..];

    if bits == 0 {
        let palette_len = decode_varint_u32(&mut cursor)?;
        if palette_len != 1 {
            return Err(StorageError::CorruptPayload(format!(
                "Uniform blocks section must have palette_len=1, got {palette_len}"
            )));
        }
        let state_id = decode_varint_u32(&mut cursor)?;
        return Ok(Blocks::Uniform(BlockStateId::new(state_id)));
    }

    let log2 = match bits {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        other => {
            return Err(StorageError::CorruptPayload(format!(
                "Invalid bit width {other} in BLOCKS section"
            )));
        }
    };

    let palette_len = decode_varint_u32(&mut cursor)? as usize;
    let max_palette_len = 1usize << bits;
    if palette_len == 0 || palette_len > max_palette_len {
        return Err(StorageError::CorruptPayload(format!(
            "Palette length {palette_len} invalid for {bits} bits (max {max_palette_len})"
        )));
    }

    let mut palette = Vec::with_capacity(palette_len);
    for _ in 0..palette_len {
        let state_id = decode_varint_u32(&mut cursor)?;
        palette.push(BlockStateId::new(state_id));
    }

    let word_count = 512usize << log2;
    let expected_words_bytes = word_count * 8;
    if cursor.len() < expected_words_bytes {
        return Err(StorageError::Truncated {
            actual: cursor.len() as u64,
            expected: expected_words_bytes as u64,
        });
    }

    let mut words = vec![0u64; word_count].into_boxed_slice();
    for word in &mut words {
        *word = u64::from_le_bytes(cursor[0..8].try_into().unwrap());
        cursor = &cursor[8..];
    }

    // Safety scan: verify that every packed index is within palette_len
    for &w in &words {
        let pw_log2 = 6 - u32::from(log2);
        let entries_per_word = 1usize << pw_log2;
        let mask = (1u64 << bits) - 1;
        for e in 0..entries_per_word {
            let slot = ((w >> (e * (bits as usize))) & mask) as usize;
            if slot >= palette_len {
                return Err(StorageError::CorruptPayload(format!(
                    "Packed voxel index {slot} out of bounds for palette length {palette_len}"
                )));
            }
        }
    }

    let packed = Packed::from_raw_parts(log2, palette.into_boxed_slice(), words);
    Ok(Blocks::Packed(Box::new(packed)))
}
