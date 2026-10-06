//! `.tlr` region container header format, preamble, and entry tables.

use crate::error::{Result, StorageError};
use bitflags::bitflags;
use xxhash_rust::xxh3::xxh3_64;

/// Byte size of each disk sector (4 KiB).
pub const SECTOR_SIZE: usize = 4096;

/// Number of sectors allocated for each header slot (5 sectors = 20,480 bytes).
pub const HEADER_SLOT_SECTORS: usize = 5;

/// Byte size of a single header slot.
pub const HEADER_SLOT_BYTES: usize = HEADER_SLOT_SECTORS * SECTOR_SIZE;

/// Starting sector for user data (after dual header slots: 0..=4 and 5..=9).
pub const DATA_START_SECTOR: u32 = 10;

/// Number of chunks per cubic region file (8 × 8 × 8 = 512).
pub const ENTRY_COUNT: usize = 512;

/// Byte length of the preamble header.
pub const PREAMBLE_BYTES: usize = 64;

/// Byte length of a single entry in the table.
pub const ENTRY_BYTES: usize = 32;

/// Byte length of the 512-entry table (16,384 bytes).
pub const TABLE_BYTES: usize = ENTRY_COUNT * ENTRY_BYTES;

/// Magic bytes identifying `.tlr` container: `\x89TLR\r\n\x1a\n`.
pub const CONTAINER_MAGIC: [u8; 8] = *b"\x89TLR\r\n\x1a\n";

/// Current container format version.
pub const CONTAINER_VERSION: u16 = 1;

/// Container kind identifier for Chunks.
pub const KIND_CHUNKS: u16 = 1;

bitflags! {
    /// Flags stored in an entry describing its storage mode.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct EntryFlags: u8 {
        /// Chunk payload is present.
        const PRESENT = 1 << 0;
        /// Chunk payload is packed into a sub-sector sharing space with others.
        const PACKED  = 1 << 1;
        /// Chunk is uniform and stored inline in the 32-byte entry (0 data sectors).
        const INLINE  = 1 << 2;
        /// Chunk was oversized and spilled to an external `.tls` file.
        const SPILLED = 1 << 3;
    }
}

/// Compression codec identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CodecId {
    /// Raw uncompressed payload.
    #[default]
    Raw = 0,
    /// Fast LZ4 compression.
    Lz4 = 1,
    /// High-ratio Zstandard compression.
    Zstd = 2,
    /// Zstandard compression with pre-trained dictionary.
    ZstdDict = 3,
}

impl TryFrom<u8> for CodecId {
    type Error = StorageError;

    fn try_from(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Zstd),
            3 => Ok(Self::ZstdDict),
            other => Err(StorageError::CorruptPayload(format!(
                "Invalid codec ID {other}"
            ))),
        }
    }
}

/// 64-byte preamble at the start of each header slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preamble {
    /// Format version (must be 1).
    pub container_version: u16,
    /// Payload kind (1 = Chunks).
    pub kind: u16,
    /// Flags (reserved, must be 0).
    pub flags: u32,
    /// Monotonically increasing commit counter.
    pub generation: u64,
    /// Region coordinates (rx, ry, rz).
    pub region_pos: (i32, i32, i32),
    /// Total entries in table (512 for chunks).
    pub entry_count: u32,
    /// Logical EOF: total sectors allocated at commit time.
    pub file_sectors: u32,
    /// Reserved field.
    pub reserved: u32,
    /// XXH3 checksum of the entire entry table (16,384 bytes).
    pub table_checksum: u64,
    /// XXH3 checksum of preamble bytes 0..56.
    pub preamble_checksum: u64,
}

/// Data stored inline in the entry for uniform chunks (all-air, all-stone, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InlineData {
    /// Save ID of the uniform block.
    pub block_save_id: u32,
    /// Save ID of the uniform biome.
    pub biome_save_id: u16,
    /// World generation status stage.
    pub status: u8,
    /// Sky light value (0..15, or 0xFF for derive).
    pub sky: u8,
    /// Reserved bytes for future expansion.
    pub reserved: [u8; 8],
}

impl InlineData {
    /// Encodes inline data into a 16-byte slice.
    #[must_use]
    pub fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&self.block_save_id.to_le_bytes());
        buf[4..6].copy_from_slice(&self.biome_save_id.to_le_bytes());
        buf[6] = self.status;
        buf[7] = self.sky;
        buf[8..16].copy_from_slice(&self.reserved);
        buf
    }

    /// Decodes inline data from a 16-byte slice.
    #[must_use]
    pub fn decode(buf: &[u8; 16]) -> Self {
        let block_save_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let biome_save_id = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        let status = buf[6];
        let sky = buf[7];
        let mut reserved = [0u8; 8];
        reserved.copy_from_slice(&buf[8..16]);
        Self {
            block_save_id,
            biome_save_id,
            status,
            sky,
            reserved,
        }
    }
}

/// 32-byte entry in the region header table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Entry {
    /// First data sector (0 if absent or INLINE).
    pub sector: u32,
    /// Byte offset within first sector if PACKED (0..4095).
    pub byte_offset: u16,
    /// Number of contiguous sectors touched by this payload.
    pub sector_span: u16,
    /// Total byte length on disk (48 B frame + compressed payload).
    pub total_len: u32,
    /// Last modification timestamp (Unix seconds).
    pub mtime: u32,
    /// XXH3 checksum of payload (or bytes 0..16 if INLINE).
    pub checksum: u64,
    /// Entry flags (PRESENT, PACKED, INLINE, SPILLED).
    pub flags: EntryFlags,
    /// Compression codec used for this payload.
    pub codec: CodecId,
    /// Dictionary ID (0 = none).
    pub dict_id: u16,
    /// Payload schema data version (starts at 1).
    pub data_version: u32,
    /// Parsed inline payload if `flags.contains(INLINE)`.
    pub inline_data: Option<InlineData>,
}

impl Entry {
    /// Creates an empty (absent) entry.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            sector: 0,
            byte_offset: 0,
            sector_span: 0,
            total_len: 0,
            mtime: 0,
            checksum: 0,
            flags: EntryFlags::empty(),
            codec: CodecId::Raw,
            dict_id: 0,
            data_version: 1,
            inline_data: None,
        }
    }

    /// Creates an INLINE uniform chunk entry (takes 0 data sectors).
    #[must_use]
    pub fn new_inline(
        block_save_id: u32,
        biome_save_id: u16,
        status: u8,
        sky: u8,
        mtime: u32,
        data_version: u32,
    ) -> Self {
        let inline = InlineData {
            block_save_id,
            biome_save_id,
            status,
            sky,
            reserved: [0u8; 8],
        };
        let bytes16 = inline.encode();
        let checksum = xxh3_64(&bytes16);
        Self {
            sector: 0,
            byte_offset: 0,
            sector_span: 0,
            total_len: 0,
            mtime,
            checksum,
            flags: EntryFlags::PRESENT | EntryFlags::INLINE,
            codec: CodecId::Raw,
            dict_id: 0,
            data_version,
            inline_data: Some(inline),
        }
    }

    /// Serializes this entry into a 32-byte array.
    #[must_use]
    pub fn encode(&self) -> [u8; ENTRY_BYTES] {
        let mut buf = [0u8; ENTRY_BYTES];
        if self.flags.contains(EntryFlags::INLINE) {
            let inline_bytes = self.inline_data.unwrap_or_default().encode();
            buf[0..16].copy_from_slice(&inline_bytes);
        } else {
            buf[0..4].copy_from_slice(&self.sector.to_le_bytes());
            buf[4..6].copy_from_slice(&self.byte_offset.to_le_bytes());
            buf[6..8].copy_from_slice(&self.sector_span.to_le_bytes());
            buf[8..12].copy_from_slice(&self.total_len.to_le_bytes());
            buf[12..16].copy_from_slice(&self.mtime.to_le_bytes());
        }
        buf[16..24].copy_from_slice(&self.checksum.to_le_bytes());
        buf[24] = self.flags.bits();
        buf[25] = self.codec as u8;
        buf[26..28].copy_from_slice(&self.dict_id.to_le_bytes());
        buf[28..32].copy_from_slice(&self.data_version.to_le_bytes());
        buf
    }

    /// Deserializes an entry from a 32-byte slice.
    pub fn decode(buf: &[u8; ENTRY_BYTES]) -> Result<Self> {
        let flags = EntryFlags::from_bits(buf[24]).ok_or_else(|| {
            StorageError::CorruptPayload(format!("Unknown entry flags bits: {:#04x}", buf[24]))
        })?;
        let codec = CodecId::try_from(buf[25])?;
        let dict_id = u16::from_le_bytes(buf[26..28].try_into().unwrap());
        let data_version = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let checksum = u64::from_le_bytes(buf[16..24].try_into().unwrap());

        if flags.contains(EntryFlags::INLINE) {
            let mut inline_bytes = [0u8; 16];
            inline_bytes.copy_from_slice(&buf[0..16]);
            let computed_checksum = xxh3_64(&inline_bytes);
            if computed_checksum != checksum {
                return Err(StorageError::ChecksumMismatch {
                    expected: checksum,
                    computed: computed_checksum,
                });
            }
            let inline_data = InlineData::decode(&inline_bytes);
            Ok(Self {
                sector: 0,
                byte_offset: 0,
                sector_span: 0,
                total_len: 0,
                mtime: 0,
                checksum,
                flags,
                codec,
                dict_id,
                data_version,
                inline_data: Some(inline_data),
            })
        } else {
            let sector = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let byte_offset = u16::from_le_bytes(buf[4..6].try_into().unwrap());
            let sector_span = u16::from_le_bytes(buf[6..8].try_into().unwrap());
            let total_len = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let mtime = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            Ok(Self {
                sector,
                byte_offset,
                sector_span,
                total_len,
                mtime,
                checksum,
                flags,
                codec,
                dict_id,
                data_version,
                inline_data: None,
            })
        }
    }
}

/// A full 20,480-byte header slot containing preamble and 512 entries.
#[derive(Debug, Clone)]
pub struct HeaderSlot {
    /// Preamble metadata.
    pub preamble: Preamble,
    /// 512 entries for the chunks in this region.
    pub entries: Box<[Entry; ENTRY_COUNT]>,
}

impl HeaderSlot {
    /// Creates a fresh empty header slot for the given region coordinates.
    #[must_use]
    pub fn new_empty(rx: i32, ry: i32, rz: i32, generation: u64) -> Self {
        let entries = alloc_empty_entries();
        let mut preamble = Preamble {
            container_version: CONTAINER_VERSION,
            kind: KIND_CHUNKS,
            flags: 0,
            generation,
            region_pos: (rx, ry, rz),
            entry_count: ENTRY_COUNT as u32,
            file_sectors: DATA_START_SECTOR,
            reserved: 0,
            table_checksum: 0,
            preamble_checksum: 0,
        };
        // Compute checksums for the initial empty table
        let mut table_bytes = [0u8; TABLE_BYTES];
        for (i, entry) in entries.iter().enumerate() {
            table_bytes[i * ENTRY_BYTES..(i + 1) * ENTRY_BYTES].copy_from_slice(&entry.encode());
        }
        preamble.table_checksum = xxh3_64(&table_bytes);

        let mut preamble_bytes = [0u8; 56];
        preamble_bytes[0..8].copy_from_slice(&CONTAINER_MAGIC);
        preamble_bytes[8..10].copy_from_slice(&preamble.container_version.to_le_bytes());
        preamble_bytes[10..12].copy_from_slice(&preamble.kind.to_le_bytes());
        preamble_bytes[12..16].copy_from_slice(&preamble.flags.to_le_bytes());
        preamble_bytes[16..24].copy_from_slice(&preamble.generation.to_le_bytes());
        preamble_bytes[24..28].copy_from_slice(&preamble.region_pos.0.to_le_bytes());
        preamble_bytes[28..32].copy_from_slice(&preamble.region_pos.1.to_le_bytes());
        preamble_bytes[32..36].copy_from_slice(&preamble.region_pos.2.to_le_bytes());
        preamble_bytes[36..40].copy_from_slice(&preamble.entry_count.to_le_bytes());
        preamble_bytes[40..44].copy_from_slice(&preamble.file_sectors.to_le_bytes());
        preamble_bytes[44..48].copy_from_slice(&preamble.reserved.to_le_bytes());
        preamble_bytes[48..56].copy_from_slice(&preamble.table_checksum.to_le_bytes());
        preamble.preamble_checksum = xxh3_64(&preamble_bytes);

        Self { preamble, entries }
    }

    /// Serializes this header slot into a 20,480-byte buffer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_SLOT_BYTES];

        // 1. Encode entries and compute table checksum
        for (i, entry) in self.entries.iter().enumerate() {
            let offset = PREAMBLE_BYTES + i * ENTRY_BYTES;
            out[offset..offset + ENTRY_BYTES].copy_from_slice(&entry.encode());
        }
        let table_checksum = xxh3_64(&out[PREAMBLE_BYTES..PREAMBLE_BYTES + TABLE_BYTES]);

        // 2. Encode preamble bytes 0..56
        out[0..8].copy_from_slice(&CONTAINER_MAGIC);
        out[8..10].copy_from_slice(&self.preamble.container_version.to_le_bytes());
        out[10..12].copy_from_slice(&self.preamble.kind.to_le_bytes());
        out[12..16].copy_from_slice(&self.preamble.flags.to_le_bytes());
        out[16..24].copy_from_slice(&self.preamble.generation.to_le_bytes());
        out[24..28].copy_from_slice(&self.preamble.region_pos.0.to_le_bytes());
        out[28..32].copy_from_slice(&self.preamble.region_pos.1.to_le_bytes());
        out[32..36].copy_from_slice(&self.preamble.region_pos.2.to_le_bytes());
        out[36..40].copy_from_slice(&self.preamble.entry_count.to_le_bytes());
        out[40..44].copy_from_slice(&self.preamble.file_sectors.to_le_bytes());
        out[44..48].copy_from_slice(&self.preamble.reserved.to_le_bytes());
        out[48..56].copy_from_slice(&table_checksum.to_le_bytes());

        // 3. Compute preamble checksum
        let preamble_checksum = xxh3_64(&out[0..56]);
        out[56..64].copy_from_slice(&preamble_checksum.to_le_bytes());

        out
    }

    /// Deserializes and validates a header slot from a slice.
    #[allow(clippy::similar_names)]
    pub fn decode(
        buf: &[u8],
        expected_rx: i32,
        expected_ry: i32,
        expected_rz: i32,
    ) -> Result<Self> {
        if buf.len() < PREAMBLE_BYTES + TABLE_BYTES {
            return Err(StorageError::Truncated {
                actual: buf.len() as u64,
                expected: (PREAMBLE_BYTES + TABLE_BYTES) as u64,
            });
        }

        // 1. Verify magic
        if buf[0..8] != CONTAINER_MAGIC {
            return Err(StorageError::InvalidMagic {
                expected: &CONTAINER_MAGIC,
                actual: buf[0..8].to_vec(),
            });
        }

        // 2. Verify preamble checksum
        let stored_preamble_checksum = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        let computed_preamble_checksum = xxh3_64(&buf[0..56]);
        if stored_preamble_checksum != computed_preamble_checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: stored_preamble_checksum,
                computed: computed_preamble_checksum,
            });
        }

        // 3. Parse preamble fields
        let container_version = u16::from_le_bytes(buf[8..10].try_into().unwrap());
        if container_version != CONTAINER_VERSION {
            return Err(StorageError::UnsupportedContainerVersion {
                version: container_version,
                max_supported: CONTAINER_VERSION,
            });
        }

        let kind = u16::from_le_bytes(buf[10..12].try_into().unwrap());
        let flags = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        if flags != 0 {
            return Err(StorageError::CorruptPayload(format!(
                "Non-zero preamble flags: {flags:#010x}"
            )));
        }

        let generation = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let rx = i32::from_le_bytes(buf[24..28].try_into().unwrap());
        let ry = i32::from_le_bytes(buf[28..32].try_into().unwrap());
        let rz = i32::from_le_bytes(buf[32..36].try_into().unwrap());

        if rx != expected_rx || ry != expected_ry || rz != expected_rz {
            return Err(StorageError::CorruptPayload(format!(
                "Region coordinates mismatch: header has ({rx}, {ry}, {rz}), expected ({expected_rx}, {expected_ry}, {expected_rz})"
            )));
        }

        let entry_count = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        if entry_count != ENTRY_COUNT as u32 {
            return Err(StorageError::CorruptPayload(format!(
                "Unexpected entry count {entry_count}, expected {ENTRY_COUNT}"
            )));
        }

        let file_sectors = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let reserved = u32::from_le_bytes(buf[44..48].try_into().unwrap());
        let table_checksum = u64::from_le_bytes(buf[48..56].try_into().unwrap());

        // 4. Verify table checksum
        let computed_table_checksum = xxh3_64(&buf[PREAMBLE_BYTES..PREAMBLE_BYTES + TABLE_BYTES]);
        if table_checksum != computed_table_checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: table_checksum,
                computed: computed_table_checksum,
            });
        }

        // 5. Decode entries
        let mut entries = alloc_empty_entries();
        for (i, entry) in entries.iter_mut().enumerate() {
            let offset = PREAMBLE_BYTES + i * ENTRY_BYTES;
            let mut entry_bytes = [0u8; ENTRY_BYTES];
            entry_bytes.copy_from_slice(&buf[offset..offset + ENTRY_BYTES]);
            *entry = Entry::decode(&entry_bytes)?;
        }

        let preamble = Preamble {
            container_version,
            kind,
            flags,
            generation,
            region_pos: (rx, ry, rz),
            entry_count,
            file_sectors,
            reserved,
            table_checksum,
            preamble_checksum: stored_preamble_checksum,
        };

        Ok(Self { preamble, entries })
    }
}

fn alloc_empty_entries() -> Box<[Entry; ENTRY_COUNT]> {
    vec![Entry::empty(); ENTRY_COUNT]
        .into_boxed_slice()
        .try_into()
        .expect("vector has exactly ENTRY_COUNT elements")
}
