//! `.tlr` region container file coordinating open, read, write, and crash-safe commits.

use crate::compression::{compress, decompress};
use crate::error::{Result, StorageError};
use crate::format::frame::{FRAME_BYTES, PayloadFrame};
use crate::format::header::{
    CodecId, DATA_START_SECTOR, ENTRY_COUNT, Entry, EntryFlags, HEADER_SLOT_BYTES, HeaderSlot,
    SECTOR_SIZE,
};
use crate::format::section::{ChunkPayload, ChunkStatus};
use crate::io::RegionIo;
use crate::region::RegionPos;
use crate::region::allocator::{PackSectorBuffer, SectorAllocator};
use std::collections::HashSet;
use telos_core::coords::ChunkPos;
use telos_voxel::{state::BlockStateId, storage::Blocks};
use xxhash_rust::xxh3::xxh3_64;

/// A `.tlr` region file managing 512 cubic chunks with double-buffered commit safety.
pub struct RegionFile<I: RegionIo> {
    io: I,
    region_pos: RegionPos,
    active_slot: HeaderSlot,
    active_slot_index: usize,
    allocator: SectorAllocator,
}

impl<I: RegionIo> RegionFile<I> {
    /// Opens an existing region file or initializes a new one with default compression (Zstd level 3).
    pub fn open(io: I, rx: i32, ry: i32, rz: i32) -> Result<Self> {
        Self::open_with_compression(io, rx, ry, rz, true, 3)
    }

    /// Returns true if this region container is configured to compress non-uniform chunks.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.active_slot
            .preamble
            .flags
            .contains(crate::format::header::RegionFlags::COMPRESSED)
    }

    /// Returns the Zstandard compression level (1..=22, or 0 if uncompressed) for this container.
    #[must_use]
    pub fn compression_level(&self) -> u32 {
        self.active_slot.preamble.compression_level
    }

    /// Sets container compression parameters for future commits.
    pub fn set_compression(&mut self, compressed: bool, level: u32) {
        if compressed {
            self.active_slot
                .preamble
                .flags
                .insert(crate::format::header::RegionFlags::COMPRESSED);
            self.active_slot.preamble.compression_level = level.clamp(1, 22);
        } else {
            self.active_slot
                .preamble
                .flags
                .remove(crate::format::header::RegionFlags::COMPRESSED);
            self.active_slot.preamble.compression_level = 0;
        }
    }

    /// Opens an existing region file or initializes a new one with explicit compression settings.
    #[allow(clippy::similar_names)]
    pub fn open_with_compression(
        io: I,
        rx: i32,
        ry: i32,
        rz: i32,
        compressed: bool,
        compression_level: u32,
    ) -> Result<Self> {
        let region_pos = RegionPos::new(rx, ry, rz);
        let len = io.len()?;

        if len == 0 {
            // Newly created container: write both initial header slots and sync
            let slot_a = HeaderSlot::new_empty_with_compression(
                rx,
                ry,
                rz,
                1,
                compressed,
                compression_level,
            );
            let slot_b = HeaderSlot::new_empty_with_compression(
                rx,
                ry,
                rz,
                0,
                compressed,
                compression_level,
            );

            let bytes_a = slot_a.encode();
            let bytes_b = slot_b.encode();

            io.write_at(0, &bytes_a)?;
            io.write_at(HEADER_SLOT_BYTES as u64, &bytes_b)?;
            io.sync_data()?;

            let allocator = SectorAllocator::from_header(&slot_a);
            return Ok(Self {
                io,
                region_pos,
                active_slot: slot_a,
                active_slot_index: 0,
                allocator,
            });
        }

        // Existing file: inspect both slots A and B
        let slot_a_res = Self::read_and_validate_slot(&io, 0, rx, ry, rz, len);
        let slot_b_res =
            Self::read_and_validate_slot(&io, HEADER_SLOT_BYTES as u64, rx, ry, rz, len);

        let (active_slot, active_slot_index) = match (slot_a_res, slot_b_res) {
            (Ok(slot_a), Ok(slot_b)) => {
                // Both slots valid: pick highest generation
                if slot_a.preamble.generation >= slot_b.preamble.generation {
                    (slot_a, 0)
                } else {
                    (slot_b, 1)
                }
            }
            (Ok(slot_a), Err(_)) => (slot_a, 0),
            (Err(_), Ok(slot_b)) => (slot_b, 1),
            (Err(err_a), Err(err_b)) => {
                return Err(StorageError::BothHeaderSlotsCorrupt {
                    slot_a: err_a.to_string(),
                    slot_b: err_b.to_string(),
                });
            }
        };

        let allocator = SectorAllocator::from_header(&active_slot);

        Ok(Self {
            io,
            region_pos,
            active_slot,
            active_slot_index,
            allocator,
        })
    }

    /// Reads and validates a header slot from disk at the given byte offset.
    fn read_and_validate_slot(
        io: &I,
        offset: u64,
        rx: i32,
        ry: i32,
        rz: i32,
        actual_file_len: u64,
    ) -> Result<HeaderSlot> {
        if actual_file_len < offset + HEADER_SLOT_BYTES as u64 {
            return Err(StorageError::Truncated {
                actual: actual_file_len,
                expected: offset + HEADER_SLOT_BYTES as u64,
            });
        }

        let mut buf = vec![0u8; HEADER_SLOT_BYTES];
        io.read_at(offset, &mut buf)?;

        let slot = HeaderSlot::decode(&buf, rx, ry, rz)?;

        // Ensure file is not truncated relative to this slot's logical EOF
        let claimed_len = u64::from(slot.preamble.file_sectors) * (SECTOR_SIZE as u64);
        if claimed_len > actual_file_len {
            return Err(StorageError::Truncated {
                actual: actual_file_len,
                expected: claimed_len,
            });
        }

        // Validate entries consistency and overlap
        Self::validate_entries(&slot.entries)?;

        Ok(slot)
    }

    /// Validates entries sanity: bounds, spans, and overlap prevention.
    fn validate_entries(entries: &[Entry; ENTRY_COUNT]) -> Result<()> {
        for (i, e1) in entries.iter().enumerate() {
            if !e1.flags.contains(EntryFlags::PRESENT) || e1.flags.contains(EntryFlags::INLINE) {
                continue;
            }

            if e1.sector < DATA_START_SECTOR {
                return Err(StorageError::CorruptPayload(format!(
                    "Entry {i} sector {} below data start sector {DATA_START_SECTOR}",
                    e1.sector
                )));
            }

            if e1.flags.contains(EntryFlags::PACKED) {
                if e1.sector_span != 1 {
                    return Err(StorageError::CorruptPayload(format!(
                        "Packed entry {i} must have sector_span=1, got {}",
                        e1.sector_span
                    )));
                }
                if usize::from(e1.byte_offset) + e1.total_len as usize > SECTOR_SIZE {
                    return Err(StorageError::CorruptPayload(format!(
                        "Packed entry {i} exceeds 4 KiB sector boundary"
                    )));
                }
            } else {
                let max_bytes = usize::from(e1.sector_span) * SECTOR_SIZE;
                if (e1.total_len as usize) > max_bytes {
                    return Err(StorageError::CorruptPayload(format!(
                        "Entry {i} total_len {} exceeds span capacity {max_bytes}",
                        e1.total_len
                    )));
                }
            }

            // Check against remaining entries
            for (j, e2) in entries.iter().enumerate().skip(i + 1) {
                if !e2.flags.contains(EntryFlags::PRESENT) || e2.flags.contains(EntryFlags::INLINE)
                {
                    continue;
                }

                let e1_packed = e1.flags.contains(EntryFlags::PACKED);
                let e2_packed = e2.flags.contains(EntryFlags::PACKED);

                if e1_packed && e2_packed {
                    if e1.sector == e2.sector {
                        let e1_start = usize::from(e1.byte_offset);
                        let e1_end = e1_start + e1.total_len as usize;
                        let e2_start = usize::from(e2.byte_offset);
                        let e2_end = e2_start + e2.total_len as usize;

                        if !(e1_end <= e2_start || e2_end <= e1_start) {
                            return Err(StorageError::CorruptPayload(format!(
                                "Packed entries {i} and {j} overlap in sector {}",
                                e1.sector
                            )));
                        }
                    }
                } else if !e1_packed && !e2_packed {
                    let e1_end = e1.sector + u32::from(e1.sector_span);
                    let e2_end = e2.sector + u32::from(e2.sector_span);
                    if !(e1_end <= e2.sector || e2_end <= e1.sector) {
                        return Err(StorageError::CorruptPayload(format!(
                            "Entries {i} and {j} sector spans overlap: [{}, {}) vs [{}, {})",
                            e1.sector, e1_end, e2.sector, e2_end
                        )));
                    }
                } else {
                    // One is packed, one is non-packed: cannot share any sector
                    let (packed_sec, non_packed_start, non_packed_span) = if e1_packed {
                        (e1.sector, e2.sector, u32::from(e2.sector_span))
                    } else {
                        (e2.sector, e1.sector, u32::from(e1.sector_span))
                    };
                    if packed_sec >= non_packed_start
                        && packed_sec < non_packed_start + non_packed_span
                    {
                        return Err(StorageError::CorruptPayload(format!(
                            "Packed entry overlaps non-packed entry sectors at sector {packed_sec}"
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Region coordinates of this container.
    #[must_use]
    pub fn region_pos(&self) -> RegionPos {
        self.region_pos
    }

    /// Generation of the currently active header slot.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.active_slot.preamble.generation
    }

    /// Reference to the active header slot.
    #[must_use]
    pub fn active_slot(&self) -> &HeaderSlot {
        &self.active_slot
    }

    /// Reference to the underlying IO implementation.
    #[must_use]
    pub fn io(&self) -> &I {
        &self.io
    }

    /// Mutable reference to the underlying IO implementation.
    pub fn io_mut(&mut self) -> &mut I {
        &mut self.io
    }

    /// Reads and decodes a chunk from this region.
    ///
    /// Returns `Ok(None)` if the chunk is not present in the container.
    pub fn read_chunk(&self, chunk_pos: ChunkPos) -> Result<Option<ChunkPayload>> {
        let expected_region = RegionPos::from_chunk(chunk_pos);
        if expected_region != self.region_pos {
            return Err(StorageError::CorruptPayload(format!(
                "Chunk {chunk_pos:?} does not belong to region {:?}",
                self.region_pos
            )));
        }

        let entry_idx = RegionPos::entry_index(chunk_pos);
        let entry = &self.active_slot.entries[entry_idx];

        if !entry.flags.contains(EntryFlags::PRESENT) {
            return Ok(None);
        }

        // Fast path: INLINE uniform chunk elision (0 disk reads, instant)
        if entry.flags.contains(EntryFlags::INLINE) {
            let inline = entry.inline_data.unwrap_or_default();
            let blocks = Blocks::Uniform(BlockStateId::new(inline.block_save_id));
            let status = ChunkStatus {
                gen_stage: inline.status,
                inhabited_ticks: 0,
                flags: 0,
            };
            return Ok(Some(ChunkPayload::new(blocks, status)));
        }

        // Disk path: read payload run or packed sub-sector
        let offset = u64::from(entry.sector) * (SECTOR_SIZE as u64) + u64::from(entry.byte_offset);
        let total_len = entry.total_len as usize;

        if total_len < FRAME_BYTES {
            return Err(StorageError::CorruptPayload(format!(
                "Entry total_len {total_len} smaller than frame header {FRAME_BYTES}"
            )));
        }

        let mut disk_bytes = vec![0u8; total_len];
        self.io.read_at(offset, &mut disk_bytes)?;

        // Verify checksum of disk bytes (frame + stored)
        let computed_checksum = xxh3_64(&disk_bytes);
        if computed_checksum != entry.checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: entry.checksum,
                computed: computed_checksum,
            });
        }

        // Decode 48-byte payload frame
        let mut frame_buf = [0u8; FRAME_BYTES];
        frame_buf.copy_from_slice(&disk_bytes[0..FRAME_BYTES]);
        let frame = PayloadFrame::decode(&frame_buf)?;

        if frame.chunk_pos != (chunk_pos.x(), chunk_pos.y(), chunk_pos.z()) {
            return Err(StorageError::CorruptPayload(format!(
                "Payload frame coords {:?} mismatch requested {chunk_pos:?}",
                frame.chunk_pos
            )));
        }

        let stored_bytes = &disk_bytes[FRAME_BYTES..];
        frame.verify_stored(stored_bytes)?;

        // Decompress
        let uncompressed = decompress(frame.codec, stored_bytes, frame.raw_len)?;

        // Decode sections
        let payload = ChunkPayload::decode(&uncompressed)?;
        Ok(Some(payload))
    }

    /// Commits a batch of modified chunks using the double-buffered crash-safe sequence.
    ///
    /// Chunks mapped to `None` are deleted.
    #[allow(clippy::too_many_lines)]
    pub fn commit_chunks(
        &mut self,
        chunks: &[(ChunkPos, Option<ChunkPayload>)],
        codec: CodecId,
        timestamp: u32,
    ) -> Result<()> {
        let mut updated_entries = Vec::new();
        let mut writes_to_perform: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut occupied_in_commit = HashSet::new();
        let mut current_pack: Option<PackSectorBuffer> = None;

        let new_generation = self.active_slot.preamble.generation + 1;

        for (pos, maybe_payload) in chunks {
            let expected_region = RegionPos::from_chunk(*pos);
            if expected_region != self.region_pos {
                return Err(StorageError::CorruptPayload(format!(
                    "Chunk {pos:?} does not belong to region {:?}",
                    self.region_pos
                )));
            }

            let entry_idx = RegionPos::entry_index(*pos);

            let Some(payload) = maybe_payload else {
                // Delete chunk
                updated_entries.push((entry_idx, Entry::empty()));
                continue;
            };

            // INLINE candidate check: uniform blocks with default status and no custom sections
            let is_default_status =
                payload.status.inhabited_ticks == 0 && payload.status.flags == 0;
            if payload.blocks.is_uniform()
                && payload.unknown_sections.is_empty()
                && is_default_status
            {
                let block_id = match payload.blocks {
                    Blocks::Uniform(id) => id.as_u32(),
                    Blocks::Packed(_) => unreachable!(),
                };
                let entry =
                    Entry::new_inline(block_id, 0, payload.status.gen_stage, 0xFF, timestamp, 1);
                updated_entries.push((entry_idx, entry));
                continue;
            }

            // Non-uniform chunk: encode, compress, and allocate
            let raw_bytes = payload.encode();
            let (effective_codec, stored_bytes) = if self.is_compressed() {
                let level = i32::try_from(self.compression_level()).unwrap_or(3);
                (
                    CodecId::Zstd,
                    crate::compression::compress_with_level(CodecId::Zstd, &raw_bytes, level)?,
                )
            } else if codec == CodecId::Zstd {
                (
                    CodecId::Raw,
                    crate::compression::compress_with_level(CodecId::Raw, &raw_bytes, 0)?,
                )
            } else {
                (codec, compress(codec, &raw_bytes)?)
            };

            let frame = PayloadFrame {
                kind: 1,
                codec: effective_codec,
                dict_id: 0,
                chunk_pos: (pos.x(), pos.y(), pos.z()),
                stored_len: stored_bytes.len() as u32,
                raw_len: raw_bytes.len() as u32,
                data_version: 1,
                generation: new_generation,
                checksum: xxh3_64(&stored_bytes),
            };

            let mut disk_bytes = Vec::with_capacity(FRAME_BYTES + stored_bytes.len());
            disk_bytes.extend_from_slice(&frame.encode());
            disk_bytes.extend_from_slice(&stored_bytes);

            let total_len = disk_bytes.len() as u32;
            let checksum = xxh3_64(&disk_bytes);

            if total_len <= 2048 {
                // Sub-sector packing
                let mut need_new_pack = false;
                if let Some(ref mut pack) = current_pack {
                    if let Some(offset) = pack.try_append(&disk_bytes) {
                        let entry = Entry {
                            sector: pack.sector,
                            byte_offset: offset,
                            sector_span: 1,
                            total_len,
                            mtime: timestamp,
                            checksum,
                            flags: EntryFlags::PRESENT | EntryFlags::PACKED,
                            codec: effective_codec,
                            dict_id: 0,
                            data_version: 1,
                            inline_data: None,
                        };
                        updated_entries.push((entry_idx, entry));
                    } else {
                        need_new_pack = true;
                    }
                } else {
                    need_new_pack = true;
                }

                if need_new_pack {
                    if let Some(prev_pack) = current_pack.take().filter(|p| !p.is_empty()) {
                        writes_to_perform.push((
                            u64::from(prev_pack.sector) * (SECTOR_SIZE as u64),
                            prev_pack.data.to_vec(),
                        ));
                    }

                    let pack_sector = self
                        .allocator
                        .allocate_contiguous(1, &mut occupied_in_commit)?;
                    let mut new_pack = PackSectorBuffer::new(pack_sector);
                    let offset = new_pack
                        .try_append(&disk_bytes)
                        .expect("payload <= 2048 must fit in fresh pack sector");

                    let entry = Entry {
                        sector: pack_sector,
                        byte_offset: offset,
                        sector_span: 1,
                        total_len,
                        mtime: timestamp,
                        checksum,
                        flags: EntryFlags::PRESENT | EntryFlags::PACKED,
                        codec: effective_codec,
                        dict_id: 0,
                        data_version: 1,
                        inline_data: None,
                    };
                    updated_entries.push((entry_idx, entry));
                    current_pack = Some(new_pack);
                }
            } else {
                // Contiguous sector run
                let span = disk_bytes.len().div_ceil(SECTOR_SIZE);
                let sector = self
                    .allocator
                    .allocate_contiguous(span as u32, &mut occupied_in_commit)?;

                let target_len = span * SECTOR_SIZE;
                if disk_bytes.len() < target_len {
                    disk_bytes.resize(target_len, 0);
                }

                writes_to_perform.push((u64::from(sector) * (SECTOR_SIZE as u64), disk_bytes));

                let entry = Entry {
                    sector,
                    byte_offset: 0,
                    sector_span: span as u16,
                    total_len,
                    mtime: timestamp,
                    checksum,
                    flags: EntryFlags::PRESENT,
                    codec: effective_codec,
                    dict_id: 0,
                    data_version: 1,
                    inline_data: None,
                };
                updated_entries.push((entry_idx, entry));
            }
        }

        // Flush trailing pack sector
        if let Some(trailing_pack) = current_pack.take().filter(|p| !p.is_empty()) {
            writes_to_perform.push((
                u64::from(trailing_pack.sector) * (SECTOR_SIZE as u64),
                trailing_pack.data.to_vec(),
            ));
        }

        // --- Crash-Safe Commit Sequence ---

        // 1. Write all payload sectors (never active sectors)
        for (offset, bytes) in &writes_to_perform {
            self.io.write_at(*offset, bytes)?;
        }

        // 2. Barrier: all payload data durable before referenced
        self.io.sync_data()?;

        // 3. Build new table and write inactive header slot
        let inactive_idx = 1 - self.active_slot_index;
        let mut new_slot = self.active_slot.clone();
        for (entry_idx, entry) in updated_entries {
            new_slot.entries[entry_idx] = entry;
        }
        new_slot.preamble.generation = new_generation;
        new_slot.preamble.file_sectors = self.allocator.max_sector();

        let encoded_header = new_slot.encode();
        let header_offset = (inactive_idx * HEADER_SLOT_BYTES) as u64;
        self.io.write_at(header_offset, &encoded_header)?;

        // 4. Commit point: sync new header slot
        self.io.sync_data()?;

        // 5. In-memory swap: activate new slot
        self.active_slot = new_slot;
        self.active_slot_index = inactive_idx;
        self.allocator = SectorAllocator::from_header(&self.active_slot);

        Ok(())
    }
}
