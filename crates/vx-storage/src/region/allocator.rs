//! Sector allocation and sub-sector pack management for `.vxr` region files.

use crate::error::Result;
use crate::format::header::{DATA_START_SECTOR, EntryFlags, HeaderSlot, SECTOR_SIZE};
use std::collections::HashSet;

/// Tracks sector usage and performs crash-safe allocations for region writes.
#[derive(Debug, Clone)]
pub struct SectorAllocator {
    /// Sectors currently referenced by the active header slot (forbidden from being overwritten).
    active_sectors: HashSet<u32>,
    /// Highest sector currently allocated.
    max_sector: u32,
}

impl SectorAllocator {
    /// Constructs a `SectorAllocator` by scanning entries of the active `HeaderSlot`.
    #[must_use]
    pub fn from_header(slot: &HeaderSlot) -> Self {
        let mut active_sectors = HashSet::new();
        let mut max_sector = DATA_START_SECTOR;

        for entry in slot.entries.iter() {
            if entry.flags.contains(EntryFlags::PRESENT)
                && !entry.flags.contains(EntryFlags::INLINE)
                && entry.sector >= DATA_START_SECTOR
            {
                for s in entry.sector..entry.sector + u32::from(entry.sector_span) {
                    active_sectors.insert(s);
                    if s + 1 > max_sector {
                        max_sector = s + 1;
                    }
                }
            }
        }

        Self {
            active_sectors,
            max_sector,
        }
    }

    /// Highest sector index allocated or reserved.
    #[must_use]
    pub fn max_sector(&self) -> u32 {
        self.max_sector
    }

    /// Checks if a sector is currently occupied by the active header slot.
    #[must_use]
    pub fn is_active_sector(&self, sector: u32) -> bool {
        self.active_sectors.contains(&sector)
    }

    /// Finds or appends a contiguous run of `span` sectors that are NOT active.
    ///
    /// `occupied_in_commit` tracks sectors already allocated during the current commit batch.
    pub fn allocate_contiguous(
        &mut self,
        span: u32,
        occupied_in_commit: &mut HashSet<u32>,
    ) -> Result<u32> {
        if span == 0 {
            return Ok(0);
        }

        // Try first-fit search in existing gaps between DATA_START_SECTOR and max_sector
        let mut candidate = DATA_START_SECTOR;
        while candidate + span <= self.max_sector {
            let mut fits = true;
            for s in candidate..candidate + span {
                if self.active_sectors.contains(&s) || occupied_in_commit.contains(&s) {
                    fits = false;
                    candidate = s + 1;
                    break;
                }
            }
            if fits {
                for s in candidate..candidate + span {
                    occupied_in_commit.insert(s);
                }
                return Ok(candidate);
            }
        }

        // No gap found; allocate at the end (append)
        let allocated = std::cmp::max(self.max_sector, candidate);
        for s in allocated..allocated + span {
            occupied_in_commit.insert(s);
        }
        self.max_sector = allocated + span;
        Ok(allocated)
    }
}

/// Buffer for packing small payloads (≤ 2048 B) back-to-back into a single 4 KiB sector.
#[derive(Debug)]
pub struct PackSectorBuffer {
    /// 4096-byte raw buffer.
    pub data: [u8; SECTOR_SIZE],
    /// Current write offset (16-byte aligned).
    pub current_offset: usize,
    /// Allocated sector for this pack buffer on disk.
    pub sector: u32,
}

impl PackSectorBuffer {
    /// Creates a new empty pack buffer bound to the given sector.
    #[must_use]
    pub fn new(sector: u32) -> Self {
        Self {
            data: [0u8; SECTOR_SIZE],
            current_offset: 0,
            sector,
        }
    }

    /// Attempts to append a payload into this pack sector.
    ///
    /// Returns `Some(byte_offset)` on success, or `None` if the payload does not fit.
    pub fn try_append(&mut self, payload: &[u8]) -> Option<u16> {
        let len = payload.len();
        if self.current_offset + len > SECTOR_SIZE {
            return None;
        }

        let offset = self.current_offset;
        self.data[offset..offset + len].copy_from_slice(payload);

        // Advance offset with 16-byte alignment
        let next_offset = (offset + len + 15) & !15;
        self.current_offset = std::cmp::min(next_offset, SECTOR_SIZE);

        Some(offset as u16)
    }

    /// Returns `true` if no bytes have been written to this pack buffer yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current_offset == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entry;

    #[test]
    fn test_allocator_contiguous_first_fit() {
        let mut slot = HeaderSlot::new_empty(0, 0, 0, 1);
        // Occupy sectors 10 and 12
        slot.entries[0] = Entry {
            sector: 10,
            byte_offset: 0,
            sector_span: 1,
            total_len: 4096,
            mtime: 0,
            checksum: 0,
            flags: EntryFlags::PRESENT,
            codec: crate::format::header::CodecId::Raw,
            dict_id: 0,
            data_version: 1,
            inline_data: None,
        };
        slot.entries[1] = Entry {
            sector: 12,
            byte_offset: 0,
            sector_span: 2,
            total_len: 8192,
            mtime: 0,
            checksum: 0,
            flags: EntryFlags::PRESENT,
            codec: crate::format::header::CodecId::Raw,
            dict_id: 0,
            data_version: 1,
            inline_data: None,
        };

        let mut alloc = SectorAllocator::from_header(&slot);
        assert_eq!(alloc.max_sector(), 14);

        let mut occupied = HashSet::new();
        // Sector 11 should be a gap of size 1
        let s = alloc.allocate_contiguous(1, &mut occupied).unwrap();
        assert_eq!(s, 11);

        // Next allocation of size 2 should be at 14 (end of file)
        let s2 = alloc.allocate_contiguous(2, &mut occupied).unwrap();
        assert_eq!(s2, 14);
        assert_eq!(alloc.max_sector(), 16);
    }

    #[test]
    fn test_pack_sector_alignment() {
        let mut pack = PackSectorBuffer::new(10);
        let payload1 = [0xAA; 500];
        let offset1 = pack.try_append(&payload1).expect("should fit");
        assert_eq!(offset1, 0);
        // 500 rounded up to 16-byte alignment is 512
        assert_eq!(pack.current_offset, 512);

        let payload2 = [0xBB; 600];
        let offset2 = pack.try_append(&payload2).expect("should fit");
        assert_eq!(offset2, 512);
        // 512 + 600 = 1112 rounded up to 16 is 1120
        assert_eq!(pack.current_offset, 1120);
    }
}
