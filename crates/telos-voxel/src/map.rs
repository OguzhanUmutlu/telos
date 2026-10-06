//! Generational chunk storage arena and concurrent snapshot publishing map.

use arc_swap::ArcSwapOption;
use hashbrown::HashMap;
use std::sync::Arc;
use telos_core::coords::ChunkPos;

use crate::chunk::{Chunk, ChunkSnapshot};

/// An ABA-safe generational handle referencing an allocated chunk slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkHandle {
    index: u32,
    generation: u32,
}

impl ChunkHandle {
    /// Slot index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Slot generation.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

struct Slot {
    generation: u32,
    chunk: Option<Chunk>,
    snapshot: ArcSwapOption<ChunkSnapshot>,
}

/// Storage arena managing chunks and publishing lock-free immutable snapshots.
pub struct ChunkMap {
    slots: Vec<Slot>,
    free_indices: Vec<u32>,
    by_pos: HashMap<ChunkPos, ChunkHandle>,
}

impl Default for ChunkMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkMap {
    /// Creates an empty chunk map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
            by_pos: HashMap::new(),
        }
    }

    /// Inserts or replaces a chunk at its position, publishing an initial snapshot.
    pub fn insert(&mut self, mut chunk: Chunk) -> ChunkHandle {
        let pos = chunk.position();
        let snapshot = chunk.publish_snapshot();

        if let Some(existing_handle) = self.by_pos.get(&pos).copied() {
            let slot = &mut self.slots[existing_handle.index as usize];
            slot.chunk = Some(chunk);
            slot.snapshot.store(Some(snapshot));
            return existing_handle;
        }

        let index = if let Some(free_idx) = self.free_indices.pop() {
            let slot = &mut self.slots[free_idx as usize];
            slot.generation = slot.generation.wrapping_add(1);
            slot.chunk = Some(chunk);
            slot.snapshot.store(Some(snapshot));
            free_idx
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                chunk: Some(chunk),
                snapshot: ArcSwapOption::from(Some(snapshot)),
            });
            idx
        };

        let generation = self.slots[index as usize].generation;
        let handle = ChunkHandle { index, generation };
        self.by_pos.insert(pos, handle);

        handle
    }

    /// Removes a chunk at `pos`, invalidating any existing handles.
    pub fn remove(&mut self, pos: ChunkPos) -> Option<Chunk> {
        let handle = self.by_pos.remove(&pos)?;
        let slot = &mut self.slots[handle.index as usize];
        if slot.generation != handle.generation {
            return None;
        }

        slot.generation = slot.generation.wrapping_add(1);
        slot.snapshot.store(None);
        self.free_indices.push(handle.index);

        slot.chunk.take()
    }

    /// Obtains a mutable reference to the active `Chunk` at `pos`.
    pub fn get_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        let handle = self.by_pos.get(&pos)?;
        let slot = &mut self.slots[handle.index as usize];
        if slot.generation == handle.generation {
            slot.chunk.as_mut()
        } else {
            None
        }
    }

    /// Retrieves the published immutable `Arc<ChunkSnapshot>` for `pos` lock-free.
    #[must_use]
    pub fn get_snapshot(&self, pos: ChunkPos) -> Option<Arc<ChunkSnapshot>> {
        let handle = self.by_pos.get(&pos)?;
        self.get_snapshot_by_handle(*handle)
    }

    /// Retrieves the published immutable `Arc<ChunkSnapshot>` by `ChunkHandle` lock-free.
    #[must_use]
    pub fn get_snapshot_by_handle(&self, handle: ChunkHandle) -> Option<Arc<ChunkSnapshot>> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation == handle.generation {
            slot.snapshot.load_full()
        } else {
            None
        }
    }

    /// Publishes snapshots for all modified chunks and resets their dirty status.
    pub fn publish_dirty(&mut self) -> usize {
        let mut count = 0;
        for slot in &mut self.slots {
            if let Some(chunk) = &mut slot.chunk
                && chunk.is_dirty()
            {
                let snapshot = chunk.publish_snapshot();
                slot.snapshot.store(Some(snapshot));
                count += 1;
            }
        }
        count
    }

    /// Total number of loaded chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_pos.len()
    }

    /// Returns `true` if map contains zero loaded chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_pos.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BlockStateId;

    #[test]
    fn test_chunk_map_lifecycle() {
        let mut map = ChunkMap::new();
        let pos = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new_uniform(pos, BlockStateId::AIR, false);

        let handle = map.insert(chunk);
        assert_eq!(map.len(), 1);

        let snap = map.get_snapshot(pos).unwrap();
        assert_eq!(snap.position(), pos);

        // Remove
        let removed = map.remove(pos).unwrap();
        assert_eq!(removed.position(), pos);
        assert_eq!(map.len(), 0);

        // Previous handle is now stale and returns None
        assert!(map.get_snapshot_by_handle(handle).is_none());
    }
}
