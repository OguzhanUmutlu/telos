//! Block entity side table, kinds, and persistent slot container data.

use crate::coords::LocalIdx;
use hashbrown::HashMap;

/// Number of item slots in a standard chest block entity.
pub const CHEST_CONTAINER_SLOTS: usize = 27;

/// Number of item slots in a furnace block entity (Input, Fuel, Output).
pub const FURNACE_CONTAINER_SLOTS: usize = 3;
/// Index of the smelting ingredient input slot.
pub const FURNACE_SLOT_INPUT: usize = 0;
/// Index of the combustible fuel slot.
pub const FURNACE_SLOT_FUEL: usize = 1;
/// Index of the smelted result output slot.
pub const FURNACE_SLOT_OUTPUT: usize = 2;

/// Discrete classification of block entity kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockEntityKind {
    /// Standard 27-slot storage chest.
    Chest,
    /// 3-slot smelting furnace.
    Furnace,
}

/// A single item slot within a block entity container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockEntitySlot {
    /// Slot index within the container (0..27).
    pub slot: u8,
    /// Item identifier (0 = Air / empty).
    pub item: u32,
    /// Stack count (0..=64).
    pub count: u16,
}

impl BlockEntitySlot {
    /// Creates an empty slot.
    pub const EMPTY: Self = Self {
        slot: 0,
        item: 0,
        count: 0,
    };

    /// Creates a slot with the specified item and count.
    #[must_use]
    pub const fn new(slot: u8, item: u32, count: u16) -> Self {
        if item == 0 || count == 0 {
            Self {
                slot,
                item: 0,
                count: 0,
            }
        } else {
            Self { slot, item, count }
        }
    }

    /// Whether this slot is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.item == 0 || self.count == 0
    }
}

/// Discrete data payload associated with a block entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockEntityData {
    /// Storage chest holding up to 27 item slots.
    Chest {
        /// Optional custom container name (e.g. renamed via anvil).
        custom_name: Option<String>,
        /// 27 inventory slots.
        items: [BlockEntitySlot; CHEST_CONTAINER_SLOTS],
    },
    /// Smelting furnace holding input, fuel, and output slots.
    Furnace {
        /// Optional custom container name (e.g. renamed via anvil).
        custom_name: Option<String>,
        /// 3 inventory slots (input, fuel, output).
        items: [BlockEntitySlot; FURNACE_CONTAINER_SLOTS],
        /// Ticks remaining for active fuel combustion (0 = not burning).
        burn_time_remaining: u16,
        /// Total burn duration of the currently consumed fuel item.
        total_burn_time: u16,
        /// Cook progress ticks for current smelting item (`0..cook_duration`).
        cook_progress: u16,
        /// Cook duration required to complete smelting (default 200 ticks = 10s).
        cook_duration: u16,
    },
}

impl BlockEntityData {
    /// Creates a new default empty chest block entity.
    #[must_use]
    pub fn new_chest() -> Self {
        let mut items = [BlockEntitySlot::EMPTY; CHEST_CONTAINER_SLOTS];
        #[allow(clippy::cast_possible_truncation)]
        for (i, slot) in items.iter_mut().enumerate() {
            slot.slot = i as u8;
        }
        Self::Chest {
            custom_name: None,
            items,
        }
    }

    /// Creates a new default empty furnace block entity.
    #[must_use]
    pub fn new_furnace() -> Self {
        let mut items = [BlockEntitySlot::EMPTY; FURNACE_CONTAINER_SLOTS];
        #[allow(clippy::cast_possible_truncation)]
        for (i, slot) in items.iter_mut().enumerate() {
            slot.slot = i as u8;
        }
        Self::Furnace {
            custom_name: None,
            items,
            burn_time_remaining: 0,
            total_burn_time: 0,
            cook_progress: 0,
            cook_duration: 200,
        }
    }

    /// The kind of this block entity.
    #[must_use]
    pub const fn kind(&self) -> BlockEntityKind {
        match self {
            Self::Chest { .. } => BlockEntityKind::Chest,
            Self::Furnace { .. } => BlockEntityKind::Furnace,
        }
    }

    /// Returns a slice of item slots if this block entity is a container.
    #[must_use]
    pub fn items(&self) -> &[BlockEntitySlot] {
        match self {
            Self::Chest { items, .. } => items,
            Self::Furnace { items, .. } => items,
        }
    }

    /// Returns a mutable slice of item slots if this block entity is a container.
    pub fn items_mut(&mut self) -> &mut [BlockEntitySlot] {
        match self {
            Self::Chest { items, .. } => items,
            Self::Furnace { items, .. } => items,
        }
    }

    /// Returns the custom name of this block entity, if present.
    #[must_use]
    pub fn custom_name(&self) -> Option<&str> {
        match self {
            Self::Chest { custom_name, .. } | Self::Furnace { custom_name, .. } => {
                custom_name.as_deref()
            }
        }
    }

    /// Returns `true` if this block entity is actively combusting fuel.
    #[must_use]
    pub const fn is_burning(&self) -> bool {
        match self {
            Self::Chest { .. } => false,
            Self::Furnace {
                burn_time_remaining,
                ..
            } => *burn_time_remaining > 0,
        }
    }
}

/// Per-chunk side table mapping local voxel indices to their associated block entity data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockEntityTable {
    entries: HashMap<LocalIdx, BlockEntityData>,
}

impl BlockEntityTable {
    /// Creates an empty block entity table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Returns the number of block entities in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether this table has no block entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a reference to the block entity at `idx`, if present.
    #[must_use]
    pub fn get(&self, idx: LocalIdx) -> Option<&BlockEntityData> {
        self.entries.get(&idx)
    }

    /// Returns a mutable reference to the block entity at `idx`, if present.
    pub fn get_mut(&mut self, idx: LocalIdx) -> Option<&mut BlockEntityData> {
        self.entries.get_mut(&idx)
    }

    /// Inserts or replaces the block entity at `idx`.
    pub fn insert(&mut self, idx: LocalIdx, data: BlockEntityData) -> Option<BlockEntityData> {
        self.entries.insert(idx, data)
    }

    /// Removes and returns the block entity at `idx`, if present.
    pub fn remove(&mut self, idx: LocalIdx) -> Option<BlockEntityData> {
        self.entries.remove(&idx)
    }

    /// Iterates over all block entities and their local voxel indices.
    pub fn iter(&self) -> impl Iterator<Item = (&LocalIdx, &BlockEntityData)> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chest_block_entity_creation_and_slots() {
        let mut chest = BlockEntityData::new_chest();
        assert_eq!(chest.kind(), BlockEntityKind::Chest);
        assert_eq!(chest.items().len(), 27);
        assert!(chest.items()[0].is_empty());

        chest.items_mut()[0] = BlockEntitySlot::new(0, 15, 64);
        assert!(!chest.items()[0].is_empty());
        assert_eq!(chest.items()[0].item, 15);
        assert_eq!(chest.items()[0].count, 64);
    }

    #[test]
    fn test_furnace_block_entity_creation_and_slots() {
        let mut furnace = BlockEntityData::new_furnace();
        assert_eq!(furnace.kind(), BlockEntityKind::Furnace);
        assert_eq!(furnace.items().len(), 3);
        assert!(furnace.items()[0].is_empty());
        assert!(!furnace.is_burning());

        furnace.items_mut()[FURNACE_SLOT_INPUT] = BlockEntitySlot::new(0, 50, 8); // 8 Raw beef
        furnace.items_mut()[FURNACE_SLOT_FUEL] = BlockEntitySlot::new(1, 54, 4); // 4 Coal
        furnace.items_mut()[FURNACE_SLOT_OUTPUT] = BlockEntitySlot::new(2, 66, 2); // 2 Cooked beef

        assert_eq!(furnace.items()[0].item, 50);
        assert_eq!(furnace.items()[0].count, 8);
        assert_eq!(furnace.items()[1].item, 54);
        assert_eq!(furnace.items()[1].count, 4);
        assert_eq!(furnace.items()[2].item, 66);
        assert_eq!(furnace.items()[2].count, 2);

        if let BlockEntityData::Furnace {
            burn_time_remaining,
            ..
        } = &mut furnace
        {
            *burn_time_remaining = 1600;
        }
        assert!(furnace.is_burning());
    }

    #[test]
    fn test_block_entity_table_operations() {
        let mut table = BlockEntityTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);

        let idx = LocalIdx::from_coords(2, 4, 6).unwrap();
        let chest = BlockEntityData::new_chest();
        assert!(table.insert(idx, chest).is_none());
        assert!(!table.is_empty());
        assert_eq!(table.len(), 1);

        assert!(table.get(idx).is_some());
        assert!(table.get_mut(idx).is_some());

        let removed = table.remove(idx);
        assert!(removed.is_some());
        assert!(table.is_empty());
    }
}
