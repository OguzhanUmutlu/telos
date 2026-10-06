//! Inventory container, item stacks, and deterministic click interaction logic.

use bevy_ecs::component::Component;
use thiserror::Error;

/// Maximum number of items in a standard stackable item stack.
pub const MAX_STACK_SIZE: u16 = 64;

/// Total number of player inventory slots (9 hotbar + 27 main inventory).
pub const PLAYER_INVENTORY_SLOTS: usize = 36;

/// A stack of items in an inventory container or carried by the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ItemStack {
    /// Item identifier (0 = Air / Empty).
    pub item: u32,
    /// Number of items in this stack (0 = Empty).
    pub count: u16,
}

impl ItemStack {
    /// Empty item stack constant.
    pub const EMPTY: Self = Self { item: 0, count: 0 };

    /// Creates a new item stack, normalizing 0-count to Air.
    #[must_use]
    pub const fn new(item: u32, count: u16) -> Self {
        if item == 0 || count == 0 {
            Self::EMPTY
        } else {
            Self { item, count }
        }
    }

    /// Whether this item stack is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.item == 0 || self.count == 0
    }

    /// Normalizes empty states so `count == 0` strictly implies `item == 0`.
    pub fn normalize(&mut self) {
        if self.item == 0 || self.count == 0 {
            self.item = 0;
            self.count = 0;
        }
    }

    /// Checks if this stack can combine with another stack without exceeding the limit.
    #[must_use]
    pub fn can_stack(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.item == other.item && self.count + other.count <= MAX_STACK_SIZE
    }
}

/// Mouse button used during an inventory click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickButton {
    /// Left mouse button (0).
    Left,
    /// Right mouse button (1).
    Right,
}

/// Click interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickMode {
    /// Normal pickup, placement, or stack merging (left/right click).
    Pickup,
    /// Quick transfer between hotbar and main inventory (shift-click).
    QuickMove,
    /// Swap with hotbar slot via number key.
    SwapHotbar,
    /// Drop item stack into world.
    Drop,
}

/// Errors returned by inventory manipulation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InventoryError {
    /// Slot index was out of bounds for the container.
    #[error("Slot index {0} is out of bounds")]
    SlotOutOfBounds(usize),
}

/// Player inventory component containing 36 item slots and a cursor carried stack.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct Inventory {
    /// 36 slots: 0..=8 (hotbar), 9..=35 (main inventory storage).
    pub slots: [ItemStack; PLAYER_INVENTORY_SLOTS],
    /// Currently selected hotbar slot index (0..=8).
    pub selected_slot: usize,
    /// Item stack currently held on the cursor.
    pub carried: ItemStack,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            slots: [ItemStack::EMPTY; PLAYER_INVENTORY_SLOTS],
            selected_slot: 0,
            carried: ItemStack::EMPTY,
        }
    }
}

impl Inventory {
    /// Returns the currently active item stack selected on the hotbar.
    #[must_use]
    pub fn selected_item(&self) -> &ItemStack {
        let idx = self.selected_slot.min(8);
        &self.slots[idx]
    }

    /// Gets a reference to an inventory slot.
    #[must_use]
    pub fn get(&self, slot: usize) -> Option<&ItemStack> {
        self.slots.get(slot)
    }

    /// Gets a mutable reference to an inventory slot.
    pub fn get_mut(&mut self, slot: usize) -> Option<&mut ItemStack> {
        self.slots.get_mut(slot)
    }
}

/// Executes a deterministic inventory click on an inventory container.
///
/// Modifies the target slot and the `carried` cursor stack according to the standard rules.
#[allow(clippy::too_many_lines)]
pub fn inventory_click(
    inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
    mode: ClickMode,
) -> Result<(), InventoryError> {
    if slot_idx >= PLAYER_INVENTORY_SLOTS {
        return Err(InventoryError::SlotOutOfBounds(slot_idx));
    }

    match mode {
        ClickMode::Pickup => match button {
            ClickButton::Left => {
                let slot = inv.slots[slot_idx];
                if inv.carried.is_empty() {
                    // Pick up entire stack
                    inv.carried = slot;
                    inv.slots[slot_idx] = ItemStack::EMPTY;
                } else if slot.is_empty() {
                    // Place entire stack
                    inv.slots[slot_idx] = inv.carried;
                    inv.carried = ItemStack::EMPTY;
                } else if slot.item == inv.carried.item {
                    // Merge stacks up to 64
                    let space = MAX_STACK_SIZE.saturating_sub(slot.count);
                    let to_move = inv.carried.count.min(space);
                    inv.slots[slot_idx].count += to_move;
                    inv.carried.count -= to_move;
                    inv.carried.normalize();
                } else {
                    // Swap slot and carried
                    inv.slots[slot_idx] = inv.carried;
                    inv.carried = slot;
                }
            }
            ClickButton::Right => {
                let slot = inv.slots[slot_idx];
                if inv.carried.is_empty() {
                    if !slot.is_empty() {
                        // Take half of the slot (rounded up)
                        let take = slot.count.div_ceil(2);
                        inv.carried = ItemStack::new(slot.item, take);
                        inv.slots[slot_idx].count -= take;
                        inv.slots[slot_idx].normalize();
                    }
                } else if slot.is_empty() {
                    // Place 1 item into empty slot
                    inv.slots[slot_idx] = ItemStack::new(inv.carried.item, 1);
                    inv.carried.count -= 1;
                    inv.carried.normalize();
                } else if slot.item == inv.carried.item && slot.count < MAX_STACK_SIZE {
                    // Place 1 item into matching slot
                    inv.slots[slot_idx].count += 1;
                    inv.carried.count -= 1;
                    inv.carried.normalize();
                } else {
                    // Swap slot and carried
                    inv.slots[slot_idx] = inv.carried;
                    inv.carried = slot;
                }
            }
        },
        ClickMode::QuickMove => {
            // Shift-click: transfer between hotbar (0..8) and storage (9..35)
            let is_hotbar = slot_idx < 9;
            let (target_start, target_end) = if is_hotbar {
                (9, PLAYER_INVENTORY_SLOTS)
            } else {
                (0, 9)
            };

            let mut source = inv.slots[slot_idx];
            if source.is_empty() {
                return Ok(());
            }

            // 1. Try to merge into existing matching stacks
            for i in target_start..target_end {
                let target = &mut inv.slots[i];
                if target.item == source.item && target.count < MAX_STACK_SIZE {
                    let space = MAX_STACK_SIZE - target.count;
                    let move_count = source.count.min(space);
                    target.count += move_count;
                    source.count -= move_count;
                    if source.count == 0 {
                        break;
                    }
                }
            }

            // 2. If remainder, place into first empty slot
            if source.count > 0 {
                for i in target_start..target_end {
                    let target = &mut inv.slots[i];
                    if target.is_empty() {
                        *target = source;
                        source = ItemStack::EMPTY;
                        break;
                    }
                }
            }

            source.normalize();
            inv.slots[slot_idx] = source;
        }
        ClickMode::SwapHotbar => {
            // Swap with hotbar slot
            let hotbar_slot = match button {
                ClickButton::Left => inv.selected_slot.min(8),
                ClickButton::Right => 0,
            };
            inv.slots.swap(slot_idx, hotbar_slot);
        }
        ClickMode::Drop => {
            // Drop item: for now, clear slot (or 1 item if right click)
            match button {
                ClickButton::Left => {
                    inv.slots[slot_idx] = ItemStack::EMPTY;
                }
                ClickButton::Right => {
                    if inv.slots[slot_idx].count > 1 {
                        inv.slots[slot_idx].count -= 1;
                    } else {
                        inv.slots[slot_idx] = ItemStack::EMPTY;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_left_click_pickup_and_place() {
        let mut inv = Inventory::default();
        inv.slots[0] = ItemStack::new(1, 10); // 10 stone in slot 0

        // Left click on slot 0: picks up all 10 stone
        inventory_click(&mut inv, 0, ClickButton::Left, ClickMode::Pickup).unwrap();
        assert_eq!(inv.slots[0], ItemStack::EMPTY);
        assert_eq!(inv.carried, ItemStack::new(1, 10));

        // Left click on slot 1: places all 10 stone
        inventory_click(&mut inv, 1, ClickButton::Left, ClickMode::Pickup).unwrap();
        assert_eq!(inv.slots[1], ItemStack::new(1, 10));
        assert_eq!(inv.carried, ItemStack::EMPTY);
    }

    #[test]
    fn test_right_click_split_and_place_one() {
        let mut inv = Inventory::default();
        inv.slots[0] = ItemStack::new(1, 7); // 7 stone

        // Right click slot 0: takes half (4 stone into carried, 3 left in slot)
        inventory_click(&mut inv, 0, ClickButton::Right, ClickMode::Pickup).unwrap();
        assert_eq!(inv.slots[0], ItemStack::new(1, 3));
        assert_eq!(inv.carried, ItemStack::new(1, 4));

        // Right click empty slot 1: places 1 stone (carried now 3, slot 1 has 1)
        inventory_click(&mut inv, 1, ClickButton::Right, ClickMode::Pickup).unwrap();
        assert_eq!(inv.slots[1], ItemStack::new(1, 1));
        assert_eq!(inv.carried, ItemStack::new(1, 3));
    }

    #[test]
    fn test_quick_move_hotbar_to_storage() {
        let mut inv = Inventory::default();
        inv.slots[0] = ItemStack::new(2, 64); // 64 dirt in hotbar slot 0

        // Shift click slot 0: moves into main storage slot 9
        inventory_click(&mut inv, 0, ClickButton::Left, ClickMode::QuickMove).unwrap();
        assert_eq!(inv.slots[0], ItemStack::EMPTY);
        assert_eq!(inv.slots[9], ItemStack::new(2, 64));

        // Shift click slot 9: moves back to hotbar slot 0
        inventory_click(&mut inv, 9, ClickButton::Left, ClickMode::QuickMove).unwrap();
        assert_eq!(inv.slots[9], ItemStack::EMPTY);
        assert_eq!(inv.slots[0], ItemStack::new(2, 64));
    }
}
