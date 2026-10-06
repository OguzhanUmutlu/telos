//! Inventory container, item stacks, and deterministic click interaction logic.

use crate::crafting::find_recipe_2x2;
use bevy_ecs::component::Component;
use std::ops::Range;
use thiserror::Error;

/// Maximum number of items in a standard stackable item stack.
pub const MAX_STACK_SIZE: u16 = 64;

/// Total number of player inventory slots:
/// - 0..9: Hotbar (9 slots)
/// - 9..36: Main storage (27 slots)
/// - 36..40: Armor slots (4 slots: Helmet, Chestplate, Leggings, Boots)
/// - 40..44: 2x2 Crafting inputs (4 slots)
/// - 44: Crafting result (1 slot)
/// - 45: Offhand (1 slot)
pub const PLAYER_INVENTORY_SLOTS: usize = 46;

/// Hotbar slot range (0..9).
pub const HOTBAR_SLOTS: Range<usize> = 0..9;
/// Main inventory storage slot range (9..36).
pub const STORAGE_SLOTS: Range<usize> = 9..36;
/// Armor slot range (36..40).
pub const ARMOR_SLOTS: Range<usize> = 36..40;
/// Helmet armor slot index (36).
pub const ARMOR_HELMET_SLOT: usize = 36;
/// Chestplate armor slot index (37).
pub const ARMOR_CHESTPLATE_SLOT: usize = 37;
/// Leggings armor slot index (38).
pub const ARMOR_LEGGINGS_SLOT: usize = 38;
/// Boots armor slot index (39).
pub const ARMOR_BOOTS_SLOT: usize = 39;
/// 2x2 Crafting input slots range (40..44).
pub const CRAFTING_INPUT_SLOTS: Range<usize> = 40..44;
/// Crafting result output slot index (44).
pub const CRAFTING_RESULT_SLOT: usize = 44;
/// Offhand slot index (45).
pub const OFFHAND_SLOT: usize = 45;

/// Returns true if the item is a helmet.
#[must_use]
pub const fn is_helmet(item: u32) -> bool {
    item == 16
}

/// Returns true if the item is a chestplate.
#[must_use]
pub const fn is_chestplate(item: u32) -> bool {
    item == 17
}

/// Returns true if the item is leggings.
#[must_use]
pub const fn is_leggings(item: u32) -> bool {
    item == 18
}

/// Returns true if the item is boots.
#[must_use]
pub const fn is_boots(item: u32) -> bool {
    item == 31
}

/// Returns true if the item is any equippable armor item.
#[must_use]
pub const fn is_armor(item: u32) -> bool {
    is_helmet(item) || is_chestplate(item) || is_leggings(item) || is_boots(item)
}

/// Returns the matching armor slot index for an armor item, or None.
#[must_use]
pub const fn matching_armor_slot(item: u32) -> Option<usize> {
    if is_helmet(item) {
        Some(ARMOR_HELMET_SLOT)
    } else if is_chestplate(item) {
        Some(ARMOR_CHESTPLATE_SLOT)
    } else if is_leggings(item) {
        Some(ARMOR_LEGGINGS_SLOT)
    } else if is_boots(item) {
        Some(ARMOR_BOOTS_SLOT)
    } else {
        None
    }
}

/// Checks if a slot accepts the specified item.
#[must_use]
pub const fn is_slot_valid_for_item(slot: usize, item: u32) -> bool {
    if item == 0 {
        return true;
    }
    match slot {
        ARMOR_HELMET_SLOT => is_helmet(item),
        ARMOR_CHESTPLATE_SLOT => is_chestplate(item),
        ARMOR_LEGGINGS_SLOT => is_leggings(item),
        ARMOR_BOOTS_SLOT => is_boots(item),
        CRAFTING_RESULT_SLOT => false, // Cannot directly place items into crafting result!
        _ => true,
    }
}

/// Returns the user-facing name of an item ID.
#[must_use]
pub fn item_name(item: u32) -> &'static str {
    match item {
        0 => "Air",
        1 => "Stone",
        2 => "Dirt",
        3 => "Grass Block",
        4 => "Cobblestone",
        5 => "Oak Log",
        6 => "Water",
        7 => "Oak Planks",
        8 => "Oak Leaves",
        9 => "Glass",
        10 => "Stone Slab",
        11 => "Oak Stairs",
        12 => "Sand",
        13 => "Stick",
        14 => "Crafting Table",
        15 => "Torch",
        16 => "Iron Helmet",
        17 => "Iron Chestplate",
        18 => "Iron Leggings",
        19 => "Logic Wire",
        20 => "Powered Wire",
        21 => "Power Block",
        22 => "Lever",
        23 => "Lever (On)",
        24 => "Logic Lamp",
        25 => "Logic Lamp (Lit)",
        26 => "Logic Repeater",
        27 => "Logic Repeater (Powered)",
        28 => "Logic Inverter",
        29 => "Logic Inverter (Off)",
        30 => "Logic Diode",
        31 => "Iron Boots",
        _ => "Unknown Item",
    }
}

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
    /// Slot does not accept the item.
    #[error("Slot {0} does not accept item {1}")]
    InvalidItemForSlot(usize, u32),
}

/// Player inventory component containing 46 item slots and a cursor carried stack.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct Inventory {
    /// 46 slots:
    /// - 0..9: Hotbar (9 slots)
    /// - 9..36: Main storage (27 slots)
    /// - 36..40: Armor (4 slots)
    /// - 40..44: Crafting inputs (4 slots)
    /// - 44: Crafting result (1 slot)
    /// - 45: Offhand (1 slot)
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

    /// Updates the crafting result slot (44) based on crafting input slots (40..44).
    pub fn update_crafting(&mut self) {
        let inputs = [
            self.slots[40],
            self.slots[41],
            self.slots[42],
            self.slots[43],
        ];
        self.slots[CRAFTING_RESULT_SLOT] = find_recipe_2x2(&inputs).unwrap_or(ItemStack::EMPTY);
    }

    /// Inserts as much of `stack` as possible into the specified slot range.
    ///
    /// Returns the number of items successfully inserted.
    pub fn insert_into_range(&mut self, stack: &mut ItemStack, range: Range<usize>) -> u16 {
        if stack.is_empty() {
            return 0;
        }
        let initial_count = stack.count;

        // 1. Merge into existing matching stacks
        for i in range.clone() {
            let target = &mut self.slots[i];
            if target.item == stack.item && target.count < MAX_STACK_SIZE {
                let space = MAX_STACK_SIZE - target.count;
                let to_move = stack.count.min(space);
                target.count += to_move;
                stack.count -= to_move;
                if stack.count == 0 {
                    stack.normalize();
                    return initial_count;
                }
            }
        }

        // 2. Place remainder into empty slots
        for i in range {
            let target = &mut self.slots[i];
            if target.is_empty() {
                *target = *stack;
                stack.count = 0;
                stack.normalize();
                return initial_count;
            }
        }

        stack.normalize();
        initial_count - stack.count
    }

    /// Inserts a stack into main storage (9..36) first, then hotbar (0..9).
    pub fn insert_into_storage_or_hotbar(&mut self, stack: &mut ItemStack) -> u16 {
        let mut inserted = self.insert_into_range(stack, STORAGE_SLOTS);
        if !stack.is_empty() {
            inserted += self.insert_into_range(stack, HOTBAR_SLOTS);
        }
        inserted
    }

    /// Returns any items remaining in crafting inputs (40..44) into storage or hotbar.
    ///
    /// Clears the crafting result slot (44).
    pub fn return_crafting_items(&mut self) {
        for i in CRAFTING_INPUT_SLOTS {
            if !self.slots[i].is_empty() {
                let mut stack = self.slots[i];
                self.insert_into_storage_or_hotbar(&mut stack);
                self.slots[i] = stack; // If full, whatever remains stays
            }
        }
        self.update_crafting();
    }

    /// Returns carried cursor stack into storage or hotbar.
    pub fn return_carried(&mut self) {
        if !self.carried.is_empty() {
            let mut stack = self.carried;
            self.insert_into_storage_or_hotbar(&mut stack);
            self.carried = stack;
        }
    }

    /// Returns both crafting items and carried cursor stack into storage or hotbar.
    pub fn return_crafting_and_carried(&mut self) {
        self.return_carried();
        self.return_crafting_items();
    }
}

/// Executes a deterministic inventory click on an inventory container.
///
/// Modifies the target slot and the `carried` cursor stack according to standard rules.
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

    // Special handling: Crafting Result Slot (44)
    if slot_idx == CRAFTING_RESULT_SLOT {
        handle_crafting_result_click(inv, button, mode);
        return Ok(());
    }

    match mode {
        ClickMode::Pickup => handle_pickup_click(inv, slot_idx, button)?,
        ClickMode::QuickMove => handle_quick_move_click(inv, slot_idx),
        ClickMode::SwapHotbar => handle_swap_hotbar_click(inv, slot_idx, button)?,
        ClickMode::Drop => handle_drop_click(inv, slot_idx, button),
    }

    // Re-evaluate crafting if any crafting input slot changed
    if CRAFTING_INPUT_SLOTS.contains(&slot_idx) {
        inv.update_crafting();
    }

    Ok(())
}

/// Handles clicking on the crafting result slot (44).
fn handle_crafting_result_click(inv: &mut Inventory, _button: ClickButton, mode: ClickMode) {
    let result = inv.slots[CRAFTING_RESULT_SLOT];
    if result.is_empty() {
        return;
    }

    match mode {
        ClickMode::Pickup => {
            // Can only take result if carried is empty or matches result item and can hold it
            if inv.carried.is_empty() {
                inv.carried = result;
            } else if inv.carried.item == result.item
                && inv.carried.count + result.count <= MAX_STACK_SIZE
            {
                inv.carried.count += result.count;
            } else {
                return; // Cannot take result
            }

            // Deduct 1 item from each non-empty crafting input slot
            for i in CRAFTING_INPUT_SLOTS {
                if !inv.slots[i].is_empty() {
                    inv.slots[i].count -= 1;
                    inv.slots[i].normalize();
                }
            }
            inv.update_crafting();
        }
        ClickMode::QuickMove => {
            // Shift-click: craft as many times as possible into storage/hotbar
            for _ in 0..64 {
                let res = inv.slots[CRAFTING_RESULT_SLOT];
                if res.is_empty() {
                    break;
                }
                let mut to_insert = res;
                let inserted = inv.insert_into_storage_or_hotbar(&mut to_insert);
                if inserted < res.count {
                    // Could not insert full result: roll back partial insert if needed
                    break;
                }

                // Deduct 1 item from each crafting input
                for i in CRAFTING_INPUT_SLOTS {
                    if !inv.slots[i].is_empty() {
                        inv.slots[i].count -= 1;
                        inv.slots[i].normalize();
                    }
                }
                inv.update_crafting();
            }
        }
        ClickMode::SwapHotbar | ClickMode::Drop => {
            // Not supported on crafting result
        }
    }
}

/// Handles standard pickup click (left or right click).
fn handle_pickup_click(
    inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
) -> Result<(), InventoryError> {
    match button {
        ClickButton::Left => {
            let slot = inv.slots[slot_idx];
            if inv.carried.is_empty() {
                // Pick up entire stack
                inv.carried = slot;
                inv.slots[slot_idx] = ItemStack::EMPTY;
            } else if slot.is_empty() {
                // Check if target slot accepts carried item (e.g. armor slots)
                if !is_slot_valid_for_item(slot_idx, inv.carried.item) {
                    return Err(InventoryError::InvalidItemForSlot(
                        slot_idx,
                        inv.carried.item,
                    ));
                }
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
                // Swap slot and carried - verify carried item is valid for this slot
                if !is_slot_valid_for_item(slot_idx, inv.carried.item) {
                    return Err(InventoryError::InvalidItemForSlot(
                        slot_idx,
                        inv.carried.item,
                    ));
                }
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
                // Check if target slot accepts carried item
                if !is_slot_valid_for_item(slot_idx, inv.carried.item) {
                    return Err(InventoryError::InvalidItemForSlot(
                        slot_idx,
                        inv.carried.item,
                    ));
                }
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
                if !is_slot_valid_for_item(slot_idx, inv.carried.item) {
                    return Err(InventoryError::InvalidItemForSlot(
                        slot_idx,
                        inv.carried.item,
                    ));
                }
                inv.slots[slot_idx] = inv.carried;
                inv.carried = slot;
            }
        }
    }
    Ok(())
}

/// Handles shift-click quick move transfer.
fn handle_quick_move_click(inv: &mut Inventory, slot_idx: usize) {
    let mut source = inv.slots[slot_idx];
    if source.is_empty() {
        return;
    }

    if HOTBAR_SLOTS.contains(&slot_idx) {
        // From Hotbar: check if armor first
        if let Some(armor_slot) = matching_armor_slot(source.item)
            && inv.slots[armor_slot].is_empty()
        {
            inv.slots[armor_slot] = ItemStack::new(source.item, 1);
            source.count -= 1;
            source.normalize();
            inv.slots[slot_idx] = source;
            return;
        }
        // Move to main storage
        inv.insert_into_range(&mut source, STORAGE_SLOTS);
        inv.slots[slot_idx] = source;
    } else if STORAGE_SLOTS.contains(&slot_idx) {
        // From Storage: check if armor first
        if let Some(armor_slot) = matching_armor_slot(source.item)
            && inv.slots[armor_slot].is_empty()
        {
            inv.slots[armor_slot] = ItemStack::new(source.item, 1);
            source.count -= 1;
            source.normalize();
            inv.slots[slot_idx] = source;
            return;
        }
        // Move to hotbar
        inv.insert_into_range(&mut source, HOTBAR_SLOTS);
        inv.slots[slot_idx] = source;
    } else if ARMOR_SLOTS.contains(&slot_idx)
        || CRAFTING_INPUT_SLOTS.contains(&slot_idx)
        || slot_idx == OFFHAND_SLOT
    {
        // From Armor, Crafting Inputs, or Offhand: move to storage then hotbar
        inv.insert_into_storage_or_hotbar(&mut source);
        inv.slots[slot_idx] = source;
    }
}

/// Handles number key hotbar swap.
fn handle_swap_hotbar_click(
    inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
) -> Result<(), InventoryError> {
    let hotbar_slot = match button {
        ClickButton::Left => inv.selected_slot.min(8),
        ClickButton::Right => 0,
    };

    // If slot is an armor slot, check compatibility
    if ARMOR_SLOTS.contains(&slot_idx) {
        let hotbar_item = inv.slots[hotbar_slot].item;
        if !is_slot_valid_for_item(slot_idx, hotbar_item) {
            return Err(InventoryError::InvalidItemForSlot(slot_idx, hotbar_item));
        }
    }

    inv.slots.swap(slot_idx, hotbar_slot);
    Ok(())
}

/// Handles dropping items.
fn handle_drop_click(inv: &mut Inventory, slot_idx: usize, button: ClickButton) {
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

    #[test]
    fn test_crafting_interaction() {
        let mut inv = Inventory::default();
        // Place 1 oak log (5) in crafting input slot 40
        inv.slots[40] = ItemStack::new(5, 1);
        inv.update_crafting();
        assert_eq!(inv.slots[CRAFTING_RESULT_SLOT], ItemStack::new(7, 4)); // 4 planks

        // Click on result slot 44: picks up 4 planks, consumes 1 log
        inventory_click(
            &mut inv,
            CRAFTING_RESULT_SLOT,
            ClickButton::Left,
            ClickMode::Pickup,
        )
        .unwrap();
        assert_eq!(inv.carried, ItemStack::new(7, 4));
        assert_eq!(inv.slots[40], ItemStack::EMPTY);
        assert_eq!(inv.slots[CRAFTING_RESULT_SLOT], ItemStack::EMPTY);
    }

    #[test]
    fn test_armor_slot_validation() {
        let mut inv = Inventory {
            carried: ItemStack::new(1, 1),
            ..Default::default()
        };
        let res = inventory_click(
            &mut inv,
            ARMOR_HELMET_SLOT,
            ClickButton::Left,
            ClickMode::Pickup,
        );
        assert!(res.is_err());

        // Carried iron helmet (16): can place in helmet slot 36
        inv.carried = ItemStack::new(16, 1);
        let res = inventory_click(
            &mut inv,
            ARMOR_HELMET_SLOT,
            ClickButton::Left,
            ClickMode::Pickup,
        );
        assert!(res.is_ok());
        assert_eq!(inv.slots[ARMOR_HELMET_SLOT], ItemStack::new(16, 1));
        assert_eq!(inv.carried, ItemStack::EMPTY);
    }

    #[test]
    fn test_quick_move_auto_equips_armor() {
        let mut inv = Inventory::default();
        // Iron chestplate in hotbar slot 0
        inv.slots[0] = ItemStack::new(17, 1);
        inventory_click(&mut inv, 0, ClickButton::Left, ClickMode::QuickMove).unwrap();
        assert_eq!(inv.slots[0], ItemStack::EMPTY);
        assert_eq!(inv.slots[ARMOR_CHESTPLATE_SLOT], ItemStack::new(17, 1));

        // Shift click armor slot: moves back to storage/hotbar
        inventory_click(
            &mut inv,
            ARMOR_CHESTPLATE_SLOT,
            ClickButton::Left,
            ClickMode::QuickMove,
        )
        .unwrap();
        assert_eq!(inv.slots[ARMOR_CHESTPLATE_SLOT], ItemStack::EMPTY);
        assert_eq!(inv.slots[9], ItemStack::new(17, 1)); // Into first storage slot
    }

    #[test]
    fn test_return_crafting_items_on_close() {
        let mut inv = Inventory::default();
        inv.slots[40] = ItemStack::new(7, 2); // 2 planks in input
        inv.slots[42] = ItemStack::new(7, 2); // 2 planks in input
        inv.update_crafting();
        assert_eq!(inv.slots[CRAFTING_RESULT_SLOT], ItemStack::new(13, 4)); // 4 sticks

        inv.return_crafting_and_carried();
        assert_eq!(inv.slots[40], ItemStack::EMPTY);
        assert_eq!(inv.slots[42], ItemStack::EMPTY);
        assert_eq!(inv.slots[CRAFTING_RESULT_SLOT], ItemStack::EMPTY);
        assert_eq!(inv.slots[9], ItemStack::new(7, 4)); // Combined into storage
    }
}
