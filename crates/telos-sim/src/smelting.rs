//! Smelting recipes, fuel combustion, and deterministic furnace container interactions.

use crate::inventory::{
    ClickButton, ClickMode, Inventory, InventoryError, ItemStack, MAX_STACK_SIZE,
};
use bevy_ecs::component::Component;
use hashbrown::HashMap;
use std::ops::Range;
use telos_voxel::block_entity::{
    BlockEntityData, BlockEntitySlot, FURNACE_CONTAINER_SLOTS, FURNACE_SLOT_FUEL,
    FURNACE_SLOT_INPUT, FURNACE_SLOT_OUTPUT,
};

/// Total number of item slots in a dual furnace container window (3 furnace + 27 storage + 9 hotbar = 39).
pub const DUAL_FURNACE_SLOT_COUNT: usize = 39;

/// Furnace container slot range (0..3).
pub const CONTAINER_FURNACE_SLOTS: Range<usize> = 0..3;
/// Player storage slot range in dual furnace container window (3..30).
pub const CONTAINER_FURNACE_STORAGE_SLOTS: Range<usize> = 3..30;
/// Player hotbar slot range in dual furnace container window (30..39).
pub const CONTAINER_FURNACE_HOTBAR_SLOTS: Range<usize> = 30..39;

/// A single smelting recipe definition mapping input ingredient to smelted output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmeltingRecipe {
    /// Input item identifier.
    pub input_item: u32,
    /// Smelted output item identifier.
    pub output_item: u32,
    /// Number of output items produced per cook operation (typically 1).
    pub output_count: u16,
    /// Cook duration in simulation ticks (default 200 ticks = 10.0 seconds).
    pub cook_duration: u16,
    /// Experience reward earned when retrieving smelted items.
    pub experience: f32,
}

/// Registry storing all deterministic smelting recipes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SmeltingRegistry {
    recipes: HashMap<u32, SmeltingRecipe>,
}

impl SmeltingRegistry {
    /// Creates a new empty smelting registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            recipes: HashMap::new(),
        }
    }

    /// Creates a standard registry populated with default survival recipes.
    #[must_use]
    pub fn standard() -> Self {
        let mut reg = Self::new();
        // Raw Porkchop (49) -> Cooked Porkchop (65)
        reg.register(SmeltingRecipe {
            input_item: 49,
            output_item: 65,
            output_count: 1,
            cook_duration: 200,
            experience: 0.35,
        });
        // Raw Beef (50) -> Cooked Beef (66)
        reg.register(SmeltingRecipe {
            input_item: 50,
            output_item: 66,
            output_count: 1,
            cook_duration: 200,
            experience: 0.35,
        });
        // Cobblestone (4) -> Stone (1)
        reg.register(SmeltingRecipe {
            input_item: 4,
            output_item: 1,
            output_count: 1,
            cook_duration: 200,
            experience: 0.1,
        });
        // Sand (12) -> Glass (9)
        reg.register(SmeltingRecipe {
            input_item: 12,
            output_item: 9,
            output_count: 1,
            cook_duration: 200,
            experience: 0.1,
        });
        // Wood Logs (5, 75, 76) -> Charcoal (67)
        for log_id in [5, 75, 76] {
            reg.register(SmeltingRecipe {
                input_item: log_id,
                output_item: 67,
                output_count: 1,
                cook_duration: 200,
                experience: 0.15,
            });
        }
        // Iron Ore (13) -> Iron Ingot (52)
        reg.register(SmeltingRecipe {
            input_item: 13,
            output_item: 52,
            output_count: 1,
            cook_duration: 200,
            experience: 0.7,
        });
        // Gold Ore (14) -> Gold Ingot (53)
        reg.register(SmeltingRecipe {
            input_item: 14,
            output_item: 53,
            output_count: 1,
            cook_duration: 200,
            experience: 1.0,
        });
        reg
    }

    /// Registers or replaces a smelting recipe.
    pub fn register(&mut self, recipe: SmeltingRecipe) {
        self.recipes.insert(recipe.input_item, recipe);
    }

    /// Finds a smelting recipe for the given input item.
    #[must_use]
    pub fn find_recipe(&self, input_item: u32) -> Option<&SmeltingRecipe> {
        self.recipes.get(&input_item)
    }

    /// Whether the given input item has a registered smelting recipe.
    #[must_use]
    pub fn has_recipe(&self, input_item: u32) -> bool {
        self.recipes.contains_key(&input_item)
    }
}

/// Registry storing combustible fuel burn durations in game ticks (20 ticks = 1 second).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FuelRegistry {
    fuels: HashMap<u32, u16>,
}

impl FuelRegistry {
    /// Creates a new empty fuel registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fuels: HashMap::new(),
        }
    }

    /// Creates a standard fuel registry populated with survival combustible items.
    #[must_use]
    pub fn standard() -> Self {
        let mut reg = Self::new();
        // Coal (54) & Charcoal (67): 1600 ticks (80s, smelts 8 items)
        reg.register(54, 1600);
        reg.register(67, 1600);
        // Wood Logs (5, 75, 76): 300 ticks (15s, smelts 1.5 items)
        for log_id in [5, 75, 76] {
            reg.register(log_id, 300);
        }
        // Oak Planks (7): 300 ticks (15s)
        reg.register(7, 300);
        // Crafting Table (14): 300 ticks (15s)
        reg.register(14, 300);
        // Chest (63): 300 ticks (15s)
        reg.register(63, 300);
        // Stick (13): 100 ticks (5s, smelts 0.5 items)
        reg.register(13, 100);
        reg
    }

    /// Registers a combustible item and its burn duration in simulation ticks.
    pub fn register(&mut self, item: u32, duration_ticks: u16) {
        self.fuels.insert(item, duration_ticks);
    }

    /// Returns the burn duration in ticks for an item, or 0 if not combustible.
    #[must_use]
    pub fn burn_duration(&self, item: u32) -> u16 {
        self.fuels.get(&item).copied().unwrap_or(0)
    }

    /// Returns `true` if the item is combustible fuel.
    #[must_use]
    pub fn is_fuel(&self, item: u32) -> bool {
        self.fuels.get(&item).is_some_and(|&d| d > 0)
    }
}

/// Active furnace container inventory component holding 3 item slots and combustion/cook state.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct FurnaceInventory {
    /// 3 slots: [0: Input, 1: Fuel, 2: Output].
    pub slots: [ItemStack; FURNACE_CONTAINER_SLOTS],
    /// Optional custom container title.
    pub custom_name: Option<String>,
    /// Fuel burn ticks remaining (0 = not burning).
    pub burn_time_remaining: u16,
    /// Total burn duration of the active fuel item.
    pub total_burn_time: u16,
    /// Ticks elapsed cooking current ingredient (`0..cook_duration`).
    pub cook_progress: u16,
    /// Ticks required to finish cooking (default 200).
    pub cook_duration: u16,
}

impl Default for FurnaceInventory {
    fn default() -> Self {
        Self {
            slots: [ItemStack::EMPTY; FURNACE_CONTAINER_SLOTS],
            custom_name: None,
            burn_time_remaining: 0,
            total_burn_time: 0,
            cook_progress: 0,
            cook_duration: 200,
        }
    }
}

impl FurnaceInventory {
    /// Creates a `FurnaceInventory` from a `BlockEntityData` payload.
    #[must_use]
    pub fn from_block_entity(be: &BlockEntityData) -> Self {
        match be {
            BlockEntityData::Furnace {
                custom_name,
                items,
                burn_time_remaining,
                total_burn_time,
                cook_progress,
                cook_duration,
            } => {
                let mut slots = [ItemStack::EMPTY; FURNACE_CONTAINER_SLOTS];
                for (i, slot) in items.iter().enumerate().take(FURNACE_CONTAINER_SLOTS) {
                    slots[i] = ItemStack::new(slot.item, slot.count);
                }
                Self {
                    slots,
                    custom_name: custom_name.clone(),
                    burn_time_remaining: *burn_time_remaining,
                    total_burn_time: *total_burn_time,
                    cook_progress: *cook_progress,
                    cook_duration: *cook_duration,
                }
            }
            BlockEntityData::Chest { custom_name, items } => {
                let mut slots = [ItemStack::EMPTY; FURNACE_CONTAINER_SLOTS];
                for (i, slot) in items.iter().enumerate().take(FURNACE_CONTAINER_SLOTS) {
                    slots[i] = ItemStack::new(slot.item, slot.count);
                }
                Self {
                    slots,
                    custom_name: custom_name.clone(),
                    burn_time_remaining: 0,
                    total_burn_time: 0,
                    cook_progress: 0,
                    cook_duration: 200,
                }
            }
        }
    }

    /// Converts this inventory into a `BlockEntityData::Furnace` payload.
    #[must_use]
    pub fn to_block_entity(&self) -> BlockEntityData {
        let mut items = [BlockEntitySlot::EMPTY; FURNACE_CONTAINER_SLOTS];
        #[allow(clippy::cast_possible_truncation)]
        for (i, slot) in self.slots.iter().enumerate() {
            items[i] = BlockEntitySlot::new(i as u8, slot.item, slot.count);
        }
        BlockEntityData::Furnace {
            custom_name: self.custom_name.clone(),
            items,
            burn_time_remaining: self.burn_time_remaining,
            total_burn_time: self.total_burn_time,
            cook_progress: self.cook_progress,
            cook_duration: self.cook_duration,
        }
    }

    /// Updates an existing `BlockEntityData` in-place with this inventory's state.
    pub fn update_block_entity(&self, be: &mut BlockEntityData) {
        *be = self.to_block_entity();
    }

    /// Returns `true` if active fuel is burning.
    #[must_use]
    pub const fn is_burning(&self) -> bool {
        self.burn_time_remaining > 0
    }
}

/// Result of a single furnace tick simulation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct FurnaceTickResult {
    /// Whether the lit block state toggled (`furnace` <-> `lit_furnace`).
    pub lit_changed: bool,
    /// Current lit state of the furnace.
    pub is_lit: bool,
    /// Whether slot contents were modified (fuel consumed or output created).
    pub contents_changed: bool,
    /// Whether progress/burn timers changed.
    pub properties_changed: bool,
}

/// Executes a single simulation tick step for a furnace block entity.
#[must_use]
pub fn tick_furnace_step(
    furnace: &mut FurnaceInventory,
    recipes: &SmeltingRegistry,
    fuels: &FuelRegistry,
) -> FurnaceTickResult {
    let was_burning = furnace.burn_time_remaining > 0;
    let mut contents_changed = false;
    let mut properties_changed = false;

    // 1. Consume remaining fuel combustion time
    if furnace.burn_time_remaining > 0 {
        furnace.burn_time_remaining -= 1;
        properties_changed = true;
    }

    // 2. Evaluate if current input can be smelted
    let input_item = furnace.slots[FURNACE_SLOT_INPUT].item;
    let recipe = if furnace.slots[FURNACE_SLOT_INPUT].is_empty() {
        None
    } else {
        recipes.find_recipe(input_item)
    };

    let can_cook = if let Some(r) = recipe {
        let output_slot = &furnace.slots[FURNACE_SLOT_OUTPUT];
        if output_slot.is_empty() {
            true
        } else if output_slot.item == r.output_item {
            output_slot.count + r.output_count <= MAX_STACK_SIZE
        } else {
            false
        }
    } else {
        false
    };

    // 3. Ignite fuel if needed and possible
    if can_cook && furnace.burn_time_remaining == 0 {
        let fuel_item = furnace.slots[FURNACE_SLOT_FUEL].item;
        let fuel_duration = fuels.burn_duration(fuel_item);
        if fuel_duration > 0 && furnace.slots[FURNACE_SLOT_FUEL].count > 0 {
            furnace.burn_time_remaining = fuel_duration;
            furnace.total_burn_time = fuel_duration;
            furnace.slots[FURNACE_SLOT_FUEL].count -= 1;
            if furnace.slots[FURNACE_SLOT_FUEL].count == 0 {
                furnace.slots[FURNACE_SLOT_FUEL] = ItemStack::EMPTY;
            }
            contents_changed = true;
            properties_changed = true;
        }
    }

    // 4. Progress cooking or decay progress
    if can_cook && furnace.burn_time_remaining > 0 {
        let r = recipe.expect("can_cook requires valid recipe");
        furnace.cook_duration = r.cook_duration;
        furnace.cook_progress += 1;
        properties_changed = true;

        if furnace.cook_progress >= furnace.cook_duration {
            furnace.cook_progress = 0;
            // Decrement input
            furnace.slots[FURNACE_SLOT_INPUT].count -= 1;
            if furnace.slots[FURNACE_SLOT_INPUT].count == 0 {
                furnace.slots[FURNACE_SLOT_INPUT] = ItemStack::EMPTY;
            }
            // Add output
            let out = &mut furnace.slots[FURNACE_SLOT_OUTPUT];
            if out.is_empty() {
                *out = ItemStack::new(r.output_item, r.output_count);
            } else {
                out.count += r.output_count;
            }
            contents_changed = true;
        }
    } else if furnace.cook_progress > 0 {
        // Slowly decay cooking progress if unlit or invalid input
        furnace.cook_progress = furnace.cook_progress.saturating_sub(2);
        properties_changed = true;
    }

    let is_burning = furnace.burn_time_remaining > 0;
    let lit_changed = was_burning != is_burning;

    FurnaceTickResult {
        lit_changed,
        is_lit: is_burning,
        contents_changed,
        properties_changed,
    }
}

/// Executes a click interaction within a dual furnace container session:
/// - Slots 0..3: Furnace storage (Input, Fuel, Output)
/// - Slots 3..30: Player main storage (27 slots)
/// - Slots 30..39: Player hotbar (9 slots)
///
/// Output slot (2) is protected from item insertion.
pub fn furnace_container_click(
    container: &mut FurnaceInventory,
    player_inv: &mut Inventory,
    slot: usize,
    button: ClickButton,
    mode: ClickMode,
    fuels: &FuelRegistry,
    recipes: &SmeltingRegistry,
) -> Result<(), InventoryError> {
    if slot >= DUAL_FURNACE_SLOT_COUNT {
        return Err(InventoryError::SlotOutOfBounds(slot));
    }

    match mode {
        ClickMode::Pickup => handle_furnace_pickup(container, player_inv, slot, button),
        ClickMode::QuickMove => {
            handle_furnace_quick_move(container, player_inv, slot, fuels, recipes)
        }
        ClickMode::SwapHotbar => handle_furnace_swap_hotbar(container, player_inv, slot, button),
        ClickMode::Drop => handle_furnace_drop(container, player_inv, slot, button),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn handle_furnace_pickup(
    container: &mut FurnaceInventory,
    player_inv: &mut Inventory,
    slot: usize,
    button: ClickButton,
) -> Result<(), InventoryError> {
    if slot == FURNACE_SLOT_OUTPUT {
        // Output slot: only allows taking out items into carried stack
        let out_stack = &mut container.slots[FURNACE_SLOT_OUTPUT];
        if out_stack.is_empty() {
            return Ok(());
        }

        if player_inv.carried.is_empty() {
            match button {
                ClickButton::Left => {
                    player_inv.carried = *out_stack;
                    *out_stack = ItemStack::EMPTY;
                }
                ClickButton::Right => {
                    let half = out_stack.count.div_ceil(2);
                    player_inv.carried = ItemStack::new(out_stack.item, half);
                    out_stack.count -= half;
                    if out_stack.count == 0 {
                        *out_stack = ItemStack::EMPTY;
                    }
                }
            }
        } else if player_inv.carried.item == out_stack.item {
            let space = MAX_STACK_SIZE.saturating_sub(player_inv.carried.count);
            let to_take = out_stack.count.min(space);
            if to_take > 0 {
                player_inv.carried.count += to_take;
                out_stack.count -= to_take;
                if out_stack.count == 0 {
                    *out_stack = ItemStack::EMPTY;
                }
            }
        }
        return Ok(());
    }

    let target_stack = if slot < 3 {
        &mut container.slots[slot]
    } else if (3..30).contains(&slot) {
        &mut player_inv.slots[9 + (slot - 3)]
    } else {
        &mut player_inv.slots[slot - 30]
    };

    match button {
        ClickButton::Left => {
            if player_inv.carried.is_empty() {
                player_inv.carried = *target_stack;
                *target_stack = ItemStack::EMPTY;
            } else if target_stack.is_empty() {
                *target_stack = player_inv.carried;
                player_inv.carried = ItemStack::EMPTY;
            } else if target_stack.item == player_inv.carried.item {
                let space = MAX_STACK_SIZE.saturating_sub(target_stack.count);
                let to_add = player_inv.carried.count.min(space);
                target_stack.count += to_add;
                player_inv.carried.count -= to_add;
                if player_inv.carried.count == 0 {
                    player_inv.carried = ItemStack::EMPTY;
                }
            } else {
                std::mem::swap(target_stack, &mut player_inv.carried);
            }
        }
        ClickButton::Right => {
            if player_inv.carried.is_empty() {
                if !target_stack.is_empty() {
                    let half = target_stack.count.div_ceil(2);
                    player_inv.carried = ItemStack::new(target_stack.item, half);
                    target_stack.count -= half;
                    if target_stack.count == 0 {
                        *target_stack = ItemStack::EMPTY;
                    }
                }
            } else if target_stack.is_empty() {
                *target_stack = ItemStack::new(player_inv.carried.item, 1);
                player_inv.carried.count -= 1;
                if player_inv.carried.count == 0 {
                    player_inv.carried = ItemStack::EMPTY;
                }
            } else if target_stack.item == player_inv.carried.item
                && target_stack.count < MAX_STACK_SIZE
            {
                target_stack.count += 1;
                player_inv.carried.count -= 1;
                if player_inv.carried.count == 0 {
                    player_inv.carried = ItemStack::EMPTY;
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::unnecessary_wraps, clippy::too_many_lines)]
fn handle_furnace_quick_move(
    container: &mut FurnaceInventory,
    player_inv: &mut Inventory,
    slot: usize,
    fuels: &FuelRegistry,
    recipes: &SmeltingRegistry,
) -> Result<(), InventoryError> {
    if slot < 3 {
        // From furnace slot to player inventory (hotbar first, then storage)
        let src = &mut container.slots[slot];
        if src.is_empty() {
            return Ok(());
        }

        // Try merge into player hotbar (0..9)
        for h_slot in 0..9 {
            if player_inv.slots[h_slot].item == src.item {
                let space = MAX_STACK_SIZE.saturating_sub(player_inv.slots[h_slot].count);
                let to_add = src.count.min(space);
                player_inv.slots[h_slot].count += to_add;
                src.count -= to_add;
                if src.count == 0 {
                    *src = ItemStack::EMPTY;
                    return Ok(());
                }
            }
        }
        // Try merge into player storage (9..36)
        for s_slot in 9..36 {
            if player_inv.slots[s_slot].item == src.item {
                let space = MAX_STACK_SIZE.saturating_sub(player_inv.slots[s_slot].count);
                let to_add = src.count.min(space);
                player_inv.slots[s_slot].count += to_add;
                src.count -= to_add;
                if src.count == 0 {
                    *src = ItemStack::EMPTY;
                    return Ok(());
                }
            }
        }
        // Place into first empty slot in hotbar
        for h_slot in 0..9 {
            if player_inv.slots[h_slot].is_empty() {
                player_inv.slots[h_slot] = *src;
                *src = ItemStack::EMPTY;
                return Ok(());
            }
        }
        // Place into first empty slot in storage
        for s_slot in 9..36 {
            if player_inv.slots[s_slot].is_empty() {
                player_inv.slots[s_slot] = *src;
                *src = ItemStack::EMPTY;
                return Ok(());
            }
        }
    } else {
        // From player inventory (3..39) to furnace or hotbar/storage swap
        let is_storage = (3..30).contains(&slot);
        let p_idx = if is_storage {
            9 + (slot - 3)
        } else {
            slot - 30
        };
        let mut src = player_inv.slots[p_idx];
        if src.is_empty() {
            return Ok(());
        }

        // 1. If combustible fuel, try fuel slot (1) first
        if fuels.is_fuel(src.item) {
            let fuel_slot = &mut container.slots[FURNACE_SLOT_FUEL];
            if fuel_slot.is_empty() {
                *fuel_slot = src;
                player_inv.slots[p_idx] = ItemStack::EMPTY;
                return Ok(());
            } else if fuel_slot.item == src.item {
                let space = MAX_STACK_SIZE.saturating_sub(fuel_slot.count);
                let to_add = src.count.min(space);
                fuel_slot.count += to_add;
                src.count -= to_add;
                player_inv.slots[p_idx] = if src.count == 0 {
                    ItemStack::EMPTY
                } else {
                    src
                };
                if src.count == 0 {
                    return Ok(());
                }
            }
        }

        // 2. If smelting ingredient, try input slot (0)
        if recipes.has_recipe(src.item) {
            let input_slot = &mut container.slots[FURNACE_SLOT_INPUT];
            if input_slot.is_empty() {
                *input_slot = src;
                player_inv.slots[p_idx] = ItemStack::EMPTY;
                return Ok(());
            } else if input_slot.item == src.item {
                let space = MAX_STACK_SIZE.saturating_sub(input_slot.count);
                let to_add = src.count.min(space);
                input_slot.count += to_add;
                src.count -= to_add;
                player_inv.slots[p_idx] = if src.count == 0 {
                    ItemStack::EMPTY
                } else {
                    src
                };
                if src.count == 0 {
                    return Ok(());
                }
            }
        }

        // 3. Fallback: swap between player storage and player hotbar
        if is_storage {
            // Storage -> Hotbar
            for h_slot in 0..9 {
                if player_inv.slots[h_slot].is_empty() {
                    player_inv.slots[h_slot] = src;
                    player_inv.slots[p_idx] = ItemStack::EMPTY;
                    return Ok(());
                }
            }
        } else {
            // Hotbar -> Storage
            for s_slot in 9..36 {
                if player_inv.slots[s_slot].is_empty() {
                    player_inv.slots[s_slot] = src;
                    player_inv.slots[p_idx] = ItemStack::EMPTY;
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn handle_furnace_swap_hotbar(
    container: &mut FurnaceInventory,
    player_inv: &mut Inventory,
    slot: usize,
    button: ClickButton,
) -> Result<(), InventoryError> {
    let hotbar_idx = match button {
        ClickButton::Left => 0,
        ClickButton::Right => 1,
    };

    if slot == FURNACE_SLOT_OUTPUT {
        let out_slot = &mut container.slots[FURNACE_SLOT_OUTPUT];
        let hotbar_slot = &mut player_inv.slots[hotbar_idx];
        if !out_slot.is_empty() && hotbar_slot.is_empty() {
            *hotbar_slot = *out_slot;
            *out_slot = ItemStack::EMPTY;
        }
    } else if slot < 3 {
        std::mem::swap(
            &mut container.slots[slot],
            &mut player_inv.slots[hotbar_idx],
        );
    } else {
        let target_idx = if (3..30).contains(&slot) {
            9 + (slot - 3)
        } else {
            slot - 30
        };
        if target_idx != hotbar_idx {
            player_inv.slots.swap(target_idx, hotbar_idx);
        }
    }

    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn handle_furnace_drop(
    container: &mut FurnaceInventory,
    player_inv: &mut Inventory,
    slot: usize,
    button: ClickButton,
) -> Result<(), InventoryError> {
    let target_stack = if slot < 3 {
        &mut container.slots[slot]
    } else if (3..30).contains(&slot) {
        &mut player_inv.slots[9 + (slot - 3)]
    } else {
        &mut player_inv.slots[slot - 30]
    };

    if target_stack.is_empty() {
        return Ok(());
    }

    match button {
        ClickButton::Left => {
            // Drop entire stack
            *target_stack = ItemStack::EMPTY;
        }
        ClickButton::Right => {
            // Drop single item
            target_stack.count -= 1;
            if target_stack.count == 0 {
                *target_stack = ItemStack::EMPTY;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smelting_recipes_lookup() {
        let reg = SmeltingRegistry::standard();
        assert!(reg.has_recipe(49)); // Raw Porkchop
        assert!(reg.has_recipe(50)); // Raw Beef
        assert!(reg.has_recipe(4)); // Cobblestone
        assert!(reg.has_recipe(12)); // Sand
        assert!(reg.has_recipe(5)); // Oak log
        assert!(reg.has_recipe(13)); // Iron Ore
        assert!(!reg.has_recipe(1)); // Stone -> no recipe

        let beef_recipe = reg.find_recipe(50).unwrap();
        assert_eq!(beef_recipe.output_item, 66); // Cooked beef
        assert_eq!(beef_recipe.output_count, 1);
        assert_eq!(beef_recipe.cook_duration, 200);
    }

    #[test]
    fn test_fuel_registry_lookup() {
        let fuels = FuelRegistry::standard();
        assert_eq!(fuels.burn_duration(54), 1600); // Coal
        assert_eq!(fuels.burn_duration(67), 1600); // Charcoal
        assert_eq!(fuels.burn_duration(5), 300); // Oak log
        assert_eq!(fuels.burn_duration(7), 300); // Planks
        assert_eq!(fuels.burn_duration(13), 100); // Stick
        assert_eq!(fuels.burn_duration(1), 0); // Stone -> not fuel
        assert!(fuels.is_fuel(54));
        assert!(!fuels.is_fuel(1));
    }

    #[test]
    fn test_tick_furnace_step_combustion_and_cooking() {
        let recipes = SmeltingRegistry::standard();
        let fuels = FuelRegistry::standard();
        let mut furnace = FurnaceInventory::default();

        furnace.slots[FURNACE_SLOT_INPUT] = ItemStack::new(50, 2); // 2 Raw beef
        furnace.slots[FURNACE_SLOT_FUEL] = ItemStack::new(54, 1); // 1 Coal

        // Tick 1: Consumes 1 coal, starts burning (1600 ticks), cook_progress becomes 1
        let res = tick_furnace_step(&mut furnace, &recipes, &fuels);
        assert!(res.lit_changed);
        assert!(res.is_lit);
        assert!(res.contents_changed);
        assert_eq!(furnace.burn_time_remaining, 1600);
        assert_eq!(furnace.total_burn_time, 1600);
        assert_eq!(furnace.cook_progress, 1);
        assert!(furnace.slots[FURNACE_SLOT_FUEL].is_empty()); // Coal consumed

        // Simulate 199 more ticks -> cook progress reaches 200 and item finishes smelting
        for _ in 1..200 {
            let _ = tick_furnace_step(&mut furnace, &recipes, &fuels);
        }

        assert_eq!(furnace.slots[FURNACE_SLOT_INPUT].count, 1); // 1 raw beef left
        assert_eq!(furnace.slots[FURNACE_SLOT_OUTPUT].item, 66); // Cooked beef produced!
        assert_eq!(furnace.slots[FURNACE_SLOT_OUTPUT].count, 1);
        assert_eq!(furnace.cook_progress, 0);
        assert!(furnace.is_burning());
    }

    #[test]
    fn test_furnace_output_slot_protection() {
        let fuels = FuelRegistry::standard();
        let recipes = SmeltingRegistry::standard();
        let mut container = FurnaceInventory::default();
        let mut inv = Inventory {
            carried: ItemStack::new(4, 10),
            ..Default::default()
        };

        // Cannot place into output slot!
        let _ = furnace_container_click(
            &mut container,
            &mut inv,
            FURNACE_SLOT_OUTPUT,
            ClickButton::Left,
            ClickMode::Pickup,
            &fuels,
            &recipes,
        );
        assert!(container.slots[FURNACE_SLOT_OUTPUT].is_empty());
        assert_eq!(inv.carried.count, 10); // Carried preserved!
    }

    #[test]
    fn test_furnace_shift_click_quick_move() {
        let fuels = FuelRegistry::standard();
        let recipes = SmeltingRegistry::standard();
        let mut container = FurnaceInventory::default();
        let mut inv = Inventory::default();

        // Put coal in player hotbar slot 0 (index 30 in container window)
        inv.slots[0] = ItemStack::new(54, 16);
        furnace_container_click(
            &mut container,
            &mut inv,
            30,
            ClickButton::Left,
            ClickMode::QuickMove,
            &fuels,
            &recipes,
        )
        .unwrap();

        assert_eq!(container.slots[FURNACE_SLOT_FUEL].item, 54);
        assert_eq!(container.slots[FURNACE_SLOT_FUEL].count, 16);
        assert!(inv.slots[0].is_empty());

        // Put raw beef in player hotbar slot 1 (index 31 in container window)
        inv.slots[1] = ItemStack::new(50, 5);
        furnace_container_click(
            &mut container,
            &mut inv,
            31,
            ClickButton::Left,
            ClickMode::QuickMove,
            &fuels,
            &recipes,
        )
        .unwrap();

        assert_eq!(container.slots[FURNACE_SLOT_INPUT].item, 50);
        assert_eq!(container.slots[FURNACE_SLOT_INPUT].count, 5);
        assert!(inv.slots[1].is_empty());
    }
}
