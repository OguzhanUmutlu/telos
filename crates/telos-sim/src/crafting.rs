//! Deterministic shaped and shapeless crafting recipe engine, 2x2 and 3x3 matrix evaluation,
//! remainder items, and crafting table container interactions.

use crate::inventory::{
    ClickButton, ClickMode, Inventory, InventoryError, ItemStack, MAX_STACK_SIZE,
};
use std::ops::Range;

/// Total number of item slots in a dual crafting table container window (46).
/// - Slot 0: Crafting result output slot
/// - Slots 1..=9: 3x3 crafting grid
/// - Slots 10..=36: Player main storage (27 slots)
/// - Slots 37..=45: Player hotbar (9 slots)
pub const DUAL_CRAFTING_TABLE_SLOT_COUNT: usize = 46;

/// Number of item slots in the crafting table portion of the container (1 result + 9 grid = 10).
pub const CRAFTING_TABLE_CONTAINER_SLOTS: usize = 10;

/// Crafting result output slot index in dual crafting table container window (0).
pub const CONTAINER_CRAFTING_RESULT_SLOT: usize = 0;

/// Crafting 3x3 grid slot range in dual crafting table container window (1..10).
pub const CONTAINER_CRAFTING_GRID_SLOTS: Range<usize> = 1..10;

/// Player storage slot range in dual crafting table container window (10..37).
pub const CONTAINER_CRAFTING_STORAGE_SLOTS: Range<usize> = 10..37;

/// Player hotbar slot range in dual crafting table container window (37..46).
pub const CONTAINER_CRAFTING_HOTBAR_SLOTS: Range<usize> = 37..46;

/// A shaped crafting recipe with grid dimensions and pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapedRecipe {
    /// Width of the pattern in grid columns (1..=3).
    pub width: usize,
    /// Height of the pattern in grid rows (1..=3).
    pub height: usize,
    /// Pattern item IDs of size `width * height`, where 0 indicates an empty slot.
    pub pattern: Vec<u32>,
    /// Whether horizontal mirroring is permitted for this recipe.
    pub mirrored: bool,
    /// Output produced by one craft operation.
    pub result: ItemStack,
    /// Optional item left behind in consumed slots (e.g. empty buckets).
    pub remainder_item: Option<u32>,
}

impl ShapedRecipe {
    /// Creates a new shaped recipe without mirroring.
    #[must_use]
    pub fn new(width: usize, height: usize, pattern: Vec<u32>, result: ItemStack) -> Self {
        assert_eq!(
            pattern.len(),
            width * height,
            "Pattern length must match width * height"
        );
        Self {
            width,
            height,
            pattern,
            mirrored: false,
            result,
            remainder_item: None,
        }
    }

    /// Creates a new shaped recipe with optional horizontal mirroring.
    #[must_use]
    pub fn new_mirrored(
        width: usize,
        height: usize,
        pattern: Vec<u32>,
        result: ItemStack,
        mirrored: bool,
    ) -> Self {
        assert_eq!(
            pattern.len(),
            width * height,
            "Pattern length must match width * height"
        );
        Self {
            width,
            height,
            pattern,
            mirrored,
            result,
            remainder_item: None,
        }
    }

    /// Checks if this recipe matches the given input grid by extracting its non-empty bounding box.
    #[must_use]
    pub fn matches_grid(&self, inputs: &[ItemStack], grid_w: usize, grid_h: usize) -> bool {
        let mut min_col = usize::MAX;
        let mut max_col = 0;
        let mut min_row = usize::MAX;
        let mut max_row = 0;
        let mut has_items = false;

        for r in 0..grid_h {
            for c in 0..grid_w {
                let slot = &inputs[r * grid_w + c];
                if !slot.is_empty() {
                    has_items = true;
                    min_col = min_col.min(c);
                    max_col = max_col.max(c);
                    min_row = min_row.min(r);
                    max_row = max_row.max(r);
                }
            }
        }

        if !has_items {
            return false;
        }

        let bbox_w = max_col - min_col + 1;
        let bbox_h = max_row - min_row + 1;

        if bbox_w != self.width || bbox_h != self.height {
            return false;
        }

        // 1. Check direct match within the bounding box
        let mut matches_direct = true;
        for r in 0..self.height {
            for c in 0..self.width {
                let expected = self.pattern[r * self.width + c];
                let actual = inputs[(min_row + r) * grid_w + (min_col + c)];
                if expected == 0 {
                    if !actual.is_empty() {
                        matches_direct = false;
                        break;
                    }
                } else if actual.is_empty() || actual.item != expected {
                    matches_direct = false;
                    break;
                }
            }
            if !matches_direct {
                break;
            }
        }

        if matches_direct {
            return true;
        }

        // 2. Check mirrored match if enabled and width > 1
        if self.mirrored && self.width > 1 {
            let mut matches_mirrored = true;
            for r in 0..self.height {
                for c in 0..self.width {
                    let expected = self.pattern[r * self.width + (self.width - 1 - c)];
                    let actual = inputs[(min_row + r) * grid_w + (min_col + c)];
                    if expected == 0 {
                        if !actual.is_empty() {
                            matches_mirrored = false;
                            break;
                        }
                    } else if actual.is_empty() || actual.item != expected {
                        matches_mirrored = false;
                        break;
                    }
                }
                if !matches_mirrored {
                    break;
                }
            }
            if matches_mirrored {
                return true;
            }
        }

        false
    }
}

/// A shapeless crafting recipe where ingredients can appear in any order or grid position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapelessRecipe {
    /// Required ingredient item IDs (multiset).
    pub ingredients: Vec<u32>,
    /// Output produced by one craft operation.
    pub result: ItemStack,
    /// Optional item left behind in consumed slots.
    pub remainder_item: Option<u32>,
}

impl ShapelessRecipe {
    /// Creates a new shapeless recipe.
    #[must_use]
    pub fn new(mut ingredients: Vec<u32>, result: ItemStack) -> Self {
        ingredients.sort_unstable();
        Self {
            ingredients,
            result,
            remainder_item: None,
        }
    }

    /// Checks if this shapeless recipe matches the provided input grid.
    #[must_use]
    pub fn matches_grid(&self, inputs: &[ItemStack]) -> bool {
        let mut actual_items: Vec<u32> = inputs
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.item)
            .collect();

        if actual_items.len() != self.ingredients.len() {
            return false;
        }

        actual_items.sort_unstable();
        actual_items == self.ingredients
    }
}

/// A generic crafting recipe (shaped or shapeless).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CraftingRecipe {
    /// Shaped pattern recipe.
    Shaped(ShapedRecipe),
    /// Shapeless recipe.
    Shapeless(ShapelessRecipe),
}

impl CraftingRecipe {
    /// Returns the output item stack produced by this recipe.
    #[must_use]
    pub const fn result(&self) -> ItemStack {
        match self {
            Self::Shaped(r) => r.result,
            Self::Shapeless(r) => r.result,
        }
    }

    /// Returns the optional remainder item left behind in crafting slots.
    #[must_use]
    pub const fn remainder_item(&self) -> Option<u32> {
        match self {
            Self::Shaped(r) => r.remainder_item,
            Self::Shapeless(r) => r.remainder_item,
        }
    }

    /// Checks if this recipe matches the given input grid.
    #[must_use]
    pub fn matches(&self, inputs: &[ItemStack], grid_w: usize, grid_h: usize) -> bool {
        match self {
            Self::Shaped(r) => r.matches_grid(inputs, grid_w, grid_h),
            Self::Shapeless(r) => r.matches_grid(inputs),
        }
    }
}

/// Deterministic recipe registry indexing shaped and shapeless recipes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RecipeRegistry {
    recipes: Vec<CraftingRecipe>,
}

impl RecipeRegistry {
    /// Creates a new empty recipe registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            recipes: Vec::new(),
        }
    }

    /// Registers a shaped recipe.
    pub fn register_shaped(&mut self, recipe: ShapedRecipe) {
        self.recipes.push(CraftingRecipe::Shaped(recipe));
    }

    /// Registers a shapeless recipe.
    pub fn register_shapeless(&mut self, recipe: ShapelessRecipe) {
        self.recipes.push(CraftingRecipe::Shapeless(recipe));
    }

    /// Finds the first matching recipe for a given grid dimension.
    #[must_use]
    pub fn find_match(
        &self,
        inputs: &[ItemStack],
        grid_w: usize,
        grid_h: usize,
    ) -> Option<&CraftingRecipe> {
        self.recipes
            .iter()
            .find(|recipe| recipe.matches(inputs, grid_w, grid_h))
    }

    /// Finds matching recipe for a 2x2 crafting input grid (4 slots).
    #[must_use]
    pub fn find_match_2x2(&self, inputs: &[ItemStack; 4]) -> Option<&CraftingRecipe> {
        self.find_match(inputs, 2, 2)
    }

    /// Finds matching recipe for a 3x3 crafting table grid (9 slots).
    #[must_use]
    pub fn find_match_3x3(&self, inputs: &[ItemStack; 9]) -> Option<&CraftingRecipe> {
        self.find_match(inputs, 3, 3)
    }

    /// Creates a standard registry populated with default survival recipes.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn standard() -> Self {
        let mut reg = Self::new();

        // -------------------------------------------------------------
        // 1. Shapeless Wood & Basic Conversions
        // -------------------------------------------------------------
        // Logs -> 4 Planks (Oak = 5, Birch = 75, Spruce = 76 -> Planks = 7)
        for log_id in [5, 75, 76] {
            reg.register_shapeless(ShapelessRecipe::new(vec![log_id], ItemStack::new(7, 4)));
        }

        // Cobblestone (4) -> 2 Stone Slabs (10) (single item fallback)
        reg.register_shapeless(ShapelessRecipe::new(vec![4], ItemStack::new(10, 2)));

        // -------------------------------------------------------------
        // 2. Shaped Basics (Sticks, Crafting Table, Torches, Slabs, Stairs)
        // -------------------------------------------------------------
        // 2 Planks (1x2 vertical) -> 4 Sticks (13)
        reg.register_shaped(ShapedRecipe::new(1, 2, vec![7, 7], ItemStack::new(13, 4)));

        // 4 Planks (2x2) -> 1 Crafting Table (14)
        reg.register_shaped(ShapedRecipe::new(
            2,
            2,
            vec![7, 7, 7, 7],
            ItemStack::new(14, 1),
        ));

        // Coal (54) or Charcoal (67) over Stick (13) -> 4 Torches (15)
        reg.register_shaped(ShapedRecipe::new(1, 2, vec![54, 13], ItemStack::new(15, 4)));
        reg.register_shaped(ShapedRecipe::new(1, 2, vec![67, 13], ItemStack::new(15, 4)));
        // Planks (7) over Stick (13) fallback torch recipe
        reg.register_shaped(ShapedRecipe::new(1, 2, vec![7, 13], ItemStack::new(15, 4)));

        // 2 Stone (2x1) or 3 Stone (3x1) -> Stone Slabs (10)
        reg.register_shaped(ShapedRecipe::new(2, 1, vec![1, 1], ItemStack::new(10, 4)));
        reg.register_shaped(ShapedRecipe::new(
            3,
            1,
            vec![1, 1, 1],
            ItemStack::new(10, 6),
        ));

        // 3 Planks (3x1) -> 6 Oak Slabs
        reg.register_shaped(ShapedRecipe::new(
            3,
            1,
            vec![7, 7, 7],
            ItemStack::new(10, 6),
        ));

        // 6 Planks -> 4 Oak Stairs (11) (mirrored)
        reg.register_shaped(ShapedRecipe::new_mirrored(
            3,
            3,
            vec![7, 0, 0, 7, 7, 0, 7, 7, 7],
            ItemStack::new(11, 4),
            true,
        ));

        // 8 Planks surrounding empty center (3x3) -> 1 Chest (63)
        reg.register_shaped(ShapedRecipe::new(
            3,
            3,
            vec![7, 7, 7, 7, 0, 7, 7, 7, 7],
            ItemStack::new(crate::inventory::ITEM_CHEST, 1),
        ));

        // 8 Cobblestone surrounding empty center (3x3) -> 1 Furnace (64)
        reg.register_shaped(ShapedRecipe::new(
            3,
            3,
            vec![4, 4, 4, 4, 0, 4, 4, 4, 4],
            ItemStack::new(crate::inventory::ITEM_FURNACE, 1),
        ));
        // 4 Cobblestone (2x2) fallback -> 1 Furnace (64)
        reg.register_shaped(ShapedRecipe::new(
            2,
            2,
            vec![4, 4, 4, 4],
            ItemStack::new(crate::inventory::ITEM_FURNACE, 1),
        ));

        // 3 Wheat (3x1) -> 1 Bread (57)
        reg.register_shaped(ShapedRecipe::new(
            3,
            1,
            vec![58, 58, 58],
            ItemStack::new(57, 1),
        ));

        // -------------------------------------------------------------
        // 3. Tools (Wooden, Stone, Iron)
        // -------------------------------------------------------------
        let tool_materials = [
            (
                7,
                crate::inventory::ITEM_WOODEN_PICKAXE,
                crate::inventory::ITEM_WOODEN_AXE,
                crate::inventory::ITEM_WOODEN_SHOVEL,
                crate::inventory::ITEM_WOODEN_SWORD,
                crate::inventory::ITEM_WOODEN_HOE,
            ),
            (
                4,
                crate::inventory::ITEM_STONE_PICKAXE,
                crate::inventory::ITEM_STONE_AXE,
                crate::inventory::ITEM_STONE_SHOVEL,
                crate::inventory::ITEM_STONE_SWORD,
                crate::inventory::ITEM_STONE_HOE,
            ),
            (
                52,
                crate::inventory::ITEM_IRON_PICKAXE,
                crate::inventory::ITEM_IRON_AXE,
                crate::inventory::ITEM_IRON_SHOVEL,
                crate::inventory::ITEM_IRON_SWORD,
                crate::inventory::ITEM_IRON_HOE,
            ),
        ];

        for (mat, pick, axe, shovel, sword, hoe) in tool_materials {
            // Pickaxe: 3 mat on top row, sticks in col 1 of rows 1 and 2
            reg.register_shaped(ShapedRecipe::new(
                3,
                3,
                vec![mat, mat, mat, 0, 13, 0, 0, 13, 0],
                ItemStack::new(pick, 1),
            ));

            // Axe: 2x3 or 3x3 mirrored
            reg.register_shaped(ShapedRecipe::new_mirrored(
                2,
                3,
                vec![mat, mat, mat, 13, 0, 13],
                ItemStack::new(axe, 1),
                true,
            ));

            // Shovel: 1x3 (mat on top, 2 sticks below)
            reg.register_shaped(ShapedRecipe::new(
                1,
                3,
                vec![mat, 13, 13],
                ItemStack::new(shovel, 1),
            ));

            // Sword: 1x3 (2 mat on top, 1 stick below)
            reg.register_shaped(ShapedRecipe::new(
                1,
                3,
                vec![mat, mat, 13],
                ItemStack::new(sword, 1),
            ));

            // Hoe: 2x3 mirrored
            reg.register_shaped(ShapedRecipe::new_mirrored(
                2,
                3,
                vec![mat, mat, 0, 13, 0, 13],
                ItemStack::new(hoe, 1),
                true,
            ));
        }

        // -------------------------------------------------------------
        // 4. Armor (Leather, Iron, Gold)
        // -------------------------------------------------------------
        let armor_materials = [
            (
                51,
                crate::inventory::ITEM_LEATHER_HELMET,
                crate::inventory::ITEM_LEATHER_CHESTPLATE,
                crate::inventory::ITEM_LEATHER_LEGGINGS,
                crate::inventory::ITEM_LEATHER_BOOTS,
            ),
            (52, 16, 17, 18, 19),
            (
                53,
                crate::inventory::ITEM_GOLDEN_HELMET,
                crate::inventory::ITEM_GOLDEN_CHESTPLATE,
                crate::inventory::ITEM_GOLDEN_LEGGINGS,
                crate::inventory::ITEM_GOLDEN_BOOTS,
            ),
        ];

        for (mat, helm, chest, legs, boots) in armor_materials {
            // Helmet (3x2): [mat, mat, mat, mat, 0, mat]
            reg.register_shaped(ShapedRecipe::new(
                3,
                2,
                vec![mat, mat, mat, mat, 0, mat],
                ItemStack::new(helm, 1),
            ));

            // Chestplate (3x3): [mat, 0, mat, mat, mat, mat, mat, mat, mat]
            reg.register_shaped(ShapedRecipe::new(
                3,
                3,
                vec![mat, 0, mat, mat, mat, mat, mat, mat, mat],
                ItemStack::new(chest, 1),
            ));

            // Leggings (3x3): [mat, mat, mat, mat, 0, mat, mat, 0, mat]
            reg.register_shaped(ShapedRecipe::new(
                3,
                3,
                vec![mat, mat, mat, mat, 0, mat, mat, 0, mat],
                ItemStack::new(legs, 1),
            ));

            // Boots (3x2): [mat, 0, mat, mat, 0, mat]
            reg.register_shaped(ShapedRecipe::new(
                3,
                2,
                vec![mat, 0, mat, mat, 0, mat],
                ItemStack::new(boots, 1),
            ));
        }

        // -------------------------------------------------------------
        // 5. Weapons & Projectiles (Bow, Arrow)
        // -------------------------------------------------------------
        // Bow (3x3 mirrored):
        // [0, 13, 55]
        // [13, 0, 55]
        // [0, 13, 55]
        reg.register_shaped(ShapedRecipe::new_mirrored(
            3,
            3,
            vec![0, 13, 55, 13, 0, 55, 0, 13, 55],
            ItemStack::new(crate::inventory::ITEM_BOW, 1),
            true,
        ));

        // Arrow (1x3): Flint/Iron Ingot (52) over Stick (13) over Feather/String (55) -> 4 Arrows (62)
        reg.register_shaped(ShapedRecipe::new(
            1,
            3,
            vec![52, 13, 55],
            ItemStack::new(crate::inventory::ITEM_ARROW, 4),
        ));
        // Fallback arrow using Cobblestone (4)
        reg.register_shaped(ShapedRecipe::new(
            1,
            3,
            vec![4, 13, 55],
            ItemStack::new(crate::inventory::ITEM_ARROW, 4),
        ));

        reg
    }
}

/// In-memory state of an active crafting table session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CraftingTableInventory {
    /// Crafting output preview slot.
    pub result: ItemStack,
    /// 3x3 crafting grid input slots (row-major: 0..3 top, 3..6 middle, 6..9 bottom).
    pub grid: [ItemStack; 9],
}

impl CraftingTableInventory {
    /// Creates a new empty crafting table inventory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            result: ItemStack::EMPTY,
            grid: [ItemStack::EMPTY; 9],
        }
    }

    /// Re-evaluates recipes against the 3x3 grid and updates the output preview slot.
    pub fn update_result(&mut self, registry: &RecipeRegistry) {
        self.result = registry
            .find_match_3x3(&self.grid)
            .map_or(ItemStack::EMPTY, CraftingRecipe::result);
    }

    /// Clears all 9 grid slots and the result slot.
    pub fn clear(&mut self) {
        self.result = ItemStack::EMPTY;
        self.grid = [ItemStack::EMPTY; 9];
    }
}

/// Executes a deterministic click interaction on a dual crafting table container window:
/// - Slot 0: Output preview slot
/// - Slots 1..=9: 3x3 crafting grid
/// - Slots 10..=36: Player main storage (27 slots)
/// - Slots 37..=45: Player hotbar (9 slots)
#[allow(clippy::too_many_lines)]
pub fn crafting_table_container_click(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
    mode: ClickMode,
    registry: &RecipeRegistry,
) -> Result<(), InventoryError> {
    if slot_idx >= DUAL_CRAFTING_TABLE_SLOT_COUNT {
        return Err(InventoryError::SlotOutOfBounds(slot_idx));
    }

    // Special handling for output preview slot (0)
    if slot_idx == CONTAINER_CRAFTING_RESULT_SLOT {
        handle_crafting_table_result_click(container, player_inv, button, mode, registry);
        return Ok(());
    }

    match mode {
        ClickMode::Pickup => {
            handle_crafting_table_pickup(container, player_inv, slot_idx, button);
            if (1..10).contains(&slot_idx) {
                container.update_result(registry);
            }
        }
        ClickMode::QuickMove => {
            handle_crafting_table_quick_move(container, player_inv, slot_idx);
            if (1..10).contains(&slot_idx) {
                container.update_result(registry);
            }
        }
        ClickMode::SwapHotbar => {
            handle_crafting_table_swap_hotbar(container, player_inv, slot_idx, button);
            if (1..10).contains(&slot_idx) {
                container.update_result(registry);
            }
        }
        ClickMode::Drop => {
            handle_crafting_table_drop(container, player_inv, slot_idx, button);
            if (1..10).contains(&slot_idx) {
                container.update_result(registry);
            }
        }
    }

    Ok(())
}

/// Handles clicking on the crafting table result slot (0).
fn handle_crafting_table_result_click(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    _button: ClickButton,
    mode: ClickMode,
    registry: &RecipeRegistry,
) {
    if container.result.is_empty() {
        return;
    }

    match mode {
        ClickMode::Pickup => {
            let res = container.result;
            // Can only take result if carried cursor stack is empty or can merge with result
            if player_inv.carried.is_empty() {
                player_inv.carried = res;
            } else if player_inv.carried.item == res.item
                && player_inv.carried.count + res.count <= MAX_STACK_SIZE
            {
                player_inv.carried.count += res.count;
            } else {
                return; // Cannot pick up result
            }

            // Deduct 1 from each non-empty crafting grid slot
            let remainder = registry
                .find_match_3x3(&container.grid)
                .and_then(CraftingRecipe::remainder_item);

            for slot in &mut container.grid {
                if !slot.is_empty() {
                    slot.count -= 1;
                    if slot.count == 0 {
                        *slot = remainder.map_or(ItemStack::EMPTY, |rem| ItemStack::new(rem, 1));
                    }
                }
            }

            container.update_result(registry);
        }
        ClickMode::QuickMove => {
            // Shift-craft: craft as many batches as possible (up to 64) into player storage/hotbar
            for _ in 0..64 {
                let res = container.result;
                if res.is_empty() {
                    break;
                }

                let mut to_insert = res;
                let inserted = player_inv.insert_into_storage_or_hotbar(&mut to_insert);
                if inserted < res.count {
                    // Full result could not fit in inventory: halt shift craft
                    break;
                }

                let remainder = registry
                    .find_match_3x3(&container.grid)
                    .and_then(CraftingRecipe::remainder_item);

                for slot in &mut container.grid {
                    if !slot.is_empty() {
                        slot.count -= 1;
                        if slot.count == 0 {
                            *slot =
                                remainder.map_or(ItemStack::EMPTY, |rem| ItemStack::new(rem, 1));
                        }
                    }
                }

                container.update_result(registry);
            }
        }
        ClickMode::SwapHotbar | ClickMode::Drop => {}
    }
}

/// Helper to get a mutable reference to a slot across crafting grid (1..=9) or player inventory (10..=45).
fn get_slot_mut<'a>(
    grid: &'a mut [ItemStack; 9],
    player_slots: &'a mut [ItemStack],
    slot_idx: usize,
) -> &'a mut ItemStack {
    if (1..=9).contains(&slot_idx) {
        &mut grid[slot_idx - 1]
    } else if (10..=36).contains(&slot_idx) {
        // Player main storage (slots 9..36 in Inventory)
        let player_slot = 9 + (slot_idx - 10);
        &mut player_slots[player_slot]
    } else {
        // Player hotbar (slots 0..9 in Inventory)
        let player_slot = slot_idx - 37;
        &mut player_slots[player_slot]
    }
}

fn handle_crafting_table_pickup(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
) {
    let target = get_slot_mut(&mut container.grid, &mut player_inv.slots, slot_idx);
    let carried = &mut player_inv.carried;
    match button {
        ClickButton::Left => {
            if carried.is_empty() {
                *carried = *target;
                *target = ItemStack::EMPTY;
            } else if target.is_empty() {
                *target = *carried;
                *carried = ItemStack::EMPTY;
            } else if target.item == carried.item {
                let space = MAX_STACK_SIZE.saturating_sub(target.count);
                let to_move = carried.count.min(space);
                target.count += to_move;
                carried.count -= to_move;
                carried.normalize();
            } else {
                std::mem::swap(target, carried);
            }
        }
        ClickButton::Right => {
            if carried.is_empty() {
                if !target.is_empty() {
                    let take = target.count.div_ceil(2);
                    *carried = ItemStack::new(target.item, take);
                    target.count -= take;
                    target.normalize();
                }
            } else if target.is_empty() {
                *target = ItemStack::new(carried.item, 1);
                carried.count -= 1;
                carried.normalize();
            } else if target.item == carried.item && target.count < MAX_STACK_SIZE {
                target.count += 1;
                carried.count -= 1;
                carried.normalize();
            } else {
                std::mem::swap(target, carried);
            }
        }
    }
}

fn handle_crafting_table_quick_move(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    slot_idx: usize,
) {
    if (1..=9).contains(&slot_idx) {
        // From grid to player storage/hotbar
        let grid_slot = &mut container.grid[slot_idx - 1];
        if !grid_slot.is_empty() {
            let mut stack = *grid_slot;
            player_inv.insert_into_storage_or_hotbar(&mut stack);
            *grid_slot = stack;
        }
    } else if (10..=36).contains(&slot_idx) {
        // From storage to hotbar
        let player_slot = 9 + (slot_idx - 10);
        let mut stack = player_inv.slots[player_slot];
        if !stack.is_empty() {
            player_inv.insert_into_range(&mut stack, crate::inventory::HOTBAR_SLOTS);
            player_inv.slots[player_slot] = stack;
        }
    } else if (37..=45).contains(&slot_idx) {
        // From hotbar to storage
        let player_slot = slot_idx - 37;
        let mut stack = player_inv.slots[player_slot];
        if !stack.is_empty() {
            player_inv.insert_into_range(&mut stack, crate::inventory::STORAGE_SLOTS);
            player_inv.slots[player_slot] = stack;
        }
    }
}

fn handle_crafting_table_swap_hotbar(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
) {
    let hotbar_slot = match button {
        ClickButton::Left => player_inv.selected_slot,
        ClickButton::Right => 8,
    }
    .min(8);

    if (1..=9).contains(&slot_idx) {
        let grid_slot = &mut container.grid[slot_idx - 1];
        std::mem::swap(grid_slot, &mut player_inv.slots[hotbar_slot]);
    } else {
        let player_slot = if (10..=36).contains(&slot_idx) {
            9 + (slot_idx - 10)
        } else {
            slot_idx - 37
        };
        if player_slot != hotbar_slot {
            player_inv.slots.swap(player_slot, hotbar_slot);
        }
    }
}

fn handle_crafting_table_drop(
    container: &mut CraftingTableInventory,
    player_inv: &mut Inventory,
    slot_idx: usize,
    button: ClickButton,
) {
    let target = get_slot_mut(&mut container.grid, &mut player_inv.slots, slot_idx);
    if target.is_empty() {
        return;
    }
    match button {
        ClickButton::Left => {
            *target = ItemStack::EMPTY;
        }
        ClickButton::Right => {
            target.count = target.count.saturating_sub(1);
            target.normalize();
        }
    }
}

// ----------------------------------------------------------------------------
// Backwards Compatibility: 2x2 Crafting Helper & Tests
// ----------------------------------------------------------------------------

/// Evaluates a 2x2 crafting input grid and returns the matching recipe output, if any.
#[must_use]
pub fn find_recipe_2x2(inputs: &[ItemStack; 4]) -> Option<ItemStack> {
    RecipeRegistry::standard()
        .find_match_2x2(inputs)
        .map(CraftingRecipe::result)
}

/// A legacy 2x2 crafting recipe definition kept for backwards compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe2x2 {
    /// Input item IDs for 4 slots.
    pub pattern: [u32; 4],
    /// Output produced.
    pub result: ItemStack,
}

impl Recipe2x2 {
    /// Creates a new 2x2 recipe.
    #[must_use]
    pub const fn new(pattern: [u32; 4], result: ItemStack) -> Self {
        Self { pattern, result }
    }

    /// Checks if inputs match pattern.
    #[must_use]
    pub fn matches(&self, inputs: &[ItemStack; 4]) -> bool {
        for (i, &expected_item) in self.pattern.iter().enumerate() {
            let input = &inputs[i];
            if expected_item == 0 {
                if !input.is_empty() {
                    return false;
                }
            } else if input.is_empty() || input.item != expected_item {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_to_planks_any_slot() {
        for slot in 0..4 {
            let mut inputs = [ItemStack::EMPTY; 4];
            inputs[slot] = ItemStack::new(5, 1);
            assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(7, 4)));
        }
    }

    #[test]
    fn test_planks_to_sticks_vertical() {
        // Left column
        let mut inputs = [ItemStack::EMPTY; 4];
        inputs[0] = ItemStack::new(7, 1);
        inputs[2] = ItemStack::new(7, 1);
        assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(13, 4)));

        // Right column
        let mut inputs = [ItemStack::EMPTY; 4];
        inputs[1] = ItemStack::new(7, 1);
        inputs[3] = ItemStack::new(7, 1);
        assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(13, 4)));

        // Horizontal planks should NOT make sticks
        let mut inputs = [ItemStack::EMPTY; 4];
        inputs[0] = ItemStack::new(7, 1);
        inputs[1] = ItemStack::new(7, 1);
        assert_eq!(find_recipe_2x2(&inputs), None);
    }

    #[test]
    fn test_planks_to_crafting_table() {
        let inputs = [
            ItemStack::new(7, 1),
            ItemStack::new(7, 1),
            ItemStack::new(7, 1),
            ItemStack::new(7, 1),
        ];
        assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(14, 1)));
    }

    #[test]
    fn test_shaped_tools_and_bounding_box_in_3x3() {
        let reg = RecipeRegistry::standard();

        // 1. Iron Pickaxe: top row iron (52), col 1 sticks (13)
        let mut grid = [ItemStack::EMPTY; 9];
        grid[0] = ItemStack::new(52, 1);
        grid[1] = ItemStack::new(52, 1);
        grid[2] = ItemStack::new(52, 1);
        grid[4] = ItemStack::new(13, 1);
        grid[7] = ItemStack::new(13, 1);

        let matched = reg.find_match_3x3(&grid);
        assert!(matched.is_some());
        assert_eq!(
            matched.unwrap().result(),
            ItemStack::new(crate::inventory::ITEM_IRON_PICKAXE, 1)
        );

        // 2. Iron Sword (1x3): can be placed in column 0, 1, or 2!
        for col in 0..3 {
            let mut sword_grid = [ItemStack::EMPTY; 9];
            sword_grid[col] = ItemStack::new(52, 1);
            sword_grid[3 + col] = ItemStack::new(52, 1);
            sword_grid[6 + col] = ItemStack::new(13, 1);

            let sword_match = reg.find_match_3x3(&sword_grid);
            assert!(sword_match.is_some(), "Sword should match in column {col}");
            assert_eq!(
                sword_match.unwrap().result(),
                ItemStack::new(crate::inventory::ITEM_IRON_SWORD, 1)
            );
        }

        // 3. Chest (8 planks ring)
        let mut chest_grid = [ItemStack::new(7, 1); 9];
        chest_grid[4] = ItemStack::EMPTY; // Middle empty
        let chest_match = reg.find_match_3x3(&chest_grid);
        assert!(chest_match.is_some());
        assert_eq!(
            chest_match.unwrap().result(),
            ItemStack::new(crate::inventory::ITEM_CHEST, 1)
        );
    }

    #[test]
    fn test_crafting_table_click_and_shift_craft() {
        let reg = RecipeRegistry::standard();
        let mut container = CraftingTableInventory::new();
        let mut player_inv = Inventory::default();

        // Set up 1x2 sticks in grid: slot 1 (idx 0) and slot 4 (idx 3)
        container.grid[0] = ItemStack::new(7, 10);
        container.grid[3] = ItemStack::new(7, 10);
        container.update_result(&reg);

        assert_eq!(container.result, ItemStack::new(13, 4));

        // Normal pickup click on slot 0
        crafting_table_container_click(
            &mut container,
            &mut player_inv,
            0,
            ClickButton::Left,
            ClickMode::Pickup,
            &reg,
        )
        .unwrap();

        assert_eq!(player_inv.carried, ItemStack::new(13, 4));
        assert_eq!(container.grid[0].count, 9);
        assert_eq!(container.grid[3].count, 9);
        assert_eq!(container.result, ItemStack::new(13, 4));

        // Put carried away
        player_inv.return_carried();

        // Shift-craft on slot 0: should craft remaining 9 batches (36 sticks) directly into inventory
        crafting_table_container_click(
            &mut container,
            &mut player_inv,
            0,
            ClickButton::Left,
            ClickMode::QuickMove,
            &reg,
        )
        .unwrap();

        assert!(container.grid[0].is_empty());
        assert!(container.grid[3].is_empty());
        assert!(container.result.is_empty());

        let total_sticks: u16 = player_inv
            .slots
            .iter()
            .filter(|s| s.item == 13)
            .map(|s| s.count)
            .sum();
        assert_eq!(total_sticks, 40); // 4 from first craft + 36 from shift craft
    }
}
