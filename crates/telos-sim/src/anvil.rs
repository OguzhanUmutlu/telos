use crate::enchantment::{EnchantmentKind, EnchantmentTarget};
use crate::inventory::{
    ClickButton, ClickMode, Inventory, InventoryError, ItemStack, MAX_STACK_SIZE, is_armor, is_axe,
    is_boots, is_bow, is_enchanted_book, is_sword, is_tool,
};

/// Total number of slots in an anvil container:
/// - 0: Input Left (item to repair/enchant)
/// - 1: Input Right (sacrifice item, enchanted book, or repair material)
/// - 2: Output Result
pub const ANVIL_CONTAINER_SLOTS: usize = 3;

/// Left input slot index (0).
pub const ANVIL_SLOT_LEFT: usize = 0;
/// Right input slot index (1).
pub const ANVIL_SLOT_RIGHT: usize = 1;
/// Output result slot index (2).
pub const ANVIL_SLOT_RESULT: usize = 2;

/// Total number of slots in a dual anvil container window (39):
/// - 0..3: Anvil container slots (Left, Right, Result)
/// - 3..30: Player main storage (27 slots)
/// - 30..39: Player hotbar (9 slots)
pub const DUAL_ANVIL_SLOT_COUNT: usize = 39;

/// Checks whether an enchantment kind can legally be applied to a target item ID.
#[must_use]
pub fn can_apply_enchantment(kind: EnchantmentKind, item: u32) -> bool {
    if is_enchanted_book(item) || item == crate::inventory::ITEM_BOOK {
        return true;
    }

    match kind.target() {
        EnchantmentTarget::Armor => is_armor(item),
        EnchantmentTarget::Boots => is_boots(item),
        EnchantmentTarget::Weapon => is_sword(item) || is_axe(item),
        EnchantmentTarget::Tool => is_tool(item),
        EnchantmentTarget::Bow => is_bow(item),
        EnchantmentTarget::Breakable => {
            is_armor(item) || is_tool(item) || is_sword(item) || is_bow(item)
        }
    }
}

/// The computed result of combining two items in an anvil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnvilCombineResult {
    /// Resulting item stack.
    pub result: ItemStack,
    /// Experience level cost required to retrieve the result.
    pub level_cost: u32,
    /// Number of items consumed from the right input slot.
    pub right_consumed: u16,
}

/// Combines two items in an anvil according to vanilla-style anvil combination rules.
///
/// Returns `None` if the items cannot be combined (incompatible types, incompatible
/// enchantments with no changes, or no valid operation).
#[must_use]
pub fn combine_anvil_items(left: ItemStack, right: ItemStack) -> Option<AnvilCombineResult> {
    if left.is_empty() || right.is_empty() {
        return None;
    }

    let is_right_book = is_enchanted_book(right.item);
    let is_left_book = is_enchanted_book(left.item);
    let is_same_item = left.item == right.item;

    if !is_same_item && !is_right_book {
        return None;
    }

    // Prepare result item
    let mut result_item = left;
    result_item.count = 1;
    let mut changed = false;
    let mut cost: u32 = 0;

    // In vanilla, combining two items of the same type has a base repair cost of 1 level
    if is_same_item && !is_left_book {
        cost = 1;
        changed = true;
    }

    // Merge enchantments from right onto result
    for (kind, level_r) in right.enchantments.iter() {
        // Check target item compatibility
        if !can_apply_enchantment(kind, result_item.item) {
            continue;
        }

        let level_l = result_item.enchantments.get_level(kind);
        if level_l == 0 {
            // New enchantment applied
            if result_item.enchantments.set_enchantment(kind, level_r) {
                cost = cost.saturating_add(u32::from(level_r) * kind.rarity_multiplier());
                changed = true;
            }
        } else if level_l == level_r {
            // Same level: upgrade by 1 if not max
            let new_level = (level_l + 1).min(kind.max_level());
            if new_level > level_l {
                result_item.enchantments.set_enchantment(kind, new_level);
                cost = cost.saturating_add(u32::from(new_level) * kind.rarity_multiplier());
                changed = true;
            } else {
                cost = cost.saturating_add(u32::from(level_l) * kind.rarity_multiplier());
            }
        } else {
            // Different levels: keep higher level
            let new_level = level_l.max(level_r);
            if new_level > level_l {
                result_item.enchantments.set_enchantment(kind, new_level);
                cost = cost.saturating_add(u32::from(new_level) * kind.rarity_multiplier());
                changed = true;
            } else {
                // Sacrifice item had lower level, minor sacrifice cost
                cost = cost.saturating_add(1);
            }
        }
    }

    if !changed {
        return None;
    }

    Some(AnvilCombineResult {
        result: result_item,
        level_cost: cost.max(1),
        right_consumed: 1,
    })
}

/// In-memory state of an active anvil container session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AnvilInventory {
    /// Left input slot (item to repair or enchant).
    pub left: ItemStack,
    /// Right input slot (sacrifice item, enchanted book, or repair material).
    pub right: ItemStack,
    /// Output result preview slot.
    pub result: ItemStack,
    /// Experience level cost required to take the result.
    pub level_cost: u32,
    /// Number of items consumed from right slot on take.
    pub right_consumed: u16,
    /// Optional custom rename string.
    pub custom_name: Option<String>,
}

impl AnvilInventory {
    /// Creates a new empty anvil inventory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            left: ItemStack::EMPTY,
            right: ItemStack::EMPTY,
            result: ItemStack::EMPTY,
            level_cost: 0,
            right_consumed: 1,
            custom_name: None,
        }
    }

    /// Re-evaluates item combination against left and right inputs and updates preview.
    pub fn update_result(&mut self) {
        if let Some(res) = combine_anvil_items(self.left, self.right) {
            self.result = res.result;
            self.level_cost = res.level_cost;
            self.right_consumed = res.right_consumed;
        } else {
            self.result = ItemStack::EMPTY;
            self.level_cost = 0;
            self.right_consumed = 1;
        }
    }

    /// Clears all slots and reset level cost.
    pub fn clear(&mut self) {
        self.left = ItemStack::EMPTY;
        self.right = ItemStack::EMPTY;
        self.result = ItemStack::EMPTY;
        self.level_cost = 0;
        self.right_consumed = 1;
        self.custom_name = None;
    }

    /// Returns the array of 3 container slots (Left, Right, Result).
    #[must_use]
    pub const fn slots(&self) -> [ItemStack; ANVIL_CONTAINER_SLOTS] {
        [self.left, self.right, self.result]
    }
}

/// Executes a click interaction on an anvil container window:
/// - Slot 0: Left input
/// - Slot 1: Right input
/// - Slot 2: Output result (protected, requires sufficient player level)
/// - Slots 3..30: Player main storage (27 slots)
/// - Slots 30..39: Player hotbar (9 slots)
///
/// Returns `Ok(Some(levels_spent))` if output was retrieved, or `Ok(None)` for standard moves.
#[allow(clippy::too_many_lines)]
pub fn anvil_container_click(
    container: &mut AnvilInventory,
    player_inv: &mut Inventory,
    player_level: u32,
    slot_idx: usize,
    button: ClickButton,
    mode: ClickMode,
) -> Result<Option<u32>, InventoryError> {
    if slot_idx >= DUAL_ANVIL_SLOT_COUNT {
        return Err(InventoryError::SlotOutOfBounds(slot_idx));
    }

    // 1. Result slot (2)
    if slot_idx == ANVIL_SLOT_RESULT {
        if container.result.is_empty() || player_level < container.level_cost {
            return Ok(None);
        }

        let cost = container.level_cost;

        match mode {
            ClickMode::Pickup => {
                if player_inv.carried.is_empty() {
                    player_inv.carried = container.result;
                    container.result = ItemStack::EMPTY;
                    container.left = ItemStack::EMPTY;
                    container.right.count = container
                        .right
                        .count
                        .saturating_sub(container.right_consumed);
                    container.right.normalize();
                    container.update_result();
                    return Ok(Some(cost));
                } else if player_inv.carried.item == container.result.item
                    && player_inv.carried.enchantments == container.result.enchantments
                    && player_inv.carried.count + container.result.count <= MAX_STACK_SIZE
                {
                    player_inv.carried.count += container.result.count;
                    container.result = ItemStack::EMPTY;
                    container.left = ItemStack::EMPTY;
                    container.right.count = container
                        .right
                        .count
                        .saturating_sub(container.right_consumed);
                    container.right.normalize();
                    container.update_result();
                    return Ok(Some(cost));
                }
            }
            ClickMode::QuickMove => {
                let mut moved = container.result;
                player_inv.insert_into_storage_or_hotbar(&mut moved);
                if moved.is_empty() {
                    container.result = ItemStack::EMPTY;
                    container.left = ItemStack::EMPTY;
                    container.right.count = container
                        .right
                        .count
                        .saturating_sub(container.right_consumed);
                    container.right.normalize();
                    container.update_result();
                    return Ok(Some(cost));
                }
            }
            _ => {}
        }

        return Ok(None);
    }

    // 2. Input slots (0 or 1)
    if slot_idx < 2 {
        let target = if slot_idx == 0 {
            &mut container.left
        } else {
            &mut container.right
        };

        match mode {
            ClickMode::Pickup => match button {
                ClickButton::Left => {
                    if player_inv.carried.is_empty() {
                        player_inv.carried = *target;
                        *target = ItemStack::EMPTY;
                    } else if target.is_empty() {
                        *target = player_inv.carried;
                        player_inv.carried = ItemStack::EMPTY;
                    } else if target.item == player_inv.carried.item
                        && target.enchantments == player_inv.carried.enchantments
                    {
                        let space = MAX_STACK_SIZE.saturating_sub(target.count);
                        let to_add = player_inv.carried.count.min(space);
                        target.count += to_add;
                        player_inv.carried.count -= to_add;
                        player_inv.carried.normalize();
                    } else {
                        std::mem::swap(target, &mut player_inv.carried);
                    }
                }
                ClickButton::Right => {
                    if player_inv.carried.is_empty() {
                        if !target.is_empty() {
                            let half = target.count.div_ceil(2);
                            player_inv.carried =
                                ItemStack::new_enchanted(target.item, half, target.enchantments);
                            target.count -= half;
                            target.normalize();
                        }
                    } else if target.is_empty() {
                        *target = ItemStack::new_enchanted(
                            player_inv.carried.item,
                            1,
                            player_inv.carried.enchantments,
                        );
                        player_inv.carried.count -= 1;
                        player_inv.carried.normalize();
                    } else if target.item == player_inv.carried.item
                        && target.enchantments == player_inv.carried.enchantments
                        && target.count < MAX_STACK_SIZE
                    {
                        target.count += 1;
                        player_inv.carried.count -= 1;
                        player_inv.carried.normalize();
                    }
                }
            },
            ClickMode::QuickMove => {
                let mut moved = *target;
                player_inv.insert_into_storage_or_hotbar(&mut moved);
                *target = moved;
            }
            ClickMode::Drop => {
                if !target.is_empty() {
                    match button {
                        ClickButton::Left => *target = ItemStack::EMPTY,
                        ClickButton::Right => {
                            target.count -= 1;
                            target.normalize();
                        }
                    }
                }
            }
            ClickMode::SwapHotbar => {}
        }

        container.update_result();
        return Ok(None);
    }

    // 3. Player inventory slots (3..39)
    let p_idx = if slot_idx < 30 {
        // Storage: 3..30 -> 9..36
        slot_idx + 6
    } else {
        // Hotbar: 30..39 -> 0..9
        slot_idx - 30
    };

    match mode {
        ClickMode::Pickup => {
            let target = &mut player_inv.slots[p_idx];
            match button {
                ClickButton::Left => {
                    if player_inv.carried.is_empty() {
                        player_inv.carried = *target;
                        *target = ItemStack::EMPTY;
                    } else if target.is_empty() {
                        *target = player_inv.carried;
                        player_inv.carried = ItemStack::EMPTY;
                    } else if target.item == player_inv.carried.item
                        && target.enchantments == player_inv.carried.enchantments
                    {
                        let space = MAX_STACK_SIZE.saturating_sub(target.count);
                        let to_add = player_inv.carried.count.min(space);
                        target.count += to_add;
                        player_inv.carried.count -= to_add;
                        player_inv.carried.normalize();
                    } else {
                        std::mem::swap(target, &mut player_inv.carried);
                    }
                }
                ClickButton::Right => {
                    if player_inv.carried.is_empty() {
                        if !target.is_empty() {
                            let half = target.count.div_ceil(2);
                            player_inv.carried =
                                ItemStack::new_enchanted(target.item, half, target.enchantments);
                            target.count -= half;
                            target.normalize();
                        }
                    } else if target.is_empty() {
                        *target = ItemStack::new_enchanted(
                            player_inv.carried.item,
                            1,
                            player_inv.carried.enchantments,
                        );
                        player_inv.carried.count -= 1;
                        player_inv.carried.normalize();
                    } else if target.item == player_inv.carried.item
                        && target.enchantments == player_inv.carried.enchantments
                        && target.count < MAX_STACK_SIZE
                    {
                        target.count += 1;
                        player_inv.carried.count -= 1;
                        player_inv.carried.normalize();
                    }
                }
            }
        }
        ClickMode::QuickMove => {
            let src = &mut player_inv.slots[p_idx];
            if !src.is_empty() {
                if container.left.is_empty() {
                    container.left = *src;
                    *src = ItemStack::EMPTY;
                } else if container.right.is_empty() {
                    container.right = *src;
                    *src = ItemStack::EMPTY;
                }
                container.update_result();
            }
        }
        ClickMode::Drop => {
            let target = &mut player_inv.slots[p_idx];
            if !target.is_empty() {
                match button {
                    ClickButton::Left => *target = ItemStack::EMPTY,
                    ClickButton::Right => {
                        target.count -= 1;
                        target.normalize();
                    }
                }
            }
        }
        ClickMode::SwapHotbar => {}
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{ITEM_ENCHANTED_BOOK, ITEM_IRON_SWORD};

    const MOCK_SWORD: u32 = ITEM_IRON_SWORD;

    #[test]
    fn test_combine_two_swords_with_enchantments() {
        let mut sword_a = ItemStack::new(MOCK_SWORD, 1);
        sword_a
            .enchantments
            .set_enchantment(EnchantmentKind::Sharpness, 2);

        let mut sword_b = ItemStack::new(MOCK_SWORD, 1);
        sword_b
            .enchantments
            .set_enchantment(EnchantmentKind::Sharpness, 2);
        sword_b
            .enchantments
            .set_enchantment(EnchantmentKind::Unbreaking, 1);

        let res = combine_anvil_items(sword_a, sword_b).expect("should combine swords");
        assert_eq!(res.result.item, MOCK_SWORD);
        // Sharpness 2 + 2 -> 3
        assert_eq!(
            res.result
                .enchantments
                .get_level(EnchantmentKind::Sharpness),
            3
        );
        // Unbreaking 1 transferred
        assert_eq!(
            res.result
                .enchantments
                .get_level(EnchantmentKind::Unbreaking),
            1
        );
        assert!(res.level_cost >= 4);
    }

    #[test]
    fn test_combine_sword_with_enchanted_book() {
        let sword = ItemStack::new(MOCK_SWORD, 1);

        let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
        book.enchantments
            .set_enchantment(EnchantmentKind::FireAspect, 2);

        let res = combine_anvil_items(sword, book).expect("should apply book to sword");
        assert_eq!(res.result.item, MOCK_SWORD);
        assert_eq!(
            res.result
                .enchantments
                .get_level(EnchantmentKind::FireAspect),
            2
        );
        // FireAspect rarity mult is 2 -> level 2 * 2 = 4 cost
        assert_eq!(res.level_cost, 4);
    }

    #[test]
    fn test_combine_incompatible_enchantments_prevented() {
        let mut sword = ItemStack::new(MOCK_SWORD, 1);
        sword
            .enchantments
            .set_enchantment(EnchantmentKind::Sharpness, 3);

        let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
        book.enchantments.set_enchantment(EnchantmentKind::Smite, 4);

        // Smite cannot combine with Sharpness
        let res = combine_anvil_items(sword, book);
        assert!(res.is_none());
    }

    #[test]
    fn test_anvil_container_click_deduction() {
        let mut anvil = AnvilInventory::new();
        let mut player_inv = Inventory::default();

        anvil.left = ItemStack::new(MOCK_SWORD, 1);
        let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
        book.enchantments
            .set_enchantment(EnchantmentKind::Sharpness, 1);
        anvil.right = book;
        anvil.update_result();

        assert!(!anvil.result.is_empty());
        let required_cost = anvil.level_cost;

        // Player with 0 levels cannot take result
        let res_fail = anvil_container_click(
            &mut anvil,
            &mut player_inv,
            0,
            ANVIL_SLOT_RESULT,
            ClickButton::Left,
            ClickMode::Pickup,
        )
        .unwrap();
        assert_eq!(res_fail, None);
        assert!(player_inv.carried.is_empty());

        // Player with sufficient level takes result
        let res_success = anvil_container_click(
            &mut anvil,
            &mut player_inv,
            required_cost + 5,
            ANVIL_SLOT_RESULT,
            ClickButton::Left,
            ClickMode::Pickup,
        )
        .unwrap();
        assert_eq!(res_success, Some(required_cost));
        assert_eq!(player_inv.carried.item, MOCK_SWORD);
        assert_eq!(
            player_inv
                .carried
                .enchantments
                .get_level(EnchantmentKind::Sharpness),
            1
        );
        assert!(anvil.left.is_empty());
        assert!(anvil.right.is_empty());
    }
}
