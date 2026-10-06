//! 2x2 Crafting engine and deterministic recipe matching.

use crate::inventory::ItemStack;

/// A 2x2 crafting recipe definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe2x2 {
    /// Input item IDs for the 4 slots: [top-left, top-right, bottom-left, bottom-right].
    /// 0 indicates an empty slot.
    pub pattern: [u32; 4],
    /// Output produced by one craft operation.
    pub result: ItemStack,
}

impl Recipe2x2 {
    /// Creates a new 2x2 recipe.
    #[must_use]
    pub const fn new(pattern: [u32; 4], result: ItemStack) -> Self {
        Self { pattern, result }
    }

    /// Checks if the provided 4 input item stacks match this recipe pattern.
    ///
    /// Every required slot must have `item == pattern[i]` and `count >= 1`.
    /// Every slot with `pattern[i] == 0` must be empty (`count == 0` or `item == 0`).
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

/// Evaluates a 2x2 crafting input grid and returns the matching recipe output, if any.
///
/// Supported default recipes:
/// - 1 Oak Log (5) anywhere -> 4 Oak Planks (7)
/// - 2 Oak Planks (7) vertically stacked -> 4 Sticks (13)
/// - 4 Oak Planks (7) -> 1 Crafting Table (14)
/// - 1 Oak Planks (7) over 1 Stick (13) vertically -> 4 Torches (15)
/// - 2 Stone (1) horizontally -> 4 Stone Slabs (10)
/// - 1 Cobblestone (4) anywhere -> 2 Stone Slabs (10)
#[must_use]
pub fn find_recipe_2x2(inputs: &[ItemStack; 4]) -> Option<ItemStack> {
    // 1. Single Oak Log (5) anywhere -> 4 Oak Planks (7)
    let non_empty_count = inputs.iter().filter(|s| !s.is_empty()).count();
    if non_empty_count == 1 {
        let single = inputs.iter().find(|s| !s.is_empty()).unwrap();
        if single.item == 5 {
            // Oak Log -> 4 Oak Planks
            return Some(ItemStack::new(7, 4));
        }
        if single.item == 4 {
            // Cobblestone -> 2 Stone Slabs
            return Some(ItemStack::new(10, 2));
        }
    }

    // 2. 4 Oak Planks (7) -> 1 Crafting Table (14)
    if non_empty_count == 4 && inputs.iter().all(|s| s.item == 7) {
        return Some(ItemStack::new(14, 1));
    }

    // 3. 2 Oak Planks (7) vertically -> 4 Sticks (13)
    if non_empty_count == 2 {
        // Left column: [0, 2]
        if inputs[0].item == 7
            && inputs[2].item == 7
            && inputs[1].is_empty()
            && inputs[3].is_empty()
        {
            return Some(ItemStack::new(13, 4));
        }
        // Right column: [1, 3]
        if inputs[1].item == 7
            && inputs[3].item == 7
            && inputs[0].is_empty()
            && inputs[2].is_empty()
        {
            return Some(ItemStack::new(13, 4));
        }

        // 4. 1 Planks (7) over 1 Stick (13) vertically -> 4 Torches (15)
        if inputs[0].item == 7
            && inputs[2].item == 13
            && inputs[1].is_empty()
            && inputs[3].is_empty()
        {
            return Some(ItemStack::new(15, 4));
        }
        if inputs[1].item == 7
            && inputs[3].item == 13
            && inputs[0].is_empty()
            && inputs[2].is_empty()
        {
            return Some(ItemStack::new(15, 4));
        }

        // 5. 2 Stone (1) horizontally -> 4 Stone Slabs (10)
        // Top row: [0, 1]
        if inputs[0].item == 1
            && inputs[1].item == 1
            && inputs[2].is_empty()
            && inputs[3].is_empty()
        {
            return Some(ItemStack::new(10, 4));
        }
        // Bottom row: [2, 3]
        if inputs[2].item == 1
            && inputs[3].item == 1
            && inputs[0].is_empty()
            && inputs[1].is_empty()
        {
            return Some(ItemStack::new(10, 4));
        }
    }

    None
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
    fn test_planks_and_stick_to_torches() {
        let mut inputs = [ItemStack::EMPTY; 4];
        inputs[0] = ItemStack::new(7, 1); // Planks on top
        inputs[2] = ItemStack::new(13, 1); // Stick on bottom
        assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(15, 4)));
    }

    #[test]
    fn test_stone_to_slabs_horizontal() {
        let mut inputs = [ItemStack::EMPTY; 4];
        inputs[0] = ItemStack::new(1, 1);
        inputs[1] = ItemStack::new(1, 1);
        assert_eq!(find_recipe_2x2(&inputs), Some(ItemStack::new(10, 4)));
    }
}
