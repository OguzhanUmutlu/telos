//! Loot table definitions and deterministic rolling engine for dungeon & ruin chests.

use crate::math::hash3;

/// A rolled item entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LootItem {
    /// Item identifier name in `telos` namespace (e.g. `"iron_ingot"`).
    pub name: &'static str,
    /// Stack count.
    pub count: u16,
}

/// An entry in a loot table with min/max count and selection weight.
#[derive(Debug, Clone, Copy)]
pub struct LootEntry {
    /// Canonical item identifier name.
    pub item_name: &'static str,
    /// Minimum stack size.
    pub min_count: u16,
    /// Maximum stack size.
    pub max_count: u16,
    /// Relative probability weight.
    pub weight: u32,
}

/// Available loot table archetypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LootTableKind {
    /// Subterranean dungeon chest with valuable metals, saddles, redstone, and mob drops.
    DungeonChest,
    /// Surface ruin chest with exploratory provisions, tools, coal, and flora.
    SurfaceRuinChest,
}

/// Dungeon chest loot entries.
pub const DUNGEON_LOOT_POOL: &[LootEntry] = &[
    LootEntry {
        item_name: "iron_ingot",
        min_count: 1,
        max_count: 4,
        weight: 100,
    },
    LootEntry {
        item_name: "gold_ingot",
        min_count: 1,
        max_count: 3,
        weight: 50,
    },
    LootEntry {
        item_name: "redstone_ore",
        min_count: 1,
        max_count: 4,
        weight: 50,
    },
    LootEntry {
        item_name: "coal",
        min_count: 1,
        max_count: 4,
        weight: 100,
    },
    LootEntry {
        item_name: "gunpowder",
        min_count: 1,
        max_count: 5,
        weight: 80,
    },
    LootEntry {
        item_name: "string",
        min_count: 1,
        max_count: 5,
        weight: 80,
    },
    LootEntry {
        item_name: "bread",
        min_count: 1,
        max_count: 3,
        weight: 70,
    },
    LootEntry {
        item_name: "wheat",
        min_count: 1,
        max_count: 4,
        weight: 70,
    },
    LootEntry {
        item_name: "rotten_flesh",
        min_count: 1,
        max_count: 8,
        weight: 100,
    },
    LootEntry {
        item_name: "torch",
        min_count: 2,
        max_count: 6,
        weight: 90,
    },
    LootEntry {
        item_name: "saddle",
        min_count: 1,
        max_count: 1,
        weight: 25,
    },
    LootEntry {
        item_name: "name_tag",
        min_count: 1,
        max_count: 1,
        weight: 25,
    },
];

/// Surface ruin chest loot entries.
pub const SURFACE_RUIN_LOOT_POOL: &[LootEntry] = &[
    LootEntry {
        item_name: "wheat",
        min_count: 2,
        max_count: 6,
        weight: 100,
    },
    LootEntry {
        item_name: "bread",
        min_count: 1,
        max_count: 3,
        weight: 80,
    },
    LootEntry {
        item_name: "coal",
        min_count: 1,
        max_count: 5,
        weight: 90,
    },
    LootEntry {
        item_name: "iron_ingot",
        min_count: 1,
        max_count: 3,
        weight: 60,
    },
    LootEntry {
        item_name: "gold_ingot",
        min_count: 1,
        max_count: 2,
        weight: 30,
    },
    LootEntry {
        item_name: "stick",
        min_count: 2,
        max_count: 8,
        weight: 100,
    },
    LootEntry {
        item_name: "leather",
        min_count: 1,
        max_count: 3,
        weight: 70,
    },
    LootEntry {
        item_name: "torch",
        min_count: 2,
        max_count: 5,
        weight: 80,
    },
    LootEntry {
        item_name: "poppy",
        min_count: 1,
        max_count: 3,
        weight: 50,
    },
];

/// Deterministically rolls loot items for a given loot table.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn roll_loot(seed: u64, salt: u64, kind: LootTableKind) -> Vec<LootItem> {
    let (pool, min_rolls, max_rolls) = match kind {
        LootTableKind::DungeonChest => (DUNGEON_LOOT_POOL, 4usize, 8usize),
        LootTableKind::SurfaceRuinChest => (SURFACE_RUIN_LOOT_POOL, 3usize, 6usize),
    };

    let total_weight: u32 = pool.iter().map(|e| e.weight).sum();
    let num_rolls_hash = hash3(seed ^ salt, 0x1234_5678, 0x7654_3210, 0);
    let num_rolls = min_rolls + (num_rolls_hash as usize % (max_rolls - min_rolls + 1));

    let mut items = Vec::with_capacity(num_rolls);
    for roll_idx in 0..num_rolls {
        let roll_hash = hash3(seed ^ salt, roll_idx as i32, 0x4152_6374, 1);
        let mut target_weight = roll_hash % u64::from(total_weight);

        for entry in pool {
            if target_weight < u64::from(entry.weight) {
                let count_range = entry.max_count - entry.min_count + 1;
                let count_hash = hash3(roll_hash, roll_idx as i32, 0x2233_4455, 2);
                let count = entry.min_count + (count_hash % u64::from(count_range)) as u16;
                items.push(LootItem {
                    name: entry.item_name,
                    count,
                });
                break;
            }
            target_weight -= u64::from(entry.weight);
        }
    }

    items
}

/// Deterministically rolls chest items mapped into distinct container slots $0..27$.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn roll_chest_slots(seed: u64, salt: u64, kind: LootTableKind) -> Vec<(u8, LootItem)> {
    let items = roll_loot(seed, salt, kind);
    let mut slots_result = Vec::with_capacity(items.len());
    let mut used_slots = [false; 27];

    for (i, item) in items.into_iter().enumerate() {
        let slot_hash = hash3(seed ^ salt, i as i32, 0x55AA_33CC, 3);
        let mut candidate_slot = (slot_hash % 27) as usize;

        // Linear probe for open slot
        let mut attempts = 0;
        while used_slots[candidate_slot] && attempts < 27 {
            candidate_slot = (candidate_slot + 1) % 27;
            attempts += 1;
        }

        if !used_slots[candidate_slot] {
            used_slots[candidate_slot] = true;
            slots_result.push((candidate_slot as u8, item));
        }
    }

    slots_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dungeon_loot_determinism() {
        let seed = 0x1234_5678_9ABC;
        let salt = 42;
        let loot1 = roll_loot(seed, salt, LootTableKind::DungeonChest);
        let loot2 = roll_loot(seed, salt, LootTableKind::DungeonChest);

        assert_eq!(
            loot1, loot2,
            "Loot rolls with identical seed must match exactly"
        );
        assert!(loot1.len() >= 4 && loot1.len() <= 8);
    }

    #[test]
    fn test_chest_slots_distinct() {
        let seed = 0x9999_8888_7777;
        let salt = 100;
        let slots = roll_chest_slots(seed, salt, LootTableKind::DungeonChest);

        let mut occupied_slots = std::collections::HashSet::new();
        for (slot, item) in slots {
            assert!(slot < 27);
            assert!(item.count > 0);
            assert!(occupied_slots.insert(slot), "Slots must be distinct");
        }
    }
}
