//! Data schema for item definitions (`data/<ns>/item/<name>.ron` or `.json`).

use serde::{Deserialize, Serialize};

/// Armor slot classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArmorSlotDef {
    /// Head / Helmet slot.
    Helmet,
    /// Torso / Chestplate slot.
    Chestplate,
    /// Legs / Leggings slot.
    Leggings,
    /// Feet / Boots slot.
    Boots,
}

/// Specialized item behavior and properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ItemTypeDef {
    /// Standard generic item or crafting ingredient.
    #[default]
    Generic,
    /// Item that places a block when used on a block face.
    Block(String),
    /// Equipable armor item.
    Armor {
        /// Target armor equipment slot.
        slot: ArmorSlotDef,
        /// Armor defense rating.
        defense: u32,
        /// Armor toughness rating.
        #[serde(default)]
        toughness: f32,
    },
    /// Mining or combat tool item.
    Tool {
        /// Tool category (e.g. "pickaxe", "axe", "shovel", "sword").
        kind: String,
        /// Material tier (e.g. "wood", "stone", "iron", "diamond").
        #[serde(default)]
        tier: String,
        /// Mining speed multiplier.
        #[serde(default = "default_mining_speed")]
        mining_speed: f32,
        /// Attack damage bonus.
        #[serde(default = "default_attack_damage")]
        attack_damage: f32,
    },
    /// Consumable potion item.
    Potion {
        /// Potion effect descriptor (e.g. "swiftness", "healing").
        potion_type: String,
    },
}

fn default_mining_speed() -> f32 {
    1.0
}

fn default_attack_damage() -> f32 {
    1.0
}

/// Data-driven definition of an item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Item")]
pub struct ItemDef {
    /// Human-readable display name.
    #[serde(default)]
    pub name: String,
    /// Maximum stack size (1..=64).
    #[serde(default = "default_max_stack_size")]
    pub max_stack_size: u16,
    /// Specialized item type and attributes.
    #[serde(default)]
    pub item_type: ItemTypeDef,
}

fn default_max_stack_size() -> u16 {
    64
}

impl Default for ItemDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            max_stack_size: default_max_stack_size(),
            item_type: ItemTypeDef::Generic,
        }
    }
}

impl ItemDef {
    /// Creates a generic item with the given display name.
    pub fn new_generic(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_stack_size: 64,
            item_type: ItemTypeDef::Generic,
        }
    }

    /// Creates a block item that places the specified block identifier.
    pub fn new_block_item(name: impl Into<String>, block_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_stack_size: 64,
            item_type: ItemTypeDef::Block(block_id.into()),
        }
    }

    /// Creates an armor item.
    pub fn new_armor(name: impl Into<String>, slot: ArmorSlotDef, defense: u32) -> Self {
        Self {
            name: name.into(),
            max_stack_size: 1,
            item_type: ItemTypeDef::Armor {
                slot,
                defense,
                toughness: 0.0,
            },
        }
    }

    /// Returns whether this item is equipable armor.
    #[must_use]
    pub fn is_armor(&self) -> bool {
        matches!(self.item_type, ItemTypeDef::Armor { .. })
    }

    /// Returns the target armor slot if this is an armor item.
    #[must_use]
    pub fn armor_slot(&self) -> Option<ArmorSlotDef> {
        match self.item_type {
            ItemTypeDef::Armor { slot, .. } => Some(slot),
            _ => None,
        }
    }

    /// Returns the armor defense value if this is an armor item.
    #[must_use]
    pub fn armor_defense(&self) -> u32 {
        match &self.item_type {
            ItemTypeDef::Armor { defense, .. } => *defense,
            _ => 0,
        }
    }

    /// Returns the armor toughness value if this is an armor item.
    #[must_use]
    pub fn armor_toughness(&self) -> f32 {
        match &self.item_type {
            ItemTypeDef::Armor { toughness, .. } => *toughness,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ron_item_deserialization() {
        let ron_str = r#"
            Item(
                name: "Ruby Gem",
                max_stack_size: 64,
                item_type: Generic,
            )
        "#;
        let def: ItemDef = ron::from_str(ron_str).expect("Valid RON ItemDef");
        assert_eq!(def.name, "Ruby Gem");
        assert_eq!(def.max_stack_size, 64);
        assert_eq!(def.item_type, ItemTypeDef::Generic);
    }

    #[test]
    fn test_ron_armor_deserialization() {
        let ron_str = r#"
            Item(
                name: "Iron Helmet",
                max_stack_size: 1,
                item_type: Armor(
                    slot: Helmet,
                    defense: 2,
                    toughness: 0.0,
                ),
            )
        "#;
        let def: ItemDef = ron::from_str(ron_str).expect("Valid RON Armor ItemDef");
        assert_eq!(def.name, "Iron Helmet");
        assert_eq!(def.max_stack_size, 1);
        assert_eq!(def.armor_slot(), Some(ArmorSlotDef::Helmet));
    }
}
