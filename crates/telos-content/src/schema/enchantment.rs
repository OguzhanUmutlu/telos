//! Data schema for enchantment definitions (`data/<ns>/enchantments/<name>.ron` or `.json`).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Rarity tier of an enchantment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnchantmentRarityDef {
    /// Common enchantment (e.g. Protection I, Sharpness I).
    #[default]
    Common,
    /// Uncommon enchantment (e.g. Fire Protection, Feather Falling).
    Uncommon,
    /// Rare enchantment (e.g. Silk Touch, Infinity).
    Rare,
    /// Very rare enchantment (e.g. Mending).
    VeryRare,
}

/// Target equipment category for an enchantment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnchantmentTargetDef {
    /// Any armor piece (helmet, chestplate, leggings, boots).
    #[default]
    Armor,
    /// Boots only.
    Boots,
    /// Melee weapons (swords, axes).
    Weapon,
    /// Digging/mining tools (pickaxes, shovels, axes).
    Tool,
    /// Any breakable item with durability.
    Breakable,
    /// Ranged bow weapons.
    Bow,
    /// Crossbow weapons.
    Crossbow,
}

/// Data definition of an enchantment loaded from a data pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnchantmentDef {
    /// Namespaced enchantment identifier (e.g. `"minecraft:sharpness"`).
    pub id: String,
    /// User-facing display name (e.g. `"Sharpness"`).
    pub name: String,
    /// Maximum natural level for this enchantment (e.g. `5`).
    #[serde(default = "default_max_level")]
    pub max_level: u8,
    /// Relative rarity weight for enchanting tables and loot generation.
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Rarity classification.
    #[serde(default)]
    pub rarity: EnchantmentRarityDef,
    /// Equipment target category.
    #[serde(default)]
    pub target: EnchantmentTargetDef,
    /// List of conflicting enchantment identifiers that cannot coexist on the same item.
    #[serde(default)]
    pub conflicts: Vec<String>,
}

const fn default_max_level() -> u8 {
    1
}

const fn default_weight() -> u32 {
    10
}

/// Error encountered when deserializing an enchantment definition.
#[derive(Debug, Error)]
pub enum EnchantmentDefError {
    /// Failed to parse JSON definition.
    #[error("failed to parse enchantment JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Failed to parse RON definition.
    #[error("failed to parse enchantment RON: {0}")]
    Ron(#[from] ron::error::SpannedError),
}

impl EnchantmentDef {
    /// Parses an `EnchantmentDef` from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, EnchantmentDefError> {
        Ok(serde_json::from_str(s)?)
    }

    /// Parses an `EnchantmentDef` from a RON string.
    pub fn from_ron_str(s: &str) -> Result<Self, EnchantmentDefError> {
        Ok(ron::from_str(s)?)
    }

    /// Returns `true` if this enchantment conflicts with another enchantment ID.
    #[must_use]
    pub fn conflicts_with(&self, other_id: &str) -> bool {
        self.conflicts.iter().any(|c| c == other_id)
    }
}
