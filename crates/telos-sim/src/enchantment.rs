//! Enchantment definitions, target matrices, compatibility rules, and compact bitpack representation.

use crate::attributes::DamageType;

/// Supported enchantment kinds in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnchantmentKind {
    /// General damage reduction across all damage types (EPF = level, max IV).
    Protection = 1,
    /// Reduces fire and lava damage (EPF = level * 2, max IV).
    FireProtection = 2,
    /// Reduces fall damage (EPF = level * 3, max IV).
    FeatherFalling = 3,
    /// Reduces explosion / blast damage (EPF = level * 2, max IV).
    BlastProtection = 4,
    /// Extra melee weapon attack damage (0.5 * level + 0.5, max V).
    Sharpness = 5,
    /// Increases melee attack knockback (+0.5 * level blocks, max II).
    Knockback = 6,
    /// Increases tool mining speed (base + level^2 + 1, max V).
    Efficiency = 7,
    /// Gives chance to avoid item durability loss (max III).
    Unbreaking = 8,
    /// Repairs item durability using collected experience (max I).
    Mending = 9,
}

impl EnchantmentKind {
    /// Numeric ID for compact packing and serialization.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Resolves numeric ID to `EnchantmentKind`.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(Self::Protection),
            2 => Some(Self::FireProtection),
            3 => Some(Self::FeatherFalling),
            4 => Some(Self::BlastProtection),
            5 => Some(Self::Sharpness),
            6 => Some(Self::Knockback),
            7 => Some(Self::Efficiency),
            8 => Some(Self::Unbreaking),
            9 => Some(Self::Mending),
            _ => None,
        }
    }

    /// User-facing display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Protection => "Protection",
            Self::FireProtection => "Fire Protection",
            Self::FeatherFalling => "Feather Falling",
            Self::BlastProtection => "Blast Protection",
            Self::Sharpness => "Sharpness",
            Self::Knockback => "Knockback",
            Self::Efficiency => "Efficiency",
            Self::Unbreaking => "Unbreaking",
            Self::Mending => "Mending",
        }
    }

    /// Maximum natural level for this enchantment.
    #[must_use]
    pub const fn max_level(self) -> u8 {
        match self {
            Self::Protection
            | Self::FireProtection
            | Self::FeatherFalling
            | Self::BlastProtection => 4,
            Self::Sharpness | Self::Efficiency => 5,
            Self::Unbreaking => 3,
            Self::Knockback => 2,
            Self::Mending => 1,
        }
    }

    /// Equipment target category for this enchantment.
    #[must_use]
    pub const fn target(self) -> EnchantmentTarget {
        match self {
            Self::Protection | Self::FireProtection | Self::BlastProtection => {
                EnchantmentTarget::Armor
            }
            Self::FeatherFalling => EnchantmentTarget::Boots,
            Self::Sharpness | Self::Knockback => EnchantmentTarget::Weapon,
            Self::Efficiency => EnchantmentTarget::Tool,
            Self::Unbreaking | Self::Mending => EnchantmentTarget::Breakable,
        }
    }

    /// Checks if two enchantments can coexist on the same item.
    #[must_use]
    pub const fn is_compatible_with(self, other: Self) -> bool {
        if (self as u8) == (other as u8) {
            return true;
        }
        // Protection, FireProtection, BlastProtection are mutually exclusive
        let is_prot_a = matches!(
            self,
            Self::Protection | Self::FireProtection | Self::BlastProtection
        );
        let is_prot_b = matches!(
            other,
            Self::Protection | Self::FireProtection | Self::BlastProtection
        );
        !(is_prot_a && is_prot_b)
    }

    /// Calculates Enchantment Protection Factor (EPF) for this enchantment against a damage type.
    #[must_use]
    pub fn epf_for_damage(self, level: u8, damage_type: DamageType) -> u8 {
        let lvl = level.min(self.max_level());
        match self {
            Self::Protection
                if matches!(
                    damage_type,
                    DamageType::Attack | DamageType::Fall | DamageType::Fire | DamageType::Generic
                ) =>
            {
                lvl
            }
            Self::FireProtection if damage_type == DamageType::Fire => lvl.saturating_mul(2),
            Self::FeatherFalling if damage_type == DamageType::Fall => lvl.saturating_mul(3),
            Self::BlastProtection if damage_type == DamageType::Generic => lvl.saturating_mul(2),
            _ => 0,
        }
    }
}

/// Target item classification for enchantments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnchantmentTarget {
    /// Any armor piece (helmet, chestplate, leggings, boots).
    Armor,
    /// Boots only.
    Boots,
    /// Swords and weapons.
    Weapon,
    /// Digging/mining tools (pickaxe, axe, shovel).
    Tool,
    /// Any item with durability.
    Breakable,
}

/// Compact 64-bit (`u64`) packed representation storing up to 4 enchantments.
///
/// Bit layout per 16-bit slot (4 slots total):
/// - Bits 0..=7: `EnchantmentKind` numeric ID (0 = Empty).
/// - Bits 8..=15: Enchantment level (1..=5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct CompactEnchantments(pub u64);

impl CompactEnchantments {
    /// Empty enchantment set constant.
    pub const EMPTY: Self = Self(0);

    /// Creates an empty enchantment set.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Returns `true` if there are no enchantments attached.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Retrieves the level of a specific enchantment (0 if absent).
    #[must_use]
    pub fn get_level(self, kind: EnchantmentKind) -> u8 {
        let target_id = kind.id();
        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            let id = (slot & 0xFF) as u8;
            let level = (slot >> 8) as u8;
            if id == target_id {
                return level;
            }
        }
        0
    }

    /// Sets or updates an enchantment. Returns `true` if successfully applied, or `false`
    /// if incompatible with existing enchantments or all 4 slots are occupied.
    pub fn set_enchantment(&mut self, kind: EnchantmentKind, level: u8) -> bool {
        let level = level.clamp(1, kind.max_level());
        let target_id = kind.id();

        // 1. Check compatibility with existing enchantments
        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            let id = (slot & 0xFF) as u8;
            if id != 0
                && id != target_id
                && let Some(existing_kind) = EnchantmentKind::from_u8(id)
                && !kind.is_compatible_with(existing_kind)
            {
                return false;
            }
        }

        // 2. If already present in a slot, update its level
        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            let id = (slot & 0xFF) as u8;
            if id == target_id {
                let new_slot = (u64::from(target_id)) | ((u64::from(level)) << 8);
                let mask = !(0xFFFFu64 << shift);
                self.0 = (self.0 & mask) | (new_slot << shift);
                return true;
            }
        }

        // 3. Otherwise find the first empty slot (slot == 0)
        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            if slot == 0 {
                let new_slot = (u64::from(target_id)) | ((u64::from(level)) << 8);
                let mask = !(0xFFFFu64 << shift);
                self.0 = (self.0 & mask) | (new_slot << shift);
                return true;
            }
        }

        // All 4 slots full
        false
    }

    /// Removes an enchantment by kind. Returns `true` if found and removed.
    pub fn remove_enchantment(&mut self, kind: EnchantmentKind) -> bool {
        let target_id = kind.id();
        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            let id = (slot & 0xFF) as u8;
            if id == target_id {
                let mask = !(0xFFFFu64 << shift);
                self.0 &= mask;
                return true;
            }
        }
        false
    }

    /// Returns an iterator over all present `(EnchantmentKind, level)` pairs.
    pub fn iter(self) -> impl Iterator<Item = (EnchantmentKind, u8)> {
        let mut list = [None; 4];
        let mut count = 0;

        for shift in (0..64).step_by(16) {
            let slot = ((self.0 >> shift) & 0xFFFF) as u16;
            let id = (slot & 0xFF) as u8;
            let level = (slot >> 8) as u8;
            if id != 0
                && let Some(kind) = EnchantmentKind::from_u8(id)
            {
                list[count] = Some((kind, level));
                count += 1;
            }
        }

        list.into_iter().flatten()
    }

    /// Calculates total EPF contribution from this item's enchantments against a damage type.
    #[must_use]
    pub fn epf_for_damage(self, damage_type: DamageType) -> u8 {
        let mut total_epf = 0u8;
        for (kind, level) in self.iter() {
            total_epf = total_epf.saturating_add(kind.epf_for_damage(level, damage_type));
        }
        total_epf
    }
}

/// Computes the aggregate Enchantment Protection Factor (EPF) across all equipped armor pieces
/// (clamped to the engine maximum of 20 EPF).
#[must_use]
pub fn calculate_total_epf(pieces: &[CompactEnchantments], damage_type: DamageType) -> u8 {
    let mut sum: u16 = 0;
    for piece in pieces {
        sum += u16::from(piece.epf_for_damage(damage_type));
    }
    (sum.min(20)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_enchantments_packing() {
        let mut ench = CompactEnchantments::new();
        assert!(ench.is_empty());

        assert!(ench.set_enchantment(EnchantmentKind::Protection, 4));
        assert_eq!(ench.get_level(EnchantmentKind::Protection), 4);
        assert!(!ench.is_empty());

        assert!(ench.set_enchantment(EnchantmentKind::Unbreaking, 3));
        assert_eq!(ench.get_level(EnchantmentKind::Unbreaking), 3);

        // Protection and FireProtection are incompatible
        assert!(!ench.set_enchantment(EnchantmentKind::FireProtection, 2));
        assert_eq!(ench.get_level(EnchantmentKind::FireProtection), 0);

        // Update level of existing
        assert!(ench.set_enchantment(EnchantmentKind::Protection, 2));
        assert_eq!(ench.get_level(EnchantmentKind::Protection), 2);

        // Remove
        assert!(ench.remove_enchantment(EnchantmentKind::Protection));
        assert_eq!(ench.get_level(EnchantmentKind::Protection), 0);
        assert_eq!(ench.get_level(EnchantmentKind::Unbreaking), 3);
    }

    #[test]
    fn test_epf_calculation() {
        let mut helmet = CompactEnchantments::new();
        helmet.set_enchantment(EnchantmentKind::Protection, 4); // EPF = 4

        let mut chest = CompactEnchantments::new();
        chest.set_enchantment(EnchantmentKind::Protection, 4); // EPF = 4

        let mut legs = CompactEnchantments::new();
        legs.set_enchantment(EnchantmentKind::Protection, 4); // EPF = 4

        let mut boots = CompactEnchantments::new();
        boots.set_enchantment(EnchantmentKind::Protection, 4); // EPF = 4
        boots.set_enchantment(EnchantmentKind::FeatherFalling, 4); // Fall EPF = 12

        // General attack: 4 pieces * 4 = 16 EPF
        let epf_attack = calculate_total_epf(&[helmet, chest, legs, boots], DamageType::Attack);
        assert_eq!(epf_attack, 16);

        // Fall damage: 16 (Protection) + 12 (Feather Falling) = 28 -> capped at 20!
        let epf_fall = calculate_total_epf(&[helmet, chest, legs, boots], DamageType::Fall);
        assert_eq!(epf_fall, 20);
    }
}
