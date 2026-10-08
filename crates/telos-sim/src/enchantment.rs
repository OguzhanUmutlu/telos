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
    /// Extra damage against undead mobs (+2.5 * level, max V).
    Smite = 10,
    /// Extra damage against spider/arthropod mobs (+2.5 * level, max V).
    BaneOfArthropods = 11,
    /// Sets target on fire for 4 * level seconds (max II).
    FireAspect = 12,
    /// Increases mob loot drops (max III).
    Looting = 13,
    /// Causes mined blocks to drop themselves rather than usual drops (max I).
    SilkTouch = 14,
    /// Increases block drop quantities for ores and crops (max III).
    Fortune = 15,
    /// Increases arrow damage (+25% * (level + 1), max V).
    Power = 16,
    /// Increases arrow knockback impulse (+0.6 * level blocks, max II).
    Punch = 17,
    /// Ignites arrows so they ignite target on hit (max I).
    Flame = 18,
    /// Shooting arrows does not consume regular arrows from inventory (max I).
    Infinity = 19,
    /// Reduces projectile damage from arrows (EPF = level * 2, max IV).
    ProjectileProtection = 20,
    /// Extends underwater breathing time and reduces drowning damage (max III).
    Respiration = 21,
    /// Normal underwater mining speed without penalty (max I).
    AquaAffinity = 22,
    /// Chance to reflect damage to melee attackers (max III).
    Thorns = 23,
    /// Increases underwater swimming movement speed (max III).
    DepthStrider = 24,
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
            10 => Some(Self::Smite),
            11 => Some(Self::BaneOfArthropods),
            12 => Some(Self::FireAspect),
            13 => Some(Self::Looting),
            14 => Some(Self::SilkTouch),
            15 => Some(Self::Fortune),
            16 => Some(Self::Power),
            17 => Some(Self::Punch),
            18 => Some(Self::Flame),
            19 => Some(Self::Infinity),
            20 => Some(Self::ProjectileProtection),
            21 => Some(Self::Respiration),
            22 => Some(Self::AquaAffinity),
            23 => Some(Self::Thorns),
            24 => Some(Self::DepthStrider),
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
            Self::Smite => "Smite",
            Self::BaneOfArthropods => "Bane of Arthropods",
            Self::FireAspect => "Fire Aspect",
            Self::Looting => "Looting",
            Self::SilkTouch => "Silk Touch",
            Self::Fortune => "Fortune",
            Self::Power => "Power",
            Self::Punch => "Punch",
            Self::Flame => "Flame",
            Self::Infinity => "Infinity",
            Self::ProjectileProtection => "Projectile Protection",
            Self::Respiration => "Respiration",
            Self::AquaAffinity => "Aqua Affinity",
            Self::Thorns => "Thorns",
            Self::DepthStrider => "Depth Strider",
        }
    }

    /// Maximum natural level for this enchantment.
    #[must_use]
    pub const fn max_level(self) -> u8 {
        match self {
            Self::Protection
            | Self::FireProtection
            | Self::FeatherFalling
            | Self::BlastProtection
            | Self::ProjectileProtection => 4,
            Self::Sharpness
            | Self::Efficiency
            | Self::Smite
            | Self::BaneOfArthropods
            | Self::Power => 5,
            Self::Unbreaking
            | Self::Looting
            | Self::Fortune
            | Self::Respiration
            | Self::Thorns
            | Self::DepthStrider => 3,
            Self::Knockback | Self::FireAspect | Self::Punch => 2,
            Self::Mending | Self::SilkTouch | Self::Flame | Self::Infinity | Self::AquaAffinity => {
                1
            }
        }
    }

    /// Equipment target category for this enchantment.
    #[must_use]
    pub const fn target(self) -> EnchantmentTarget {
        match self {
            Self::Protection
            | Self::FireProtection
            | Self::BlastProtection
            | Self::ProjectileProtection
            | Self::Respiration
            | Self::Thorns => EnchantmentTarget::Armor,
            Self::FeatherFalling | Self::DepthStrider => EnchantmentTarget::Boots,
            Self::Sharpness
            | Self::Smite
            | Self::BaneOfArthropods
            | Self::Knockback
            | Self::FireAspect
            | Self::Looting => EnchantmentTarget::Weapon,
            Self::Efficiency | Self::SilkTouch | Self::Fortune | Self::AquaAffinity => {
                EnchantmentTarget::Tool
            }
            Self::Power | Self::Punch | Self::Flame | Self::Infinity => EnchantmentTarget::Bow,
            Self::Unbreaking | Self::Mending => EnchantmentTarget::Breakable,
        }
    }

    /// Checks if two enchantments can coexist on the same item.
    #[must_use]
    pub const fn is_compatible_with(self, other: Self) -> bool {
        if (self as u8) == (other as u8) {
            return true;
        }
        // Protection, FireProtection, BlastProtection, ProjectileProtection are mutually exclusive
        let is_prot_a = matches!(
            self,
            Self::Protection
                | Self::FireProtection
                | Self::BlastProtection
                | Self::ProjectileProtection
        );
        let is_prot_b = matches!(
            other,
            Self::Protection
                | Self::FireProtection
                | Self::BlastProtection
                | Self::ProjectileProtection
        );
        if is_prot_a && is_prot_b {
            return false;
        }

        // Sharpness, Smite, BaneOfArthropods are mutually exclusive
        let is_dmg_a = matches!(self, Self::Sharpness | Self::Smite | Self::BaneOfArthropods);
        let is_dmg_b = matches!(
            other,
            Self::Sharpness | Self::Smite | Self::BaneOfArthropods
        );
        if is_dmg_a && is_dmg_b {
            return false;
        }

        // SilkTouch and Fortune are mutually exclusive
        let is_drop_a = matches!(self, Self::SilkTouch | Self::Fortune);
        let is_drop_b = matches!(other, Self::SilkTouch | Self::Fortune);
        if is_drop_a && is_drop_b {
            return false;
        }

        // Infinity and Mending are mutually exclusive
        let is_inf_mend_a = matches!(self, Self::Infinity | Self::Mending);
        let is_inf_mend_b = matches!(other, Self::Infinity | Self::Mending);
        if is_inf_mend_a && is_inf_mend_b {
            return false;
        }

        true
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
            Self::ProjectileProtection
                if matches!(damage_type, DamageType::Attack | DamageType::Generic) =>
            {
                lvl.saturating_mul(2)
            }
            _ => 0,
        }
    }
    /// Returns the anvil level cost multiplier based on enchantment rarity.
    #[must_use]
    pub const fn rarity_multiplier(self) -> u32 {
        match self {
            Self::Protection | Self::Sharpness | Self::Efficiency | Self::Unbreaking => 1,
            Self::FireProtection
            | Self::FeatherFalling
            | Self::Smite
            | Self::BaneOfArthropods
            | Self::Knockback
            | Self::FireAspect
            | Self::Punch => 2,
            Self::BlastProtection
            | Self::ProjectileProtection
            | Self::Respiration
            | Self::AquaAffinity
            | Self::Thorns
            | Self::DepthStrider
            | Self::Looting
            | Self::Fortune
            | Self::Power
            | Self::Flame => 4,
            Self::SilkTouch | Self::Infinity | Self::Mending => 8,
        }
    }

    /// Converts a level (1..=5) to standard Roman numerals ("I", "II", "III", "IV", "V").
    #[must_use]
    pub const fn level_to_roman(level: u8) -> &'static str {
        match level {
            1 => "I",
            2 => "II",
            3 => "III",
            4 => "IV",
            5 => "V",
            _ => "",
        }
    }
}

/// Calculates modified melee damage based on weapon enchantments.
#[must_use]
pub fn calculate_melee_damage(
    base_damage: f32,
    enchants: CompactEnchantments,
    is_undead: bool,
    is_arthropod: bool,
) -> f32 {
    let mut bonus = 0.0f32;
    let sharpness = enchants.get_level(EnchantmentKind::Sharpness);
    if sharpness > 0 {
        bonus += 0.5 + f32::from(sharpness) * 0.5;
    }
    let smite = enchants.get_level(EnchantmentKind::Smite);
    if smite > 0 && is_undead {
        bonus += f32::from(smite) * 2.5;
    }
    let bane = enchants.get_level(EnchantmentKind::BaneOfArthropods);
    if bane > 0 && is_arthropod {
        bonus += f32::from(bane) * 2.5;
    }
    base_damage + bonus
}

/// Returns fire duration in seconds inflicted by a melee attack using Fire Aspect.
#[must_use]
pub fn calculate_fire_aspect_seconds(enchants: CompactEnchantments) -> f32 {
    let level = enchants.get_level(EnchantmentKind::FireAspect);
    f32::from(level) * 4.0
}

/// Returns horizontal knockback multiplier bonus from Knockback enchantment.
#[must_use]
pub fn calculate_knockback_bonus(enchants: CompactEnchantments) -> f32 {
    let level = enchants.get_level(EnchantmentKind::Knockback);
    f32::from(level) * 0.5
}

/// Calculates tool mining speed multiplier with Efficiency enchantment.
#[must_use]
pub fn calculate_mining_speed(base_speed: f32, enchants: CompactEnchantments) -> f32 {
    let level = enchants.get_level(EnchantmentKind::Efficiency);
    if level > 0 {
        let bonus = (u32::from(level) * u32::from(level) + 1) as f32;
        base_speed + bonus
    } else {
        base_speed
    }
}

/// Checks if Unbreaking prevents durability reduction given a uniform random sample in `[0.0, 1.0)`.
#[must_use]
pub fn should_prevent_durability_loss(enchants: CompactEnchantments, rng_sample: f32) -> bool {
    let level = enchants.get_level(EnchantmentKind::Unbreaking);
    if level == 0 {
        return false;
    }
    // Chance to ignore durability loss: level / (level + 1)
    let ignore_chance = f32::from(level) / (f32::from(level) + 1.0);
    rng_sample < ignore_chance
}

/// Calculates bow arrow impact damage scaled with Power enchantment.
#[must_use]
pub fn calculate_arrow_damage(base_damage: f32, enchants: CompactEnchantments) -> f32 {
    let level = enchants.get_level(EnchantmentKind::Power);
    if level > 0 {
        base_damage * (1.0 + 0.25 * (f32::from(level) + 1.0))
    } else {
        base_damage
    }
}

/// Calculates arrow extra knockback velocity added by Punch enchantment.
#[must_use]
pub fn calculate_arrow_knockback(enchants: CompactEnchantments) -> f32 {
    let level = enchants.get_level(EnchantmentKind::Punch);
    f32::from(level) * 0.6
}

/// Returns true if arrows shot with this bow should be ignited by Flame enchantment.
#[must_use]
pub fn is_arrow_flaming(enchants: CompactEnchantments) -> bool {
    enchants.get_level(EnchantmentKind::Flame) > 0
}

/// Returns true if arrows are not consumed from player inventory when shooting (Infinity).
#[must_use]
pub fn has_infinite_arrows(enchants: CompactEnchantments) -> bool {
    enchants.get_level(EnchantmentKind::Infinity) > 0
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
    /// Bow weapons.
    Bow,
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

    #[test]
    fn test_damage_modifiers() {
        let mut enchants = CompactEnchantments::new();
        enchants.set_enchantment(EnchantmentKind::Sharpness, 5);
        let dmg = calculate_melee_damage(7.0, enchants, false, false);
        // Base 7.0 + (0.5 + 5 * 0.5 = 3.0) = 10.0
        assert!((dmg - 10.0).abs() < 1e-4);

        let mut smite_ench = CompactEnchantments::new();
        smite_ench.set_enchantment(EnchantmentKind::Smite, 4);
        let dmg_undead = calculate_melee_damage(5.0, smite_ench, true, false);
        // Base 5.0 + 4 * 2.5 = 15.0
        assert!((dmg_undead - 15.0).abs() < 1e-4);
        let dmg_living = calculate_melee_damage(5.0, smite_ench, false, false);
        assert!((dmg_living - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_tool_and_bow_modifiers() {
        let mut enchants = CompactEnchantments::new();
        enchants.set_enchantment(EnchantmentKind::Efficiency, 4);
        // Base 8.0 + (4*4 + 1 = 17.0) = 25.0
        assert!((calculate_mining_speed(8.0, enchants) - 25.0).abs() < 1e-4);

        let mut bow_ench = CompactEnchantments::new();
        bow_ench.set_enchantment(EnchantmentKind::Power, 5);
        bow_ench.set_enchantment(EnchantmentKind::Flame, 1);
        bow_ench.set_enchantment(EnchantmentKind::Infinity, 1);
        // Base 6.0 * (1.0 + 0.25 * (5 + 1) = 2.5) = 15.0
        assert!((calculate_arrow_damage(6.0, bow_ench) - 15.0).abs() < 1e-4);
        assert!(is_arrow_flaming(bow_ench));
        assert!(has_infinite_arrows(bow_ench));
    }

    #[test]
    fn test_unbreaking_durability_prevention() {
        let mut enchants = CompactEnchantments::new();
        enchants.set_enchantment(EnchantmentKind::Unbreaking, 3);
        // Chance is 3 / 4 = 0.75
        assert!(should_prevent_durability_loss(enchants, 0.5));
        assert!(should_prevent_durability_loss(enchants, 0.74));
        assert!(!should_prevent_durability_loss(enchants, 0.76));
    }
}
