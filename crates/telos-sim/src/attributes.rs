//! Health, damage pipeline, and combat attributes.

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::event::Event;
use hashbrown::HashMap;

/// Category of damage dealt to an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DamageType {
    /// Generic or untyped damage.
    Generic,
    /// Damage sustained from falling onto solid ground.
    Fall,
    /// Damage sustained when hunger reaches 0.
    Starvation,
    /// Damage sustained from falling into the void.
    Void,
    /// Damage dealt by a player or mob attack.
    Attack,
    /// Direct command or debug damage.
    Command,
    /// Damage from fire, lava, or burning.
    Fire,
    /// Damage from poison effect.
    Poison,
    /// Damage from wither effect.
    Wither,
}

/// Event dispatched when an entity takes damage.
#[derive(Debug, Clone, PartialEq, Event)]
pub struct DamageEvent {
    /// Entity receiving the damage.
    pub target: Entity,
    /// Raw damage amount in half-hearts (HP).
    pub amount: f32,
    /// Source / category of damage.
    pub damage_type: DamageType,
}

/// Living health component.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Health {
    /// Current health points (typically 0.0..=20.0).
    pub cur: f32,
    /// Maximum health points (typically 20.0).
    pub max: f32,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            cur: 20.0,
            max: 20.0,
        }
    }
}

impl Health {
    /// Creates a new health component with cur equal to max.
    #[must_use]
    pub const fn new(max: f32) -> Self {
        Self { cur: max, max }
    }

    /// Whether the entity is still alive (`cur > 0.0`).
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.cur > 0.0
    }

    /// Heals the entity by `amount`, capped at `max`.
    pub fn heal(&mut self, amount: f32) {
        if amount > 0.0 && self.is_alive() {
            self.cur = (self.cur + amount).min(self.max);
        }
    }

    /// Directly reduces health by `amount`, clamped to 0.0.
    pub fn reduce(&mut self, amount: f32) {
        if amount > 0.0 {
            self.cur = (self.cur - amount).max(0.0);
        }
    }
}

/// Tracks hurt cooldowns and invulnerability frames.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct CombatTracker {
    /// Number of ticks remaining in the invulnerability window (starts at 10 ticks = 0.5s).
    pub invulnerable_ticks: u32,
    /// Last damage amount taken during the active invulnerability window.
    pub last_damage_amount: f32,
}

impl Default for CombatTracker {
    fn default() -> Self {
        Self {
            invulnerable_ticks: 0,
            last_damage_amount: 0.0,
        }
    }
}

impl CombatTracker {
    /// Ticks down the invulnerability timer by 1 tick.
    pub fn tick(&mut self) {
        if self.invulnerable_ticks > 0 {
            self.invulnerable_ticks -= 1;
            if self.invulnerable_ticks == 0 {
                self.last_damage_amount = 0.0;
            }
        }
    }
}

/// Applies damage to an entity, taking into account the invulnerability window.
///
/// Returns the actual amount of damage dealt to health.
pub fn apply_damage(
    health: &mut Health,
    tracker: &mut CombatTracker,
    amount: f32,
    damage_type: DamageType,
) -> f32 {
    if !health.is_alive() || amount <= 0.0 {
        return 0.0;
    }

    // Void and Command bypass invulnerability
    let bypass_invuln = matches!(damage_type, DamageType::Void | DamageType::Command);

    let actual_damage = if bypass_invuln || tracker.invulnerable_ticks == 0 {
        // Full damage applies
        tracker.invulnerable_ticks = 10;
        tracker.last_damage_amount = amount;
        amount
    } else if amount > tracker.last_damage_amount {
        // Within invulnerability window: only excess damage applies
        let excess = amount - tracker.last_damage_amount;
        tracker.last_damage_amount = amount;
        excess
    } else {
        // In cooldown and not higher than previous damage
        0.0
    };

    if actual_damage > 0.0 {
        health.reduce(actual_damage);
    }

    actual_damage
}

/// Calculates mitigated damage according to the armor, toughness, resistance, and enchantment protection factor (EPF) pipeline.
///
/// Follows `analysis/gameplay-simulation/SKILL.md` §9.1:
/// 1. Armor: `d *= 1 - clamp(max(armor/5, armor - 4d/(toughness + 8)), 0, 20) / 25`
/// 2. Resistance: `d *= 1 - 0.2 * level` (>= 5 is immune)
/// 3. Enchantment protection factor: `epf = min(epf, 20); d *= 1 - epf / 25`
#[must_use]
pub fn calculate_damage_mitigation(
    raw_damage: f32,
    damage_type: DamageType,
    armor: f32,
    toughness: f32,
    resistance_level: u8,
    epf: u8,
) -> f32 {
    if raw_damage <= 0.0 {
        return 0.0;
    }
    if matches!(
        damage_type,
        DamageType::Void | DamageType::Command | DamageType::Starvation
    ) {
        return raw_damage;
    }

    let mut d = raw_damage;

    // 1. Armor mitigation (unless fall/void/starvation/command)
    if !matches!(damage_type, DamageType::Fall) && armor > 0.0 {
        let defense = (armor - (4.0 * d) / (toughness + 8.0)).max(armor / 5.0);
        let clamped_defense = defense.clamp(0.0, 20.0);
        d *= 1.0 - (clamped_defense / 25.0);
    }

    // 2. Resistance effect (20% per level; level >= 5 means immune)
    if resistance_level > 0 {
        let reduction = 0.2 * f32::from(resistance_level);
        d *= (1.0 - reduction).max(0.0);
    }

    // 3. Enchantment protection factor (EPF, clamped to 20; 4% reduction per point)
    if epf > 0 {
        let capped_epf = f32::from(epf.min(20)) / 25.0;
        d *= (1.0 - capped_epf).max(0.0);
    }

    d.max(0.0)
}

/// Applies mitigated damage to an entity, running through the armor, resistance, and EPF pipeline
/// before feeding into the combat tracker invulnerability window.
pub fn apply_mitigated_damage(
    health: &mut Health,
    tracker: &mut CombatTracker,
    raw_damage: f32,
    damage_type: DamageType,
    armor: f32,
    toughness: f32,
    resistance_level: u8,
    epf: u8,
) -> f32 {
    let mitigated = calculate_damage_mitigation(
        raw_damage,
        damage_type,
        armor,
        toughness,
        resistance_level,
        epf,
    );
    apply_damage(health, tracker, mitigated, damage_type)
}

/// Core attribute types supported on entities and players.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeKind {
    /// Maximum health points (base 20.0).
    MaxHealth,
    /// Base movement speed in blocks/tick (base 0.1).
    MovementSpeed,
    /// Base melee attack damage in half-hearts (base 1.0).
    AttackDamage,
    /// Attack cooldown recovery rate (base 4.0).
    AttackSpeed,
    /// Armor defense points (base 0.0, max 30.0).
    Armor,
    /// Armor toughness points (base 0.0, max 20.0).
    ArmorToughness,
    /// Knockback resistance in [0.0, 1.0] (base 0.0).
    KnockbackResistance,
}

impl AttributeKind {
    /// Returns the default base value for this attribute.
    #[must_use]
    pub const fn default_base(self) -> f64 {
        match self {
            Self::MaxHealth => 20.0,
            Self::MovementSpeed => 0.1,
            Self::AttackDamage | Self::AttackSpeed => 4.0,
            Self::Armor | Self::ArmorToughness | Self::KnockbackResistance => 0.0,
        }
    }

    /// Allowed range `[min, max]` for this attribute.
    #[must_use]
    pub const fn value_bounds(self) -> (f64, f64) {
        match self {
            Self::MaxHealth => (1.0, 1024.0),
            Self::MovementSpeed | Self::AttackSpeed => (0.0, 1024.0),
            Self::AttackDamage => (0.0, 2048.0),
            Self::Armor => (0.0, 30.0),
            Self::ArmorToughness => (0.0, 20.0),
            Self::KnockbackResistance => (0.0, 1.0),
        }
    }
}

/// Operation applied by an attribute modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModifierOperation {
    /// Adds amount directly to the base value: `(base + sum(AddValue))`.
    AddValue = 0,
    /// Multiplies base value by amount: `(base * sum(AddMultipliedBase))`.
    AddMultipliedBase = 1,
    /// Multiplies the running total: `total * product(1.0 + AddMultipliedTotal)`.
    AddMultipliedTotal = 2,
}

/// A named modifier modifying an attribute's value.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeModifier {
    /// Unique name or identifier for this modifier (e.g. "effect.speed", "item.armor").
    pub name: String,
    /// Numeric modifier amount.
    pub amount: f64,
    /// Mathematical operation.
    pub operation: ModifierOperation,
}

impl AttributeModifier {
    /// Creates a new attribute modifier.
    pub fn new(name: impl Into<String>, amount: f64, operation: ModifierOperation) -> Self {
        Self {
            name: name.into(),
            amount,
            operation,
        }
    }
}

/// An attribute instance with a base value, registered modifiers, and a cached final value.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Which attribute this represents.
    pub kind: AttributeKind,
    /// Base unmodified value.
    pub base_value: f64,
    /// Minimum allowed clamped value.
    pub min_value: f64,
    /// Maximum allowed clamped value.
    pub max_value: f64,
    /// Collection of attached modifiers.
    pub modifiers: Vec<AttributeModifier>,
    cached_value: f64,
    dirty: bool,
}

impl Attribute {
    /// Creates a new attribute with default bounds and no modifiers.
    #[must_use]
    pub fn new(kind: AttributeKind, base_value: f64) -> Self {
        let (min_value, max_value) = kind.value_bounds();
        let mut attr = Self {
            kind,
            base_value,
            min_value,
            max_value,
            modifiers: Vec::new(),
            cached_value: base_value,
            dirty: true,
        };
        attr.recompute();
        attr
    }

    /// Adds or replaces a modifier by name.
    pub fn add_modifier(&mut self, modifier: AttributeModifier) {
        if let Some(existing) = self.modifiers.iter_mut().find(|m| m.name == modifier.name) {
            *existing = modifier;
        } else {
            self.modifiers.push(modifier);
        }
        self.dirty = true;
    }

    /// Removes a modifier by name, returning true if removed.
    pub fn remove_modifier(&mut self, name: &str) -> bool {
        let initial_len = self.modifiers.len();
        self.modifiers.retain(|m| m.name != name);
        if self.modifiers.len() == initial_len {
            false
        } else {
            self.dirty = true;
            true
        }
    }

    /// Clears all modifiers.
    pub fn clear_modifiers(&mut self) {
        if !self.modifiers.is_empty() {
            self.modifiers.clear();
            self.dirty = true;
        }
    }

    /// Returns the computed and clamped final attribute value.
    pub fn value(&mut self) -> f64 {
        if self.dirty {
            self.recompute();
        }
        self.cached_value
    }

    /// Forces recomputation of the cached value according to standard formula:
    /// `total = (base + sum(AddValue)) * (1.0 + sum(AddMultipliedBase)) * product(1.0 + AddMultipliedTotal)`
    pub fn recompute(&mut self) {
        let mut add_value = 0.0;
        let mut add_mult_base = 0.0;
        let mut mult_total = 1.0;

        for m in &self.modifiers {
            match m.operation {
                ModifierOperation::AddValue => add_value += m.amount,
                ModifierOperation::AddMultipliedBase => add_mult_base += m.amount,
                ModifierOperation::AddMultipliedTotal => mult_total *= 1.0 + m.amount,
            }
        }

        let step1 = self.base_value + add_value;
        let step2 = step1 * (1.0 + add_mult_base);
        let step3 = step2 * mult_total;

        self.cached_value = step3.clamp(self.min_value, self.max_value);
        self.dirty = false;
    }
}

/// Living entity / player attributes component.
#[derive(Debug, Clone, PartialEq, Component)]
pub struct Attributes {
    /// Active attributes by kind.
    pub attributes: HashMap<AttributeKind, Attribute>,
}

impl Default for Attributes {
    fn default() -> Self {
        Self::player_default()
    }
}

impl Attributes {
    /// Creates the standard attributes set for a player.
    #[must_use]
    pub fn player_default() -> Self {
        let mut attributes = HashMap::new();
        for kind in [
            AttributeKind::MaxHealth,
            AttributeKind::MovementSpeed,
            AttributeKind::AttackDamage,
            AttributeKind::AttackSpeed,
            AttributeKind::Armor,
            AttributeKind::ArmorToughness,
            AttributeKind::KnockbackResistance,
        ] {
            attributes.insert(kind, Attribute::new(kind, kind.default_base()));
        }
        Self { attributes }
    }

    /// Returns the computed value of an attribute, or its default base if not present.
    pub fn get_value(&mut self, kind: AttributeKind) -> f64 {
        if let Some(attr) = self.attributes.get_mut(&kind) {
            attr.value()
        } else {
            kind.default_base()
        }
    }

    /// Sets base value for an attribute.
    pub fn set_base(&mut self, kind: AttributeKind, base: f64) {
        if let Some(attr) = self.attributes.get_mut(&kind) {
            attr.base_value = base;
            attr.dirty = true;
        } else {
            self.attributes.insert(kind, Attribute::new(kind, base));
        }
    }

    /// Adds or replaces a modifier on an attribute.
    pub fn add_modifier(&mut self, kind: AttributeKind, modifier: AttributeModifier) {
        if let Some(attr) = self.attributes.get_mut(&kind) {
            attr.add_modifier(modifier);
        } else {
            let mut attr = Attribute::new(kind, kind.default_base());
            attr.add_modifier(modifier);
            self.attributes.insert(kind, attr);
        }
    }

    /// Removes a modifier from an attribute by name.
    pub fn remove_modifier(&mut self, kind: AttributeKind, name: &str) -> bool {
        if let Some(attr) = self.attributes.get_mut(&kind) {
            attr.remove_modifier(name)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_heal_and_reduce() {
        let mut h = Health::new(20.0);
        assert!(h.is_alive());
        h.reduce(5.0);
        assert!((h.cur - 15.0).abs() < f32::EPSILON);
        h.heal(10.0);
        assert!((h.cur - 20.0).abs() < f32::EPSILON);
        h.reduce(30.0);
        assert!(!h.is_alive());
        assert!((h.cur - 0.0).abs() < f32::EPSILON);
        h.heal(5.0);
        assert!(!h.is_alive()); // Dead entities cannot heal
    }

    #[test]
    fn test_apply_damage_invulnerability() {
        let mut h = Health::new(20.0);
        let mut tracker = CombatTracker::default();

        // Hit 1: 4.0 damage
        let dealt1 = apply_damage(&mut h, &mut tracker, 4.0, DamageType::Attack);
        assert!((dealt1 - 4.0).abs() < f32::EPSILON);
        assert!((h.cur - 16.0).abs() < f32::EPSILON);
        assert_eq!(tracker.invulnerable_ticks, 10);

        // Hit 2 in same window with lower damage: 2.0 -> 0 dealt
        let dealt2 = apply_damage(&mut h, &mut tracker, 2.0, DamageType::Attack);
        assert!((dealt2 - 0.0).abs() < f32::EPSILON);
        assert!((h.cur - 16.0).abs() < f32::EPSILON);

        // Hit 3 in same window with higher damage: 6.0 -> 2.0 dealt (excess)
        let dealt3 = apply_damage(&mut h, &mut tracker, 6.0, DamageType::Attack);
        assert!((dealt3 - 2.0).abs() < f32::EPSILON);
        assert!((h.cur - 14.0).abs() < f32::EPSILON);

        // Tick down tracker
        for _ in 0..10 {
            tracker.tick();
        }
        assert_eq!(tracker.invulnerable_ticks, 0);

        // Hit 4 after cooldown: full damage
        let dealt4 = apply_damage(&mut h, &mut tracker, 4.0, DamageType::Attack);
        assert!((dealt4 - 4.0).abs() < f32::EPSILON);
        assert!((h.cur - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attribute_modifiers_pipeline() {
        let mut speed = Attribute::new(AttributeKind::MovementSpeed, 0.1);
        assert!((speed.value() - 0.1).abs() < 1e-6);

        // AddValue: +0.02 -> 0.12
        speed.add_modifier(AttributeModifier::new(
            "boots",
            0.02,
            ModifierOperation::AddValue,
        ));
        assert!((speed.value() - 0.12).abs() < 1e-6);

        // AddMultipliedBase: +20% of base (0.12 * 1.2 = 0.144)
        speed.add_modifier(AttributeModifier::new(
            "speed_effect",
            0.20,
            ModifierOperation::AddMultipliedBase,
        ));
        assert!((speed.value() - 0.144).abs() < 1e-6);

        // AddMultipliedTotal: +10% total (0.144 * 1.1 = 0.1584)
        speed.add_modifier(AttributeModifier::new(
            "sprint",
            0.10,
            ModifierOperation::AddMultipliedTotal,
        ));
        assert!((speed.value() - 0.1584).abs() < 1e-6);

        // Remove modifier
        assert!(speed.remove_modifier("speed_effect"));
        // 0.12 * 1.0 * 1.1 = 0.132
        assert!((speed.value() - 0.132).abs() < 1e-6);
    }

    #[test]
    fn test_damage_mitigation_formula() {
        // Raw damage 10.0, no armor, no resistance -> 10.0
        let d0 = calculate_damage_mitigation(10.0, DamageType::Attack, 0.0, 0.0, 0, 0);
        assert!((d0 - 10.0).abs() < 1e-4);

        // Raw damage 10.0, 20 armor, 0 toughness:
        // defense = max(20/5, 20 - 40/8) = max(4, 15) = 15.0
        // d = 10 * (1 - 15/25) = 10 * 0.4 = 4.0
        let d1 = calculate_damage_mitigation(10.0, DamageType::Attack, 20.0, 0.0, 0, 0);
        assert!((d1 - 4.0).abs() < 1e-4);

        // Plus Resistance I (20% reduction): 4.0 * 0.8 = 3.2
        let d2 = calculate_damage_mitigation(10.0, DamageType::Attack, 20.0, 0.0, 1, 0);
        assert!((d2 - 3.2).abs() < 1e-4);

        // Plus EPF 10 (40% reduction): 3.2 * (1 - 10/25) = 3.2 * 0.6 = 1.92
        let d3 = calculate_damage_mitigation(10.0, DamageType::Attack, 20.0, 0.0, 1, 10);
        assert!((d3 - 1.92).abs() < 1e-4);

        // Void bypasses armor and resistance
        let d_void = calculate_damage_mitigation(10.0, DamageType::Void, 20.0, 10.0, 2, 20);
        assert!((d_void - 10.0).abs() < 1e-4);
    }
}
