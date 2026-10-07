//! Status effects, duration countdown, periodic effect ticking, and attribute synchronization.

use crate::attributes::{AttributeKind, AttributeModifier, Attributes, Health, ModifierOperation};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::system::Query;

/// Standard status effect categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusEffectKind {
    /// Increases movement speed (+20% per amplifier).
    Speed = 1,
    /// Decreases movement speed (-15% per amplifier).
    Slowness = 2,
    /// Increases melee attack damage (+3.0 per amplifier).
    Strength = 3,
    /// Decreases melee attack damage (-4.0 flat).
    Weakness = 4,
    /// Periodic health regeneration (heals 1 HP every `50 >> amplifier` ticks).
    Regeneration = 5,
    /// Periodic damage (deals 1 HP every `25 >> amplifier` ticks, cannot kill / stops at 1 HP).
    Poison = 6,
    /// Periodic damage (deals 1 HP every `40 >> amplifier` ticks, can kill).
    Wither = 7,
    /// Damage reduction (20% per amplifier; immune at level 5).
    Resistance = 8,
    /// Complete immunity to fire and lava damage.
    FireResistance = 9,
    /// Prevents drowning underwater.
    WaterBreathing = 10,
    /// Increases mining and attack speed (+20% per amplifier).
    Haste = 11,
    /// Decreases mining speed (-10% per amplifier).
    MiningFatigue = 12,
    /// Renders entity invisible to players and mob aggro.
    Invisibility = 13,
    /// Increases jump height (+0.1 blocks per amplifier).
    JumpBoost = 14,
    /// Instant health restoration (heals `4 << amplifier` HP).
    InstantHealth = 15,
    /// Instant damage dealing (deals `6 << amplifier` HP).
    InstantDamage = 16,
}

impl StatusEffectKind {
    /// Numeric identifier for wire serialization and network replication.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Resolves numeric ID into a `StatusEffectKind`.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(Self::Speed),
            2 => Some(Self::Slowness),
            3 => Some(Self::Strength),
            4 => Some(Self::Weakness),
            5 => Some(Self::Regeneration),
            6 => Some(Self::Poison),
            7 => Some(Self::Wither),
            8 => Some(Self::Resistance),
            9 => Some(Self::FireResistance),
            10 => Some(Self::WaterBreathing),
            11 => Some(Self::Haste),
            12 => Some(Self::MiningFatigue),
            13 => Some(Self::Invisibility),
            14 => Some(Self::JumpBoost),
            15 => Some(Self::InstantHealth),
            16 => Some(Self::InstantDamage),
            _ => None,
        }
    }

    /// User-facing display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Speed => "Speed",
            Self::Slowness => "Slowness",
            Self::Strength => "Strength",
            Self::Weakness => "Weakness",
            Self::Regeneration => "Regeneration",
            Self::Poison => "Poison",
            Self::Wither => "Wither",
            Self::Resistance => "Resistance",
            Self::FireResistance => "Fire Resistance",
            Self::WaterBreathing => "Water Breathing",
            Self::Haste => "Haste",
            Self::MiningFatigue => "Mining Fatigue",
            Self::Invisibility => "Invisibility",
            Self::JumpBoost => "Jump Boost",
            Self::InstantHealth => "Instant Health",
            Self::InstantDamage => "Instant Damage",
        }
    }

    /// Ambient particle color `[R, G, B]` for swirls and bubbles.
    #[must_use]
    pub const fn particle_color(self) -> [u8; 3] {
        match self {
            Self::Speed => [124, 175, 198],
            Self::Slowness => [90, 108, 129],
            Self::Strength => [147, 36, 35],
            Self::Weakness => [72, 77, 72],
            Self::Regeneration => [205, 92, 171],
            Self::Poison => [78, 147, 49],
            Self::Wither => [53, 42, 39],
            Self::Resistance => [153, 69, 58],
            Self::FireResistance => [228, 154, 58],
            Self::WaterBreathing => [46, 82, 153],
            Self::Haste => [217, 192, 67],
            Self::MiningFatigue => [74, 66, 23],
            Self::Invisibility => [127, 131, 146],
            Self::JumpBoost => [34, 255, 76],
            Self::InstantHealth => [248, 36, 35],
            Self::InstantDamage => [67, 10, 9],
        }
    }

    /// Whether this is an instant effect (applied immediately without persistent duration).
    #[must_use]
    pub const fn is_instant(self) -> bool {
        matches!(self, Self::InstantHealth | Self::InstantDamage)
    }

    /// Whether this effect is beneficial / positive.
    #[must_use]
    pub const fn is_beneficial(self) -> bool {
        matches!(
            self,
            Self::Speed
                | Self::Strength
                | Self::Regeneration
                | Self::Resistance
                | Self::FireResistance
                | Self::WaterBreathing
                | Self::Haste
                | Self::Invisibility
                | Self::JumpBoost
                | Self::InstantHealth
        )
    }
}

/// An active instance of a status effect on an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectInstance {
    /// Effect kind.
    pub kind: StatusEffectKind,
    /// Duration in ticks remaining (-1 for permanent/infinite, > 0 for finite).
    pub duration_ticks: i32,
    /// Amplifier / power level (0 = Level I, 1 = Level II, etc.).
    pub amplifier: u8,
    /// Whether this effect was applied by an ambient source (e.g. Beacon/Conduit).
    pub ambient: bool,
    /// Whether to render particle swirls.
    pub show_particles: bool,
    /// Whether to show the icon in HUD.
    pub show_icon: bool,
}

impl EffectInstance {
    /// Creates a standard status effect instance with particles and icon enabled.
    #[must_use]
    pub const fn new(kind: StatusEffectKind, duration_ticks: i32, amplifier: u8) -> Self {
        Self {
            kind,
            duration_ticks,
            amplifier,
            ambient: false,
            show_particles: true,
            show_icon: true,
        }
    }

    /// Converts amplifier to standard Roman numeral string.
    #[must_use]
    pub const fn roman_numeral(amplifier: u8) -> &'static str {
        match amplifier {
            0 => "I",
            1 => "II",
            2 => "III",
            3 => "IV",
            4 => "V",
            _ => "+",
        }
    }

    /// Formats the effect name with Roman numeral (e.g. "Speed II", "Poison").
    #[must_use]
    pub fn display_name(&self) -> String {
        if self.amplifier == 0 {
            self.kind.name().to_string()
        } else {
            format!(
                "{} {}",
                self.kind.name(),
                Self::roman_numeral(self.amplifier)
            )
        }
    }

    /// Formats duration as `M:SS` or `**:**` if infinite.
    #[must_use]
    pub fn format_duration(&self) -> String {
        if self.duration_ticks < 0 {
            return "**:**".to_string();
        }
        let total_seconds = self.duration_ticks / 20;
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{minutes}:{seconds:02}")
    }

    /// Ticks the duration down by 1. Returns `true` if expired (`duration_ticks == 0`).
    pub fn tick(&mut self) -> bool {
        if self.duration_ticks > 0 {
            self.duration_ticks -= 1;
            self.duration_ticks == 0
        } else {
            false
        }
    }
}

/// Living entity component holding active status effects.
#[derive(Debug, Clone, PartialEq, Default, Component)]
pub struct StatusEffects {
    /// Active effects list.
    pub effects: Vec<EffectInstance>,
}

impl StatusEffects {
    /// Creates an empty status effects container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Applies a status effect instance:
    /// - If the effect is not currently active, inserts it.
    /// - If the effect is active:
    ///   - Higher amplifier replaces the existing instance.
    ///   - Equal amplifier takes the maximum duration.
    ///   - Lower amplifier is ignored.
    pub fn apply(&mut self, instance: EffectInstance) {
        if let Some(existing) = self.effects.iter_mut().find(|e| e.kind == instance.kind) {
            if instance.amplifier > existing.amplifier {
                *existing = instance;
            } else if instance.amplifier == existing.amplifier
                && (instance.duration_ticks < 0
                    || (existing.duration_ticks >= 0
                        && instance.duration_ticks > existing.duration_ticks))
            {
                existing.duration_ticks = instance.duration_ticks;
                existing.ambient = instance.ambient;
                existing.show_particles = instance.show_particles;
                existing.show_icon = instance.show_icon;
            }
        } else {
            self.effects.push(instance);
        }
    }

    /// Removes an effect by kind, returning `true` if removed.
    pub fn remove(&mut self, kind: StatusEffectKind) -> bool {
        let initial_len = self.effects.len();
        self.effects.retain(|e| e.kind != kind);
        self.effects.len() != initial_len
    }

    /// Clears all active status effects.
    pub fn clear(&mut self) {
        self.effects.clear();
    }

    /// Looks up an active effect instance.
    #[must_use]
    pub fn get(&self, kind: StatusEffectKind) -> Option<&EffectInstance> {
        self.effects.iter().find(|e| e.kind == kind)
    }

    /// Checks if a status effect is active.
    #[must_use]
    pub fn has(&self, kind: StatusEffectKind) -> bool {
        self.effects.iter().any(|e| e.kind == kind)
    }

    /// Returns the amplifier level of an effect if active.
    #[must_use]
    pub fn amplifier(&self, kind: StatusEffectKind) -> Option<u8> {
        self.get(kind).map(|e| e.amplifier)
    }

    /// Returns `true` if there are no active status effects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

/// Simulation system executing within `TickSet::Status`:
/// 1. Ticks down remaining duration of active status effects.
/// 2. Evaluates periodic effect intervals (Regeneration, Poison, Wither).
/// 3. Cleans up expired effects.
/// 4. Synchronizes dynamic attribute modifiers with `Attributes`.
pub fn status_effect_system(
    mut query: Query<(
        Entity,
        &mut StatusEffects,
        Option<&mut Health>,
        Option<&mut Attributes>,
    )>,
) {
    for (_entity, mut effects, mut health_opt, mut attr_opt) in &mut query {
        if effects.is_empty() {
            if let Some(ref mut attrs) = attr_opt {
                attrs.remove_modifier(AttributeKind::MovementSpeed, "effect.speed");
                attrs.remove_modifier(AttributeKind::MovementSpeed, "effect.slowness");
                attrs.remove_modifier(AttributeKind::AttackDamage, "effect.strength");
                attrs.remove_modifier(AttributeKind::AttackDamage, "effect.weakness");
            }
            continue;
        }

        let mut expired = Vec::new();

        // 1. Process periodic ticks and duration countdown
        for (idx, effect) in effects.effects.iter_mut().enumerate() {
            let amp = effect.amplifier;
            let duration = effect.duration_ticks;

            // Periodic effects trigger on fixed tick intervals based on duration countdown
            if let Some(ref mut health) = health_opt {
                match effect.kind {
                    StatusEffectKind::Regeneration => {
                        let interval = (50 >> amp).max(1);
                        if duration > 0 && duration % interval == 0 {
                            health.heal(1.0);
                        }
                    }
                    StatusEffectKind::Poison => {
                        let interval = (25 >> amp).max(1);
                        if duration > 0 && duration % interval == 0 && health.cur > 1.0 {
                            let dmg = 1.0f32.min(health.cur - 1.0);
                            health.reduce(dmg);
                        }
                    }
                    StatusEffectKind::Wither => {
                        let interval = (40 >> amp).max(1);
                        if duration > 0 && duration % interval == 0 {
                            health.reduce(1.0);
                        }
                    }
                    _ => {}
                }
            }

            // Advance duration countdown
            if effect.tick() {
                expired.push(idx);
            }
        }

        // 2. Remove expired effects in reverse order
        for idx in expired.into_iter().rev() {
            effects.effects.remove(idx);
        }

        // 3. Synchronize attribute modifiers
        if let Some(ref mut attrs) = attr_opt {
            // Speed / Slowness -> MovementSpeed
            if let Some(speed_amp) = effects.amplifier(StatusEffectKind::Speed) {
                let bonus = 0.20 * f64::from(speed_amp + 1);
                attrs.add_modifier(
                    AttributeKind::MovementSpeed,
                    AttributeModifier::new(
                        "effect.speed",
                        bonus,
                        ModifierOperation::AddMultipliedBase,
                    ),
                );
            } else {
                attrs.remove_modifier(AttributeKind::MovementSpeed, "effect.speed");
            }

            if let Some(slow_amp) = effects.amplifier(StatusEffectKind::Slowness) {
                let penalty = -0.15 * f64::from(slow_amp + 1);
                attrs.add_modifier(
                    AttributeKind::MovementSpeed,
                    AttributeModifier::new(
                        "effect.slowness",
                        penalty,
                        ModifierOperation::AddMultipliedBase,
                    ),
                );
            } else {
                attrs.remove_modifier(AttributeKind::MovementSpeed, "effect.slowness");
            }

            // Strength / Weakness -> AttackDamage
            if let Some(str_amp) = effects.amplifier(StatusEffectKind::Strength) {
                let bonus = 3.0 * f64::from(str_amp + 1);
                attrs.add_modifier(
                    AttributeKind::AttackDamage,
                    AttributeModifier::new("effect.strength", bonus, ModifierOperation::AddValue),
                );
            } else {
                attrs.remove_modifier(AttributeKind::AttackDamage, "effect.strength");
            }

            if effects.has(StatusEffectKind::Weakness) {
                attrs.add_modifier(
                    AttributeKind::AttackDamage,
                    AttributeModifier::new("effect.weakness", -4.0, ModifierOperation::AddValue),
                );
            } else {
                attrs.remove_modifier(AttributeKind::AttackDamage, "effect.weakness");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_effects_application_rules() {
        let mut effects = StatusEffects::new();

        // 1. Initial application
        effects.apply(EffectInstance::new(StatusEffectKind::Speed, 100, 0));
        assert_eq!(effects.amplifier(StatusEffectKind::Speed), Some(0));
        assert_eq!(
            effects.get(StatusEffectKind::Speed).unwrap().duration_ticks,
            100
        );

        // 2. Lower amplifier is ignored
        effects.apply(EffectInstance::new(StatusEffectKind::Speed, 500, 0));
        assert_eq!(
            effects.get(StatusEffectKind::Speed).unwrap().duration_ticks,
            500
        ); // Equal amp extends duration

        // 3. Higher amplifier overwrites
        effects.apply(EffectInstance::new(StatusEffectKind::Speed, 60, 1));
        assert_eq!(effects.amplifier(StatusEffectKind::Speed), Some(1));
        assert_eq!(
            effects.get(StatusEffectKind::Speed).unwrap().duration_ticks,
            60
        );

        // 4. Remove effect
        assert!(effects.remove(StatusEffectKind::Speed));
        assert!(!effects.has(StatusEffectKind::Speed));
    }

    #[test]
    fn test_poison_does_not_kill() {
        let health = Health::new(2.0);
        let mut effects = StatusEffects::new();
        effects.apply(EffectInstance::new(StatusEffectKind::Poison, 25, 0));

        let mut world = bevy_ecs::world::World::new();
        let entity = world.spawn((health, effects)).id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(status_effect_system);

        // Tick 25 ticks: Poison I interval is 25 ticks
        for _ in 0..25 {
            schedule.run(&mut world);
        }

        let h = world.get::<Health>(entity).unwrap();
        // Should have taken 1.0 damage, leaving 1.0 HP
        assert!((h.cur - 1.0).abs() < 1e-4);

        // Another poison effect at 1.0 HP
        let mut eff = world.get_mut::<StatusEffects>(entity).unwrap();
        eff.apply(EffectInstance::new(StatusEffectKind::Poison, 25, 0));

        for _ in 0..25 {
            schedule.run(&mut world);
        }

        let h2 = world.get::<Health>(entity).unwrap();
        // Still alive at exactly 1.0 HP!
        assert!((h2.cur - 1.0).abs() < 1e-4);
        assert!(h2.is_alive());
    }

    #[test]
    fn test_speed_attribute_synchronization() {
        let health = Health::new(20.0);
        let mut effects = StatusEffects::new();
        effects.apply(EffectInstance::new(StatusEffectKind::Speed, 10, 0)); // Speed I = +20%
        let attrs = Attributes::player_default();

        let mut world = bevy_ecs::world::World::new();
        let entity = world.spawn((health, effects, attrs)).id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(status_effect_system);

        // Run 1 tick
        schedule.run(&mut world);

        let mut a = world.get_mut::<Attributes>(entity).unwrap();
        // Base 0.1 * 1.2 = 0.12
        assert!((a.get_value(AttributeKind::MovementSpeed) - 0.12).abs() < 1e-6);

        // Advance 10 ticks until effect expires
        for _ in 0..10 {
            schedule.run(&mut world);
        }

        let mut a_expired = world.get_mut::<Attributes>(entity).unwrap();
        // Back to base 0.1
        assert!((a_expired.get_value(AttributeKind::MovementSpeed) - 0.1).abs() < 1e-6);
    }
}
