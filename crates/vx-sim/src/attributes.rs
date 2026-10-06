//! Health, damage pipeline, and combat attributes.

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::event::Event;

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
}
