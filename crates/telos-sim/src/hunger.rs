//! Hunger, exhaustion, saturation, and natural health regeneration.

use crate::attributes::Health;
use bevy_ecs::component::Component;
use bevy_ecs::system::Resource;

/// Simulation parameters governing hunger and regeneration rates.
#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct SimParams {
    /// Exhaustion points required to consume 1 point of saturation/food.
    pub exhaustion_threshold: f32,
    /// Exhaustion added per meter sprinted.
    pub sprint_exhaustion_per_meter: f32,
    /// Ticks between saturated regeneration ticks (default 10 ticks = 0.5s).
    pub saturated_regen_interval: u32,
    /// Ticks between normal regeneration ticks (default 80 ticks = 4.0s).
    pub normal_regen_interval: u32,
    /// Ticks between starvation damage ticks (default 80 ticks = 4.0s).
    pub starvation_interval: u32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            exhaustion_threshold: 4.0,
            sprint_exhaustion_per_meter: 0.1,
            saturated_regen_interval: 10,
            normal_regen_interval: 80,
            starvation_interval: 80,
        }
    }
}

/// Player hunger and nutrition component.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Hunger {
    /// Food level in half-drumsticks (0..=20).
    pub food: u32,
    /// Hidden saturation buffer protecting food level from dropping (0.0..=food as f32).
    pub saturation: f32,
    /// Accumulated exhaustion counter (0.0..).
    pub exhaustion: f32,
    /// Internal tick timer for pacing regeneration and starvation.
    pub timer: u32,
}

impl Default for Hunger {
    fn default() -> Self {
        Self {
            food: 20,
            saturation: 5.0,
            exhaustion: 0.0,
            timer: 0,
        }
    }
}

impl Hunger {
    /// Creates a hunger component with specified food and saturation levels.
    #[must_use]
    pub fn new(food: u32, saturation: f32) -> Self {
        let food = food.min(20);
        #[allow(clippy::cast_precision_loss)]
        let saturation = saturation.clamp(0.0, food as f32);
        Self {
            food,
            saturation,
            exhaustion: 0.0,
            timer: 0,
        }
    }

    /// Whether the player has enough food to initiate or maintain sprinting (`food > 6`).
    #[must_use]
    pub fn can_sprint(&self) -> bool {
        self.food > 6
    }

    /// Adds exhaustion points from physical activities (e.g. sprinting, jumping).
    pub fn add_exhaustion(&mut self, amount: f32) {
        if amount > 0.0 {
            self.exhaustion += amount;
        }
    }

    /// Adds food and saturation (e.g. from eating).
    pub fn feed(&mut self, food_points: u32, saturation_bonus: f32) {
        self.food = (self.food + food_points).min(20);
        #[allow(clippy::cast_precision_loss)]
        let max_sat = self.food as f32;
        self.saturation = (self.saturation + saturation_bonus).clamp(0.0, max_sat);
    }
}

/// Result of a single tick of the hunger simulation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HungerTickResult {
    /// Health points regenerated this tick (if any).
    pub healed: f32,
    /// Damage dealt by starvation this tick (if any).
    pub starved: f32,
}

/// Advances hunger by 1 tick, applying exhaustion drains, natural regeneration, or starvation.
pub fn tick_hunger(
    health: &mut Health,
    hunger: &mut Hunger,
    params: &SimParams,
) -> HungerTickResult {
    let mut result = HungerTickResult::default();

    // 1. Drain exhaustion into saturation, then food
    while hunger.exhaustion >= params.exhaustion_threshold {
        hunger.exhaustion -= params.exhaustion_threshold;
        if hunger.saturation > 0.0 {
            hunger.saturation = (hunger.saturation - 1.0).max(0.0);
        } else if hunger.food > 0 {
            hunger.food -= 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let max_sat = hunger.food as f32;
    if hunger.saturation > max_sat {
        hunger.saturation = max_sat;
    }

    // 2. Health regeneration & Starvation
    hunger.timer = hunger.timer.saturating_add(1);

    if hunger.food == 20 && hunger.saturation > 0.0 {
        // Fast saturated regeneration (every 10 ticks)
        if hunger.timer >= params.saturated_regen_interval {
            hunger.timer = 0;
            if health.cur < health.max {
                let heal_amount = (hunger.saturation / 6.0).clamp(0.0, 1.0);
                health.heal(heal_amount);
                hunger.add_exhaustion(6.0);
                result.healed = heal_amount;
            }
        }
    } else if hunger.food >= 18 {
        // Normal regeneration (every 80 ticks)
        if hunger.timer >= params.normal_regen_interval {
            hunger.timer = 0;
            if health.cur < health.max {
                health.heal(1.0);
                hunger.add_exhaustion(6.0);
                result.healed = 1.0;
            }
        }
    } else if hunger.food == 0 {
        // Starvation damage (every 80 ticks)
        if hunger.timer >= params.starvation_interval {
            hunger.timer = 0;
            // Starve down to 0.0 HP (can cause death)
            health.reduce(1.0);
            result.starved = 1.0;
        }
    } else {
        // Reset timer if not in any active regen/starvation state
        if hunger.timer >= 80 {
            hunger.timer = 0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exhaustion_depletes_saturation_then_food() {
        let mut hunger = Hunger::new(20, 2.0);
        let mut health = Health::new(20.0);
        let params = SimParams::default();

        // Add 5.0 exhaustion -> triggers 1 threshold drain (removes 4.0, leaves 1.0)
        hunger.add_exhaustion(5.0);
        tick_hunger(&mut health, &mut hunger, &params);
        assert_eq!(hunger.food, 20);
        assert!((hunger.saturation - 1.0).abs() < 0.01);
        assert!((hunger.exhaustion - 1.0).abs() < 0.01);

        // Add another 7.0 exhaustion -> total 8.0 -> triggers 2 threshold drains
        // 1st drains remaining saturation (1.0 -> 0.0)
        // 2nd drains food (20 -> 19)
        hunger.add_exhaustion(7.0);
        tick_hunger(&mut health, &mut hunger, &params);
        assert_eq!(hunger.food, 19);
        assert!(hunger.saturation.abs() < f32::EPSILON);
        assert!(hunger.exhaustion.abs() < f32::EPSILON);
    }

    #[test]
    fn test_saturated_regeneration() {
        let mut health = Health::new(20.0);
        health.reduce(5.0); // health at 15.0
        let mut hunger = Hunger::new(20, 6.0);
        let params = SimParams::default();

        for _ in 0..10 {
            tick_hunger(&mut health, &mut hunger, &params);
        }

        // At 10 ticks, should have healed
        assert!(health.cur > 15.0);
        assert!(hunger.exhaustion > 0.0);
    }

    #[test]
    fn test_starvation_damage() {
        let mut health = Health::new(20.0);
        let mut hunger = Hunger::new(0, 0.0);
        let params = SimParams::default();

        for _ in 0..80 {
            tick_hunger(&mut health, &mut hunger, &params);
        }

        assert!((health.cur - 19.0).abs() < f32::EPSILON);
    }
}
