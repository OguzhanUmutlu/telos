//! Bevy ECS simulation schedules, system sets, and standard tick systems.

use crate::attributes::{CombatTracker, Health};
use crate::effect::status_effect_system;
use crate::entity::{
    aquatic_movement_system, mob_ai_system, mob_hurt_decay_system, mob_movement_system,
};
use crate::hunger::{Hunger, SimParams, tick_hunger};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{ScheduleLabel, SystemSet};

/// Schedule label for the fixed 20 TPS simulation tick.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SimTick;

/// System sets defining the strictly ordered stages of a single simulation tick.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TickSet {
    /// Artificial intelligence decision making for mobs.
    Ai,
    /// Physical velocity integration, gravity, and drag.
    Movement,
    /// Processing player and environment interactions.
    Interaction,
    /// Damage resolution, invulnerability frames, and combat effects.
    Combat,
    /// Status effects, hunger, saturation, exhaustion, natural regeneration, and starvation.
    Status,
}

/// System that advances combat trackers (invulnerability cooldowns).
pub fn combat_tracker_system(mut query: Query<&mut CombatTracker>) {
    for mut tracker in &mut query {
        tracker.tick();
    }
}

/// System that runs the hunger, exhaustion, natural regeneration, and starvation simulation.
#[allow(clippy::needless_pass_by_value)]
pub fn hunger_system(mut query: Query<(&mut Health, &mut Hunger)>, params: Res<SimParams>) {
    for (mut health, mut hunger) in &mut query {
        tick_hunger(&mut health, &mut hunger, params.as_ref());
    }
}

/// Builds and configures the standard simulation tick schedule.
#[must_use]
pub fn build_sim_schedule() -> Schedule {
    let mut schedule = Schedule::new(SimTick);

    schedule.configure_sets(
        (
            TickSet::Ai,
            TickSet::Movement,
            TickSet::Interaction,
            TickSet::Combat,
            TickSet::Status,
        )
            .chain(),
    );

    schedule.add_systems(mob_ai_system.in_set(TickSet::Ai));
    schedule.add_systems((mob_movement_system, aquatic_movement_system).in_set(TickSet::Movement));
    schedule.add_systems((combat_tracker_system, mob_hurt_decay_system).in_set(TickSet::Combat));
    schedule.add_systems((hunger_system, status_effect_system).in_set(TickSet::Status));

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_schedule_execution() {
        let mut world = World::new();
        world.insert_resource(SimParams::default());

        let mut schedule = build_sim_schedule();

        let entity = world
            .spawn((
                Health::new(20.0),
                CombatTracker {
                    invulnerable_ticks: 5,
                    last_damage_amount: 4.0,
                },
                Hunger::new(20, 5.0),
            ))
            .id();

        // Run 1 tick
        schedule.run(&mut world);

        let tracker = world.get::<CombatTracker>(entity).unwrap();
        assert_eq!(tracker.invulnerable_ticks, 4);
    }
}
