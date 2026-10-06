//! Standard entity bundles for spawning players and entities.

use crate::attributes::{CombatTracker, Health};
use crate::experience::Experience;
use crate::hunger::Hunger;
use crate::inventory::Inventory;
use bevy_ecs::bundle::Bundle;

/// Component bundle required for a fully simulated player entity.
#[derive(Debug, Clone, Default, Bundle)]
pub struct PlayerBundle {
    /// Living health and max health points.
    pub health: Health,
    /// Combat invulnerability and cooldown tracker.
    pub combat: CombatTracker,
    /// Food, saturation, and exhaustion levels.
    pub hunger: Hunger,
    /// Experience points and leveling stats.
    pub experience: Experience,
    /// 36-slot inventory container and cursor stack.
    pub inventory: Inventory,
}

impl PlayerBundle {
    /// Creates a fresh player bundle with default survival parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
