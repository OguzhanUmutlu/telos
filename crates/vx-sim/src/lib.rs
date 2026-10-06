//! Pure, deterministic gameplay simulation core for voxel.
//!
//! Provides survival attributes, hunger/exhaustion mechanics, experience leveling formulas,
//! inventory containers, and Bevy ECS simulation schedules.

#![forbid(unsafe_code)]

pub mod attributes;
pub mod bundles;
pub mod experience;
pub mod hunger;
pub mod inventory;
pub mod schedule;

pub use attributes::{CombatTracker, DamageEvent, DamageType, Health, apply_damage};
pub use bundles::PlayerBundle;
pub use experience::{
    Experience, level_from_total_points, points_for_next_level, total_points_for_level,
};
pub use hunger::{Hunger, HungerTickResult, SimParams, tick_hunger};
pub use inventory::{
    ClickButton, ClickMode, Inventory, InventoryError, ItemStack, MAX_STACK_SIZE,
    PLAYER_INVENTORY_SLOTS, inventory_click,
};
pub use schedule::{SimTick, TickSet, build_sim_schedule};
