//! Pure, deterministic gameplay simulation core for voxel.
//!
//! Provides survival attributes, hunger/exhaustion mechanics, experience leveling formulas,
//! inventory containers, crafting logic, and Bevy ECS simulation schedules.

#![forbid(unsafe_code)]

pub mod attributes;
pub mod bundles;
pub mod crafting;
pub mod experience;
pub mod hunger;
pub mod inventory;
pub mod schedule;
pub mod weather;

pub use attributes::{CombatTracker, DamageEvent, DamageType, Health, apply_damage};
pub use bundles::PlayerBundle;
pub use crafting::{Recipe2x2, find_recipe_2x2};
pub use experience::{
    Experience, level_from_total_points, points_for_next_level, total_points_for_level,
};
pub use hunger::{Hunger, HungerTickResult, SimParams, tick_hunger};
pub use inventory::{
    ARMOR_BOOTS_SLOT, ARMOR_CHESTPLATE_SLOT, ARMOR_HELMET_SLOT, ARMOR_LEGGINGS_SLOT, ARMOR_SLOTS,
    CRAFTING_INPUT_SLOTS, CRAFTING_RESULT_SLOT, ClickButton, ClickMode, HOTBAR_SLOTS, Inventory,
    InventoryError, ItemStack, MAX_STACK_SIZE, OFFHAND_SLOT, PLAYER_INVENTORY_SLOTS, STORAGE_SLOTS,
    inventory_click, is_armor, is_boots, is_chestplate, is_helmet, is_leggings,
    is_slot_valid_for_item, item_name, matching_armor_slot,
};
pub use schedule::{SimTick, TickSet, build_sim_schedule};
pub use weather::{
    ALTITUDE_LAPSE_RATE, LEVEL_FADE_PER_TICK, LIGHTNING_FLASH_DURATION_TICKS, PrecipitationKind,
    SEA_LEVEL_REF, SNOW_TEMP_THRESHOLD, WeatherKind, WeatherState, precipitation_at,
};
