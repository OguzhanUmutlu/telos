//! Pure, deterministic gameplay simulation core for voxel.
//!
//! Provides survival attributes, hunger/exhaustion mechanics, experience leveling formulas,
//! inventory containers, crafting logic, and Bevy ECS simulation schedules.

#![forbid(unsafe_code)]

pub mod attributes;
pub mod bundles;
pub mod command;
pub mod crafting;
pub mod effect;
pub mod enchantment;
pub mod entity;
pub mod event;
pub mod experience;
pub mod hunger;
pub mod inventory;
pub mod logic;
pub mod movement;
pub mod particle;
pub mod potion;
pub mod prediction;
pub mod schedule;
pub mod weather;

pub use attributes::{
    Attribute, AttributeKind, AttributeModifier, Attributes, CombatTracker, DamageEvent,
    DamageType, Health, ModifierOperation, apply_damage, apply_mitigated_damage,
    calculate_damage_mitigation,
};
pub use bundles::PlayerBundle;
pub use command::{
    CommandContext, CommandDispatcher, CommandNode, CommandOutput, CommandSuggestions,
    register_builtins,
};
pub use crafting::{Recipe2x2, find_recipe_2x2};
pub use effect::{EffectInstance, StatusEffectKind, StatusEffects, status_effect_system};
pub use enchantment::{
    CompactEnchantments, EnchantmentKind, EnchantmentTarget, calculate_total_epf,
};
pub use entity::{
    AiState, EntityAabb, EntityType, HurtTime, Mob, MobBundle, MobKind, NetEntity, PlayerPositions,
    Position, Rotation, SimulationFrozen, Velocity, mob_ai_system, mob_hurt_decay_system,
    mob_movement_system,
};
pub use event::{EventFilter, EventQueue, GameEvent};
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
pub use logic::{
    LogicChunk, LogicComponent, LogicEngine, LogicKind, MAX_SIGNAL_DISTANCE, ScheduledLogicTick,
};
pub use movement::{
    MAX_LEGAL_SPEED, MoveMode, MoveState, angles_to_direction, dequantize_pitch, dequantize_yaw,
    quantize_pitch, quantize_yaw, simulate_movement_step,
};
pub use particle::{Particle, ParticleGpu, ParticleKind, ParticleSystem};
pub use potion::{BrewingRecipe, BrewingRegistry, PotionType};
pub use prediction::{
    PREDICTION_BUFFER_CAPACITY, PredictionBuffer, PredictionEntry, ReconciliationResult,
    VisualSmoothing,
};
pub use schedule::{SimTick, TickSet, build_sim_schedule};
pub use weather::{
    ALTITUDE_LAPSE_RATE, LEVEL_FADE_PER_TICK, LIGHTNING_FLASH_DURATION_TICKS, PrecipitationKind,
    SEA_LEVEL_REF, SNOW_TEMP_THRESHOLD, WeatherKind, WeatherState, precipitation_at,
};
