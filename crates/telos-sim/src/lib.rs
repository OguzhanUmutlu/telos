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
pub mod fluid;
pub mod foliage;
pub mod hunger;
pub mod inventory;
pub mod logic;
pub mod movement;
pub mod nav;
pub mod particle;
pub mod potion;
pub mod prediction;
pub mod schedule;
pub mod smelting;
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
    ARROW_AIR_DRAG, ARROW_DESPAWN_FLYING_TICKS, ARROW_DESPAWN_STUCK_TICKS, ARROW_GRAVITY,
    ARROW_PICKUP_RADIUS, AiState, ArrowEntity, ArrowStepOutcome, AttackCooldown,
    BOW_FULL_CHARGE_TICKS, BOW_MAX_RELEASE_SPEED, BOW_MIN_CHARGE_TICKS, BOW_MIN_RELEASE_SPEED,
    EntityAabb, EntityType, HurtTime, ITEM_DESPAWN_TICKS, ITEM_MERGE_RADIUS, ITEM_PICKUP_RADIUS,
    ItemEntity, Mob, MobBundle, MobKind, NetEntity, PLAYER_DROP_PICKUP_DELAY, PathFollower,
    PlayerPositions, Position, Rotation, SimulationFrozen, TargetablePlayer, Velocity,
    has_line_of_sight, merge_item_stacks, mob_ai_system, mob_hurt_decay_system,
    mob_movement_system, tick_arrow_physics_step, tick_item_physics_step,
};
pub use event::{EventFilter, EventQueue, GameEvent};
pub use experience::{
    Experience, level_from_total_points, points_for_next_level, total_points_for_level,
};
pub use fluid::{
    FluidEngine, FluidKind, FluidReaction, FluidState, FluidWorldReader, LAVA_MAX_DECAY,
    LAVA_TICK_RATE, SLOPE_SEARCH_DISTANCE, ScheduledFluidTick, WATER_MAX_DECAY, WATER_TICK_RATE,
};
pub use foliage::{LEAF_DECAY_RADIUS, is_leaf_decaying};
pub use hunger::{Hunger, HungerTickResult, SimParams, tick_hunger};
pub use inventory::{
    ARMOR_BOOTS_SLOT, ARMOR_CHESTPLATE_SLOT, ARMOR_HELMET_SLOT, ARMOR_LEGGINGS_SLOT, ARMOR_SLOTS,
    CHEST_CONTAINER_SLOTS, CRAFTING_INPUT_SLOTS, CRAFTING_RESULT_SLOT, ChestInventory, ClickButton,
    ClickMode, DUAL_CONTAINER_SLOTS, HOTBAR_SLOTS, ITEM_ARROW, ITEM_BOW, ITEM_CHARCOAL, ITEM_CHEST,
    ITEM_COOKED_BEEF, ITEM_COOKED_PORKCHOP, ITEM_FURNACE, Inventory, InventoryError, ItemStack,
    MAX_STACK_SIZE, OFFHAND_SLOT, PLAYER_INVENTORY_SLOTS, STORAGE_SLOTS, block_to_drop_item,
    container_click, inventory_click, is_armor, is_arrow, is_boots, is_bow, is_chest,
    is_chestplate, is_furnace, is_helmet, is_leggings, is_slot_valid_for_item, item_name,
    matching_armor_slot,
};
pub use logic::{
    LogicChunk, LogicComponent, LogicEngine, LogicKind, MAX_SIGNAL_DISTANCE, ScheduledLogicTick,
};
pub use movement::{
    MAX_LEGAL_SPEED, MoveMode, MoveState, angles_to_direction, dequantize_pitch, dequantize_yaw,
    quantize_pitch, quantize_yaw, simulate_movement_step,
};
pub use nav::{
    ChunkPortalGraph, ChunkPortals, LocalAStar, NavPath, NavWorldReader, PathProfile, is_hazard,
    is_passable, is_solid_ground, is_walkable_node, octile_heuristic, update_mob_navigation_paths,
};
pub use particle::{Particle, ParticleGpu, ParticleKind, ParticleSystem};
pub use potion::{BrewingRecipe, BrewingRegistry, PotionType};
pub use prediction::{
    PREDICTION_BUFFER_CAPACITY, PredictionBuffer, PredictionEntry, ReconciliationResult,
    VisualSmoothing,
};
pub use schedule::{SimTick, TickSet, build_sim_schedule};
pub use smelting::{
    CONTAINER_FURNACE_HOTBAR_SLOTS, CONTAINER_FURNACE_SLOTS, CONTAINER_FURNACE_STORAGE_SLOTS,
    DUAL_FURNACE_SLOT_COUNT, FuelRegistry, FurnaceInventory, FurnaceTickResult, SmeltingRecipe,
    SmeltingRegistry, furnace_container_click, tick_furnace_step,
};
pub use weather::{
    ALTITUDE_LAPSE_RATE, LEVEL_FADE_PER_TICK, LIGHTNING_FLASH_DURATION_TICKS, PrecipitationKind,
    SEA_LEVEL_REF, SNOW_TEMP_THRESHOLD, WeatherKind, WeatherState, precipitation_at,
};
