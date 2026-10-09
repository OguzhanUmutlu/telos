//! Pure, deterministic gameplay simulation core for voxel.
//!
//! Provides survival attributes, hunger/exhaustion mechanics, experience leveling formulas,
//! inventory containers, crafting logic, and Bevy ECS simulation schedules.

#![forbid(unsafe_code)]

pub mod advancement;
/// Anvil combining, repairing, and enchanting mechanics.
pub mod anvil;
pub mod attributes;
pub mod bundles;
pub mod capabilities;
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

pub use advancement::{
    Advancement, AdvancementCategory, AdvancementCriterion, AdvancementFrame, AdvancementRegistry,
    CriterionTrigger, PlayerAdvancements,
};
pub use anvil::{
    ANVIL_CONTAINER_SLOTS, ANVIL_SLOT_LEFT, ANVIL_SLOT_RESULT, ANVIL_SLOT_RIGHT,
    AnvilCombineResult, AnvilInventory, DUAL_ANVIL_SLOT_COUNT, anvil_container_click,
    can_apply_enchantment, combine_anvil_items,
};
pub use attributes::{
    Attribute, AttributeKind, AttributeModifier, Attributes, CombatTracker, DamageEvent,
    DamageType, Health, ModifierOperation, apply_damage, apply_mitigated_damage,
    calculate_damage_mitigation,
};
pub use bundles::PlayerBundle;
pub use capabilities::{
    CAP_FLAG_ALLOW_FLIGHT, CAP_FLAG_CAN_BUILD, CAP_FLAG_FLYING, CAP_FLAG_INSTABREAK,
    CAP_FLAG_INVINCIBLE, CAP_FLAG_NOCLIP, GameMode, PlayerCapabilities,
};
pub use command::{
    CommandContext, CommandDispatcher, CommandNode, CommandOutput, CommandSuggestions,
    register_builtins,
};
pub use crafting::{
    CONTAINER_CRAFTING_GRID_SLOTS, CONTAINER_CRAFTING_HOTBAR_SLOTS, CONTAINER_CRAFTING_RESULT_SLOT,
    CONTAINER_CRAFTING_STORAGE_SLOTS, CRAFTING_TABLE_CONTAINER_SLOTS, CraftingRecipe,
    CraftingTableInventory, DUAL_CRAFTING_TABLE_SLOT_COUNT, Recipe2x2, RecipeRegistry,
    ShapedRecipe, ShapelessRecipe, crafting_table_container_click, find_recipe_2x2,
    find_recipe_2x2_with_registry,
};
pub use effect::{
    EffectInstance, StatusEffectDef, StatusEffectKind, StatusEffectRegistry, StatusEffects,
    status_effect_system,
};
pub use enchantment::{
    CompactEnchantments, EnchantmentKind, EnchantmentTarget, calculate_arrow_damage,
    calculate_arrow_knockback, calculate_fire_aspect_seconds, calculate_knockback_bonus,
    calculate_melee_damage, calculate_mining_speed, calculate_total_epf, has_infinite_arrows,
    is_arrow_flaming, should_prevent_durability_loss,
};
pub use entity::{
    ARROW_AIR_DRAG, ARROW_DESPAWN_FLYING_TICKS, ARROW_DESPAWN_STUCK_TICKS, ARROW_GRAVITY,
    ARROW_PICKUP_RADIUS, AiState, AquaticMob, ArrowEntity, ArrowStepOutcome, AttackCooldown,
    BOW_FULL_CHARGE_TICKS, BOW_MAX_RELEASE_SPEED, BOW_MIN_CHARGE_TICKS, BOW_MIN_RELEASE_SPEED,
    BoatEntity, EntityAabb, EntityType, HurtTime, ITEM_DESPAWN_TICKS, ITEM_MERGE_RADIUS,
    ITEM_PICKUP_RADIUS, ItemEntity, Mob, MobBundle, MobKind, NetEntity, PLAYER_DROP_PICKUP_DELAY,
    PathFollower, PlayerPositions, Position, Riding, Rotation, SimulationFrozen, TargetablePlayer,
    Velocity, aquatic_movement_system, boat_movement_system, has_line_of_sight, merge_item_stacks,
    mob_ai_system, mob_hurt_decay_system, mob_movement_system, tick_arrow_physics_step,
    tick_item_physics_step, tick_item_physics_step_fluid,
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
    ClickMode, DUAL_CONTAINER_SLOTS, HOTBAR_SLOTS, ITEM_ANVIL, ITEM_ARROW, ITEM_BOOK, ITEM_BOW,
    ITEM_CHARCOAL, ITEM_CHEST, ITEM_COD, ITEM_COOKED_BEEF, ITEM_COOKED_COD, ITEM_COOKED_PORKCHOP,
    ITEM_CRAFTING_TABLE, ITEM_ENCHANTED_BOOK, ITEM_FURNACE, ITEM_GLASS_BOTTLE, ITEM_GOLDEN_BOOTS,
    ITEM_GOLDEN_CHESTPLATE, ITEM_GOLDEN_HELMET, ITEM_GOLDEN_LEGGINGS, ITEM_INK_SAC, ITEM_IRON_AXE,
    ITEM_IRON_BOOTS, ITEM_IRON_CHESTPLATE, ITEM_IRON_HELMET, ITEM_IRON_HOE, ITEM_IRON_LEGGINGS,
    ITEM_IRON_PICKAXE, ITEM_IRON_SHOVEL, ITEM_IRON_SWORD, ITEM_LEATHER_BOOTS,
    ITEM_LEATHER_CHESTPLATE, ITEM_LEATHER_HELMET, ITEM_LEATHER_LEGGINGS, ITEM_OAK_BOAT,
    ITEM_POTION, ITEM_STONE_AXE, ITEM_STONE_HOE, ITEM_STONE_PICKAXE, ITEM_STONE_SHOVEL,
    ITEM_STONE_SWORD, ITEM_WOODEN_AXE, ITEM_WOODEN_HOE, ITEM_WOODEN_PICKAXE, ITEM_WOODEN_SHOVEL,
    ITEM_WOODEN_SWORD, Inventory, InventoryError, ItemStack, MAX_STACK_SIZE, OFFHAND_SLOT,
    PLAYER_INVENTORY_SLOTS, STORAGE_SLOTS, block_to_drop_item, container_click, inventory_click,
    inventory_click_with_registry, is_anvil, is_armor, is_arrow, is_axe, is_boat, is_book,
    is_boots, is_bow, is_chest, is_chestplate, is_enchanted_book, is_furnace, is_helmet, is_hoe,
    is_leggings, is_pickaxe, is_potion, is_shovel, is_slot_valid_for_item, is_sword, is_tool,
    item_name, matching_armor_slot,
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
pub use potion::{BrewingRecipe, BrewingRegistry, PotionType, consume_item_effects};
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
