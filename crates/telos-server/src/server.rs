//! Authoritative game server ticking at 20 TPS with session management and chunk streaming.

use crate::form::{
    FormHandlerKind, HELP_COMMANDS, PendingForm, build_help_detail_form, build_help_index_form,
};
use glam::{DVec3, Vec3};
use hashbrown::HashMap;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use telos_content::{
    FrozenRegistries, ModSide, RegistryBuilder, discover_packs, resolve_load_order,
};
use telos_core::coords::{BlockPos, ChunkPos, Face};
use telos_core::form::{FormResponseData, ModalFormData};
use telos_mod::{JsPlugin, JsPluginEngine, ModConfig, ModManager, ModPermissions, ModResult};
use telos_net::{Connection, Lane, Payload};
use telos_protocol::bounded::{BoundedString, BoundedVec};
use telos_protocol::messages::{
    BlockActionKind, C2sBlockAction, C2sChatMessage, C2sCloseContainer, C2sCommandSuggest,
    C2sInteractEntity, C2sInventoryClick, C2sMessage, C2sModalFormResponse, C2sPlayerCommand,
    ChunkPayload, ConnectionPhase, LodPayload, NetworkEffect, ParticleEffectKind,
    PlayerCommandKind, S2cBlockActionAck, S2cBlockEvent, S2cBlockUpdate, S2cChatMessage,
    S2cChunkData, S2cChunkUnload, S2cCloseContainer, S2cCommandSuggestions, S2cConfigDone,
    S2cContainerProperty, S2cDespawnEntity, S2cEntityMove, S2cEntityStatus, S2cGameMode,
    S2cHelloReply, S2cInventoryBulk, S2cInventorySlot, S2cJoinGame, S2cLodNodeData,
    S2cLodNodeUnload, S2cLoginSuccess, S2cMessage, S2cModalFormRequest, S2cOpenContainer,
    S2cParticleEvent, S2cPlayerMovementAck, S2cRegistryData, S2cSpawnArrow, S2cSpawnEntity,
    S2cSpawnItem, S2cUniformChunk, S2cUpdateEffects, S2cUpdateStats, S2cUpdateTime,
    S2cUpdateWeather, SlotData,
};
use telos_sim::command::{
    ArgumentType, CommandContext, CommandDispatcher, CommandNode, CommandOutput, register_builtins,
};
use telos_sim::event::{EventQueue, GameEvent};
use telos_sim::{
    ARMOR_SLOTS, ARROW_DESPAWN_FLYING_TICKS, ARROW_DESPAWN_STUCK_TICKS, ARROW_PICKUP_RADIUS,
    AiState, ArrowEntity, ArrowStepOutcome, AttackCooldown, AttributeKind, Attributes,
    BOW_FULL_CHARGE_TICKS, BOW_MAX_RELEASE_SPEED, BOW_MIN_CHARGE_TICKS, BOW_MIN_RELEASE_SPEED,
    CombatTracker, DamageType, EffectInstance, EnchantmentKind, EntityType, Experience, GameMode,
    Health, Hunger, HurtTime, ITEM_ARROW, ITEM_BOW, ITEM_DESPAWN_TICKS, ITEM_MERGE_RADIUS,
    ITEM_PICKUP_RADIUS, Inventory, ItemEntity, ItemStack, Mob, MobBundle, MoveMode, NetEntity,
    PLAYER_DROP_PICKUP_DELAY, PlayerPositions, Position, PotionType, Rotation, SimParams,
    SimulationFrozen, StatusEffectKind, StatusEffects, TargetablePlayer, Velocity, WeatherKind,
    WeatherState, apply_mitigated_damage, block_to_drop_item, build_sim_schedule,
    calculate_total_epf, merge_item_stacks, tick_arrow_physics_step, tick_item_physics_step,
};
use telos_voxel::state::BlockStateId;
use telos_voxel::storage::Blocks;
use tracing::{debug, info};

use crate::builder::ServerBuilder;
use crate::config::ServerConfig;
use crate::error::ServerError;
use crate::multi_world::{MultiWorldManager, WorldError};
use crate::session::{ActiveContainerSession, PlayerSession};
use crate::world::ServerWorld;

fn register_world_command(dispatcher: &mut CommandDispatcher) {
    let world_node = CommandNode::literal("world")
        .with_tooltip("Manage worlds and transfer players")
        .executes(|_| {
            CommandOutput::success("Usage: /world list or /world tp <world_name> [x y z]")
        })
        .then(
            CommandNode::literal("list")
                .with_tooltip("List all active worlds")
                .executes(|_| CommandOutput::success("Listing worlds...")),
        )
        .then(
            CommandNode::literal("tp")
                .with_tooltip("Teleport to another world")
                .then(
                    CommandNode::argument("name", ArgumentType::Word)
                        .with_tooltip("Destination world name")
                        .executes(|_| CommandOutput::success("Teleporting to world..."))
                        .then(
                            CommandNode::argument("pos", ArgumentType::Vec3)
                                .with_tooltip("Optional coordinates in destination world")
                                .executes(|_| {
                                    CommandOutput::success("Teleporting to world coordinates...")
                                }),
                        ),
                ),
        );
    dispatcher.register(world_node);
}

/// Runtime state of an active monster spawner block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnerState {
    /// Type of mob spawned by this block.
    pub mob_type: EntityType,
    /// Ticks remaining until next spawn attempt.
    pub spawn_delay: u16,
}

/// Top-level authoritative server orchestrating worlds, simulation, and client streaming.
pub struct Server {
    config: ServerConfig,
    worlds: MultiWorldManager,
    /// Active frozen registries.
    pub registries: Arc<FrozenRegistries>,
    sessions: HashMap<u64, PlayerSession>,
    next_session_id: u64,
    next_entity_id: u32,
    tick_count: u64,
    time_of_day: u64,
    /// Active server weather simulation state.
    pub weather: WeatherState,
    ecs_world: bevy_ecs::world::World,
    sim_schedule: bevy_ecs::schedule::Schedule,
    /// Active mob entities keyed by their network ID.
    pub tracked_mobs: HashMap<u32, bevy_ecs::entity::Entity>,
    /// Last known positions of mobs for delta move broadcasting.
    pub mob_positions: HashMap<u32, DVec3>,
    /// Last known yaw of mobs for delta rotation broadcasting.
    pub mob_yaws: HashMap<u32, f32>,
    /// Active dropped item entities keyed by their network ID.
    pub tracked_items: HashMap<u32, bevy_ecs::entity::Entity>,
    /// Last known positions of items for delta move broadcasting.
    pub item_positions: HashMap<u32, DVec3>,
    /// Active arrow entities keyed by their network ID.
    pub tracked_arrows: HashMap<u32, bevy_ecs::entity::Entity>,
    /// Last known positions of arrows for delta move broadcasting.
    pub arrow_positions: HashMap<u32, DVec3>,
    /// Tick count when natural mob spawning last ran.
    pub last_mob_spawn_tick: u64,
    /// Active monster spawners tracked by the server.
    pub active_spawners: HashMap<BlockPos, SpawnerState>,
    /// Authoritative command dispatcher and syntax tree.
    pub command_dispatcher: Arc<CommandDispatcher>,
    /// Central manager for sandboxed WebAssembly mods and scripts.
    pub mod_manager: ModManager,
    /// Central manager for sandboxed JavaScript server plugins.
    pub js_plugins: JsPluginEngine,
    /// Bounded event queue buffering simulation events during the tick.
    pub event_queue: EventQueue,
    /// Remote QUIC network listener accepting external client connections.
    pub listener: Option<telos_net::QuicListener>,
    /// LAN discovery beacon emitter broadcasting over UDP.
    pub lan_emitter: Option<telos_net::LanBeaconEmitter>,
    /// Registry of deterministic smelting recipes.
    pub smelting_recipes: telos_sim::SmeltingRegistry,
    /// Registry of combustible fuels and burn durations.
    pub fuel_registry: telos_sim::FuelRegistry,
    /// Registry of deterministic crafting recipes.
    pub recipe_registry: telos_sim::RecipeRegistry,
    /// Next unique modal form identifier.
    pub next_form_id: u32,
    /// Outstanding pending modal forms dispatched to clients.
    pub pending_forms: HashMap<(u64, u32), PendingForm>,
    /// Thread-safe signal to request server shutdown.
    shutdown_requested: Arc<AtomicBool>,
}

impl Server {
    /// Creates a new `Server` instance with procedural worldgen seed and configuration.
    #[must_use]
    #[allow(clippy::collapsible_if)]
    pub fn new(seed: u64, config: ServerConfig) -> Self {
        let mut builder = RegistryBuilder::new();
        let _ = builder.load_core_pack();
        if !config.data_pack_directories.is_empty() {
            if let Ok(discovered) = discover_packs(&config.data_pack_directories) {
                if let Ok(sorted) = resolve_load_order(discovered, ModSide::Server) {
                    for pack in &sorted {
                        let _ = builder.load_pack(pack);
                    }
                }
            }
        }
        let registries = Arc::new(builder.freeze().unwrap_or_else(|_| {
            let mut fallback = RegistryBuilder::new();
            let _ = fallback.load_core_pack();
            fallback.freeze().expect("Core pack always freezes")
        }));

        Self::with_registries(seed, config, registries)
    }

    /// Creates a new `Server` instance with pre-configured frozen registries.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_registries(
        seed: u64,
        config: ServerConfig,
        registries: Arc<FrozenRegistries>,
    ) -> Self {
        let mut worlds_vec = Vec::new();
        if config.worlds.is_empty() {
            let default_world = if let Some(dir) = &config.save_directory {
                match ServerWorld::with_content_storage_and_compression(
                    seed,
                    &registries,
                    dir,
                    config.region_compression,
                    config.region_compression_level,
                ) {
                    Ok(w) => w,
                    Err(err) => {
                        tracing::error!(
                            "Failed to initialize storage at {dir:?}: {err}; falling back to memory"
                        );
                        ServerWorld::new(seed, registries.block_registry().clone())
                    }
                }
            } else {
                ServerWorld::new(seed, registries.block_registry().clone())
            };
            worlds_vec.push(("overworld".to_string(), default_world));
        } else {
            for (idx, w_cfg) in config.worlds.iter().enumerate() {
                let mut resolved_w_cfg = w_cfg.clone();
                if let (None, Some(base_dir)) =
                    (&resolved_w_cfg.save_directory, &config.save_directory)
                {
                    resolved_w_cfg.region_compression = config.region_compression;
                    resolved_w_cfg.region_compression_level = config.region_compression_level;
                    if idx == 0 {
                        resolved_w_cfg.save_directory = Some(base_dir.clone());
                    } else {
                        resolved_w_cfg.save_directory =
                            Some(base_dir.join("dimensions").join(&w_cfg.name));
                    }
                }
                let world = match ServerWorld::with_config(&resolved_w_cfg, &registries) {
                    Ok(w) => w,
                    Err(err) => {
                        tracing::error!(
                            "Failed to initialize world '{}' storage: {err}; falling back to memory",
                            w_cfg.name
                        );
                        ServerWorld::with_generator(
                            w_cfg.name.clone(),
                            w_cfg.seed,
                            registries.block_registry().clone(),
                            w_cfg.generator,
                        )
                    }
                };
                worlds_vec.push((w_cfg.name.clone(), world));
            }
        }

        let (first_name, first_world) = worlds_vec.remove(0);
        let mut worlds = MultiWorldManager::new(first_name, first_world);
        for (name, world) in worlds_vec {
            let _ = worlds.add_world(name, world);
        }

        let lan_emitter = if config.lan_broadcast {
            let port = config
                .bind_address
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(25565);
            let beacon = telos_net::LanBeacon::new(port, &config.motd);
            telos_net::LanBeaconEmitter::spawn(beacon, port).ok()
        } else {
            None
        };

        let mut ecs_world = bevy_ecs::world::World::new();
        ecs_world.insert_resource(SimParams::default());
        let sim_schedule = build_sim_schedule();

        let mut dispatcher = CommandDispatcher::new();
        register_builtins(&mut dispatcher);
        register_world_command(&mut dispatcher);
        let command_dispatcher = Arc::new(dispatcher);

        let mod_manager = ModManager::new().expect("ModManager engine initialization");
        let mut js_plugins = JsPluginEngine::new();
        if let Some(save_dir) = &config.save_directory {
            for sub_dir in ["plugins", "scripts"] {
                let dir = save_dir.join(sub_dir);
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("js")
                            && let Ok(content) = std::fs::read_to_string(&path)
                        {
                            let id = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("plugin");
                            if let Ok(plugin) = JsPlugin::new(id, &content) {
                                tracing::info!("Loaded JavaScript server plugin: {id}");
                                js_plugins.add_plugin(plugin);
                            }
                        }
                    }
                }
            }
        }

        let event_queue = EventQueue::default();

        Self {
            config,
            worlds,
            registries,
            sessions: HashMap::new(),
            next_session_id: 1,
            next_entity_id: 1,
            tick_count: 0,
            time_of_day: telos_core::NOON_TICKS,
            weather: WeatherState::new(seed),
            ecs_world,
            sim_schedule,
            tracked_mobs: HashMap::new(),
            mob_positions: HashMap::new(),
            mob_yaws: HashMap::new(),
            tracked_items: HashMap::new(),
            item_positions: HashMap::new(),
            tracked_arrows: HashMap::new(),
            arrow_positions: HashMap::new(),
            last_mob_spawn_tick: 0,
            active_spawners: HashMap::new(),
            command_dispatcher,
            mod_manager,
            js_plugins,
            event_queue,
            listener: None,
            lan_emitter,
            smelting_recipes: telos_sim::SmeltingRegistry::standard(),
            fuel_registry: telos_sim::FuelRegistry::standard(),
            recipe_registry: telos_sim::RecipeRegistry::standard(),
            next_form_id: 1,
            pending_forms: HashMap::new(),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Loads and activates a sandboxed JavaScript plugin from source script.
    pub fn load_js_plugin(&mut self, id: impl Into<String>, source: &str) -> ModResult<()> {
        let plugin = JsPlugin::new(id, source)?;
        self.js_plugins.add_plugin(plugin);
        Ok(())
    }

    /// Loads a sandboxed mod from WebAssembly text format (`.wat`).
    pub fn load_mod_from_wat(
        &mut self,
        mod_id: &str,
        wat: &str,
        permissions: ModPermissions,
        config: ModConfig,
    ) -> ModResult<()> {
        self.mod_manager
            .load_mod_from_wat(mod_id, wat, permissions, config)
    }

    /// Loads a sandboxed mod from WebAssembly binary bytes (`.wasm`).
    pub fn load_mod_from_bytes(
        &mut self,
        mod_id: &str,
        bytes: &[u8],
        permissions: ModPermissions,
        config: ModConfig,
    ) -> ModResult<()> {
        self.mod_manager
            .load_mod_from_bytes(mod_id, bytes, permissions, config)
    }

    /// Returns the active frozen registries.
    #[must_use]
    pub fn registries(&self) -> &Arc<FrozenRegistries> {
        &self.registries
    }

    /// Spawns a mob entity of the given type at `pos`.
    pub fn spawn_mob(&mut self, entity_type: EntityType, pos: DVec3) -> u32 {
        let net_id = self.next_entity_id;
        self.next_entity_id += 1;

        let seed = self
            .worlds
            .default_world()
            .seed()
            .wrapping_add(u64::from(net_id));
        let bundle = match entity_type {
            EntityType::Pig => MobBundle::new_pig(net_id, pos, seed),
            EntityType::Cow => MobBundle::new_cow(net_id, pos, seed),
            EntityType::Zombie | EntityType::Player | EntityType::Item | EntityType::Arrow => {
                MobBundle::new_zombie(net_id, pos, seed)
            }
        };
        let health = bundle.health.cur;
        let entity = self
            .ecs_world
            .spawn((bundle, Attributes::player_default(), StatusEffects::new()))
            .id();

        self.tracked_mobs.insert(net_id, entity);
        self.mob_positions.insert(net_id, pos);
        self.mob_yaws.insert(net_id, 0.0);

        let spawn_msg = S2cMessage::SpawnEntity(S2cSpawnEntity {
            net_id,
            entity_type: entity_type.to_u8(),
            x: pos.x,
            y: pos.y,
            z: pos.z,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            health,
            max_health: health,
        });

        for s in self.sessions.values_mut() {
            if s.phase == ConnectionPhase::Play {
                let _ = s
                    .connection
                    .send(Lane::Control, Payload::Msg(spawn_msg.clone()));
            }
        }

        net_id
    }

    /// Despawns a mob entity by its network ID.
    pub fn despawn_mob(&mut self, net_id: u32) {
        if let Some(entity) = self.tracked_mobs.remove(&net_id) {
            self.ecs_world.despawn(entity);
            self.mob_positions.remove(&net_id);
            self.mob_yaws.remove(&net_id);

            let despawn_msg = S2cMessage::DespawnEntity(S2cDespawnEntity {
                net_ids: BoundedVec::new(vec![net_id]).expect("single net_id"),
            });

            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(despawn_msg.clone()));
                }
            }
        }
    }

    /// Sets the game mode and capabilities of a player session and notifies the client.
    pub fn apply_game_mode(&mut self, session_id: u64, mode: GameMode) {
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.game_mode = mode;
            session.capabilities = mode.default_capabilities();
            session.move_mode = match mode {
                GameMode::Creative | GameMode::Spectator => MoveMode::NoClipFly,
                GameMode::Survival | GameMode::Adventure => MoveMode::Walk,
            };
            session.move_state.flying = session.capabilities.flying;
            let s2c_msg = S2cMessage::GameMode(S2cGameMode {
                game_mode: mode.id(),
                flags: session.capabilities.flags(),
                fly_speed: session.capabilities.fly_speed,
                walk_speed: session.capabilities.walk_speed,
                reach_distance: session.capabilities.reach_distance,
            });
            let _ = session
                .connection
                .send(Lane::Control, Payload::Msg(s2c_msg));
            info!(session_id, ?mode, "Applied game mode to session");
        }
    }

    /// Spawns a dropped item entity in the specified world at `pos` with initial `vel`.
    pub fn spawn_item_entity(
        &mut self,
        world_name: &str,
        pos: DVec3,
        vel: Vec3,
        stack: ItemStack,
        pickup_delay: u16,
    ) -> u32 {
        if stack.is_empty() {
            return 0;
        }

        let net_id = self.next_entity_id;
        self.next_entity_id += 1;

        let item_component = ItemEntity::new(stack, pickup_delay);
        let entity = self
            .ecs_world
            .spawn((
                NetEntity {
                    net_id,
                    entity_type: EntityType::Item,
                },
                Position(pos),
                Velocity(vel),
                item_component,
            ))
            .id();

        self.tracked_items.insert(net_id, entity);
        self.item_positions.insert(net_id, pos);

        let spawn_msg = S2cMessage::SpawnItem(S2cSpawnItem {
            net_id,
            item_id: stack.item,
            count: stack.count,
            x: pos.x,
            y: pos.y,
            z: pos.z,
            vel_x: vel.x,
            vel_y: vel.y,
            vel_z: vel.z,
        });

        for s in self.sessions.values_mut() {
            if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                let _ = s
                    .connection
                    .send(Lane::Control, Payload::Msg(spawn_msg.clone()));
            }
        }

        net_id
    }

    /// Despawns a dropped item entity by its network ID.
    pub fn despawn_item_entity(&mut self, net_id: u32) {
        if let Some(entity) = self.tracked_items.remove(&net_id) {
            self.ecs_world.despawn(entity);
            self.item_positions.remove(&net_id);

            let despawn_msg = S2cMessage::DespawnEntity(S2cDespawnEntity {
                net_ids: BoundedVec::new(vec![net_id]).expect("single net_id"),
            });

            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(despawn_msg.clone()));
                }
            }
        }
    }

    /// Closes an open container session for the specified session, returning any crafting grid
    /// items or carried items to the player's inventory and dropping any overflow into the world.
    ///
    /// Returns `Some((world_name, block_pos, container_kind))` if a container was open.
    pub fn close_session_container(&mut self, session_id: u64) -> Option<(String, BlockPos, u8)> {
        let session = self.sessions.get_mut(&session_id)?;
        let active_cont = session.active_container.take()?;
        let world_name = session.world_name.clone();
        let block_pos = active_cont.block_pos;
        let container_kind = active_cont.container_kind;
        let entity = session.ecs_entity;
        let pos = session.position;

        let mut dropped = Vec::new();
        if container_kind == 2 {
            // Crafting table: refund 3x3 grid items to player inventory, drop overflow
            let mut grid = std::mem::take(&mut session.active_crafting_table.grid);
            session.active_crafting_table.result = ItemStack::EMPTY;

            if let Some(entity) = entity
                && let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(entity)
            {
                if !inv.carried.is_empty() {
                    inv.return_carried();
                }
                for stack in &mut grid {
                    if !stack.is_empty() {
                        inv.insert_into_storage_or_hotbar(stack);
                        if !stack.is_empty() {
                            dropped.push(*stack);
                        }
                        *stack = ItemStack::EMPTY;
                    }
                }
                let mut slot_vec = Vec::with_capacity(inv.slots.len());
                for slot in &inv.slots {
                    slot_vec.push(SlotData {
                        item: slot.item,
                        count: slot.count,
                    });
                }
                let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                    carried: SlotData {
                        item: inv.carried.item,
                        count: inv.carried.count,
                    },
                });
                let _ = session
                    .connection
                    .send(Lane::Control, Payload::Msg(bulk_msg));
            } else {
                for stack in grid {
                    if !stack.is_empty() {
                        dropped.push(stack);
                    }
                }
            }
        } else if let Some(entity) = entity
            && let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(entity)
            && !inv.carried.is_empty()
        {
            inv.return_carried();
            let mut slot_vec = Vec::with_capacity(inv.slots.len());
            for slot in &inv.slots {
                slot_vec.push(SlotData {
                    item: slot.item,
                    count: slot.count,
                });
            }
            let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                carried: SlotData {
                    item: inv.carried.item,
                    count: inv.carried.count,
                },
            });
            let _ = session
                .connection
                .send(Lane::Control, Payload::Msg(bulk_msg));
        }

        for stack in dropped {
            self.spawn_item_entity(
                &world_name,
                pos,
                Vec3::new(0.0, 0.1, 0.0),
                stack,
                PLAYER_DROP_PICKUP_DELAY,
            );
        }

        Some((world_name, block_pos, container_kind))
    }

    /// Spawns an arrow entity into the world, broadcasts `S2cSpawnArrow`, and returns its `net_id`.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_arrow_entity(
        &mut self,
        world_name: &str,
        pos: DVec3,
        vel: Vec3,
        yaw: f32,
        pitch: f32,
        shooter_id: Option<u32>,
        damage: f32,
        pickup_allowed: bool,
    ) -> u32 {
        let net_id = self.next_entity_id;
        self.next_entity_id += 1;

        let arrow_comp = ArrowEntity {
            shooter_id,
            in_ground: false,
            stuck_block: None,
            age: 0,
            damage,
            pickup_allowed,
        };

        let entity = self
            .ecs_world
            .spawn((
                NetEntity {
                    net_id,
                    entity_type: EntityType::Arrow,
                },
                Position(pos),
                Velocity(vel),
                Rotation {
                    yaw,
                    pitch,
                    head_yaw: yaw,
                },
                arrow_comp,
            ))
            .id();

        self.tracked_arrows.insert(net_id, entity);
        self.arrow_positions.insert(net_id, pos);

        let spawn_msg = S2cMessage::SpawnArrow(S2cSpawnArrow {
            net_id,
            x: pos.x,
            y: pos.y,
            z: pos.z,
            vel_x: vel.x,
            vel_y: vel.y,
            vel_z: vel.z,
            yaw,
            pitch,
        });

        for s in self.sessions.values_mut() {
            if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                let _ = s
                    .connection
                    .send(Lane::Control, Payload::Msg(spawn_msg.clone()));
            }
        }

        net_id
    }

    /// Despawns an arrow entity by its network ID.
    pub fn despawn_arrow_entity(&mut self, net_id: u32) {
        if let Some(entity) = self.tracked_arrows.remove(&net_id) {
            self.ecs_world.despawn(entity);
            self.arrow_positions.remove(&net_id);

            let despawn_msg = S2cMessage::DespawnEntity(S2cDespawnEntity {
                net_ids: BoundedVec::new(vec![net_id]).expect("single net_id"),
            });

            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(despawn_msg.clone()));
                }
            }
        }
    }

    /// Handles mob death transitions, broadcasting death animation and particles,
    /// awarding experience, dropping loot, and despawning the mob.
    pub fn handle_mob_death(&mut self, net_id: u32, killer_session_id: Option<u64>) {
        let Some(&entity) = self.tracked_mobs.get(&net_id) else {
            return;
        };

        let entity_type = self
            .ecs_world
            .get::<NetEntity>(entity)
            .map_or(EntityType::Zombie, |n| n.entity_type);
        let mob_pos = self.ecs_world.get::<Position>(entity).map(|p| p.0);

        // 1. Broadcast death animation status (status: 3)
        let death_msg = S2cMessage::EntityStatus(S2cEntityStatus { net_id, status: 3 });
        for s in self.sessions.values_mut() {
            if s.phase == ConnectionPhase::Play {
                let _ = s
                    .connection
                    .send(Lane::Control, Payload::Msg(death_msg.clone()));
            }
        }

        // 2. Broadcast death smoke poof particle burst
        if let Some(pos) = mob_pos {
            #[allow(clippy::cast_possible_truncation)]
            let smoke_msg = S2cMessage::ParticleEvent(S2cParticleEvent {
                effect: ParticleEffectKind::Smoke,
                x: pos.x as f32,
                y: (pos.y + 0.5) as f32,
                z: pos.z as f32,
                count: 20,
                speed: 0.15,
                block_state_id: 0,
            });
            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(smoke_msg.clone()));
                }
            }
        }

        // 3. Loot drops and XP determination
        let drop_seed = self.tick_count.wrapping_add(u64::from(net_id));
        let count_1_to_2 = (drop_seed % 2 + 1) as u16;
        let count_1_to_3 = (drop_seed % 3 + 1) as u16;
        let count_0_to_2 = ((drop_seed >> 2) % 3) as u16;

        let (xp_reward, drops): (u32, Vec<(&str, u16)>) = match entity_type {
            EntityType::Zombie => (5, vec![("rotten_flesh", count_1_to_2)]),
            EntityType::Pig => (2, vec![("porkchop", count_1_to_3)]),
            EntityType::Cow => (2, vec![("beef", count_1_to_3), ("leather", count_0_to_2)]),
            EntityType::Player | EntityType::Item | EntityType::Arrow => (0, Vec::new()),
        };

        // 4. Award XP to killer if within reach
        if let Some(sid) = killer_session_id
            && let Some(session) = self.sessions.get_mut(&sid)
        {
            let dist = mob_pos.map_or(0.0, |p| session.position.distance(p));
            if dist <= 5.0
                && let Some(ecs_ent) = session.ecs_entity
                && let Some(mut xp) = self.ecs_world.get_mut::<Experience>(ecs_ent)
            {
                xp.add_xp(xp_reward);
            }
        }

        // 5. Spawn dropped item entities in world at mob position
        if let Some(pos) = mob_pos {
            let mut rng_pop = self.tick_count.wrapping_add(u64::from(net_id));
            for (item_name, count) in drops {
                if count > 0 {
                    let ident = telos_core::ident::Identifier::new("telos", item_name).ok();
                    if let Some(ident) = ident
                        && let Some(item_id) = self.registries.item_registry().get_by_ident(&ident)
                    {
                        rng_pop = rng_pop
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        let angle = (rng_pop % 360) as f32;
                        let rad = angle.to_radians();
                        let speed = 0.15f32;
                        let vel = glam::Vec3::new(rad.cos() * speed, 0.25, rad.sin() * speed);
                        self.spawn_item_entity(
                            "overworld",
                            pos + DVec3::new(0.0, 0.5, 0.0),
                            vel,
                            ItemStack::new(item_id, count),
                            10,
                        );
                    }
                }
            }
        }

        // 6. Despawn mob
        self.despawn_mob(net_id);
    }

    /// Clears and despawns all currently active mobs.
    pub fn clear_mobs(&mut self) {
        let mob_ids: Vec<u32> = self.tracked_mobs.keys().copied().collect();
        for id in mob_ids {
            self.despawn_mob(id);
        }
    }

    /// Advances physics simulation, proximity stack merging, player pickup,
    /// and despawn lifecycles for all active dropped item entities.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn tick_item_entities(&mut self) {
        let item_entries: Vec<(u32, bevy_ecs::entity::Entity)> =
            self.tracked_items.iter().map(|(&k, &v)| (k, v)).collect();

        if item_entries.is_empty() {
            return;
        }

        let mut to_despawn = Vec::new();

        // 1. Physics step and aging
        for (net_id, entity) in &item_entries {
            let Some((pos, vel, mut item_comp)) =
                self.ecs_world.get_entity(*entity).ok().and_then(|ent| {
                    let p = *ent.get::<Position>()?;
                    let v = *ent.get::<Velocity>()?;
                    let i = *ent.get::<ItemEntity>()?;
                    Some((p, v, i))
                })
            else {
                continue;
            };

            // Increment age and decrement pickup delay
            item_comp.age = item_comp.age.saturating_add(1);
            item_comp.pickup_delay = item_comp.pickup_delay.saturating_sub(1);

            if item_comp.age >= ITEM_DESPAWN_TICKS || item_comp.stack.is_empty() {
                to_despawn.push(*net_id);
                continue;
            }

            // Physics step against world terrain
            let world = self.worlds.default_world();
            let is_solid = |bx: i32, by: i32, bz: i32| -> bool {
                let state = world.get_loaded_block(BlockPos::new(bx, by, bz));
                state.0 != 0 && state.0 != 8 && state.0 != 9
            };

            let mut cur_pos = pos.0;
            let mut cur_vel = vel.0;
            tick_item_physics_step(&mut cur_pos, &mut cur_vel, is_solid);

            // Write back updated components
            if let Ok(mut ent) = self.ecs_world.get_entity_mut(*entity) {
                if let Some(mut p) = ent.get_mut::<Position>() {
                    p.0 = cur_pos;
                }
                if let Some(mut v) = ent.get_mut::<Velocity>() {
                    v.0 = cur_vel;
                }
                if let Some(mut i) = ent.get_mut::<ItemEntity>() {
                    *i = item_comp;
                }
            }
        }

        for id in to_despawn {
            self.despawn_item_entity(id);
        }

        // 2. Proximity stack merging (matching items within 1.5 blocks)
        let active_items: Vec<(u32, bevy_ecs::entity::Entity, DVec3, u32)> = self
            .tracked_items
            .iter()
            .filter_map(|(&net_id, &entity)| {
                let pos = self.ecs_world.get::<Position>(entity)?.0;
                let item = self.ecs_world.get::<ItemEntity>(entity)?.stack.item;
                Some((net_id, entity, pos, item))
            })
            .collect();

        let mut merged_despawns = Vec::new();
        for i in 0..active_items.len() {
            let (net_a, ent_a, pos_a, item_a) = active_items[i];
            if merged_despawns.contains(&net_a) {
                continue;
            }

            for &(net_b, ent_b, pos_b, item_b) in active_items.iter().skip(i + 1) {
                if merged_despawns.contains(&net_b) || item_a != item_b {
                    continue;
                }

                if pos_a.distance_squared(pos_b) <= ITEM_MERGE_RADIUS * ITEM_MERGE_RADIUS {
                    let stacks = {
                        let sa = self.ecs_world.get::<ItemEntity>(ent_a).map(|e| e.stack);
                        let sb = self.ecs_world.get::<ItemEntity>(ent_b).map(|e| e.stack);
                        match (sa, sb) {
                            (Some(a), Some(b)) => Some((a, b)),
                            _ => None,
                        }
                    };
                    if let Some((mut stack_a, mut stack_b)) = stacks
                        && merge_item_stacks(&mut stack_a, &mut stack_b)
                    {
                        if let Some(mut comp_a) = self.ecs_world.get_mut::<ItemEntity>(ent_a) {
                            comp_a.stack = stack_a;
                        }
                        if let Some(mut comp_b) = self.ecs_world.get_mut::<ItemEntity>(ent_b) {
                            comp_b.stack = stack_b;
                        }
                        if stack_b.count == 0 {
                            merged_despawns.push(net_b);
                        }
                    }
                }
            }
        }

        for id in merged_despawns {
            self.despawn_item_entity(id);
        }

        // 3. Player proximity pickup check (survival players <= 1.5 blocks with pickup_delay == 0)
        let mut picked_up_despawns = Vec::new();
        let player_candidates: Vec<(u64, bevy_ecs::entity::Entity, DVec3)> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.phase == ConnectionPhase::Play && s.game_mode != GameMode::Spectator)
            .filter_map(|(&sid, s)| s.ecs_entity.map(|e| (sid, e, s.position)))
            .collect();

        for (&net_id, &entity) in &self.tracked_items {
            let Some((item_pos, item_comp)) =
                self.ecs_world.get_entity(entity).ok().and_then(|ent| {
                    let p = ent.get::<Position>()?.0;
                    let i = *ent.get::<ItemEntity>()?;
                    Some((p, i))
                })
            else {
                continue;
            };

            if item_comp.pickup_delay > 0 || item_comp.stack.is_empty() {
                continue;
            }

            for &(sid, player_ent, player_pos) in &player_candidates {
                if player_pos.distance_squared(item_pos) <= ITEM_PICKUP_RADIUS * ITEM_PICKUP_RADIUS
                {
                    let mut stack_to_give = item_comp.stack;
                    let inserted =
                        if let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(player_ent) {
                            inv.insert_into_storage_or_hotbar(&mut stack_to_give)
                        } else {
                            0
                        };

                    if inserted > 0 {
                        if let Some(mut comp) = self.ecs_world.get_mut::<ItemEntity>(entity) {
                            comp.stack = stack_to_give;
                        }

                        if let Some(inv) = self.ecs_world.get::<Inventory>(player_ent) {
                            let mut slot_vec = Vec::with_capacity(inv.slots.len());
                            for slot in &inv.slots {
                                slot_vec.push(SlotData {
                                    item: slot.item,
                                    count: slot.count,
                                });
                            }
                            let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                                slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                carried: SlotData {
                                    item: inv.carried.item,
                                    count: inv.carried.count,
                                },
                            });
                            if let Some(session) = self.sessions.get_mut(&sid) {
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(bulk_msg));
                            }
                        }

                        let particle_msg = S2cMessage::ParticleEvent(S2cParticleEvent {
                            effect: ParticleEffectKind::Smoke,
                            x: item_pos.x as f32,
                            y: item_pos.y as f32 + 0.2,
                            z: item_pos.z as f32,
                            count: 6,
                            speed: 0.08,
                            block_state_id: 0,
                        });
                        for s in self.sessions.values_mut() {
                            if s.phase == ConnectionPhase::Play {
                                let _ = s
                                    .connection
                                    .send(Lane::Control, Payload::Msg(particle_msg.clone()));
                            }
                        }

                        if stack_to_give.is_empty() {
                            picked_up_despawns.push(net_id);
                            break;
                        }
                    }
                }
            }
        }

        for id in picked_up_despawns {
            self.despawn_item_entity(id);
        }

        // 4. Movement delta broadcast if item moved noticeably
        for (&net_id, &entity) in &self.tracked_items {
            if let Some(pos) = self.ecs_world.get::<Position>(entity) {
                let prev_pos = self.item_positions.get(&net_id).copied().unwrap_or(pos.0);
                if pos.0.distance_squared(prev_pos) > 0.001 {
                    self.item_positions.insert(net_id, pos.0);
                    let move_msg = S2cMessage::EntityMove(S2cEntityMove {
                        net_id,
                        x: pos.0.x,
                        y: pos.0.y,
                        z: pos.0.z,
                        yaw: 0.0,
                        pitch: 0.0,
                        head_yaw: 0.0,
                        on_ground: false,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(move_msg.clone()));
                        }
                    }
                }
            }
        }
    }

    /// Advances physics simulation, entity hit detection, block embedding, player pickup,
    /// and despawn lifecycles for all active projectile arrow entities.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn tick_arrow_entities(&mut self) {
        let arrow_entries: Vec<(u32, bevy_ecs::entity::Entity)> =
            self.tracked_arrows.iter().map(|(&k, &v)| (k, v)).collect();

        if arrow_entries.is_empty() {
            return;
        }

        let mut to_despawn = Vec::new();
        let mut embedded_moves = Vec::new();

        // 1. Collect potential living target candidates (active mobs)
        let mob_targets: Vec<(u32, DVec3, telos_sim::EntityAabb)> = self
            .tracked_mobs
            .iter()
            .filter_map(|(&net_id, &entity)| {
                let pos = self.ecs_world.get::<Position>(entity)?.0;
                let net = self.ecs_world.get::<NetEntity>(entity)?;
                Some((net_id, pos, net.entity_type.default_aabb()))
            })
            .collect();

        // 2. Physics step, aging, and hit detection
        for (net_id, entity) in &arrow_entries {
            let Some((pos, vel, rot, mut arrow_comp)) =
                self.ecs_world.get_entity(*entity).ok().and_then(|ent| {
                    let p = *ent.get::<Position>()?;
                    let v = *ent.get::<Velocity>()?;
                    let r = *ent.get::<Rotation>()?;
                    let a = *ent.get::<ArrowEntity>()?;
                    Some((p, v, r, a))
                })
            else {
                continue;
            };

            arrow_comp.age = arrow_comp.age.saturating_add(1);

            // Despawn checks
            if (!arrow_comp.in_ground && arrow_comp.age >= ARROW_DESPAWN_FLYING_TICKS)
                || (arrow_comp.in_ground && arrow_comp.age >= ARROW_DESPAWN_STUCK_TICKS)
            {
                to_despawn.push(*net_id);
                continue;
            }

            if arrow_comp.in_ground {
                if let Ok(mut ent) = self.ecs_world.get_entity_mut(*entity)
                    && let Some(mut a) = ent.get_mut::<ArrowEntity>()
                {
                    *a = arrow_comp;
                }
                continue;
            }

            // Flying arrow: physics step
            let world = self.worlds.default_world();
            let is_solid = |b: BlockPos| -> bool {
                let state = world.get_loaded_block(b);
                state.0 != 0 && state.0 != 8 && state.0 != 9
            };

            let check_entities =
                |ray_origin: DVec3, ray_dir: Vec3, max_dist: f32| -> Option<(u32, f32)> {
                    let mut closest: Option<(u32, f32)> = None;
                    for (tid, tpos, taabb) in &mob_targets {
                        if Some(*tid) == arrow_comp.shooter_id {
                            continue;
                        }
                        if let Some(dist) =
                            taabb.intersects_ray(*tpos, ray_origin, ray_dir, max_dist)
                            && closest.as_ref().is_none_or(|&(_, d)| dist < d)
                        {
                            closest = Some((*tid, dist));
                        }
                    }
                    closest
                };

            let mut cur_pos = pos.0;
            let mut cur_vel = vel.0.as_dvec3();
            let mut cur_yaw = rot.yaw;
            let mut cur_pitch = rot.pitch;

            let outcome = tick_arrow_physics_step(
                &mut cur_pos,
                &mut cur_vel,
                &mut cur_yaw,
                &mut cur_pitch,
                is_solid,
                check_entities,
            );

            match outcome {
                ArrowStepOutcome::HitEntity { target_id, .. } => {
                    if let Some(&target_entity) = self.tracked_mobs.get(&target_id) {
                        let mut is_dead = false;
                        let target_armor = self
                            .ecs_world
                            .get_mut::<Attributes>(target_entity)
                            .map_or(0.0, |mut a| a.get_value(AttributeKind::Armor) as f32);
                        let target_toughness = self
                            .ecs_world
                            .get_mut::<Attributes>(target_entity)
                            .map_or(0.0, |mut a| {
                                a.get_value(AttributeKind::ArmorToughness) as f32
                            });
                        let target_resistance = self
                            .ecs_world
                            .get::<StatusEffects>(target_entity)
                            .and_then(|eff| eff.amplifier(StatusEffectKind::Resistance))
                            .map_or(0, |amp| amp + 1);

                        let mut query = self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                        if let Ok((mut health, mut tracker)) =
                            query.get_mut(&mut self.ecs_world, target_entity)
                        {
                            apply_mitigated_damage(
                                &mut health,
                                &mut tracker,
                                arrow_comp.damage,
                                DamageType::Attack,
                                target_armor,
                                target_toughness,
                                target_resistance,
                                0,
                            );
                            if !health.is_alive() {
                                is_dead = true;
                            }
                        }

                        self.event_queue.push(GameEvent::EntityDamage {
                            target_net_id: target_id,
                            damage: arrow_comp.damage,
                            attacker_net_id: arrow_comp.shooter_id,
                        });

                        if let Some(mut hurt_time) =
                            self.ecs_world.get_mut::<HurtTime>(target_entity)
                        {
                            hurt_time.0 = 10;
                        }

                        if let Some(mut target_vel) =
                            self.ecs_world.get_mut::<Velocity>(target_entity)
                        {
                            let flight_dir = vel.0.normalize_or_zero();
                            let kb = Vec3::new(flight_dir.x, 0.35, flight_dir.z) * 0.4;
                            target_vel.0 += kb;
                        }

                        if let Some(mut mob) =
                            self.ecs_world.get_mut::<telos_sim::Mob>(target_entity)
                            && mob.kind == telos_sim::MobKind::Passive
                        {
                            mob.ai_state = telos_sim::AiState::Fleeing {
                                away_from: cur_pos,
                                timer: 60,
                            };
                        }

                        let hurt_msg = S2cMessage::EntityStatus(S2cEntityStatus {
                            net_id: target_id,
                            status: 2,
                        });
                        for s in self.sessions.values_mut() {
                            if s.phase == ConnectionPhase::Play {
                                let _ = s
                                    .connection
                                    .send(Lane::Control, Payload::Msg(hurt_msg.clone()));
                            }
                        }

                        if is_dead {
                            let shooter_session = self.sessions.iter().find_map(|(&sid, s)| {
                                if Some(s.entity_id) == arrow_comp.shooter_id {
                                    Some(sid)
                                } else {
                                    None
                                }
                            });
                            self.handle_mob_death(target_id, shooter_session);
                        }
                    }

                    to_despawn.push(*net_id);
                }
                ArrowStepOutcome::HitBlock {
                    hit_block, hit_pos, ..
                } => {
                    arrow_comp.in_ground = true;
                    arrow_comp.stuck_block = Some(hit_block);
                    if let Ok(mut ent) = self.ecs_world.get_entity_mut(*entity) {
                        if let Some(mut p) = ent.get_mut::<Position>() {
                            p.0 = hit_pos;
                        }
                        if let Some(mut v) = ent.get_mut::<Velocity>() {
                            v.0 = Vec3::ZERO;
                        }
                        if let Some(mut a) = ent.get_mut::<ArrowEntity>() {
                            *a = arrow_comp;
                        }
                    }
                    self.arrow_positions.insert(*net_id, hit_pos);
                    embedded_moves.push((*net_id, hit_pos, cur_yaw, cur_pitch));
                }
                ArrowStepOutcome::Flying => {
                    if let Ok(mut ent) = self.ecs_world.get_entity_mut(*entity) {
                        if let Some(mut p) = ent.get_mut::<Position>() {
                            p.0 = cur_pos;
                        }
                        if let Some(mut v) = ent.get_mut::<Velocity>() {
                            v.0 = cur_vel.as_vec3();
                        }
                        if let Some(mut r) = ent.get_mut::<Rotation>() {
                            r.yaw = cur_yaw;
                            r.pitch = cur_pitch;
                        }
                        if let Some(mut a) = ent.get_mut::<ArrowEntity>() {
                            *a = arrow_comp;
                        }
                    }
                }
            }
        }

        // Despawn arrows that hit entities or expired
        for id in to_despawn {
            self.despawn_arrow_entity(id);
        }

        // Broadcast embedded moves (on_ground = true)
        for (net_id, pos, yaw, pitch) in embedded_moves {
            let move_msg = S2cMessage::EntityMove(S2cEntityMove {
                net_id,
                x: pos.x,
                y: pos.y,
                z: pos.z,
                yaw,
                pitch,
                head_yaw: yaw,
                on_ground: true,
            });
            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(move_msg.clone()));
                }
            }
        }

        // 3. Player proximity pickup check for stuck arrows
        let mut picked_up_arrows = Vec::new();
        let player_candidates: Vec<(u64, bevy_ecs::entity::Entity, DVec3)> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.phase == ConnectionPhase::Play && s.game_mode != GameMode::Spectator)
            .filter_map(|(&sid, s)| s.ecs_entity.map(|e| (sid, e, s.position)))
            .collect();

        for (&net_id, &entity) in &self.tracked_arrows {
            let Some((arrow_pos, arrow_comp)) =
                self.ecs_world.get_entity(entity).ok().and_then(|ent| {
                    let p = ent.get::<Position>()?.0;
                    let a = *ent.get::<ArrowEntity>()?;
                    Some((p, a))
                })
            else {
                continue;
            };

            if !arrow_comp.in_ground || !arrow_comp.pickup_allowed {
                continue;
            }

            for (sid, player_ent, player_pos) in &player_candidates {
                if arrow_pos.distance(*player_pos) <= ARROW_PICKUP_RADIUS {
                    let mut arrow_stack = ItemStack::new(ITEM_ARROW, 1);
                    let inserted =
                        if let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(*player_ent) {
                            inv.insert_into_storage_or_hotbar(&mut arrow_stack)
                        } else {
                            0
                        };
                    if inserted > 0 {
                        if let Some(inv) = self.ecs_world.get::<Inventory>(*player_ent)
                            && let Some(session) = self.sessions.get_mut(sid)
                        {
                            for (slot_idx, slot) in inv.slots.iter().enumerate() {
                                if slot.item == ITEM_ARROW {
                                    #[allow(clippy::cast_possible_truncation)]
                                    let _ = session.connection.send(
                                        Lane::Control,
                                        Payload::Msg(S2cMessage::InventorySlot(S2cInventorySlot {
                                            slot: slot_idx as u16,
                                            item: slot.item,
                                            count: slot.count,
                                        })),
                                    );
                                }
                            }
                        }
                        picked_up_arrows.push(net_id);
                        break;
                    }
                }
            }
        }

        for id in picked_up_arrows {
            self.despawn_arrow_entity(id);
        }

        // 4. Movement delta broadcast for flying arrows
        for (&net_id, &entity) in &self.tracked_arrows {
            let Some((pos, rot, arrow_comp)) =
                self.ecs_world.get_entity(entity).ok().and_then(|ent| {
                    let p = ent.get::<Position>()?.0;
                    let r = *ent.get::<Rotation>()?;
                    let a = *ent.get::<ArrowEntity>()?;
                    Some((p, r, a))
                })
            else {
                continue;
            };

            if arrow_comp.in_ground {
                continue;
            }

            let prev_pos = self.arrow_positions.get(&net_id).copied().unwrap_or(pos);
            if pos.distance_squared(prev_pos) > 0.001 {
                self.arrow_positions.insert(net_id, pos);
                let move_msg = S2cMessage::EntityMove(S2cEntityMove {
                    net_id,
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    yaw: rot.yaw,
                    pitch: rot.pitch,
                    head_yaw: rot.yaw,
                    on_ground: false,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(move_msg.clone()));
                    }
                }
            }
        }
    }

    /// Checks whether a chunk is within the horizontal Chebyshev simulation distance
    /// of any active player session currently in the specified world.
    #[must_use]
    pub fn is_chunk_simulated(&self, world_name: &str, chunk: ChunkPos) -> bool {
        for session in self.sessions.values() {
            if session.phase != ConnectionPhase::Play || session.world_name != world_name {
                continue;
            }
            let player_chunk = session.player_chunk();
            let dx = (player_chunk.x() - chunk.x()).abs();
            let dz = (player_chunk.z() - chunk.z()).abs();
            #[allow(clippy::cast_possible_wrap)]
            let sim_dist = session.simulation_distance as i32;
            if dx <= sim_dist && dz <= sim_dist {
                return true;
            }
        }
        false
    }

    /// Spawns natural mobs around players according to light and surface rules.
    pub fn tick_natural_spawner(&mut self) {
        let player_count = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .count();
        if player_count == 0 {
            return;
        }

        // Cap of 20 mobs per connected player
        let mob_cap = player_count * 20;
        if self.tracked_mobs.len() >= mob_cap {
            return;
        }

        let players: Vec<(u32, DVec3)> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| (s.entity_id, s.position))
            .collect();

        let is_night = self.time_of_day > 13_000 && self.time_of_day < 23_000;

        for (idx, &(player_id, player_pos)) in players.iter().enumerate() {
            if self.tracked_mobs.len() >= mob_cap {
                break;
            }

            // Pseudo-random angle and distance around player
            let hash = self
                .tick_count
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(u64::from(player_id))
                .wrapping_add(idx as u64);
            let angle = ((hash & 0xFFFF) as f64 / 65535.0) * std::f64::consts::TAU;
            let dist = 18.0 + (((hash >> 16) & 0xFFFF) as f64 / 65535.0) * 22.0;

            let spawn_x = (player_pos.x + angle.cos() * dist).floor() as i32;
            let spawn_z = (player_pos.z + angle.sin() * dist).floor() as i32;

            let surface_y = self
                .worlds
                .default_world_mut()
                .get_surface_y(spawn_x, spawn_z);
            if !(-500..=1000).contains(&surface_y) {
                continue;
            }

            let check_pos = BlockPos::new(spawn_x, surface_y + 1, spawn_z);
            if !self.is_chunk_simulated("overworld", check_pos.chunk()) {
                continue;
            }
            let (sky_light, block_light) = self.worlds.default_world().get_light(check_pos);

            let spawn_pos = DVec3::new(
                f64::from(spawn_x) + 0.5,
                f64::from(surface_y) + 1.0,
                f64::from(spawn_z) + 0.5,
            );

            // Spawning conditions:
            // Hostile (Zombie): dark caves (sky_light <= 4 && block_light <= 7) or night time (block_light <= 7)
            // Passive (Pig/Cow): daylight (sky_light >= 10 && !is_night)
            if (sky_light <= 4 || is_night) && block_light <= 7 {
                self.spawn_mob(EntityType::Zombie, spawn_pos);
            } else if sky_light >= 10 && !is_night {
                let mob_type = if (hash >> 32) & 1 == 0 {
                    EntityType::Pig
                } else {
                    EntityType::Cow
                };
                self.spawn_mob(mob_type, spawn_pos);
            }
        }
    }

    /// Registers an active monster spawner block at `pos`.
    pub fn register_spawner(&mut self, pos: BlockPos, mob_type: EntityType) {
        self.active_spawners.insert(
            pos,
            SpawnerState {
                mob_type,
                spawn_delay: 100,
            },
        );
    }

    /// Unregisters a monster spawner block at `pos`.
    pub fn unregister_spawner(&mut self, pos: BlockPos) {
        self.active_spawners.remove(&pos);
    }

    /// Ticks active monster spawners, checking player proximity and spawning mobs in dark conditions.
    #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
    pub fn tick_monster_spawners(&mut self) {
        if self.sessions.is_empty() || self.active_spawners.is_empty() {
            return;
        }

        let players: Vec<DVec3> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| s.position)
            .collect();

        if players.is_empty() {
            return;
        }

        let mut to_remove = Vec::new();
        let mut spawns = Vec::new();
        let mut particle_events = Vec::new();

        let default_world = self.worlds.default_world_mut();

        for (&spawner_pos, state) in &mut self.active_spawners {
            let cur_block = default_world.get_block(spawner_pos);
            if !default_world.registry().is_spawner(cur_block) {
                to_remove.push(spawner_pos);
                continue;
            }

            let spawner_center = DVec3::new(
                f64::from(spawner_pos.x()) + 0.5,
                f64::from(spawner_pos.y()) + 0.5,
                f64::from(spawner_pos.z()) + 0.5,
            );

            let is_player_near = players.iter().any(|&p| p.distance(spawner_center) <= 16.0);

            if !is_player_near {
                continue;
            }

            if state.spawn_delay > 0 {
                state.spawn_delay -= 1;
            }

            if state.spawn_delay == 0 {
                state.spawn_delay = 200;

                let hash =
                    self.tick_count.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (spawner_pos.x() as u64);
                let dx = ((hash % 9) as i32) - 4;
                let dz = (((hash >> 8) % 9) as i32) - 4;
                let dy = (((hash >> 16) % 3) as i32) - 1;

                let candidate_pos = BlockPos::new(
                    spawner_pos.x() + dx,
                    spawner_pos.y() + dy,
                    spawner_pos.z() + dz,
                );

                let cand_block = default_world.get_block(candidate_pos);
                let below_block = default_world.get_block(BlockPos::new(
                    candidate_pos.x(),
                    candidate_pos.y() - 1,
                    candidate_pos.z(),
                ));

                let (_sky, block_light) = default_world.get_light(candidate_pos);

                let is_cand_passable = cand_block == BlockStateId::AIR
                    || !default_world
                        .registry()
                        .flags(cand_block)
                        .contains(telos_voxel::state::StateFlags::OPAQUE_CUBE);
                let is_below_solid = below_block != BlockStateId::AIR
                    && default_world
                        .registry()
                        .flags(below_block)
                        .contains(telos_voxel::state::StateFlags::OPAQUE_CUBE);

                if is_cand_passable && is_below_solid && block_light <= 7 {
                    let spawn_pos = DVec3::new(
                        f64::from(candidate_pos.x()) + 0.5,
                        f64::from(candidate_pos.y()),
                        f64::from(candidate_pos.z()) + 0.5,
                    );
                    spawns.push((state.mob_type, spawn_pos));
                }

                particle_events.push(S2cParticleEvent {
                    effect: ParticleEffectKind::Smoke,
                    x: spawner_center.x as f32,
                    y: spawner_center.y as f32,
                    z: spawner_center.z as f32,
                    count: 12,
                    speed: 0.2,
                    block_state_id: 0,
                });
            }
        }

        for pos in to_remove {
            self.active_spawners.remove(&pos);
        }

        for (mob_type, pos) in spawns {
            self.spawn_mob(mob_type, pos);
        }

        for p_msg in particle_events {
            let msg = S2cMessage::ParticleEvent(p_msg);
            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s.connection.send(Lane::Control, Payload::Msg(msg.clone()));
                }
            }
        }
    }

    /// Ticks active furnace block entities, advancing fuel combustion, cook progress, and lit state transitions.
    #[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
    pub fn tick_furnace_entities(&mut self) {
        let mut furnace_targets = Vec::new();
        for (world_name, world) in self.worlds.iter() {
            for &pos in &world.furnace_positions {
                furnace_targets.push((world_name.to_string(), pos));
            }
        }

        if furnace_targets.is_empty() {
            return;
        }

        for (world_name, pos) in furnace_targets {
            let world = self.worlds.get_or_default_mut(&world_name);
            let cur_block = world.get_loaded_block(pos);
            if !world.registry().is_furnace(cur_block) {
                world.furnace_positions.remove(&pos);
                continue;
            }

            let Some(be) = world.get_block_entity_mut(pos) else {
                world.furnace_positions.remove(&pos);
                continue;
            };

            let mut furnace_inv = telos_sim::FurnaceInventory::from_block_entity(be);
            let tick_res = telos_sim::tick_furnace_step(
                &mut furnace_inv,
                &self.smelting_recipes,
                &self.fuel_registry,
            );
            furnace_inv.update_block_entity(be);

            // 1. If lit state changed, toggle block state furnace <-> lit_furnace
            if tick_res.lit_changed {
                let target_ident_path = if tick_res.is_lit {
                    "lit_furnace"
                } else {
                    "furnace"
                };
                let target_ident = telos_core::Identifier::new("telos", target_ident_path).unwrap();
                if let Some(entry) = world.registry().get(&target_ident) {
                    let target_state = entry.default_state();
                    if let Some((_snapshot, version)) = world.set_block(pos, target_state) {
                        let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                            x: pos.x(),
                            y: pos.y(),
                            z: pos.z(),
                            state_id: target_state,
                            version,
                        });
                        for s in self.sessions.values_mut() {
                            if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                                let _ = s
                                    .connection
                                    .send(Lane::Control, Payload::Msg(update_msg.clone()));
                            }
                        }
                    }
                }
            }

            // 2. If contents changed, broadcast slots to viewing players
            if tick_res.contents_changed {
                let mut slot_vec = Vec::with_capacity(furnace_inv.slots.len());
                for slot in &furnace_inv.slots {
                    slot_vec.push(SlotData {
                        item: slot.item,
                        count: slot.count,
                    });
                }
                let title_str = furnace_inv.custom_name.as_deref().unwrap_or("Furnace");
                let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                    window_id: 1,
                    container_kind: 1,
                    title: BoundedString::new(title_str)
                        .unwrap_or_else(|_| BoundedString::new("Furnace").unwrap()),
                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                    x: pos.x(),
                    y: pos.y(),
                    z: pos.z(),
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play
                        && s.world_name == world_name
                        && s.active_container.as_ref().map(|c| c.block_pos) == Some(pos)
                    {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(open_msg.clone()));
                    }
                }
            }

            // 3. If properties changed, broadcast properties to viewing players
            if tick_res.properties_changed {
                let p0 = S2cMessage::ContainerProperty(S2cContainerProperty {
                    window_id: 1,
                    property_id: 0,
                    value: furnace_inv.burn_time_remaining as i16,
                });
                let p1 = S2cMessage::ContainerProperty(S2cContainerProperty {
                    window_id: 1,
                    property_id: 1,
                    value: furnace_inv.total_burn_time as i16,
                });
                let p2 = S2cMessage::ContainerProperty(S2cContainerProperty {
                    window_id: 1,
                    property_id: 2,
                    value: furnace_inv.cook_progress as i16,
                });
                let p3 = S2cMessage::ContainerProperty(S2cContainerProperty {
                    window_id: 1,
                    property_id: 3,
                    value: furnace_inv.cook_duration as i16,
                });

                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play
                        && s.world_name == world_name
                        && s.active_container.as_ref().map(|c| c.block_pos) == Some(pos)
                    {
                        let _ = s.connection.send(Lane::Control, Payload::Msg(p0.clone()));
                        let _ = s.connection.send(Lane::Control, Payload::Msg(p1.clone()));
                        let _ = s.connection.send(Lane::Control, Payload::Msg(p2.clone()));
                        let _ = s.connection.send(Lane::Control, Payload::Msg(p3.clone()));
                    }
                }
            }
        }
    }

    /// Dispatches a server-driven modal form request to a connected player session.
    pub fn send_modal_form(
        &mut self,
        session_id: u64,
        form: &ModalFormData,
        handler: FormHandlerKind,
    ) -> Option<u32> {
        let session = self.sessions.get_mut(&session_id)?;
        if session.phase != ConnectionPhase::Play {
            return None;
        }

        let form_id = self.next_form_id;
        self.next_form_id = self.next_form_id.wrapping_add(1).max(1);

        let req = match S2cModalFormRequest::new(form_id, form) {
            Ok(r) => r,
            Err(err) => {
                tracing::warn!(?err, "Failed to serialize modal form payload");
                return None;
            }
        };

        let _ = session.connection.send(
            Lane::Control,
            Payload::Msg(S2cMessage::ModalFormRequest(req)),
        );

        self.pending_forms.insert(
            (session_id, form_id),
            PendingForm {
                sent_tick: self.tick_count,
                handler,
            },
        );

        Some(form_id)
    }

    /// Dispatches a paginated `/help` command index form to a player session.
    pub fn send_help_form(&mut self, session_id: u64, page: usize) -> Option<u32> {
        let (form, current_page) = build_help_index_form(page);
        self.send_modal_form(
            session_id,
            &form,
            FormHandlerKind::HelpIndex { page: current_page },
        )
    }

    /// Dispatches a command syntax and example details form to a player session.
    pub fn send_help_command_detail(
        &mut self,
        session_id: u64,
        command_name: &str,
        return_page: usize,
    ) -> Option<u32> {
        let (form, default_example) = build_help_detail_form(command_name);
        self.send_modal_form(
            session_id,
            &form,
            FormHandlerKind::HelpCommandDetail {
                command_name: command_name.trim_start_matches('/').to_lowercase(),
                return_page,
                example_command: default_example.map(ToString::to_string),
            },
        )
    }

    /// Accesses the ECS simulation world immutably.
    #[must_use]
    pub fn ecs_world(&self) -> &bevy_ecs::world::World {
        &self.ecs_world
    }

    /// Accesses the ECS simulation world mutably.
    pub fn ecs_world_mut(&mut self) -> &mut bevy_ecs::world::World {
        &mut self.ecs_world
    }

    /// Accesses the server configuration.
    #[must_use]
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Accesses the server's primary default world container immutably.
    #[must_use]
    pub fn world(&self) -> &ServerWorld {
        self.worlds.default_world()
    }

    /// Accesses the server's primary default world container mutably.
    pub fn world_mut(&mut self) -> &mut ServerWorld {
        self.worlds.default_world_mut()
    }

    /// Accesses a specific world by name immutably.
    #[must_use]
    pub fn world_named(&self, name: &str) -> Option<&ServerWorld> {
        self.worlds.get(name)
    }

    /// Accesses a specific world by name mutably.
    pub fn world_named_mut(&mut self, name: &str) -> Option<&mut ServerWorld> {
        self.worlds.get_mut(name)
    }

    /// Accesses the multi-world manager immutably.
    #[must_use]
    pub fn worlds(&self) -> &MultiWorldManager {
        &self.worlds
    }

    /// Accesses the multi-world manager mutably.
    pub fn worlds_mut(&mut self) -> &mut MultiWorldManager {
        &mut self.worlds
    }

    /// Registers an additional dimension / world with the server.
    pub fn add_world(
        &mut self,
        name: impl Into<String>,
        world: ServerWorld,
    ) -> Result<(), WorldError> {
        self.worlds.add_world(name, world)
    }

    /// Unregisters an additional world by name.
    pub fn remove_world(&mut self, name: &str) -> Result<ServerWorld, WorldError> {
        self.worlds.remove_world(name)
    }

    /// Creates a fluent `ServerBuilder` for customizing and launching a server.
    #[must_use]
    pub fn builder() -> ServerBuilder {
        ServerBuilder::new()
    }

    /// Binds the remote QUIC listener to the specified port on `0.0.0.0:<port>`.
    pub fn listen(&mut self, port: u16) -> Result<(), ServerError> {
        let addr = SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED), port);
        self.listen_addr(addr)
    }

    /// Binds the remote QUIC listener to the specified socket address.
    pub fn listen_addr(&mut self, addr: SocketAddr) -> Result<(), ServerError> {
        let listener = telos_net::QuicListener::bind(addr)?;
        tracing::info!(%addr, "Server QUIC listener bound successfully");
        if self.config.lan_broadcast {
            let beacon = telos_net::LanBeacon::new(addr.port(), &self.config.motd);
            self.lan_emitter = telos_net::LanBeaconEmitter::spawn(beacon, addr.port()).ok();
        }

        self.listener = Some(listener);
        Ok(())
    }

    /// Runs the authoritative server tick loop blocking the current thread until shutdown is requested.
    pub fn run_blocking(&mut self) -> Result<(), ServerError> {
        let tick_duration =
            std::time::Duration::from_secs_f64(1.0 / f64::from(self.config.tps.max(1)));

        let cancel = self.shutdown_requested.clone();
        let _ = ctrlc::set_handler(move || {
            tracing::info!("Received interrupt signal (Ctrl+C), requesting server shutdown...");
            cancel.store(true, Ordering::SeqCst);
        });

        tracing::info!(
            tps = self.config.tps,
            worlds = self.worlds.len(),
            "Server main loop started"
        );

        while !self.is_shutdown_requested() {
            let start = std::time::Instant::now();
            self.tick();
            let elapsed = start.elapsed();
            if let Some(remaining) = tick_duration.checked_sub(elapsed) {
                std::thread::sleep(remaining);
            }
        }

        tracing::info!("Server main loop stopped; flushing all worlds...");
        let _ = self.save_and_flush();
        Ok(())
    }

    /// Requests graceful shutdown of the server.
    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if a graceful shutdown has been requested.
    #[must_use]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    /// Transfers a connected player to a different dimension / world.
    pub fn transfer_player_world(
        &mut self,
        session_id: u64,
        target_world: &str,
        target_pos: Option<DVec3>,
    ) -> Result<(), WorldError> {
        if !self.worlds.contains(target_world) {
            return Err(WorldError::NotFound(target_world.to_string()));
        }

        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| WorldError::NotFound(format!("Session {session_id}")))?;

        // 1. Clear old subscriptions and send unload packets
        let (unloads, lod_unloads) = session.clear_subscriptions();
        for pos in unloads {
            let unload_msg = S2cMessage::ChunkUnload(S2cChunkUnload {
                chunk_x: pos.x(),
                chunk_y: pos.y(),
                chunk_z: pos.z(),
            });
            let _ = session
                .connection
                .send(Lane::Chunk { priority: 0 }, Payload::Msg(unload_msg));
        }
        for key in lod_unloads {
            let lod_unload = S2cMessage::LodNodeUnload(S2cLodNodeUnload {
                level: key.level,
                node_x: key.x,
                node_y: key.y,
                node_z: key.z,
            });
            let _ = session
                .connection
                .send(Lane::Chunk { priority: 1 }, Payload::Msg(lod_unload));
        }

        // 2. Set new world name
        session.world_name = target_world.to_string();

        // 3. Determine spawn position in target world
        let dest_pos = if let Some(pos) = target_pos {
            pos
        } else {
            let world = self.worlds.get_mut(target_world).unwrap();
            let surface_y = f64::from(world.get_surface_y(128, 160));
            DVec3::new(128.0, surface_y + 1.0, 160.0)
        };

        // 4. Teleport player
        session.teleport(dest_pos);

        // 5. Trigger immediate subscription recomputation
        let _ = session.recompute_subscriptions();

        tracing::info!(
            session_id,
            target_world,
            ?dest_pos,
            "Transferred player to world"
        );
        Ok(())
    }

    /// Number of active player sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns a reference to a player session by its ID.
    #[must_use]
    pub fn get_session(&self, id: u64) -> Option<&PlayerSession> {
        self.sessions.get(&id)
    }

    /// Returns a mutable reference to a player session by its ID.
    pub fn get_session_mut(&mut self, id: u64) -> Option<&mut PlayerSession> {
        self.sessions.get_mut(&id)
    }

    /// Registers a new incoming connection and returns its unique `session_id`.
    pub fn add_connection(&mut self, conn: Box<dyn Connection<S2cMessage, C2sMessage>>) -> u64 {
        let session_id = self.next_session_id;
        self.next_session_id += 1;

        let entity_id = self.next_entity_id;
        self.next_entity_id += 1;

        let spawn_pos = DVec3::new(128.0, 45.0, 160.0);

        let session =
            PlayerSession::new_with_spawn(session_id, entity_id, conn, &self.config, spawn_pos);
        self.sessions.insert(session_id, session);

        info!(
            session_id,
            entity_id,
            ?spawn_pos,
            "Registered new client session"
        );
        session_id
    }

    /// Current time of day in ticks `[0..24000)`.
    #[must_use]
    pub fn time_of_day(&self) -> u64 {
        self.time_of_day
    }

    /// Sets the current time of day in ticks.
    pub fn set_time_of_day(&mut self, time: u64) {
        self.time_of_day = time % telos_core::DAY_TICKS;
    }

    /// Advances the server simulation by one tick (20 TPS fixed).
    #[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
    pub fn tick(&mut self) {
        self.tick_count += 1;
        self.js_plugins.dispatch_tick(self.tick_count);

        // 0. Accept incoming QUIC network connections from listener
        let mut incoming_conns = Vec::new();
        if let Some(listener) = &mut self.listener {
            while let Some(conn) = listener.try_accept() {
                incoming_conns.push(conn);
            }
        }
        for conn in incoming_conns {
            let session_id = self.add_connection(conn);
            info!(session_id, "Accepted incoming QUIC connection");
        }

        // 1. Process inbound network packets across all sessions

        let mut disconnected = Vec::new();
        let mut player_joined = Vec::new();
        let mut block_actions: Vec<(u64, C2sBlockAction)> = Vec::new();
        let mut inventory_clicks: Vec<(u64, C2sInventoryClick)> = Vec::new();
        let mut container_closes: Vec<(u64, C2sCloseContainer)> = Vec::new();
        let mut player_commands: Vec<(u64, C2sPlayerCommand)> = Vec::new();
        let mut entity_interactions: Vec<(u64, C2sInteractEntity)> = Vec::new();
        let mut chat_messages: Vec<(u64, C2sChatMessage)> = Vec::new();
        let mut command_suggests: Vec<(u64, C2sCommandSuggest)> = Vec::new();
        let mut modal_form_responses: Vec<(u64, C2sModalFormResponse)> = Vec::new();

        for (session_id, session) in &mut self.sessions {
            while let Ok(Some(incoming)) = session.connection.try_recv() {
                let Some(msg) = incoming.into_msg() else {
                    continue;
                };

                match session.phase {
                    ConnectionPhase::Hello => {
                        if let C2sMessage::Hello(hello) = msg {
                            debug!(session_id, protocol = hello.protocol, "Received Hello");
                            let reply = S2cMessage::HelloReply(S2cHelloReply {
                                protocol: 1,
                                features: hello.features,
                                server_id: [0x56; 16],
                            });
                            let _ = session.connection.send(Lane::Control, Payload::Msg(reply));
                            session.phase = ConnectionPhase::Login;
                        } else if let C2sMessage::Disconnect(_) = msg {
                            disconnected.push(*session_id);
                        }
                    }
                    ConnectionPhase::Login => {
                        if let C2sMessage::LoginStart(login) = msg {
                            info!(
                                session_id,
                                username = login.username.as_str(),
                                "Client login"
                            );
                            session.username = login.username.to_string();
                            let success = S2cMessage::LoginSuccess(S2cLoginSuccess {
                                player_uuid: [0x42; 16],
                                username: login.username,
                            });
                            let _ = session
                                .connection
                                .send(Lane::Control, Payload::Msg(success));
                            session.phase = ConnectionPhase::Config;

                            // Transmit block registry table & hash
                            let mut block_entries = Vec::new();
                            for ident in self.registries.block_states().keys() {
                                if let Ok(s) = BoundedString::new(ident.to_string()) {
                                    block_entries.push(s);
                                }
                            }
                            let block_msg = S2cMessage::RegistryData(S2cRegistryData {
                                registry_id: BoundedString::new("telos:block").unwrap(),
                                content_hash: *self.registries.content_hash(),
                                entries: BoundedVec::new(block_entries).unwrap_or_default(),
                            });
                            let _ = session
                                .connection
                                .send(Lane::Control, Payload::Msg(block_msg));

                            // Transmit item registry table & hash
                            let mut item_entries = Vec::new();
                            for (_id, ident, _def) in self.registries.item_registry().iter() {
                                if let Ok(s) = BoundedString::new(ident.to_string()) {
                                    item_entries.push(s);
                                }
                            }
                            let item_msg = S2cMessage::RegistryData(S2cRegistryData {
                                registry_id: BoundedString::new("telos:item").unwrap(),
                                content_hash: *self.registries.content_hash(),
                                entries: BoundedVec::new(item_entries).unwrap_or_default(),
                            });
                            let _ = session
                                .connection
                                .send(Lane::Control, Payload::Msg(item_msg));

                            // Signal configuration complete
                            let _ = session.connection.send(
                                Lane::Control,
                                Payload::Msg(S2cMessage::ConfigDone(S2cConfigDone)),
                            );
                        } else if let C2sMessage::Disconnect(_) = msg {
                            disconnected.push(*session_id);
                        }
                    }
                    ConnectionPhase::Config => {
                        match msg {
                            C2sMessage::ClientSettings(settings) => {
                                session.view_distance = u32::from(settings.view_distance);
                                session.simulation_distance =
                                    u32::from(settings.simulation_distance).clamp(2, 32);
                            }
                            C2sMessage::ConfigAck(_) => {
                                info!(session_id, "Client config acked, entering Play");
                                session.phase = ConnectionPhase::Play;
                                player_joined.push((session.entity_id, session.username.clone()));

                                let join = S2cMessage::JoinGame(S2cJoinGame {
                                    entity_id: session.entity_id,
                                    spawn_x: session.position.x,
                                    spawn_y: session.position.y,
                                    spawn_z: session.position.z,
                                    view_distance: session.view_distance,
                                });
                                let _ = session.connection.send(Lane::Control, Payload::Msg(join));

                                let gm_msg = S2cMessage::GameMode(S2cGameMode {
                                    game_mode: session.game_mode.id(),
                                    flags: session.capabilities.flags(),
                                    fly_speed: session.capabilities.fly_speed,
                                    walk_speed: session.capabilities.walk_speed,
                                    reach_distance: session.capabilities.reach_distance,
                                });
                                let _ =
                                    session.connection.send(Lane::Control, Payload::Msg(gm_msg));

                                let time_msg = S2cMessage::UpdateTime(S2cUpdateTime {
                                    world_age: self.tick_count,
                                    time_of_day: self.time_of_day,
                                });
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(time_msg));

                                let weather_msg = S2cMessage::UpdateWeather(S2cUpdateWeather {
                                    rain_level: self.weather.rain_level,
                                    thunder_level: self.weather.thunder_level,
                                    lightning_flash: self.weather.lightning_flash_ticks,
                                });
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(weather_msg));

                                // Spawn player entity in ECS simulation
                                let mut inv = Inventory::default();
                                inv.slots[0] = ItemStack::new(1, 64); // Stone
                                inv.slots[1] = ItemStack::new(2, 64); // Dirt
                                inv.slots[2] = ItemStack::new(5, 64); // Oak Log
                                inv.slots[3] = ItemStack::new(4, 64); // Cobblestone
                                inv.slots[4] = ItemStack::new(7, 64); // Oak Planks
                                inv.slots[5] = ItemStack::new(19, 64); // Wire (logic_wire)
                                inv.slots[6] = ItemStack::new(21, 64); // Power Block (logic_power_block)
                                inv.slots[7] = ItemStack::new(22, 64); // Lever (logic_lever)
                                inv.slots[8] = ItemStack::new(24, 64); // Lamp (logic_lamp)
                                inv.slots[9] = ItemStack::new(26, 64); // Repeater (logic_repeater)
                                inv.slots[10] = ItemStack::new(28, 64); // Inverter (logic_inverter)
                                inv.slots[11] = ItemStack::new(30, 64); // Diode (logic_diode)
                                inv.update_crafting();

                                let ecs_entity = self
                                    .ecs_world
                                    .spawn((
                                        Health::new(20.0),
                                        CombatTracker::default(),
                                        Hunger::new(20, 5.0),
                                        Experience::default(),
                                        Attributes::player_default(),
                                        StatusEffects::new(),
                                        HurtTime::default(),
                                        inv.clone(),
                                    ))
                                    .id();
                                session.ecs_entity = Some(ecs_entity);

                                let stats_msg = S2cMessage::UpdateStats(S2cUpdateStats {
                                    health: 20.0,
                                    max_health: 20.0,
                                    food: 20,
                                    saturation: 5.0,
                                    xp_level: 0,
                                    xp_progress: 0.0,
                                });
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(stats_msg));

                                let mut slot_vec = Vec::with_capacity(inv.slots.len());
                                for slot in &inv.slots {
                                    slot_vec.push(SlotData {
                                        item: slot.item,
                                        count: slot.count,
                                    });
                                }
                                let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                    carried: SlotData {
                                        item: inv.carried.item,
                                        count: inv.carried.count,
                                    },
                                });
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(bulk_msg));

                                // Send active mobs to joining client
                                for (&net_id, &entity) in &self.tracked_mobs {
                                    if let (Some(net), Some(pos), Some(rot), Some(health)) = (
                                        self.ecs_world.get::<NetEntity>(entity),
                                        self.ecs_world.get::<Position>(entity),
                                        self.ecs_world.get::<Rotation>(entity),
                                        self.ecs_world.get::<Health>(entity),
                                    ) {
                                        let spawn_msg = S2cMessage::SpawnEntity(S2cSpawnEntity {
                                            net_id,
                                            entity_type: net.entity_type.to_u8(),
                                            x: pos.0.x,
                                            y: pos.0.y,
                                            z: pos.0.z,
                                            yaw: rot.yaw,
                                            pitch: rot.pitch,
                                            head_yaw: rot.head_yaw,
                                            health: health.cur,
                                            max_health: health.max,
                                        });
                                        let _ = session
                                            .connection
                                            .send(Lane::Control, Payload::Msg(spawn_msg));
                                    }
                                }

                                // Send active items to joining client
                                for (&net_id, &entity) in &self.tracked_items {
                                    if let (Some(pos), Some(vel), Some(item)) = (
                                        self.ecs_world.get::<Position>(entity),
                                        self.ecs_world.get::<Velocity>(entity),
                                        self.ecs_world.get::<ItemEntity>(entity),
                                    ) {
                                        let spawn_msg = S2cMessage::SpawnItem(S2cSpawnItem {
                                            net_id,
                                            item_id: item.stack.item,
                                            count: item.stack.count,
                                            x: pos.0.x,
                                            y: pos.0.y,
                                            z: pos.0.z,
                                            vel_x: vel.0.x,
                                            vel_y: vel.0.y,
                                            vel_z: vel.0.z,
                                        });
                                        let _ = session
                                            .connection
                                            .send(Lane::Control, Payload::Msg(spawn_msg));
                                    }
                                }

                                // Send active arrows to joining client
                                for (&net_id, &entity) in &self.tracked_arrows {
                                    if let (Some(pos), Some(vel), Some(rot)) = (
                                        self.ecs_world.get::<Position>(entity),
                                        self.ecs_world.get::<Velocity>(entity),
                                        self.ecs_world.get::<Rotation>(entity),
                                    ) {
                                        let spawn_msg = S2cMessage::SpawnArrow(S2cSpawnArrow {
                                            net_id,
                                            x: pos.0.x,
                                            y: pos.0.y,
                                            z: pos.0.z,
                                            vel_x: vel.0.x,
                                            vel_y: vel.0.y,
                                            vel_z: vel.0.z,
                                            yaw: rot.yaw,
                                            pitch: rot.pitch,
                                        });
                                        let _ = session
                                            .connection
                                            .send(Lane::Control, Payload::Msg(spawn_msg));

                                        if let Some(arrow) =
                                            self.ecs_world.get::<ArrowEntity>(entity)
                                            && arrow.in_ground
                                        {
                                            let move_msg = S2cMessage::EntityMove(S2cEntityMove {
                                                net_id,
                                                x: pos.0.x,
                                                y: pos.0.y,
                                                z: pos.0.z,
                                                yaw: rot.yaw,
                                                pitch: rot.pitch,
                                                head_yaw: rot.yaw,
                                                on_ground: true,
                                            });
                                            let _ = session
                                                .connection
                                                .send(Lane::Control, Payload::Msg(move_msg));
                                        }
                                    }
                                }

                                // Send initial movement state to synchronize client prediction
                                let initial_ack =
                                    S2cMessage::PlayerMovementAck(S2cPlayerMovementAck {
                                        client_tick_ack: 0,
                                        #[allow(clippy::cast_possible_truncation)]
                                        server_tick: self.tick_count as u32,
                                        x: session.move_state.pos.x,
                                        y: session.move_state.pos.y,
                                        z: session.move_state.pos.z,
                                        vx: session.move_state.vel.x,
                                        vy: session.move_state.vel.y,
                                        vz: session.move_state.vel.z,
                                        yaw: session.move_state.yaw,
                                        pitch: session.move_state.pitch,
                                        on_ground: session.move_state.on_ground,
                                        flying: session.move_state.flying,
                                        teleport_id: 0,
                                    });
                                let _ = session
                                    .connection
                                    .send(Lane::Control, Payload::Msg(initial_ack));

                                // Force initial chunk subscriptions
                                let _ = session.recompute_subscriptions();
                            }
                            C2sMessage::Disconnect(_) => {
                                disconnected.push(*session_id);
                            }
                            _ => {}
                        }
                    }
                    ConnectionPhase::Play => match msg {
                        C2sMessage::PlayerInput(input) => {
                            for frame in input.frames.iter() {
                                session.queue_input(*frame);
                            }
                        }
                        C2sMessage::TeleportAck(tp) => {
                            if session.awaiting_teleport == Some(tp.teleport_id) {
                                session.awaiting_teleport = None;
                            }
                        }
                        C2sMessage::PlayerPosition(pos) => {
                            let old_y = session.position.y;
                            let was_ground = session.on_ground;
                            session.update_position(
                                DVec3::new(pos.x, pos.y, pos.z),
                                pos.yaw,
                                pos.pitch,
                                pos.on_ground,
                            );
                            if session.capabilities.invincible
                                || session.move_mode == MoveMode::NoClipFly
                                || session.move_state.flying
                            {
                                session.fall_distance = 0.0;
                            } else if pos.on_ground {
                                if !was_ground && session.fall_distance > 3.0 {
                                    let fall_dmg = session.fall_distance - 3.0;
                                    if let Some(entity) = session.ecs_entity {
                                        let mut query =
                                            self.ecs_world
                                                .query::<(&mut Health, &mut CombatTracker)>();
                                        if let Ok((mut health, mut combat)) =
                                            query.get_mut(&mut self.ecs_world, entity)
                                        {
                                            telos_sim::apply_damage(
                                                &mut health,
                                                &mut combat,
                                                fall_dmg,
                                                DamageType::Fall,
                                            );
                                        }
                                    }
                                }
                                session.fall_distance = 0.0;
                            } else if pos.y < old_y {
                                #[allow(clippy::cast_possible_truncation)]
                                {
                                    session.fall_distance += (old_y - pos.y) as f32;
                                }
                            }
                        }
                        C2sMessage::BlockAction(action) => {
                            block_actions.push((*session_id, action));
                        }
                        C2sMessage::InventoryClick(click) => {
                            inventory_clicks.push((*session_id, click));
                        }
                        C2sMessage::CloseContainer(close) => {
                            container_closes.push((*session_id, close));
                        }
                        C2sMessage::PlayerCommand(cmd) => {
                            player_commands.push((*session_id, cmd));
                        }
                        C2sMessage::InteractEntity(interact) => {
                            entity_interactions.push((*session_id, interact));
                        }
                        C2sMessage::ChatMessage(chat) => {
                            chat_messages.push((*session_id, chat));
                        }
                        C2sMessage::CommandSuggest(suggest) => {
                            command_suggests.push((*session_id, suggest));
                        }
                        C2sMessage::ModalFormResponse(resp) => {
                            modal_form_responses.push((*session_id, resp));
                        }
                        C2sMessage::Disconnect(_) => {
                            disconnected.push(*session_id);
                        }
                        _ => {}
                    },
                }
            }
        }

        for (entity_id, username) in player_joined {
            self.event_queue.push(GameEvent::PlayerJoined {
                entity_net_id: entity_id,
                username,
            });
        }

        for id in disconnected {
            if let Some((world_name, chest_pos, container_kind)) = self.close_session_container(id)
                && container_kind == 0
            {
                let remaining_viewers = self.sessions.values().any(|s| {
                    s.phase == ConnectionPhase::Play
                        && s.world_name == world_name
                        && s.active_container.as_ref().map(|c| c.block_pos) == Some(chest_pos)
                });
                if !remaining_viewers {
                    let block_event_msg = S2cMessage::BlockEvent(S2cBlockEvent {
                        x: chest_pos.x(),
                        y: chest_pos.y(),
                        z: chest_pos.z(),
                        action: 1,
                        param: 0,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(block_event_msg.clone()));
                        }
                    }
                }
            }
            if let Some(session) = self.sessions.remove(&id) {
                self.event_queue.push(GameEvent::PlayerLeft {
                    entity_net_id: session.entity_id,
                    username: session.username.clone(),
                });
                if let Some(entity) = session.ecs_entity {
                    self.ecs_world.despawn(entity);
                }
                self.pending_forms.retain(|&(s_id, _), _| s_id != id);
                info!(session_id = id, "Session disconnected");
            }
        }

        // Periodic purge of expired pending forms (older than 60s / 1200 ticks)
        if self.tick_count.is_multiple_of(100) {
            self.pending_forms
                .retain(|_, pending| self.tick_count.saturating_sub(pending.sent_tick) <= 1200);
        }

        // 2. Process player debug and action commands
        for (session_id, cmd) in player_commands {
            let Some((entity, session_pos, session_yaw)) = self
                .sessions
                .get(&session_id)
                .map(|s| (s.ecs_entity, s.position, s.yaw))
            else {
                continue;
            };

            match cmd.command {
                PlayerCommandKind::Damage(amt) => {
                    if let Some(entity) = entity {
                        let mut query = self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                        if let Ok((mut health, mut combat)) =
                            query.get_mut(&mut self.ecs_world, entity)
                        {
                            telos_sim::apply_damage(
                                &mut health,
                                &mut combat,
                                amt,
                                DamageType::Command,
                            );
                        }
                    }
                }
                PlayerCommandKind::Heal(amt) => {
                    if let Some(entity) = entity
                        && let Some(mut health) = self.ecs_world.get_mut::<Health>(entity)
                    {
                        health.heal(amt);
                    }
                }
                PlayerCommandKind::SetFood(food) => {
                    if let Some(entity) = entity
                        && let Some(mut hunger) = self.ecs_world.get_mut::<Hunger>(entity)
                    {
                        hunger.food = food.min(20);
                        #[allow(clippy::cast_precision_loss)]
                        let max_sat = hunger.food as f32;
                        hunger.saturation = hunger.saturation.min(max_sat);
                    }
                }
                PlayerCommandKind::AddXp(pts) => {
                    if let Some(entity) = entity
                        && let Some(mut exp) = self.ecs_world.get_mut::<Experience>(entity)
                    {
                        exp.add_xp(pts);
                    }
                }
                PlayerCommandKind::SetWeather(w) => {
                    let kind = WeatherKind::from_u8(w);
                    self.weather.set_weather(kind, 24_000);
                }
                PlayerCommandKind::TriggerLightning => {
                    self.weather.trigger_lightning();
                }
                PlayerCommandKind::SpawnMob { mob_type, x, y, z } => {
                    let entity_type = EntityType::from_u8(mob_type).unwrap_or(EntityType::Zombie);
                    let spawn_pos = if x.abs() < 0.001 && y.abs() < 0.001 && z.abs() < 0.001 {
                        let rad = session_yaw.to_radians();
                        let fwd = DVec3::new(-f64::from(rad.sin()), 0.0, f64::from(rad.cos()));
                        session_pos + fwd * 3.0
                    } else {
                        DVec3::new(x, y, z)
                    };
                    self.spawn_mob(entity_type, spawn_pos);
                }
                PlayerCommandKind::ClearMobs => {
                    self.clear_mobs();
                }
                PlayerCommandKind::SetGameMode(mode) => {
                    let game_mode = GameMode::from_id(mode).unwrap_or(GameMode::Survival);
                    self.apply_game_mode(session_id, game_mode);
                }
                PlayerCommandKind::ApplyEffect {
                    effect_id,
                    duration_ticks,
                    amplifier,
                } => {
                    if let Some(entity) = entity
                        && let Some(kind) = StatusEffectKind::from_u8(effect_id)
                        && let Some(mut effects) = self.ecs_world.get_mut::<StatusEffects>(entity)
                    {
                        effects.apply(EffectInstance::new(
                            kind,
                            duration_ticks.cast_signed(),
                            amplifier,
                        ));
                    }
                }
                PlayerCommandKind::ClearEffects => {
                    if let Some(entity) = entity
                        && let Some(mut effects) = self.ecs_world.get_mut::<StatusEffects>(entity)
                    {
                        effects.clear();
                    }
                }
                PlayerCommandKind::DrinkPotion { potion_type } => {
                    if let Some(entity) = entity
                        && let Some(potion) = PotionType::from_u8(potion_type)
                        && let Some(effect) = potion.effect()
                        && let Some(mut effects) = self.ecs_world.get_mut::<StatusEffects>(entity)
                    {
                        effects.apply(effect);
                    }
                }
                PlayerCommandKind::DropItem { entire_stack } => {
                    let (selected_slot, world_name, pitch) = match self.sessions.get(&session_id) {
                        Some(s) => (s.selected_slot, s.world_name.clone(), s.pitch),
                        None => (0, "overworld".to_string(), 0.0),
                    };
                    let drop_info = if let Some(entity) = entity
                        && let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(entity)
                    {
                        let slot_idx = selected_slot as usize;
                        if slot_idx < inv.slots.len() && !inv.slots[slot_idx].is_empty() {
                            let slot = &mut inv.slots[slot_idx];
                            let drop_count = if entire_stack { slot.count } else { 1 };
                            let item_id = slot.item;
                            slot.count -= drop_count;
                            slot.normalize();
                            Some((item_id, drop_count, slot.item, slot.count))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some((drop_item, drop_count, remaining_item, remaining_count)) =
                        drop_info
                    {
                        let yaw_rad = session_yaw.to_radians();
                        let pitch_rad = pitch.to_radians();
                        let throw_speed = 0.3f32;
                        let forward_x = -yaw_rad.sin() * pitch_rad.cos();
                        let forward_y = -pitch_rad.sin() + 0.1;
                        let forward_z = -yaw_rad.cos() * pitch_rad.cos();
                        let vel = glam::Vec3::new(
                            forward_x * throw_speed,
                            forward_y * throw_speed + 0.1,
                            forward_z * throw_speed,
                        );

                        let drop_pos = session_pos + DVec3::new(0.0, 1.3, 0.0);
                        self.spawn_item_entity(
                            &world_name,
                            drop_pos,
                            vel,
                            ItemStack::new(drop_item, drop_count),
                            PLAYER_DROP_PICKUP_DELAY,
                        );

                        let slot_update = S2cMessage::InventorySlot(S2cInventorySlot {
                            slot: u16::from(selected_slot),
                            item: remaining_item,
                            count: remaining_count,
                        });
                        if let Some(session) = self.sessions.get_mut(&session_id) {
                            let _ = session
                                .connection
                                .send(Lane::Control, Payload::Msg(slot_update));
                        }
                    }
                }
                PlayerCommandKind::ShootBow { charge_ticks } => {
                    if charge_ticks < BOW_MIN_CHARGE_TICKS {
                        continue;
                    }

                    let (selected_slot, world_name, pitch, game_mode) = match self
                        .sessions
                        .get(&session_id)
                    {
                        Some(s) => (s.selected_slot, s.world_name.clone(), s.pitch, s.game_mode),
                        None => continue,
                    };

                    let is_creative = game_mode == GameMode::Creative;
                    let mut can_shoot = false;
                    let mut arrow_slot_to_consume: Option<usize> = None;

                    if let Some(entity) = entity
                        && let Some(inv) = self.ecs_world.get::<Inventory>(entity)
                    {
                        let slot_idx = selected_slot as usize;
                        if slot_idx < inv.slots.len() && inv.slots[slot_idx].item == ITEM_BOW {
                            if is_creative {
                                can_shoot = true;
                            } else {
                                for (idx, slot) in inv.slots.iter().enumerate() {
                                    if slot.item == ITEM_ARROW && slot.count > 0 {
                                        arrow_slot_to_consume = Some(idx);
                                        can_shoot = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if !can_shoot {
                        continue;
                    }

                    // Consume arrow in survival mode
                    if let Some(slot_idx) = arrow_slot_to_consume {
                        let (rem_item, rem_count) = if let Some(entity) = entity
                            && let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(entity)
                        {
                            let slot = &mut inv.slots[slot_idx];
                            slot.count = slot.count.saturating_sub(1);
                            slot.normalize();
                            (slot.item, slot.count)
                        } else {
                            (0, 0)
                        };

                        #[allow(clippy::cast_possible_truncation)]
                        let slot_update = S2cMessage::InventorySlot(S2cInventorySlot {
                            slot: slot_idx as u16,
                            item: rem_item,
                            count: rem_count,
                        });
                        if let Some(session) = self.sessions.get_mut(&session_id) {
                            let _ = session
                                .connection
                                .send(Lane::Control, Payload::Msg(slot_update));
                        }
                    }

                    // Calculate release velocity from charge ticks
                    #[allow(clippy::cast_precision_loss)]
                    let charge_ratio = ((f32::from(charge_ticks)
                        - f32::from(BOW_MIN_CHARGE_TICKS))
                        / (f32::from(BOW_FULL_CHARGE_TICKS) - f32::from(BOW_MIN_CHARGE_TICKS)))
                    .clamp(0.0, 1.0);

                    let speed_mps = BOW_MIN_RELEASE_SPEED
                        + charge_ratio * (BOW_MAX_RELEASE_SPEED - BOW_MIN_RELEASE_SPEED);
                    let speed_per_tick = speed_mps / 20.0;
                    let damage = (speed_per_tick * 3.0).ceil().max(2.0);

                    let yaw_rad = session_yaw.to_radians();
                    let pitch_rad = pitch.to_radians();
                    let forward_x = -yaw_rad.sin() * pitch_rad.cos();
                    let forward_y = -pitch_rad.sin();
                    let forward_z = -yaw_rad.cos() * pitch_rad.cos();
                    let dir = Vec3::new(forward_x, forward_y, forward_z).normalize_or_zero();
                    let vel = dir * speed_per_tick;

                    let launch_pos = session_pos + DVec3::new(0.0, 1.62, 0.0);
                    let shooter_id = self.sessions.get(&session_id).map(|s| s.entity_id);

                    self.spawn_arrow_entity(
                        &world_name,
                        launch_pos,
                        vel,
                        session_yaw,
                        pitch,
                        shooter_id,
                        damage,
                        !is_creative,
                    );
                }
            }
        }

        // 2b. Process entity interactions (combat & interaction)
        for (session_id, interact) in entity_interactions {
            let Some(session_pos) = self.sessions.get(&session_id).map(|s| s.position) else {
                continue;
            };
            let Some(&target_entity) = self.tracked_mobs.get(&interact.target_net_id) else {
                continue;
            };

            let Some(target_pos) = self.ecs_world.get::<Position>(target_entity).map(|p| p.0)
            else {
                continue;
            };

            // Reach validation: 4.5 blocks max
            let dist = session_pos.distance(target_pos);
            if dist > 4.5 {
                continue;
            }

            if interact.action == 0 {
                // Action 0 = Attack
                let mut is_dead = false;

                // Calculate attacker damage from Attributes
                let attacker_entity = self.sessions.get(&session_id).and_then(|s| s.ecs_entity);
                let mut attack_damage = 4.0f32;
                if let Some(att_ent) = attacker_entity
                    && let Some(mut attrs) = self.ecs_world.get_mut::<Attributes>(att_ent)
                {
                    attack_damage = (attrs.get_value(AttributeKind::AttackDamage) as f32).max(0.5);
                }

                // Query target defense
                let target_armor = self
                    .ecs_world
                    .get_mut::<Attributes>(target_entity)
                    .map_or(0.0, |mut a| a.get_value(AttributeKind::Armor) as f32);
                let target_toughness = self
                    .ecs_world
                    .get_mut::<Attributes>(target_entity)
                    .map_or(0.0, |mut a| {
                        a.get_value(AttributeKind::ArmorToughness) as f32
                    });
                let target_resistance = self
                    .ecs_world
                    .get::<StatusEffects>(target_entity)
                    .and_then(|eff| eff.amplifier(StatusEffectKind::Resistance))
                    .map_or(0, |amp| amp + 1);

                let mut query = self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                if let Ok((mut health, mut tracker)) =
                    query.get_mut(&mut self.ecs_world, target_entity)
                {
                    apply_mitigated_damage(
                        &mut health,
                        &mut tracker,
                        attack_damage,
                        DamageType::Attack,
                        target_armor,
                        target_toughness,
                        target_resistance,
                        0,
                    );
                    if !health.is_alive() {
                        is_dead = true;
                    }
                }

                self.event_queue.push(GameEvent::EntityDamage {
                    target_net_id: interact.target_net_id,
                    damage: attack_damage,
                    attacker_net_id: self.sessions.get(&session_id).map(|s| s.entity_id),
                });

                if let Some(mut hurt_time) = self.ecs_world.get_mut::<HurtTime>(target_entity) {
                    hurt_time.0 = 10;
                }

                // Apply knockback away from player
                if let Some(mut vel) = self.ecs_world.get_mut::<Velocity>(target_entity) {
                    let diff = target_pos - session_pos;
                    let horiz_dist = (diff.x * diff.x + diff.z * diff.z).sqrt().max(0.1);
                    let kb_dir = glam::Vec3::new(
                        (diff.x / horiz_dist) as f32,
                        0.35,
                        (diff.z / horiz_dist) as f32,
                    );
                    vel.0 += kb_dir * 0.4;
                }

                // If passive mob, enter fleeing state
                if let Some(mut mob) = self.ecs_world.get_mut::<telos_sim::Mob>(target_entity)
                    && mob.kind == telos_sim::MobKind::Passive
                {
                    mob.ai_state = telos_sim::AiState::Fleeing {
                        away_from: session_pos,
                        timer: 60,
                    };
                }

                // Broadcast hurt status (status = 2)
                let hurt_msg = S2cMessage::EntityStatus(S2cEntityStatus {
                    net_id: interact.target_net_id,
                    status: 2,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(hurt_msg.clone()));
                    }
                }

                if is_dead {
                    self.handle_mob_death(interact.target_net_id, Some(session_id));
                }
            }
        }

        // 3. Process command tab-completion suggestions
        for (session_id, req) in command_suggests {
            if let Some(session) = self.sessions.get_mut(&session_id) {
                #[allow(clippy::cast_possible_truncation)]
                let suggestions = self
                    .command_dispatcher
                    .suggest(&req.command, req.cursor as usize);
                let mut matches = Vec::new();
                let mut tooltips = Vec::new();
                for cand in suggestions.candidates.into_iter().take(16) {
                    if let Ok(val) = BoundedString::new(cand.value) {
                        matches.push(val);
                        let tip = cand.tooltip.unwrap_or_default();
                        let tip_bounded = BoundedString::new(tip)
                            .unwrap_or_else(|_| BoundedString::new("").unwrap());
                        tooltips.push(tip_bounded);
                    }
                }
                #[allow(clippy::cast_possible_truncation)]
                let resp = S2cMessage::CommandSuggestions(S2cCommandSuggestions {
                    id: req.id,
                    start: suggestions.start as u32,
                    length: suggestions.length as u32,
                    matches: BoundedVec::new(matches).unwrap_or_default(),
                    tooltips: BoundedVec::new(tooltips).unwrap_or_default(),
                });
                let _ = session.connection.send(Lane::Control, Payload::Msg(resp));
            }
        }

        // 3.5. Process modal form responses
        for (session_id, resp) in modal_form_responses {
            let Some(pending) = self.pending_forms.remove(&(session_id, resp.form_id)) else {
                debug!(
                    session_id,
                    form_id = resp.form_id,
                    "Received response for unknown or expired form"
                );
                continue;
            };

            if !resp.has_response {
                debug!(
                    session_id,
                    form_id = resp.form_id,
                    cancel_reason = ?resp.cancel_reason,
                    "Client dismissed modal form"
                );
                continue;
            }

            match pending.handler {
                FormHandlerKind::HelpIndex { page } => {
                    if let Ok(FormResponseData::Action { button_index }) =
                        resp.parse_action_response()
                    {
                        let total_pages = HELP_COMMANDS.len().div_ceil(crate::form::HELP_PAGE_SIZE);
                        let start_idx = page * crate::form::HELP_PAGE_SIZE;
                        let end_idx =
                            (start_idx + crate::form::HELP_PAGE_SIZE).min(HELP_COMMANDS.len());
                        let count_on_page = end_idx - start_idx;

                        let button_idx = button_index as usize;
                        if button_idx < count_on_page {
                            let cmd_name = HELP_COMMANDS[start_idx + button_idx].name;
                            self.send_help_command_detail(session_id, cmd_name, page);
                        } else {
                            let nav_idx = button_idx - count_on_page;
                            let has_next = page + 1 < total_pages;
                            let has_prev = page > 0;

                            if has_next && nav_idx == 0 {
                                self.send_help_form(session_id, page + 1);
                            } else if has_prev
                                && ((has_next && nav_idx == 1) || (!has_next && nav_idx == 0))
                            {
                                self.send_help_form(session_id, page - 1);
                            }
                        }
                    }
                }
                FormHandlerKind::HelpCommandDetail {
                    command_name: _,
                    return_page,
                    example_command,
                } => {
                    if let Ok(FormResponseData::Action { button_index }) =
                        resp.parse_action_response()
                    {
                        match button_index {
                            0 => {
                                self.send_help_form(session_id, return_page);
                            }
                            1 => {
                                if let Some(cmd) = example_command
                                    && let Ok(bounded) = BoundedString::new(cmd)
                                {
                                    chat_messages
                                        .push((session_id, C2sChatMessage { message: bounded }));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                FormHandlerKind::CustomAction { .. } => {}
            }
        }

        // 4. Process chat messages and commands
        for (session_id, chat) in chat_messages {
            let Some(session) = self.sessions.get_mut(&session_id) else {
                continue;
            };

            // Security SKILL §4: Rate limiting on chat messages (burst cap 5, 200ms window)
            if self.tick_count.saturating_sub(session.last_chat_tick) < 4 {
                session.chat_burst_count += 1;
                if session.chat_burst_count > 5 {
                    debug!(session_id, "Rate limit exceeded for chat messages");
                    continue;
                }
            } else {
                session.chat_burst_count = 0;
            }
            session.last_chat_tick = self.tick_count;

            let text = chat.message.trim();
            if text.is_empty() {
                continue;
            }

            if text.starts_with('/') {
                // Command execution
                let session_pos = session.position;
                let session_yaw = session.yaw;
                let session_pitch = session.pitch;
                let username = session.username.clone();

                #[allow(clippy::cast_possible_truncation)]
                let mut cmd_ctx = CommandContext {
                    executor_id: Some(session_id),
                    executor_pos: glam::Vec3::new(
                        session_pos.x as f32,
                        session_pos.y as f32,
                        session_pos.z as f32,
                    ),
                    executor_rot: glam::Vec2::new(session_yaw, session_pitch),
                    executor_name: username,
                    args: hashbrown::HashMap::new(),
                };

                let mut output = self.command_dispatcher.execute(text, &mut cmd_ctx);

                // Apply world side-effects for built-in commands
                if output.success {
                    if text.starts_with("/help") {
                        let sub = text.trim_start_matches("/help").trim();
                        if sub.is_empty() || sub == "1" {
                            self.send_help_form(session_id, 0);
                            output.message = "Opening command reference form...".to_string();
                        } else if let Ok(page_num) = sub.parse::<usize>() {
                            self.send_help_form(session_id, page_num.saturating_sub(1));
                            output.message =
                                format!("Opening command reference page {page_num}...");
                        } else {
                            self.send_help_command_detail(session_id, sub, 0);
                            output.message = format!("Opening /{sub} command details...");
                        }
                    } else if text.starts_with("/time set ") {
                        let sub = text.trim_start_matches("/time set ").trim();
                        let new_time = match sub {
                            "day" => Some(1_000),
                            "noon" => Some(telos_core::NOON_TICKS),
                            "night" => Some(13_000),
                            "midnight" => Some(telos_core::MIDNIGHT_TICKS),
                            ticks_str => ticks_str.parse::<u64>().ok(),
                        };
                        if let Some(t) = new_time {
                            self.time_of_day = t;
                            let time_msg = S2cMessage::UpdateTime(S2cUpdateTime {
                                world_age: self.tick_count,
                                time_of_day: self.time_of_day,
                            });
                            for s in self.sessions.values_mut() {
                                if s.phase == ConnectionPhase::Play {
                                    let _ = s
                                        .connection
                                        .send(Lane::Control, Payload::Msg(time_msg.clone()));
                                }
                            }
                        }
                    } else if text.starts_with("/weather ") {
                        let sub = text.trim_start_matches("/weather ").trim();
                        let kind = match sub {
                            "rain" => Some(WeatherKind::Rain),
                            "thunder" => Some(WeatherKind::Thunder),
                            _ => Some(WeatherKind::Clear),
                        };
                        if let Some(k) = kind {
                            self.weather.set_weather(k, 24_000);
                            let weather_msg = S2cMessage::UpdateWeather(S2cUpdateWeather {
                                rain_level: self.weather.rain_level,
                                thunder_level: self.weather.thunder_level,
                                lightning_flash: self.weather.lightning_flash_ticks,
                            });
                            for s in self.sessions.values_mut() {
                                if s.phase == ConnectionPhase::Play {
                                    let _ = s
                                        .connection
                                        .send(Lane::Control, Payload::Msg(weather_msg.clone()));
                                }
                            }
                        }
                    } else if text.starts_with("/tp ")
                        && let Some(pos_arg) = cmd_ctx.get_vec3("destination")
                    {
                        #[allow(clippy::cast_possible_truncation)]
                        let origin = glam::Vec3::new(
                            session_pos.x as f32,
                            session_pos.y as f32,
                            session_pos.z as f32,
                        );
                        let target =
                            pos_arg.resolve(origin, glam::Vec2::new(session_yaw, session_pitch));
                        if let Some(s) = self.sessions.get_mut(&session_id) {
                            let target_dvec = DVec3::new(
                                f64::from(target.x),
                                f64::from(target.y),
                                f64::from(target.z),
                            );
                            s.teleport(target_dvec);
                        }
                    } else if text.starts_with("/world") {
                        let sub = text.trim_start_matches("/world").trim();
                        if sub.is_empty() || sub == "list" {
                            let names = self.worlds.world_names();
                            let mut msg = format!("Loaded worlds ({}): ", names.len());
                            for (i, name) in names.iter().enumerate() {
                                let count = self
                                    .sessions
                                    .values()
                                    .filter(|s| s.world_name == *name)
                                    .count();
                                if i > 0 {
                                    msg.push_str(", ");
                                }
                                let _ = write!(msg, "{name} ({count} players)");
                            }
                            output.success = true;
                            output.message = msg;
                        } else if let Some(rest) = sub.strip_prefix("tp ") {
                            let parts: Vec<&str> = rest.split_whitespace().collect();
                            if let Some(target_world) = parts.first() {
                                if self.worlds.contains(target_world) {
                                    let target_pos = if parts.len() >= 4 {
                                        if let (Ok(x), Ok(y), Ok(z)) = (
                                            parts[1].parse::<f64>(),
                                            parts[2].parse::<f64>(),
                                            parts[3].parse::<f64>(),
                                        ) {
                                            Some(DVec3::new(x, y, z))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    if let Err(err) = self.transfer_player_world(
                                        session_id,
                                        target_world,
                                        target_pos,
                                    ) {
                                        output.success = false;
                                        output.message = format!("Transfer failed: {err}");
                                    } else {
                                        output.success = true;
                                        output.message =
                                            format!("Transferred to world '{target_world}'");
                                    }
                                } else {
                                    output.success = false;
                                    output.message =
                                        format!("World '{target_world}' does not exist");
                                }
                            }
                        } else {
                            output.success = false;
                            output.message =
                                "Usage: /world list or /world tp <world_name> [x y z]".to_string();
                        }
                    } else if text.starts_with("/effect clear") {
                        if let Some(s) = self.sessions.get(&session_id)
                            && let Some(ent) = s.ecs_entity
                            && let Some(mut effs) = self.ecs_world.get_mut::<StatusEffects>(ent)
                        {
                            effs.clear();
                        }
                    } else if text.starts_with("/effect give") {
                        let parts: Vec<&str> = text.split_whitespace().collect();
                        let (effect_name, seconds, amplifier) = if parts.len() >= 4
                            && (parts[2].starts_with('@') || parts[2] == "self")
                        {
                            let eff = parts[3];
                            let sec = parts
                                .get(4)
                                .and_then(|s| s.parse::<i32>().ok())
                                .unwrap_or(30);
                            let amp = parts.get(5).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
                            (eff, sec, amp)
                        } else if parts.len() >= 3 {
                            let eff = parts[2];
                            let sec = parts
                                .get(3)
                                .and_then(|s| s.parse::<i32>().ok())
                                .unwrap_or(30);
                            let amp = parts.get(4).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
                            (eff, sec, amp)
                        } else {
                            ("unknown", 30, 0)
                        };

                        let kind = match effect_name.to_lowercase().as_str() {
                            "speed" => Some(StatusEffectKind::Speed),
                            "slowness" => Some(StatusEffectKind::Slowness),
                            "strength" => Some(StatusEffectKind::Strength),
                            "weakness" => Some(StatusEffectKind::Weakness),
                            "regeneration" | "regen" => Some(StatusEffectKind::Regeneration),
                            "poison" => Some(StatusEffectKind::Poison),
                            "wither" => Some(StatusEffectKind::Wither),
                            "resistance" => Some(StatusEffectKind::Resistance),
                            "fire_resistance" => Some(StatusEffectKind::FireResistance),
                            "water_breathing" => Some(StatusEffectKind::WaterBreathing),
                            "haste" => Some(StatusEffectKind::Haste),
                            "mining_fatigue" => Some(StatusEffectKind::MiningFatigue),
                            "invisibility" => Some(StatusEffectKind::Invisibility),
                            "jump_boost" => Some(StatusEffectKind::JumpBoost),
                            "instant_health" => Some(StatusEffectKind::InstantHealth),
                            "instant_damage" => Some(StatusEffectKind::InstantDamage),
                            _ => None,
                        };

                        if let Some(k) = kind
                            && let Some(s) = self.sessions.get(&session_id)
                            && let Some(ent) = s.ecs_entity
                            && let Some(mut effs) = self.ecs_world.get_mut::<StatusEffects>(ent)
                        {
                            effs.apply(EffectInstance::new(k, seconds * 20, amplifier));
                        }
                    } else if text.starts_with("/enchant") {
                        let parts: Vec<&str> = text.split_whitespace().collect();
                        let (ench_name, level) = if parts.len() >= 4
                            && (parts[1].starts_with('@') || parts[1] == "self")
                        {
                            let name = parts[2];
                            let lvl = parts.get(3).and_then(|s| s.parse::<u8>().ok()).unwrap_or(1);
                            (name, lvl)
                        } else if parts.len() >= 2 {
                            let name = parts[1];
                            let lvl = parts.get(2).and_then(|s| s.parse::<u8>().ok()).unwrap_or(1);
                            (name, lvl)
                        } else {
                            ("unknown", 1)
                        };

                        let ench_kind = match ench_name.to_lowercase().as_str() {
                            "protection" => Some(EnchantmentKind::Protection),
                            "fire_protection" => Some(EnchantmentKind::FireProtection),
                            "feather_falling" => Some(EnchantmentKind::FeatherFalling),
                            "blast_protection" => Some(EnchantmentKind::BlastProtection),
                            "sharpness" => Some(EnchantmentKind::Sharpness),
                            "knockback" => Some(EnchantmentKind::Knockback),
                            "efficiency" => Some(EnchantmentKind::Efficiency),
                            "unbreaking" => Some(EnchantmentKind::Unbreaking),
                            "mending" => Some(EnchantmentKind::Mending),
                            _ => None,
                        };

                        if let Some(k) = ench_kind
                            && let Some(s) = self.sessions.get(&session_id)
                            && let Some(ent) = s.ecs_entity
                            && let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(ent)
                            && let Some(stack) = inv.get_mut(0)
                            && !stack.is_empty()
                        {
                            stack.enchantments.set_enchantment(k, level);
                        }
                    } else if text.starts_with("/gamemode") {
                        let parts: Vec<&str> = text.split_whitespace().collect();
                        if let Some(&mode_str) = parts.get(1) {
                            if let Some(mode) = GameMode::from_name(mode_str) {
                                let target_session_id = if let Some(&target_name) = parts.get(2) {
                                    if target_name == "@s"
                                        || target_name == "@p"
                                        || target_name == "self"
                                    {
                                        Some(session_id)
                                    } else {
                                        self.sessions.iter().find_map(|(&id, s)| {
                                            if s.username.eq_ignore_ascii_case(target_name) {
                                                Some(id)
                                            } else {
                                                None
                                            }
                                        })
                                    }
                                } else {
                                    Some(session_id)
                                };

                                if let Some(target_id) = target_session_id {
                                    self.apply_game_mode(target_id, mode);
                                    output.success = true;
                                    output.message =
                                        format!("Set game mode to {} Mode", mode.name());
                                } else {
                                    output.success = false;
                                    output.message = format!("Player not found: {}", parts[2]);
                                }
                            } else {
                                output.success = false;
                                output.message = format!(
                                    "Unknown game mode '{mode_str}'. Valid modes: survival, creative, adventure, spectator (0-3)"
                                );
                            }
                        } else {
                            output.success = false;
                            output.message =
                                "Usage: /gamemode <survival|creative|adventure|spectator|0-3> [player]".to_string();
                        }
                    }
                }

                if !output.success {
                    let cmd_str = text.trim_start_matches('/');
                    let mut parts = cmd_str.splitn(2, ' ');
                    let cmd_name = parts.next().unwrap_or("");
                    let cmd_args = parts.next().unwrap_or("");
                    if self.mod_manager.dispatch_command(cmd_name, cmd_args) {
                        output.success = true;
                        output.message = format!("Command '{cmd_name}' executed by mod");
                    }
                }

                // Send feedback back to executor
                if let Some(s) = self.sessions.get_mut(&session_id)
                    && let Ok(msg_bounded) = BoundedString::new(output.message)
                {
                    let reply = S2cMessage::ChatMessage(S2cChatMessage {
                        sender: BoundedString::new("Server").unwrap(),
                        message: msg_bounded,
                        timestamp: self.tick_count,
                    });
                    let _ = s.connection.send(Lane::Control, Payload::Msg(reply));
                }
            } else {
                let username = session.username.clone();
                let Some(processed_text) = self
                    .js_plugins
                    .dispatch_player_chat(&username, text.to_string())
                else {
                    continue; // Suppressed by JS plugin
                };

                // Broadcast standard chat message: <username> message
                let sender_bounded = BoundedString::new(username)
                    .unwrap_or_else(|_| BoundedString::new("Player").unwrap());
                if let Ok(msg_bounded) = BoundedString::new(processed_text) {
                    let chat_msg = S2cMessage::ChatMessage(S2cChatMessage {
                        sender: sender_bounded,
                        message: msg_bounded,
                        timestamp: self.tick_count,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(chat_msg.clone()));
                        }
                    }
                }
            }
        }

        // 3. Process inventory and container clicks
        for (session_id, click) in inventory_clicks {
            let (session_world_name, active_container, entity) =
                match self.sessions.get(&session_id) {
                    Some(s) => (s.world_name.clone(), s.active_container, s.ecs_entity),
                    None => continue,
                };
            let Some(entity) = entity else {
                continue;
            };
            let Some(mut inv) = self.ecs_world.get_mut::<Inventory>(entity) else {
                continue;
            };

            let button = match click.button {
                0 => telos_sim::ClickButton::Left,
                _ => telos_sim::ClickButton::Right,
            };
            let mode = match click.mode {
                0 => telos_sim::ClickMode::Pickup,
                1 => telos_sim::ClickMode::QuickMove,
                2 => telos_sim::ClickMode::SwapHotbar,
                _ => telos_sim::ClickMode::Drop,
            };

            let slot_idx = click.slot as usize;
            if let Some(active_cont) = active_container {
                if active_cont.container_kind == 2 {
                    if let Some(s) = self.sessions.get_mut(&session_id)
                        && telos_sim::crafting_table_container_click(
                            &mut s.active_crafting_table,
                            &mut inv,
                            slot_idx,
                            button,
                            mode,
                            &self.recipe_registry,
                        )
                        .is_ok()
                    {
                        let mut slot_vec = Vec::with_capacity(inv.slots.len());
                        for slot in &inv.slots {
                            slot_vec.push(SlotData {
                                item: slot.item,
                                count: slot.count,
                            });
                        }
                        let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                            slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                            carried: SlotData {
                                item: inv.carried.item,
                                count: inv.carried.count,
                            },
                        });
                        let _ = s.connection.send(Lane::Control, Payload::Msg(bulk_msg));

                        let mut container_slot_vec = Vec::with_capacity(10);
                        container_slot_vec.push(SlotData {
                            item: s.active_crafting_table.result.item,
                            count: s.active_crafting_table.result.count,
                        });
                        for slot in &s.active_crafting_table.grid {
                            container_slot_vec.push(SlotData {
                                item: slot.item,
                                count: slot.count,
                            });
                        }
                        let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                            window_id: active_cont.window_id,
                            container_kind: 2,
                            title: BoundedString::new("Crafting")
                                .unwrap_or_else(|_| BoundedString::new("Crafting").unwrap()),
                            slots: BoundedVec::new(container_slot_vec).expect("slots <= 64"),
                            x: active_cont.block_pos.x(),
                            y: active_cont.block_pos.y(),
                            z: active_cont.block_pos.z(),
                        });
                        let _ = s.connection.send(Lane::Control, Payload::Msg(open_msg));
                    }
                } else {
                    let world = self.worlds.get_or_default_mut(&session_world_name);
                    if let Some(be) = world.get_block_entity_mut(active_cont.block_pos) {
                        if active_cont.container_kind == 1 {
                            let mut furnace_inv =
                                telos_sim::FurnaceInventory::from_block_entity(be);
                            if telos_sim::furnace_container_click(
                                &mut furnace_inv,
                                &mut inv,
                                slot_idx,
                                button,
                                mode,
                                &self.fuel_registry,
                                &self.smelting_recipes,
                            )
                            .is_ok()
                            {
                                *be = furnace_inv.to_block_entity();

                                let mut slot_vec = Vec::with_capacity(inv.slots.len());
                                for slot in &inv.slots {
                                    slot_vec.push(SlotData {
                                        item: slot.item,
                                        count: slot.count,
                                    });
                                }
                                let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                    carried: SlotData {
                                        item: inv.carried.item,
                                        count: inv.carried.count,
                                    },
                                });
                                if let Some(s) = self.sessions.get_mut(&session_id) {
                                    let _ =
                                        s.connection.send(Lane::Control, Payload::Msg(bulk_msg));
                                }

                                // Broadcast updated container slots to all sessions viewing this container
                                let mut container_slot_vec =
                                    Vec::with_capacity(furnace_inv.slots.len());
                                for slot in &furnace_inv.slots {
                                    container_slot_vec.push(SlotData {
                                        item: slot.item,
                                        count: slot.count,
                                    });
                                }
                                let title_str =
                                    furnace_inv.custom_name.as_deref().unwrap_or("Furnace");
                                let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                                    window_id: active_cont.window_id,
                                    container_kind: 1,
                                    title: BoundedString::new(title_str)
                                        .unwrap_or_else(|_| BoundedString::new("Furnace").unwrap()),
                                    slots: BoundedVec::new(container_slot_vec)
                                        .expect("slots <= 64"),
                                    x: active_cont.block_pos.x(),
                                    y: active_cont.block_pos.y(),
                                    z: active_cont.block_pos.z(),
                                });
                                for s in self.sessions.values_mut() {
                                    if s.phase == ConnectionPhase::Play
                                        && s.world_name == session_world_name
                                        && s.active_container.as_ref().map(|c| c.block_pos)
                                            == Some(active_cont.block_pos)
                                    {
                                        let _ = s
                                            .connection
                                            .send(Lane::Control, Payload::Msg(open_msg.clone()));
                                    }
                                }
                            }
                        } else {
                            let mut chest_inv = telos_sim::ChestInventory::from_block_entity(be);
                            if telos_sim::container_click(
                                &mut chest_inv,
                                &mut inv,
                                slot_idx,
                                button,
                                mode,
                            )
                            .is_ok()
                            {
                                *be = chest_inv.to_block_entity();

                                let mut slot_vec = Vec::with_capacity(inv.slots.len());
                                for slot in &inv.slots {
                                    slot_vec.push(SlotData {
                                        item: slot.item,
                                        count: slot.count,
                                    });
                                }
                                let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                    carried: SlotData {
                                        item: inv.carried.item,
                                        count: inv.carried.count,
                                    },
                                });
                                if let Some(s) = self.sessions.get_mut(&session_id) {
                                    let _ =
                                        s.connection.send(Lane::Control, Payload::Msg(bulk_msg));
                                }

                                // Broadcast updated container slots to all sessions viewing this container
                                let mut container_slot_vec =
                                    Vec::with_capacity(chest_inv.slots.len());
                                for slot in &chest_inv.slots {
                                    container_slot_vec.push(SlotData {
                                        item: slot.item,
                                        count: slot.count,
                                    });
                                }
                                let title_str = chest_inv.custom_name.as_deref().unwrap_or("Chest");
                                let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                                    window_id: active_cont.window_id,
                                    container_kind: 0,
                                    title: BoundedString::new(title_str)
                                        .unwrap_or_else(|_| BoundedString::new("Chest").unwrap()),
                                    slots: BoundedVec::new(container_slot_vec)
                                        .expect("slots <= 64"),
                                    x: active_cont.block_pos.x(),
                                    y: active_cont.block_pos.y(),
                                    z: active_cont.block_pos.z(),
                                });
                                for s in self.sessions.values_mut() {
                                    if s.phase == ConnectionPhase::Play
                                        && s.world_name == session_world_name
                                        && s.active_container.as_ref().map(|c| c.block_pos)
                                            == Some(active_cont.block_pos)
                                    {
                                        let _ = s
                                            .connection
                                            .send(Lane::Control, Payload::Msg(open_msg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            } else if telos_sim::inventory_click(&mut inv, slot_idx, button, mode).is_ok() {
                let mut slot_vec = Vec::with_capacity(inv.slots.len());
                for slot in &inv.slots {
                    slot_vec.push(SlotData {
                        item: slot.item,
                        count: slot.count,
                    });
                }
                let bulk_msg = S2cMessage::InventoryBulk(S2cInventoryBulk {
                    slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                    carried: SlotData {
                        item: inv.carried.item,
                        count: inv.carried.count,
                    },
                });
                if let Some(s) = self.sessions.get_mut(&session_id) {
                    let _ = s.connection.send(Lane::Control, Payload::Msg(bulk_msg));
                }
            }
        }

        // 3b. Process container close requests
        for (session_id, close) in container_closes {
            let matches_window = self
                .sessions
                .get(&session_id)
                .and_then(|s| s.active_container)
                .is_some_and(|c| c.window_id == close.window_id);
            if matches_window
                && let Some((world_name, chest_pos, container_kind)) =
                    self.close_session_container(session_id)
                && container_kind == 0
            {
                let remaining_viewers = self.sessions.values().any(|s| {
                    s.phase == ConnectionPhase::Play
                        && s.world_name == world_name
                        && s.active_container.as_ref().map(|c| c.block_pos) == Some(chest_pos)
                });
                if !remaining_viewers {
                    let block_event_msg = S2cMessage::BlockEvent(S2cBlockEvent {
                        x: chest_pos.x(),
                        y: chest_pos.y(),
                        z: chest_pos.z(),
                        action: 1,
                        param: 0,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(block_event_msg.clone()));
                        }
                    }
                }
            }
        }

        // 4. Process block actions (authoritative validation & simulation)
        let mut dropped_items_to_spawn = Vec::new();
        let mut containers_to_close = Vec::new();
        for (session_id, action) in block_actions {
            let (
                session_pos,
                session_entity_id,
                _session_move_mode,
                session_world_name,
                session_can_build,
                session_game_mode,
            ) = {
                let Some(session) = self.sessions.get(&session_id) else {
                    continue;
                };
                (
                    session.position,
                    session.entity_id,
                    session.move_mode,
                    session.world_name.clone(),
                    session.capabilities.can_build,
                    session.game_mode,
                )
            };

            let (target_pos, new_state) = match action.action {
                BlockActionKind::Break | BlockActionKind::Interact => (
                    BlockPos::new(action.x, action.y, action.z),
                    BlockStateId::AIR,
                ),
                BlockActionKind::Place { state_id, hit_face } => {
                    let clicked_pos = BlockPos::new(action.x, action.y, action.z);
                    let face = match hit_face {
                        0 => Face::Down,
                        1 => Face::Up,
                        2 => Face::North,
                        3 => Face::South,
                        4 => Face::West,
                        _ => Face::East,
                    };
                    let norm = face.normal_ivec();
                    (
                        BlockPos::new(
                            clicked_pos.x() + norm.x,
                            clicked_pos.y() + norm.y,
                            clicked_pos.z() + norm.z,
                        ),
                        state_id,
                    )
                }
            };

            // Reach validation: squared distance from player pos to block center <= reach^2
            let block_center = DVec3::new(
                f64::from(target_pos.x()) + 0.5,
                f64::from(target_pos.y()) + 0.5,
                f64::from(target_pos.z()) + 0.5,
            );
            let dist_sq = (session_pos - block_center).length_squared();
            let max_reach = 6.0; // 5.0 blocks + 1.0 tolerance for latency
            let is_in_reach = dist_sq <= max_reach * max_reach;
            let is_in_bounds = target_pos.y() >= -1024 && target_pos.y() < 2048;

            let world = self.worlds.get_or_default_mut(&session_world_name);
            let old_state = world.get_block(target_pos);

            if is_in_reach && is_in_bounds {
                if !session_can_build
                    && matches!(
                        action.action,
                        BlockActionKind::Break | BlockActionKind::Place { .. }
                    )
                {
                    // Block editing forbidden by capabilities (Adventure / Spectator)
                    let rollback = S2cMessage::BlockUpdate(S2cBlockUpdate {
                        x: target_pos.x(),
                        y: target_pos.y(),
                        z: target_pos.z(),
                        state_id: old_state,
                        version: 0,
                    });
                    if let Some(s) = self.sessions.get_mut(&session_id) {
                        let _ = s.connection.send(Lane::Control, Payload::Msg(rollback));
                    }
                    continue;
                }

                if matches!(action.action, BlockActionKind::Break)
                    && !self.js_plugins.dispatch_block_break(
                        session_entity_id,
                        old_state.0,
                        target_pos.x(),
                        target_pos.y(),
                        target_pos.z(),
                    )
                {
                    // Block break rejected by JavaScript server plugin
                    let rollback = S2cMessage::BlockUpdate(S2cBlockUpdate {
                        x: target_pos.x(),
                        y: target_pos.y(),
                        z: target_pos.z(),
                        state_id: old_state,
                        version: 0,
                    });
                    if let Some(s) = self.sessions.get_mut(&session_id) {
                        let _ = s.connection.send(Lane::Control, Payload::Msg(rollback));
                    }
                    continue;
                }

                if matches!(action.action, BlockActionKind::Interact) {
                    if let Some(_new_lever_state) = world.logic_engine.toggle_lever(target_pos) {
                        let cur_block = world.get_block(target_pos);
                        if let Some(comp) = world.logic_engine.get_component(target_pos)
                            && let Some(target_state) =
                                world.resolve_logic_block_state(cur_block, &comp)
                            && let Some((_snapshot, version)) =
                                world.set_block(target_pos, target_state)
                        {
                            let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                                x: target_pos.x(),
                                y: target_pos.y(),
                                z: target_pos.z(),
                                state_id: target_state,
                                version,
                            });
                            for s in self.sessions.values_mut() {
                                if s.phase == ConnectionPhase::Play
                                    && s.world_name == session_world_name
                                {
                                    let _ = s
                                        .connection
                                        .send(Lane::Control, Payload::Msg(update_msg.clone()));
                                }
                            }
                        }
                    } else if world.registry().is_chest(old_state) {
                        if world.get_block_entity(target_pos).is_none() {
                            world.set_block_entity(
                                target_pos,
                                telos_voxel::block_entity::BlockEntityData::new_chest(),
                            );
                        }
                        if let Some(chest_be) = world.get_block_entity(target_pos) {
                            let chest_inv = telos_sim::ChestInventory::from_block_entity(chest_be);
                            let mut slot_vec = Vec::with_capacity(chest_inv.slots.len());
                            for slot in &chest_inv.slots {
                                slot_vec.push(SlotData {
                                    item: slot.item,
                                    count: slot.count,
                                });
                            }
                            let title_str = chest_inv.custom_name.as_deref().unwrap_or("Chest");
                            let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                                window_id: 1,
                                container_kind: 0,
                                title: BoundedString::new(title_str)
                                    .unwrap_or_else(|_| BoundedString::new("Chest").unwrap()),
                                slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                x: target_pos.x(),
                                y: target_pos.y(),
                                z: target_pos.z(),
                            });
                            if let Some(s) = self.sessions.get_mut(&session_id) {
                                s.active_container = Some(ActiveContainerSession {
                                    window_id: 1,
                                    block_pos: target_pos,
                                    container_kind: 0,
                                });
                                let _ = s.connection.send(Lane::Control, Payload::Msg(open_msg));
                            }

                            // Broadcast chest open block event (action 1, param 1)
                            let block_event_msg = S2cMessage::BlockEvent(S2cBlockEvent {
                                x: target_pos.x(),
                                y: target_pos.y(),
                                z: target_pos.z(),
                                action: 1,
                                param: 1,
                            });
                            for s in self.sessions.values_mut() {
                                if s.phase == ConnectionPhase::Play
                                    && s.world_name == session_world_name
                                {
                                    let _ = s
                                        .connection
                                        .send(Lane::Control, Payload::Msg(block_event_msg.clone()));
                                }
                            }
                        }
                    } else if world.registry().is_furnace(old_state) {
                        if world.get_block_entity(target_pos).is_none() {
                            world.set_block_entity(
                                target_pos,
                                telos_voxel::block_entity::BlockEntityData::new_furnace(),
                            );
                        }
                        if let Some(furnace_be) = world.get_block_entity(target_pos) {
                            let furnace_inv =
                                telos_sim::FurnaceInventory::from_block_entity(furnace_be);
                            let mut slot_vec = Vec::with_capacity(furnace_inv.slots.len());
                            for slot in &furnace_inv.slots {
                                slot_vec.push(SlotData {
                                    item: slot.item,
                                    count: slot.count,
                                });
                            }
                            let title_str = furnace_inv.custom_name.as_deref().unwrap_or("Furnace");
                            let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                                window_id: 1,
                                container_kind: 1,
                                title: BoundedString::new(title_str)
                                    .unwrap_or_else(|_| BoundedString::new("Furnace").unwrap()),
                                slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                                x: target_pos.x(),
                                y: target_pos.y(),
                                z: target_pos.z(),
                            });
                            if let Some(s) = self.sessions.get_mut(&session_id) {
                                s.active_container = Some(ActiveContainerSession {
                                    window_id: 1,
                                    block_pos: target_pos,
                                    container_kind: 1,
                                });
                                let _ = s.connection.send(Lane::Control, Payload::Msg(open_msg));

                                // Transmit initial combustion & cooking properties
                                let p0 = S2cMessage::ContainerProperty(S2cContainerProperty {
                                    window_id: 1,
                                    property_id: 0,
                                    value: furnace_inv.burn_time_remaining as i16,
                                });
                                let p1 = S2cMessage::ContainerProperty(S2cContainerProperty {
                                    window_id: 1,
                                    property_id: 1,
                                    value: furnace_inv.total_burn_time as i16,
                                });
                                let p2 = S2cMessage::ContainerProperty(S2cContainerProperty {
                                    window_id: 1,
                                    property_id: 2,
                                    value: furnace_inv.cook_progress as i16,
                                });
                                let p3 = S2cMessage::ContainerProperty(S2cContainerProperty {
                                    window_id: 1,
                                    property_id: 3,
                                    value: furnace_inv.cook_duration as i16,
                                });
                                let _ = s.connection.send(Lane::Control, Payload::Msg(p0));
                                let _ = s.connection.send(Lane::Control, Payload::Msg(p1));
                                let _ = s.connection.send(Lane::Control, Payload::Msg(p2));
                                let _ = s.connection.send(Lane::Control, Payload::Msg(p3));
                            }
                        }
                    } else if world.registry().is_crafting_table(old_state) {
                        let mut slot_vec = Vec::with_capacity(10);
                        for _ in 0..10 {
                            slot_vec.push(SlotData { item: 0, count: 0 });
                        }
                        let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
                            window_id: 1,
                            container_kind: 2,
                            title: BoundedString::new("Crafting")
                                .unwrap_or_else(|_| BoundedString::new("Crafting").unwrap()),
                            slots: BoundedVec::new(slot_vec).expect("slots <= 64"),
                            x: target_pos.x(),
                            y: target_pos.y(),
                            z: target_pos.z(),
                        });
                        if let Some(s) = self.sessions.get_mut(&session_id) {
                            s.active_crafting_table = telos_sim::CraftingTableInventory::new();
                            s.active_container = Some(ActiveContainerSession {
                                window_id: 1,
                                block_pos: target_pos,
                                container_kind: 2,
                            });
                            let _ = s.connection.send(Lane::Control, Payload::Msg(open_msg));
                        }
                    }
                } else {
                    let is_chest_or_furnace = world.registry().is_chest(old_state)
                        || world.registry().is_furnace(old_state);
                    let is_crafting_table = world.registry().is_crafting_table(old_state);
                    let is_container_block = is_chest_or_furnace || is_crafting_table;

                    let container_contents_to_drop =
                        if matches!(action.action, BlockActionKind::Break) && is_chest_or_furnace {
                            world.get_block_entity(target_pos).cloned()
                        } else {
                            None
                        };

                    if let Some((_snapshot, version)) = world.set_block(target_pos, new_state) {
                        match action.action {
                            BlockActionKind::Break => {
                                self.event_queue.push(GameEvent::BlockBroken {
                                    pos: target_pos,
                                    old_state,
                                    actor_net_id: Some(u64::from(session_entity_id)),
                                });

                                if let Some(be) = container_contents_to_drop {
                                    for slot in be.items() {
                                        if !slot.is_empty() {
                                            let spawn_pos = DVec3::new(
                                                f64::from(target_pos.x()) + 0.5,
                                                f64::from(target_pos.y()) + 0.5,
                                                f64::from(target_pos.z()) + 0.5,
                                            );
                                            let vel = glam::Vec3::new(0.0, 0.1, 0.0);
                                            dropped_items_to_spawn.push((
                                                session_world_name.clone(),
                                                spawn_pos,
                                                vel,
                                                ItemStack::new(slot.item, slot.count),
                                            ));
                                        }
                                    }
                                }

                                if is_container_block {
                                    let viewers: Vec<u64> = self
                                        .sessions
                                        .iter()
                                        .filter(|(_, s)| {
                                            s.phase == ConnectionPhase::Play
                                                && s.world_name == session_world_name
                                                && s.active_container.as_ref().map(|c| c.block_pos)
                                                    == Some(target_pos)
                                        })
                                        .map(|(&id, _)| id)
                                        .collect();
                                    containers_to_close.extend(viewers);
                                }

                                if session_game_mode == GameMode::Survival
                                    && let Some(drop_stack) = block_to_drop_item(old_state.0)
                                {
                                    let spawn_pos = DVec3::new(
                                        f64::from(target_pos.x()) + 0.5,
                                        f64::from(target_pos.y()) + 0.25,
                                        f64::from(target_pos.z()) + 0.5,
                                    );
                                    let vel = glam::Vec3::new(0.0, 0.1, 0.0);
                                    dropped_items_to_spawn.push((
                                        session_world_name.clone(),
                                        spawn_pos,
                                        vel,
                                        drop_stack,
                                    ));
                                }
                            }
                            BlockActionKind::Place { .. } => {
                                self.event_queue.push(GameEvent::BlockPlaced {
                                    pos: target_pos,
                                    new_state,
                                    actor_net_id: Some(u64::from(session_entity_id)),
                                });
                            }
                            BlockActionKind::Interact => {}
                        }

                        if world.registry().is_spawner(old_state) {
                            self.active_spawners.remove(&target_pos);
                        }
                        if world.registry().is_spawner(new_state) {
                            self.active_spawners.insert(
                                target_pos,
                                SpawnerState {
                                    mob_type: EntityType::Zombie,
                                    spawn_delay: 100,
                                },
                            );
                        }

                        let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                            x: target_pos.x(),
                            y: target_pos.y(),
                            z: target_pos.z(),
                            state_id: new_state,
                            version,
                        });

                        let particle_msg = match action.action {
                            BlockActionKind::Break => {
                                Some(S2cMessage::ParticleEvent(S2cParticleEvent {
                                    effect: ParticleEffectKind::BlockBreak,
                                    x: target_pos.x() as f32 + 0.5,
                                    y: target_pos.y() as f32 + 0.5,
                                    z: target_pos.z() as f32 + 0.5,
                                    count: 24,
                                    speed: 1.0,
                                    block_state_id: old_state.0,
                                }))
                            }
                            BlockActionKind::Place { .. } => {
                                Some(S2cMessage::ParticleEvent(S2cParticleEvent {
                                    effect: ParticleEffectKind::BlockPlace,
                                    x: target_pos.x() as f32 + 0.5,
                                    y: target_pos.y() as f32 + 0.5,
                                    z: target_pos.z() as f32 + 0.5,
                                    count: 10,
                                    speed: 0.5,
                                    block_state_id: new_state.0,
                                }))
                            }
                            BlockActionKind::Interact => None,
                        };

                        // Broadcast block update and particle effects to players in the same world in Play phase
                        for s in self.sessions.values_mut() {
                            if s.phase == ConnectionPhase::Play
                                && s.world_name == session_world_name
                            {
                                let _ = s
                                    .connection
                                    .send(Lane::Control, Payload::Msg(update_msg.clone()));

                                // Broadcast visual particle bursts to other players (local player predicts immediately)
                                if s.session_id != session_id
                                    && let Some(ref p_msg) = particle_msg
                                {
                                    let _ = s
                                        .connection
                                        .send(Lane::Control, Payload::Msg(p_msg.clone()));
                                }
                            }
                        }
                    }
                }
            } else {
                // Out of reach or invalid: send true block state to revert client prediction
                let real_state = world.get_block(target_pos);
                let rollback_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                    x: target_pos.x(),
                    y: target_pos.y(),
                    z: target_pos.z(),
                    state_id: real_state,
                    version: 0,
                });
                if let Some(s) = self.sessions.get_mut(&session_id) {
                    let _ = s.connection.send(Lane::Control, Payload::Msg(rollback_msg));
                }
            }

            // Always acknowledge action sequence to the client
            let ack_msg = S2cMessage::BlockActionAck(S2cBlockActionAck {
                sequence: action.sequence,
            });
            if let Some(s) = self.sessions.get_mut(&session_id) {
                let _ = s.connection.send(Lane::Control, Payload::Msg(ack_msg));
            }
        }

        for (w_name, pos, vel, stack) in dropped_items_to_spawn {
            self.spawn_item_entity(&w_name, pos, vel, stack, 0);
        }

        for viewer_id in containers_to_close {
            if let Some(s) = self.sessions.get_mut(&viewer_id) {
                let close_msg = S2cMessage::CloseContainer(S2cCloseContainer { window_id: 1 });
                let _ = s.connection.send(Lane::Control, Payload::Msg(close_msg));
            }
            self.close_session_container(viewer_id);
        }

        // 4c. Process player movement inputs and authoritative simulation
        for session in self.sessions.values_mut() {
            if session.phase != ConnectionPhase::Play {
                continue;
            }

            let mut processed_any = false;
            // Drain up to 2 input frames per tick (allows catching up on packet bursts)
            for _ in 0..2 {
                let Some(frame) = session.pending_inputs.pop_front() else {
                    break;
                };

                session.selected_slot = frame.hotbar;
                if session.awaiting_teleport.is_none() {
                    telos_sim::simulate_movement_step(
                        &mut session.move_state,
                        &frame,
                        session.move_mode,
                        0.05,
                    );
                    session.last_processed_client_tick = frame.tick;
                    processed_any = true;
                }
            }

            if processed_any
                || session.awaiting_teleport.is_some()
                || self.tick_count.is_multiple_of(20)
            {
                let old_y = session.position.y;
                let was_ground = session.on_ground;
                session.position = session.move_state.pos;
                session.yaw = session.move_state.yaw;
                session.pitch = session.move_state.pitch;
                session.on_ground = session.move_state.on_ground;

                if session.capabilities.invincible
                    || session.move_mode == MoveMode::NoClipFly
                    || session.move_state.flying
                {
                    session.fall_distance = 0.0;
                } else if session.on_ground {
                    if !was_ground && session.fall_distance > 3.0 {
                        let fall_dmg = session.fall_distance - 3.0;
                        if let Some(entity) = session.ecs_entity {
                            let mut query =
                                self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                            if let Ok((mut health, mut combat)) =
                                query.get_mut(&mut self.ecs_world, entity)
                            {
                                telos_sim::apply_damage(
                                    &mut health,
                                    &mut combat,
                                    fall_dmg,
                                    DamageType::Fall,
                                );
                            }
                        }
                    }
                    session.fall_distance = 0.0;
                } else if session.position.y < old_y {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        session.fall_distance += (old_y - session.position.y) as f32;
                    }
                }

                let ack_msg = S2cMessage::PlayerMovementAck(S2cPlayerMovementAck {
                    client_tick_ack: session.last_processed_client_tick,
                    #[allow(clippy::cast_possible_truncation)]
                    server_tick: self.tick_count as u32,
                    x: session.move_state.pos.x,
                    y: session.move_state.pos.y,
                    z: session.move_state.pos.z,
                    vx: session.move_state.vel.x,
                    vy: session.move_state.vel.y,
                    vz: session.move_state.vel.z,
                    yaw: session.move_state.yaw,
                    pitch: session.move_state.pitch,
                    on_ground: session.move_state.on_ground,
                    flying: session.move_state.flying,
                    teleport_id: session.awaiting_teleport.unwrap_or(0),
                });
                let _ = session
                    .connection
                    .send(Lane::Control, Payload::Msg(ack_msg));
            }
        }

        // 4d. Auto-close container sessions if player moved beyond reach (> 5.0m)
        let mut distance_closed_containers = Vec::new();
        for session in self.sessions.values() {
            if session.phase != ConnectionPhase::Play {
                continue;
            }
            if let Some(active_cont) = session.active_container {
                let block_center = DVec3::new(
                    f64::from(active_cont.block_pos.x()) + 0.5,
                    f64::from(active_cont.block_pos.y()) + 0.5,
                    f64::from(active_cont.block_pos.z()) + 0.5,
                );
                let dist_sq = (session.position - block_center).length_squared();
                if dist_sq > 5.0 * 5.0 {
                    distance_closed_containers.push((session.session_id, active_cont.window_id));
                }
            }
        }
        for (session_id, window_id) in distance_closed_containers {
            if let Some(s) = self.sessions.get_mut(&session_id) {
                let close_msg = S2cMessage::CloseContainer(S2cCloseContainer { window_id });
                let _ = s.connection.send(Lane::Control, Payload::Msg(close_msg));
            }
            if let Some((world_name, block_pos, container_kind)) =
                self.close_session_container(session_id)
                && container_kind == 0
            {
                let remaining = self.sessions.values().any(|s| {
                    s.phase == ConnectionPhase::Play
                        && s.world_name == world_name
                        && s.active_container.as_ref().map(|c| c.block_pos) == Some(block_pos)
                });
                if !remaining {
                    let block_event_msg = S2cMessage::BlockEvent(S2cBlockEvent {
                        x: block_pos.x(),
                        y: block_pos.y(),
                        z: block_pos.z(),
                        action: 1,
                        param: 0,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play && s.world_name == world_name {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(block_event_msg.clone()));
                        }
                    }
                }
            }
        }

        // 5. Compute movement exhaustion, run ECS simulation tick, and sync stats
        for session in self.sessions.values_mut() {
            if session.phase != ConnectionPhase::Play {
                session.prev_position = session.position;
                continue;
            }
            let dist = session.position.distance(session.prev_position);
            session.prev_position = session.position;
            if !session.capabilities.invincible
                && dist > 0.001
                && let Some(entity) = session.ecs_entity
                && let Some(mut hunger) = self.ecs_world.get_mut::<Hunger>(entity)
            {
                #[allow(clippy::cast_precision_loss)]
                hunger.add_exhaustion(dist as f32 * 0.1);
            }
        }

        let player_pos_list: Vec<TargetablePlayer> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| TargetablePlayer {
                net_id: s.entity_id,
                pos: s.position,
                targetable: !s.capabilities.invincible
                    && s.game_mode != GameMode::Spectator
                    && !s.move_state.flying,
            })
            .collect();
        self.ecs_world
            .insert_resource(PlayerPositions(player_pos_list));

        // Freeze mobs outside active simulation distance to zero out CPU cost
        let mut to_freeze = Vec::new();
        let mut to_unfreeze = Vec::new();
        {
            let mut mob_sim_query = self.ecs_world.query::<(
                bevy_ecs::entity::Entity,
                &Position,
                Option<&SimulationFrozen>,
            )>();
            for (entity, pos, frozen) in mob_sim_query.iter(&self.ecs_world) {
                #[allow(clippy::cast_possible_truncation)]
                let chunk = BlockPos::new(
                    pos.0.x.floor() as i32,
                    pos.0.y.floor() as i32,
                    pos.0.z.floor() as i32,
                )
                .chunk();
                let is_sim = self.is_chunk_simulated("overworld", chunk);
                if is_sim && frozen.is_some() {
                    to_unfreeze.push(entity);
                } else if !is_sim && frozen.is_none() {
                    to_freeze.push(entity);
                }
            }
        }
        for entity in to_freeze {
            self.ecs_world.entity_mut(entity).insert(SimulationFrozen);
        }
        for entity in to_unfreeze {
            self.ecs_world
                .entity_mut(entity)
                .remove::<SimulationFrozen>();
        }

        // Update 3D A* navigation paths for active mobs
        let default_world = self.worlds.default_world();
        telos_sim::update_mob_navigation_paths(default_world, &mut self.ecs_world);

        self.sim_schedule.run(&mut self.ecs_world);

        // 5a. Authoritative mob melee attacks on survival players
        let mut attacks_to_resolve = Vec::new();
        {
            let mut mob_attack_query =
                self.ecs_world
                    .query::<(&NetEntity, &Position, &Mob, &mut AttackCooldown)>();

            for (net, pos, mob, mut cooldown) in mob_attack_query.iter_mut(&mut self.ecs_world) {
                if mob.kind != telos_sim::MobKind::Hostile || !cooldown.can_attack() {
                    continue;
                }

                let target_player_id = match mob.ai_state {
                    AiState::Chasing { target_net_id } => Some(target_net_id),
                    _ => None,
                };

                let Some(target_id) = target_player_id else {
                    continue;
                };

                if let Some(target_session) =
                    self.sessions.values().find(|s| s.entity_id == target_id)
                {
                    if target_session.phase != ConnectionPhase::Play {
                        continue;
                    }
                    if target_session.capabilities.invincible
                        || target_session.game_mode == GameMode::Spectator
                        || target_session.move_state.flying
                    {
                        continue;
                    }

                    let dist = target_session.position.distance(pos.0);
                    if dist <= f64::from(cooldown.reach) {
                        cooldown.reset();
                        attacks_to_resolve.push((
                            net.net_id,
                            target_session.session_id,
                            pos.0,
                            cooldown.damage,
                        ));
                    }
                }
            }
        }

        for (mob_net_id, target_session_id, mob_pos, mob_damage) in attacks_to_resolve {
            let Some(target_session) = self.sessions.get_mut(&target_session_id) else {
                continue;
            };
            if target_session.capabilities.invincible {
                continue;
            }
            let player_net_id = target_session.entity_id;
            let target_ecs = target_session.ecs_entity;
            let player_pos = target_session.position;

            if let Some(ecs_ent) = target_ecs {
                let mut total_armor = 0.0f32;
                let mut total_toughness = 0.0f32;
                let mut pieces = Vec::new();

                if let Some(inv) = self.ecs_world.get::<Inventory>(ecs_ent) {
                    for i in ARMOR_SLOTS {
                        let slot = &inv.slots[i];
                        if !slot.is_empty() {
                            #[allow(clippy::cast_precision_loss)]
                            {
                                total_armor +=
                                    self.registries.item_registry().armor_defense(slot.item) as f32;
                            }
                            total_toughness +=
                                self.registries.item_registry().armor_toughness(slot.item);
                            pieces.push(slot.enchantments);
                        }
                    }
                }
                let total_epf = calculate_total_epf(&pieces, DamageType::Attack);

                let resistance = self
                    .ecs_world
                    .get::<StatusEffects>(ecs_ent)
                    .and_then(|eff| eff.amplifier(StatusEffectKind::Resistance))
                    .map_or(0, |amp| amp + 1);

                let mut query = self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                if let Ok((mut health, mut tracker)) = query.get_mut(&mut self.ecs_world, ecs_ent) {
                    apply_mitigated_damage(
                        &mut health,
                        &mut tracker,
                        mob_damage,
                        DamageType::Attack,
                        total_armor,
                        total_toughness,
                        resistance,
                        total_epf,
                    );
                }

                if let Some(mut hurt_time) = self.ecs_world.get_mut::<HurtTime>(ecs_ent) {
                    hurt_time.0 = 10;
                }
            }

            // Directional knockback impulse
            let diff = player_pos - mob_pos;
            let horiz_dist = (diff.x * diff.x + diff.z * diff.z).sqrt().max(0.01);
            #[allow(clippy::cast_possible_truncation)]
            let kb_x = (diff.x / horiz_dist * 0.35) as f32;
            let kb_y = 0.25f32;
            #[allow(clippy::cast_possible_truncation)]
            let kb_z = (diff.z / horiz_dist * 0.35) as f32;

            if let Some(session) = self.sessions.get_mut(&target_session_id) {
                session.move_state.vel.x += kb_x;
                session.move_state.vel.y += kb_y;
                session.move_state.vel.z += kb_z;

                let ack_msg = S2cMessage::PlayerMovementAck(S2cPlayerMovementAck {
                    client_tick_ack: session.last_processed_client_tick,
                    #[allow(clippy::cast_possible_truncation)]
                    server_tick: self.tick_count as u32,
                    x: session.move_state.pos.x,
                    y: session.move_state.pos.y,
                    z: session.move_state.pos.z,
                    vx: session.move_state.vel.x,
                    vy: session.move_state.vel.y,
                    vz: session.move_state.vel.z,
                    yaw: session.move_state.yaw,
                    pitch: session.move_state.pitch,
                    on_ground: session.move_state.on_ground,
                    flying: session.move_state.flying,
                    teleport_id: session.awaiting_teleport.unwrap_or(0),
                });
                let _ = session
                    .connection
                    .send(Lane::Control, Payload::Msg(ack_msg));
            }

            // Broadcast attack swing (status: 4) for mob & hurt flash (status: 2) for player
            let swing_msg = S2cMessage::EntityStatus(S2cEntityStatus {
                net_id: mob_net_id,
                status: 4,
            });
            let hurt_msg = S2cMessage::EntityStatus(S2cEntityStatus {
                net_id: player_net_id,
                status: 2,
            });

            for s in self.sessions.values_mut() {
                if s.phase == ConnectionPhase::Play {
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(swing_msg.clone()));
                    let _ = s
                        .connection
                        .send(Lane::Control, Payload::Msg(hurt_msg.clone()));
                }
            }
        }

        // Check for dead mobs
        let mut dead_mobs = Vec::new();
        {
            let mut dead_query = self.ecs_world.query::<(&NetEntity, &Health)>();
            for (net, health) in dead_query.iter(&self.ecs_world) {
                if !health.is_alive() {
                    dead_mobs.push(net.net_id);
                }
            }
        }
        for mob_id in dead_mobs {
            self.handle_mob_death(mob_id, None);
        }

        // Voxel terrain collision and floor adherence for mobs
        let default_world = self.worlds.default_world_mut();
        let mut mob_query = self
            .ecs_world
            .query::<(&NetEntity, &mut Position, &mut Velocity)>();
        let mut fallen_mobs = Vec::new();
        for (net, mut pos, mut vel) in mob_query.iter_mut(&mut self.ecs_world) {
            #[allow(clippy::cast_possible_truncation)]
            let bx = pos.0.x.floor() as i32;
            #[allow(clippy::cast_possible_truncation)]
            let by = pos.0.y.floor() as i32;
            #[allow(clippy::cast_possible_truncation)]
            let bz = pos.0.z.floor() as i32;

            let foot_block = default_world.get_loaded_block(BlockPos::new(bx, by, bz));
            let ground_block = default_world.get_loaded_block(BlockPos::new(bx, by - 1, bz));
            let reg = default_world.registry();

            if telos_sim::is_solid_ground(ground_block, reg) {
                let floor_y = f64::from(by);
                if pos.0.y <= floor_y + 0.15 && vel.0.y <= 0.0 {
                    pos.0.y = floor_y;
                    vel.0.y = 0.0;
                }
            } else if telos_sim::is_solid_ground(foot_block, reg) {
                // Step-up over 1-block obstacle if headroom is clear
                let head_block = default_world.get_loaded_block(BlockPos::new(bx, by + 1, bz));
                if telos_sim::is_solid_ground(head_block, reg) {
                    // Blocked by 2+ block tall obstacle
                    vel.0.x = 0.0;
                    vel.0.z = 0.0;
                } else {
                    let floor_y = f64::from(by + 1);
                    pos.0.y = floor_y;
                    vel.0.y = 0.0;
                }
            } else {
                // Fallback to surface height if floating above unloaded chunks
                let surface_y = default_world.get_surface_y(bx, bz);
                let floor_y = f64::from(surface_y) + 1.0;
                if pos.0.y < floor_y {
                    pos.0.y = floor_y;
                    vel.0.y = 0.0;
                }
            }

            if pos.0.y < -100.0 {
                fallen_mobs.push(net.net_id);
            }
        }
        for id in fallen_mobs {
            self.despawn_mob(id);
        }

        // 5b. Mod event dispatching and world edit application
        self.event_queue.push(GameEvent::Tick {
            tick: self.tick_count,
        });
        let events = self.event_queue.drain();
        let mod_edits = self.mod_manager.dispatch_events(&events);
        for edit in mod_edits {
            if let Some((_snapshot, version)) = self
                .worlds
                .default_world_mut()
                .set_block(edit.pos, edit.state)
            {
                let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                    x: edit.pos.x(),
                    y: edit.pos.y(),
                    z: edit.pos.z(),
                    state_id: edit.state,
                    version,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(update_msg.clone()));
                    }
                }
            }
        }

        for session in self.sessions.values_mut() {
            if session.phase != ConnectionPhase::Play {
                continue;
            }
            let Some(entity) = session.ecs_entity else {
                continue;
            };

            let health = self.ecs_world.get::<Health>(entity);
            let hunger = self.ecs_world.get::<Hunger>(entity);
            let xp = self.ecs_world.get::<Experience>(entity);

            let health_val = health.map_or(20.0, |h| h.cur);
            let max_health = health.map_or(20.0, |h| h.max);
            let food_val = hunger.map_or(20, |h| h.food);
            let sat_val = hunger.map_or(5.0, |h| h.saturation);
            let (xp_lvl, xp_prog) = xp.map_or((0, 0.0), |x| (x.level(), x.progress()));

            let stats_changed = (session.cached_health - health_val).abs() > 0.001
                || session.cached_food != food_val
                || (session.cached_saturation - sat_val).abs() > 0.001
                || session.cached_xp_level != xp_lvl
                || (session.cached_xp_progress - xp_prog).abs() > 0.001;

            if stats_changed {
                session.cached_health = health_val;
                session.cached_food = food_val;
                session.cached_saturation = sat_val;
                session.cached_xp_level = xp_lvl;
                session.cached_xp_progress = xp_prog;

                let stats_msg = S2cMessage::UpdateStats(S2cUpdateStats {
                    health: health_val,
                    max_health,
                    food: food_val,
                    saturation: sat_val,
                    xp_level: xp_lvl,
                    xp_progress: xp_prog,
                });
                let _ = session
                    .connection
                    .send(Lane::Control, Payload::Msg(stats_msg));
            }

            let effects_comp = self.ecs_world.get::<StatusEffects>(entity);
            let current_effects: Vec<NetworkEffect> = effects_comp.map_or_else(Vec::new, |effs| {
                effs.effects
                    .iter()
                    .map(|e| NetworkEffect {
                        effect_id: e.kind.id(),
                        amplifier: e.amplifier,
                        duration_ticks: e.duration_ticks,
                        ambient: e.ambient,
                        particle_color: e.kind.particle_color(),
                    })
                    .collect()
            });

            if session.cached_effects != current_effects {
                session.cached_effects.clone_from(&current_effects);
                let effects_msg = S2cMessage::UpdateEffects(S2cUpdateEffects {
                    entity_id: 0,
                    effects: current_effects,
                });
                let _ = session
                    .connection
                    .send(Lane::Control, Payload::Msg(effects_msg));
            }
        }

        // 6. Process active chunk subscriptions and deliveries
        let quota = self.config.chunks_per_tick_per_player;

        for session in self.sessions.values_mut() {
            if session.phase != ConnectionPhase::Play {
                continue;
            }

            // Recompute subscriptions if player moved chunk or rotated view significantly
            if session.should_recompute_subscriptions() {
                let (unloads, lod_unloads) = session.recompute_subscriptions();
                for pos in unloads {
                    let unload_msg = S2cMessage::ChunkUnload(S2cChunkUnload {
                        chunk_x: pos.x(),
                        chunk_y: pos.y(),
                        chunk_z: pos.z(),
                    });
                    let _ = session
                        .connection
                        .send(Lane::Control, Payload::Msg(unload_msg));
                }
                for key in lod_unloads {
                    let unload_msg = S2cMessage::LodNodeUnload(S2cLodNodeUnload {
                        level: key.level,
                        node_x: key.x,
                        node_y: key.y,
                        node_z: key.z,
                    });
                    let _ = session
                        .connection
                        .send(Lane::Control, Payload::Msg(unload_msg));
                }
            }

            // Deliver up to quota chunks
            for _ in 0..quota {
                let Some(pos) = session.pop_next_chunk() else {
                    break;
                };

                let world = self.worlds.get_or_default_mut(&session.world_name);
                let snap = world.get_or_generate_chunk(pos);
                session.mark_chunk_sent(pos);

                if let Blocks::Uniform(state) = snap.blocks() {
                    let (sky, block) = snap
                        .light()
                        .map_or((15, 0), |l| (l.sky.get(0), l.block.get(0)));

                    let uniform_msg = S2cMessage::UniformChunk(S2cUniformChunk {
                        chunk_x: pos.x(),
                        chunk_y: pos.y(),
                        chunk_z: pos.z(),
                        version: snap.content_version() as u32,
                        block_state: *state,
                        sky_light: sky,
                        block_light: block,
                    });
                    let _ = session
                        .connection
                        .send(Lane::Chunk { priority: 0 }, Payload::Msg(uniform_msg));
                } else {
                    let data_msg = S2cMessage::ChunkData(S2cChunkData {
                        chunk_x: pos.x(),
                        chunk_y: pos.y(),
                        chunk_z: pos.z(),
                        version: snap.content_version() as u32,
                        epoch: snap.world_epoch() as u32,
                        payload: ChunkPayload::Snapshot(snap),
                    });
                    let _ = session
                        .connection
                        .send(Lane::Chunk { priority: 0 }, Payload::Msg(data_msg));
                }
            }

            // Deliver up to lod_quota LOD nodes
            let lod_quota = self.config.lod_nodes_per_tick_per_player;
            for _ in 0..lod_quota {
                let Some(key) = session.pop_next_lod_node() else {
                    break;
                };

                let world = self.worlds.get_or_default_mut(&session.world_name);
                let mesh = world.get_or_mesh_lod_node(key);
                session.mark_lod_node_sent(key);

                let mut words = Vec::new();
                mesh.write_to_u32_buffer(&mut words);

                let data_msg = S2cMessage::LodNodeData(S2cLodNodeData {
                    level: key.level,
                    node_x: key.x,
                    node_y: key.y,
                    node_z: key.z,
                    version: 1,
                    quad_count: mesh.quads.len() as u32,
                    palette_count: mesh.palette.len() as u32,
                    payload: LodPayload::Memory(Arc::new(words)),
                });
                let _ = session
                    .connection
                    .send(Lane::Chunk { priority: 1 }, Payload::Msg(data_msg));
            }
        }

        // 7. Natural mob spawning (runs every 100 ticks = 5 seconds)
        if self.tick_count.saturating_sub(self.last_mob_spawn_tick) >= 100 {
            self.last_mob_spawn_tick = self.tick_count;
            self.tick_natural_spawner();
        }

        // 7b. Monster mob spawners (runs every tick)
        self.tick_monster_spawners();

        // 7c. Furnace block entities combustion & smelting (runs every tick)
        self.tick_furnace_entities();

        // 8. Distance despawning based on simulation distance ((sim_dist * 32.0) + 16.0 blocks)
        let active_players: Vec<DVec3> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| s.position)
            .collect();

        if !active_players.is_empty() {
            let max_sim_dist = self
                .sessions
                .values()
                .filter(|s| s.phase == ConnectionPhase::Play)
                .map(|s| s.simulation_distance)
                .max()
                .unwrap_or(self.config.simulation_distance);
            let despawn_dist = (f64::from(max_sim_dist) * 32.0) + 16.0;
            let despawn_dist_sq = despawn_dist * despawn_dist;

            let mut to_despawn = Vec::new();
            for (&net_id, &entity) in &self.tracked_mobs {
                if let Some(pos) = self.ecs_world.get::<Position>(entity) {
                    let min_dist_sq = active_players
                        .iter()
                        .map(|p| p.distance_squared(pos.0))
                        .fold(f64::INFINITY, f64::min);
                    if min_dist_sq > despawn_dist_sq {
                        to_despawn.push(net_id);
                    }
                }
            }
            for id in to_despawn {
                self.despawn_mob(id);
            }
        }

        // 9. Broadcast mob movement deltas
        for (&net_id, &entity) in &self.tracked_mobs {
            if let (Some(pos), Some(rot)) = (
                self.ecs_world.get::<Position>(entity),
                self.ecs_world.get::<Rotation>(entity),
            ) {
                let prev_pos = self.mob_positions.get(&net_id).copied().unwrap_or(pos.0);
                let prev_yaw = self.mob_yaws.get(&net_id).copied().unwrap_or(rot.yaw);

                let moved = pos.0.distance_squared(prev_pos) > 0.0001;
                let rotated = (rot.yaw - prev_yaw).abs() > 0.5;

                if moved || rotated {
                    self.mob_positions.insert(net_id, pos.0);
                    self.mob_yaws.insert(net_id, rot.yaw);

                    let move_msg = S2cMessage::EntityMove(S2cEntityMove {
                        net_id,
                        x: pos.0.x,
                        y: pos.0.y,
                        z: pos.0.z,
                        yaw: rot.yaw,
                        pitch: rot.pitch,
                        head_yaw: rot.head_yaw,
                        on_ground: true,
                    });

                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(move_msg.clone()));
                        }
                    }
                }
            }
        }

        // 9b. Advance dropped item physics, merging, and player pickup
        self.tick_item_entities();

        // 9c. Advance projectile arrow physics, hit detection, block embedding, and player pickup
        self.tick_arrow_entities();

        // 10. Tick deterministic logic simulation engine across all loaded worlds
        for (world_name, world) in self.worlds.iter_mut() {
            let logic_updates = world.tick_logic(self.tick_count);
            for (pos, new_state) in logic_updates {
                let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                    x: pos.x(),
                    y: pos.y(),
                    z: pos.z(),
                    state_id: new_state,
                    version: 0,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play && s.world_name == *world_name {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(update_msg.clone()));
                    }
                }
            }
        }

        // 10b. Tick real-time cellular automata fluid simulation across all loaded worlds
        for (world_name, world) in self.worlds.iter_mut() {
            let fluid_updates = world.tick_fluids(self.tick_count);
            for (pos, new_state) in fluid_updates {
                let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                    x: pos.x(),
                    y: pos.y(),
                    z: pos.z(),
                    state_id: new_state,
                    version: 0,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play && s.world_name == *world_name {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(update_msg.clone()));
                    }
                }
            }
        }

        // 10c. Tick random block updates (leaf decay, plant ticks) across all loaded worlds
        for (world_name, world) in self.worlds.iter_mut() {
            let random_updates = world.tick_random_blocks(self.tick_count, 3);
            for (pos, new_state) in random_updates {
                let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                    x: pos.x(),
                    y: pos.y(),
                    z: pos.z(),
                    state_id: new_state,
                    version: 0,
                });
                let particle_msg = S2cMessage::ParticleEvent(S2cParticleEvent {
                    effect: ParticleEffectKind::BlockBreak,
                    x: pos.x() as f32 + 0.5,
                    y: pos.y() as f32 + 0.5,
                    z: pos.z() as f32 + 0.5,
                    count: 12,
                    speed: 0.5,
                    block_state_id: 8,
                });
                for s in self.sessions.values_mut() {
                    if s.phase == ConnectionPhase::Play && s.world_name == *world_name {
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(update_msg.clone()));
                        let _ = s
                            .connection
                            .send(Lane::Control, Payload::Msg(particle_msg.clone()));
                    }
                }
            }
        }

        // 11. Periodic autosave across all worlds
        if self.config.autosave_interval_ticks > 0
            && self
                .tick_count
                .is_multiple_of(u64::from(self.config.autosave_interval_ticks))
            && let Err(err) = self.worlds.save_all()
        {
            tracing::error!("Autosave failed: {err}");
        }

        // 5. Time progression and periodic synchronization
        self.time_of_day = (self.time_of_day + 1) % telos_core::DAY_TICKS;
        if self.tick_count.is_multiple_of(20) {
            let time_msg = S2cMessage::UpdateTime(S2cUpdateTime {
                world_age: self.tick_count,
                time_of_day: self.time_of_day,
            });
            for session in self.sessions.values_mut() {
                if session.phase == ConnectionPhase::Play {
                    let _ = session
                        .connection
                        .send(Lane::Control, Payload::Msg(time_msg.clone()));
                }
            }
        }

        // 6. Weather progression and periodic synchronization
        self.weather.tick();
        if self.tick_count.is_multiple_of(20) || self.weather.lightning_flash_ticks > 0 {
            let weather_msg = S2cMessage::UpdateWeather(S2cUpdateWeather {
                rain_level: self.weather.rain_level,
                thunder_level: self.weather.thunder_level,
                lightning_flash: self.weather.lightning_flash_ticks,
            });
            for session in self.sessions.values_mut() {
                if session.phase == ConnectionPhase::Play {
                    let _ = session
                        .connection
                        .send(Lane::Control, Payload::Msg(weather_msg.clone()));
                }
            }
        }
    }

    /// Saves all dirty chunks to `.tlr` region files and syncs data to disk across all worlds.
    pub fn save_and_flush(&mut self) -> Result<usize, telos_storage::StorageError> {
        self.worlds.save_all()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.worlds.save_all();
    }
}
