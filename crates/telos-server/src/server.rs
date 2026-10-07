//! Authoritative game server ticking at 20 TPS with session management and chunk streaming.

use glam::DVec3;
use hashbrown::HashMap;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use telos_content::{
    FrozenRegistries, ModSide, RegistryBuilder, discover_packs, resolve_load_order,
};
use telos_core::coords::{BlockPos, ChunkPos, Face};
use telos_mod::{JsPlugin, JsPluginEngine, ModConfig, ModManager, ModPermissions, ModResult};
use telos_net::{Connection, Lane, Payload};
use telos_protocol::bounded::{BoundedString, BoundedVec};
use telos_protocol::messages::{
    BlockActionKind, C2sBlockAction, C2sChatMessage, C2sCommandSuggest, C2sInteractEntity,
    C2sInventoryClick, C2sMessage, C2sPlayerCommand, ChunkPayload, ConnectionPhase, LodPayload,
    NetworkEffect, ParticleEffectKind, PlayerCommandKind, S2cBlockActionAck, S2cBlockUpdate,
    S2cChatMessage, S2cChunkData, S2cChunkUnload, S2cCommandSuggestions, S2cConfigDone,
    S2cDespawnEntity, S2cEntityMove, S2cEntityStatus, S2cHelloReply, S2cInventoryBulk, S2cJoinGame,
    S2cLodNodeData, S2cLodNodeUnload, S2cLoginSuccess, S2cMessage, S2cParticleEvent,
    S2cPlayerMovementAck, S2cRegistryData, S2cSpawnEntity, S2cUniformChunk, S2cUpdateEffects,
    S2cUpdateStats, S2cUpdateTime, S2cUpdateWeather, SlotData,
};
use telos_sim::command::{
    ArgumentType, CommandContext, CommandDispatcher, CommandNode, CommandOutput, register_builtins,
};
use telos_sim::event::{EventQueue, GameEvent};
use telos_sim::{
    AttributeKind, Attributes, CombatTracker, DamageType, EffectInstance, EnchantmentKind,
    EntityType, Experience, Health, Hunger, HurtTime, Inventory, ItemStack, MobBundle, MoveMode,
    NetEntity, PlayerPositions, Position, PotionType, Rotation, SimParams, SimulationFrozen,
    StatusEffectKind, StatusEffects, Velocity, WeatherKind, WeatherState, apply_mitigated_damage,
    build_sim_schedule,
};
use telos_voxel::state::BlockStateId;
use telos_voxel::storage::Blocks;
use tracing::{debug, info};

use crate::builder::ServerBuilder;
use crate::config::ServerConfig;
use crate::error::ServerError;
use crate::multi_world::{MultiWorldManager, WorldError};
use crate::session::PlayerSession;
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
    /// Tick count when natural mob spawning last ran.
    pub last_mob_spawn_tick: u64,
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
            last_mob_spawn_tick: 0,
            command_dispatcher,
            mod_manager,
            js_plugins,
            event_queue,
            listener: None,
            lan_emitter,
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
            EntityType::Zombie | EntityType::Player => MobBundle::new_zombie(net_id, pos, seed),
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

    /// Clears and despawns all currently active mobs.
    pub fn clear_mobs(&mut self) {
        let mob_ids: Vec<u32> = self.tracked_mobs.keys().copied().collect();
        for id in mob_ids {
            self.despawn_mob(id);
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
    #[allow(clippy::too_many_lines)]
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
        let mut player_commands: Vec<(u64, C2sPlayerCommand)> = Vec::new();
        let mut entity_interactions: Vec<(u64, C2sInteractEntity)> = Vec::new();
        let mut chat_messages: Vec<(u64, C2sChatMessage)> = Vec::new();
        let mut command_suggests: Vec<(u64, C2sCommandSuggest)> = Vec::new();

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
                            if session.move_mode == MoveMode::NoClipFly || session.move_state.flying
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
            if let Some(session) = self.sessions.remove(&id) {
                self.event_queue.push(GameEvent::PlayerLeft {
                    entity_net_id: session.entity_id,
                    username: session.username.clone(),
                });
                if let Some(entity) = session.ecs_entity {
                    self.ecs_world.despawn(entity);
                }
                info!(session_id = id, "Session disconnected");
            }
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
                    if let Some(session) = self.sessions.get_mut(&session_id) {
                        session.move_mode = if mode == 1 {
                            telos_sim::MoveMode::NoClipFly
                        } else {
                            telos_sim::MoveMode::Walk
                        };
                        info!(mode, "Updated player session game mode");
                    }
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
                    // Broadcast death status (status = 3)
                    let death_msg = S2cMessage::EntityStatus(S2cEntityStatus {
                        net_id: interact.target_net_id,
                        status: 3,
                    });
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(death_msg.clone()));
                        }
                    }

                    // Award experience to attacking player
                    if let Some(player_ecs) =
                        self.sessions.get(&session_id).and_then(|s| s.ecs_entity)
                        && let Some(mut xp) = self.ecs_world.get_mut::<Experience>(player_ecs)
                    {
                        xp.add_xp(5);
                    }

                    // Despawn mob
                    self.despawn_mob(interact.target_net_id);
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
                    if text.starts_with("/time set ") {
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

        // 5. Process inventory clicks
        for (session_id, click) in inventory_clicks {
            let Some(session) = self.sessions.get(&session_id) else {
                continue;
            };
            let Some(entity) = session.ecs_entity else {
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
            if telos_sim::inventory_click(&mut inv, slot_idx, button, mode).is_ok() {
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
        }

        // 4. Process block actions (authoritative validation & simulation)
        for (session_id, action) in block_actions {
            let Some(session) = self.sessions.get(&session_id) else {
                continue;
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
            let dist_sq = (session.position - block_center).length_squared();
            let max_reach = 6.0; // 5.0 blocks + 1.0 tolerance for latency
            let is_in_reach = dist_sq <= max_reach * max_reach;
            let is_in_bounds = target_pos.y() >= -1024 && target_pos.y() < 2048;

            let session_world_name = session.world_name.clone();
            let world = self.worlds.get_or_default_mut(&session_world_name);
            let old_state = world.get_block(target_pos);

            if is_in_reach && is_in_bounds {
                if matches!(action.action, BlockActionKind::Break)
                    && !self.js_plugins.dispatch_block_break(
                        session.entity_id,
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
                    let _ = session
                        .connection
                        .send(Lane::Control, Payload::Msg(rollback));
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
                    }
                } else if let Some((_snapshot, version)) = world.set_block(target_pos, new_state) {
                    match action.action {
                        BlockActionKind::Break => {
                            self.event_queue.push(GameEvent::BlockBroken {
                                pos: target_pos,
                                old_state,
                                actor_net_id: Some(u64::from(session.entity_id)),
                            });
                        }
                        BlockActionKind::Place { .. } => {
                            self.event_queue.push(GameEvent::BlockPlaced {
                                pos: target_pos,
                                new_state,
                                actor_net_id: Some(u64::from(session.entity_id)),
                            });
                        }
                        BlockActionKind::Interact => {}
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
                        if s.phase == ConnectionPhase::Play && s.world_name == session_world_name {
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

                if session.move_mode == MoveMode::NoClipFly || session.move_state.flying {
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

        // 5. Compute movement exhaustion, run ECS simulation tick, and sync stats
        for session in self.sessions.values_mut() {
            if session.phase != ConnectionPhase::Play {
                session.prev_position = session.position;
                continue;
            }
            let dist = session.position.distance(session.prev_position);
            session.prev_position = session.position;
            if dist > 0.001
                && let Some(entity) = session.ecs_entity
                && let Some(mut hunger) = self.ecs_world.get_mut::<Hunger>(entity)
            {
                #[allow(clippy::cast_precision_loss)]
                hunger.add_exhaustion(dist as f32 * 0.1);
            }
        }

        let player_pos_list: Vec<(u32, DVec3)> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| (s.entity_id, s.position))
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
