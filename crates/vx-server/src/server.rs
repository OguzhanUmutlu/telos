//! Authoritative game server ticking at 20 TPS with session management and chunk streaming.

use glam::DVec3;
use hashbrown::HashMap;
use std::sync::Arc;
use tracing::{debug, info};
use vx_content::{FrozenRegistries, ModSide, RegistryBuilder, discover_packs, resolve_load_order};
use vx_core::coords::{BlockPos, Face};
use vx_net::{Connection, Lane, Payload};
use vx_protocol::bounded::{BoundedString, BoundedVec};
use vx_protocol::messages::{
    BlockActionKind, C2sBlockAction, C2sInteractEntity, C2sInventoryClick, C2sMessage,
    C2sPlayerCommand, ChunkPayload, ConnectionPhase, LodPayload, PlayerCommandKind,
    S2cBlockActionAck, S2cBlockUpdate, S2cChunkData, S2cChunkUnload, S2cConfigDone,
    S2cDespawnEntity, S2cEntityMove, S2cEntityStatus, S2cHelloReply, S2cInventoryBulk, S2cJoinGame,
    S2cLodNodeData, S2cLodNodeUnload, S2cLoginSuccess, S2cMessage, S2cRegistryData, S2cSpawnEntity,
    S2cUniformChunk, S2cUpdateStats, S2cUpdateTime, S2cUpdateWeather, SlotData,
};
use vx_sim::{
    CombatTracker, DamageType, EntityType, Experience, Health, Hunger, HurtTime, Inventory,
    ItemStack, MobBundle, NetEntity, PlayerPositions, Position, Rotation, SimParams, Velocity,
    WeatherKind, WeatherState, build_sim_schedule,
};
use vx_voxel::state::BlockStateId;
use vx_voxel::storage::Blocks;

use crate::config::ServerConfig;
use crate::session::PlayerSession;
use crate::world::ServerWorld;

/// Top-level authoritative server orchestrating worlds, simulation, and client streaming.
pub struct Server {
    config: ServerConfig,
    world: ServerWorld,
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
    pub fn with_registries(
        seed: u64,
        config: ServerConfig,
        registries: Arc<FrozenRegistries>,
    ) -> Self {
        let world = if let Some(dir) = &config.save_directory {
            match ServerWorld::with_content_storage(seed, &registries, dir) {
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

        let mut ecs_world = bevy_ecs::world::World::new();
        ecs_world.insert_resource(SimParams::default());
        let sim_schedule = build_sim_schedule();

        Self {
            config,
            world,
            registries,
            sessions: HashMap::new(),
            next_session_id: 1,
            next_entity_id: 1,
            tick_count: 0,
            time_of_day: vx_core::NOON_TICKS,
            weather: WeatherState::new(seed),
            ecs_world,
            sim_schedule,
            tracked_mobs: HashMap::new(),
            mob_positions: HashMap::new(),
            mob_yaws: HashMap::new(),
            last_mob_spawn_tick: 0,
        }
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

        let seed = self.world.seed().wrapping_add(u64::from(net_id));
        let bundle = match entity_type {
            EntityType::Pig => MobBundle::new_pig(net_id, pos, seed),
            EntityType::Cow => MobBundle::new_cow(net_id, pos, seed),
            EntityType::Zombie | EntityType::Player => MobBundle::new_zombie(net_id, pos, seed),
        };
        let health = bundle.health.cur;
        let entity = self.ecs_world.spawn(bundle).id();

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

            let surface_y = self.world.get_surface_y(spawn_x, spawn_z);
            if !(-500..=1000).contains(&surface_y) {
                continue;
            }

            let check_pos = BlockPos::new(spawn_x, surface_y + 1, spawn_z);
            let (sky_light, block_light) = self.world.get_light(check_pos);

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

    /// Accesses the server's world container immutably.
    #[must_use]
    pub fn world(&self) -> &ServerWorld {
        &self.world
    }

    /// Accesses the server's world container mutably.
    pub fn world_mut(&mut self) -> &mut ServerWorld {
        &mut self.world
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

        let session = PlayerSession::new(session_id, entity_id, conn, &self.config);
        self.sessions.insert(session_id, session);

        info!(session_id, entity_id, "Registered new client session");
        session_id
    }

    /// Current time of day in ticks `[0..24000)`.
    #[must_use]
    pub fn time_of_day(&self) -> u64 {
        self.time_of_day
    }

    /// Sets the current time of day in ticks.
    pub fn set_time_of_day(&mut self, time: u64) {
        self.time_of_day = time % vx_core::DAY_TICKS;
    }

    /// Advances the server simulation by one tick (20 TPS fixed).
    #[allow(clippy::too_many_lines)]
    pub fn tick(&mut self) {
        self.tick_count += 1;

        // 1. Process inbound network packets across all sessions
        let mut disconnected = Vec::new();
        let mut block_actions: Vec<(u64, C2sBlockAction)> = Vec::new();
        let mut inventory_clicks: Vec<(u64, C2sInventoryClick)> = Vec::new();
        let mut player_commands: Vec<(u64, C2sPlayerCommand)> = Vec::new();
        let mut entity_interactions: Vec<(u64, C2sInteractEntity)> = Vec::new();

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
                                registry_id: BoundedString::new("voxel:block").unwrap(),
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
                                registry_id: BoundedString::new("voxel:item").unwrap(),
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
                            }
                            C2sMessage::ConfigAck(_) => {
                                info!(session_id, "Client config acked, entering Play");
                                session.phase = ConnectionPhase::Play;

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
                                inv.update_crafting();

                                let ecs_entity = self
                                    .ecs_world
                                    .spawn((
                                        Health::new(20.0),
                                        CombatTracker::default(),
                                        Hunger::new(20, 5.0),
                                        Experience::default(),
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
                        C2sMessage::PlayerPosition(pos) => {
                            session.update_position(
                                DVec3::new(pos.x, pos.y, pos.z),
                                pos.yaw,
                                pos.pitch,
                                pos.on_ground,
                            );
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
                        C2sMessage::Disconnect(_) => {
                            disconnected.push(*session_id);
                        }
                        _ => {}
                    },
                }
            }
        }

        for id in disconnected {
            if let Some(session) = self.sessions.remove(&id) {
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
                            vx_sim::apply_damage(
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

                let mut query = self.ecs_world.query::<(&mut Health, &mut CombatTracker)>();
                if let Ok((mut health, mut tracker)) =
                    query.get_mut(&mut self.ecs_world, target_entity)
                {
                    vx_sim::apply_damage(&mut health, &mut tracker, 4.0, DamageType::Attack);
                    if !health.is_alive() {
                        is_dead = true;
                    }
                }

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
                if let Some(mut mob) = self.ecs_world.get_mut::<vx_sim::Mob>(target_entity)
                    && mob.kind == vx_sim::MobKind::Passive
                {
                    mob.ai_state = vx_sim::AiState::Fleeing {
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

        // 3. Process inventory clicks
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
                0 => vx_sim::ClickButton::Left,
                _ => vx_sim::ClickButton::Right,
            };
            let mode = match click.mode {
                0 => vx_sim::ClickMode::Pickup,
                1 => vx_sim::ClickMode::QuickMove,
                2 => vx_sim::ClickMode::SwapHotbar,
                _ => vx_sim::ClickMode::Drop,
            };

            let slot_idx = click.slot as usize;
            if vx_sim::inventory_click(&mut inv, slot_idx, button, mode).is_ok() {
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
                BlockActionKind::Break => (
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

            if is_in_reach && is_in_bounds {
                if let Some((_snapshot, version)) = self.world.set_block(target_pos, new_state) {
                    let update_msg = S2cMessage::BlockUpdate(S2cBlockUpdate {
                        x: target_pos.x(),
                        y: target_pos.y(),
                        z: target_pos.z(),
                        state_id: new_state,
                        version,
                    });

                    // Broadcast block update to all players in Play phase
                    for s in self.sessions.values_mut() {
                        if s.phase == ConnectionPhase::Play {
                            let _ = s
                                .connection
                                .send(Lane::Control, Payload::Msg(update_msg.clone()));
                        }
                    }
                }
            } else {
                // Out of reach or invalid: send true block state to revert client prediction
                let real_state = self.world.get_block(target_pos);
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

        self.sim_schedule.run(&mut self.ecs_world);

        // Terrain floor clamp for mobs
        let mut mob_query = self
            .ecs_world
            .query::<(&NetEntity, &mut Position, &mut Velocity)>();
        let mut fallen_mobs = Vec::new();
        for (net, mut pos, mut vel) in mob_query.iter_mut(&mut self.ecs_world) {
            #[allow(clippy::cast_possible_truncation)]
            let surface_y = self
                .world
                .get_surface_y(pos.0.x.floor() as i32, pos.0.z.floor() as i32);
            let floor_y = f64::from(surface_y) + 1.0;
            if pos.0.y < floor_y {
                pos.0.y = floor_y;
                vel.0.y = 0.0;
            }
            if pos.0.y < -100.0 {
                fallen_mobs.push(net.net_id);
            }
        }
        for id in fallen_mobs {
            self.despawn_mob(id);
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

                let snap = self.world.get_or_generate_chunk(pos);
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

                let mesh = self.world.get_or_mesh_lod_node(key);
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

        // 8. Distance despawning (> 72 blocks from all players)
        let active_players: Vec<DVec3> = self
            .sessions
            .values()
            .filter(|s| s.phase == ConnectionPhase::Play)
            .map(|s| s.position)
            .collect();

        if !active_players.is_empty() {
            let mut to_despawn = Vec::new();
            for (&net_id, &entity) in &self.tracked_mobs {
                if let Some(pos) = self.ecs_world.get::<Position>(entity) {
                    let min_dist_sq = active_players
                        .iter()
                        .map(|p| p.distance_squared(pos.0))
                        .fold(f64::INFINITY, f64::min);
                    if min_dist_sq > 72.0 * 72.0 {
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

        // 10. Periodic autosave
        if self.config.autosave_interval_ticks > 0
            && self
                .tick_count
                .is_multiple_of(u64::from(self.config.autosave_interval_ticks))
            && let Err(err) = self.world.save_dirty_chunks()
        {
            tracing::error!("Autosave failed: {err}");
        }

        // 5. Time progression and periodic synchronization
        self.time_of_day = (self.time_of_day + 1) % vx_core::DAY_TICKS;
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

    /// Saves all dirty chunks to `.vxr` region files and syncs data to disk.
    pub fn save_and_flush(&mut self) -> Result<usize, vx_storage::StorageError> {
        let saved = self.world.save_dirty_chunks()?;
        self.world.flush_storage()?;
        Ok(saved)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.world.save_dirty_chunks();
        let _ = self.world.flush_storage();
    }
}
