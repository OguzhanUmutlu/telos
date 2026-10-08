//! Client player session management, state machine, and chunk delivery queues.

use glam::{DVec3, Vec3};
use hashbrown::HashSet;
use std::collections::{BinaryHeap, VecDeque};
use telos_core::coords::{BlockPos, ChunkPos};
use telos_lod::coords::LodNodeKey;
use telos_lod::selection::{LodClipmap, LodClipmapConfig};
use telos_net::Connection;
use telos_protocol::messages::{
    C2sMessage, ConnectionPhase, InputFrame, NetworkEffect, S2cMessage,
};
use telos_sim::{MoveMode, MoveState};

use crate::config::ServerConfig;
use crate::priority::{QueuedChunk, compute_chunk_priority};

/// Represents an active player connection session on the server.
pub struct PlayerSession {
    /// Unique session identifier.
    pub session_id: u64,
    /// Authoritative entity ID assigned to player in the world.
    pub entity_id: u32,
    /// Active lifecycle phase of the connection.
    pub phase: ConnectionPhase,
    /// Bidirectional network or in-memory transport connection.
    pub connection: Box<dyn Connection<S2cMessage, C2sMessage>>,
    /// Current world position in double precision.
    pub position: DVec3,
    /// Camera look yaw in degrees.
    pub yaw: f32,
    /// Camera look pitch in degrees.
    pub pitch: f32,
    /// Ground contact flag.
    pub on_ground: bool,
    /// Horizontal view distance in chunks.
    pub view_distance: u32,
    /// Simulation distance in chunks.
    pub simulation_distance: u32,
    /// Vertical chunk radius above and below player chunk.
    pub vertical_view_distance: u32,
    /// Set of chunk coordinates currently held / active on the client.
    pub sent_chunks: HashSet<ChunkPos>,
    /// Chunks queued for transmission prioritized by distance and view cone.
    pub queued_chunks: BinaryHeap<QueuedChunk>,
    /// Set of chunks currently in the queue to prevent duplicate insertion.
    pub queued_set: HashSet<ChunkPos>,
    /// Clipmap selector for far-field LOD nodes.
    pub clipmap: LodClipmap,
    /// Set of LOD node keys currently delivered to client.
    pub sent_lod_nodes: HashSet<LodNodeKey>,
    /// Queue of LOD node keys awaiting generation and delivery.
    pub queued_lod_nodes: Vec<LodNodeKey>,
    /// Set of LOD node keys currently in the queue.
    pub queued_lod_set: HashSet<LodNodeKey>,
    /// Chunk position when subscriptions were last recomputed.
    pub last_subscription_chunk: ChunkPos,
    /// Yaw angle when subscriptions were last recomputed.
    pub last_subscription_yaw: f32,
    /// ECS Entity representing the player in telos-sim.
    pub ecs_entity: Option<bevy_ecs::entity::Entity>,
    /// Previous position for calculating distance traveled in the tick.
    pub prev_position: DVec3,
    /// Cached health sent to client.
    pub cached_health: f32,
    /// Cached food sent to client.
    pub cached_food: u32,
    /// Cached saturation sent to client.
    pub cached_saturation: f32,
    /// Cached experience level sent to client.
    pub cached_xp_level: u32,
    /// Cached experience progress fraction sent to client.
    pub cached_xp_progress: f32,
    /// Cached active status effects sent to client.
    pub cached_effects: Vec<NetworkEffect>,
    /// Player display username.
    pub username: String,
    /// Tick count when client last sent a chat or command message.
    pub last_chat_tick: u64,
    /// Number of chat messages sent in current burst window.
    pub chat_burst_count: u32,
    /// Authoritative movement physics state.
    pub move_state: MoveState,
    /// Pending client movement inputs awaiting simulation.
    pub pending_inputs: VecDeque<InputFrame>,
    /// Last client tick simulated and acknowledged.
    pub last_processed_client_tick: u32,
    /// Sequential counter for server-initiated teleports.
    pub teleport_id_counter: u32,
    /// Pending teleport ID awaiting client `C2sTeleportAck`.
    pub awaiting_teleport: Option<u32>,
    /// Active movement simulation mode.
    pub move_mode: MoveMode,
    /// Accumulated fall distance in blocks for fall damage calculation.
    pub fall_distance: f32,
    /// Currently selected hotbar slot index (0..=8).
    pub selected_slot: u8,
    /// Name of the world/dimension this session currently resides in.
    pub world_name: String,
    /// Active game mode (Survival, Creative, Adventure, Spectator).
    pub game_mode: telos_sim::GameMode,
    /// Player game mode capabilities (flight, invincibility, build permission, etc.).
    pub capabilities: telos_sim::PlayerCapabilities,
    /// Currently open container session (if any).
    pub active_container: Option<ActiveContainerSession>,
    /// Active 3x3 crafting table grid state (if crafting table container is open).
    pub active_crafting_table: telos_sim::CraftingTableInventory,
    /// Player advancement progress state.
    pub advancements: telos_sim::PlayerAdvancements,
}

/// Tracks an open container window for a player session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveContainerSession {
    /// Window ID (typically 1).
    pub window_id: u8,
    /// World coordinates of the container block.
    pub block_pos: BlockPos,
    /// Container kind (0: Chest, 1: Furnace, 2: Crafting Table).
    pub container_kind: u8,
}

impl PlayerSession {
    /// Creates a new `PlayerSession` starting in the `Hello` phase with default spawn.
    #[must_use]
    pub fn new(
        session_id: u64,
        entity_id: u32,
        connection: Box<dyn Connection<S2cMessage, C2sMessage>>,
        config: &ServerConfig,
    ) -> Self {
        Self::new_with_spawn(
            session_id,
            entity_id,
            connection,
            config,
            DVec3::new(128.0, 45.0, 160.0),
        )
    }

    /// Creates a new `PlayerSession` starting in the `Hello` phase with a custom spawn point.
    #[must_use]
    pub fn new_with_spawn(
        session_id: u64,
        entity_id: u32,
        connection: Box<dyn Connection<S2cMessage, C2sMessage>>,
        config: &ServerConfig,
        spawn_pos: DVec3,
    ) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        let clipmap_config = LodClipmapConfig {
            r0: (config.view_distance as f32) * 32.0,
            max_level: config.max_lod_level,
            hysteresis: 0.12,
            vertical_span: config.vertical_view_distance as i32,
        };

        let move_state = MoveState::new(spawn_pos, -90.0, 0.0, false);

        Self {
            session_id,
            entity_id,
            phase: ConnectionPhase::Hello,
            connection,
            position: spawn_pos,
            yaw: -90.0,
            pitch: 0.0,
            on_ground: false,
            fall_distance: 0.0,
            view_distance: config.view_distance,
            simulation_distance: config.simulation_distance,
            vertical_view_distance: config.vertical_view_distance,
            sent_chunks: HashSet::new(),
            queued_chunks: BinaryHeap::new(),
            queued_set: HashSet::new(),
            clipmap: LodClipmap::new(clipmap_config),
            sent_lod_nodes: HashSet::new(),
            queued_lod_nodes: Vec::new(),
            queued_lod_set: HashSet::new(),
            last_subscription_chunk: ChunkPos::new(i32::MAX, i32::MAX, i32::MAX),
            last_subscription_yaw: -90.0,
            ecs_entity: None,
            prev_position: spawn_pos,
            cached_health: 20.0,
            cached_food: 20,
            cached_saturation: 5.0,
            cached_xp_level: 0,
            cached_xp_progress: 0.0,
            cached_effects: Vec::new(),
            username: format!("Player{session_id}"),
            last_chat_tick: 0,
            chat_burst_count: 0,
            move_state,
            pending_inputs: VecDeque::with_capacity(16),
            last_processed_client_tick: 0,
            teleport_id_counter: 0,
            awaiting_teleport: None,
            move_mode: MoveMode::Walk,
            game_mode: telos_sim::GameMode::Survival,
            capabilities: telos_sim::PlayerCapabilities::survival(),
            selected_slot: 0,
            world_name: "overworld".to_string(),
            active_container: None,
            active_crafting_table: telos_sim::CraftingTableInventory::new(),
            advancements: telos_sim::PlayerAdvancements::default(),
        }
    }

    /// Current chunk containing the player's position.
    #[must_use]
    pub fn player_chunk(&self) -> ChunkPos {
        BlockPos::new(
            self.position.x.floor() as i32,
            self.position.y.floor() as i32,
            self.position.z.floor() as i32,
        )
        .chunk()
    }

    /// Unit forward vector derived from current yaw and pitch.
    #[must_use]
    pub fn look_direction(&self) -> Vec3 {
        let cos_pitch = self.pitch.to_radians().cos();
        Vec3::new(
            self.yaw.to_radians().cos() * cos_pitch,
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * cos_pitch,
        )
        .normalize_or_zero()
    }

    /// Updates the player's pose from client input.
    pub fn update_position(&mut self, pos: DVec3, yaw: f32, pitch: f32, on_ground: bool) {
        self.position = pos;
        self.yaw = yaw;
        self.pitch = pitch;
        self.on_ground = on_ground;
        self.move_state.pos = pos;
        self.move_state.yaw = yaw;
        self.move_state.pitch = pitch;
        self.move_state.on_ground = on_ground;
    }

    /// Enqueues a client input frame, enforcing tick freshness and rejecting packets sent too far ahead.
    pub fn queue_input(&mut self, frame: InputFrame) {
        if self.awaiting_teleport.is_some() {
            // Discard inputs that were generated before acknowledging the teleport
            return;
        }

        if frame.tick <= self.last_processed_client_tick {
            return;
        }

        // Anti-cheat speed / timer hack guard: drop frames > 10 ticks in the future
        if frame.tick > self.last_processed_client_tick + 10 && self.last_processed_client_tick > 0
        {
            return;
        }

        // Keep pending inputs sorted by tick
        if let Some(last) = self.pending_inputs.back() {
            if frame.tick > last.tick {
                self.pending_inputs.push_back(frame);
            } else if !self.pending_inputs.iter().any(|f| f.tick == frame.tick) {
                let idx = self.pending_inputs.partition_point(|f| f.tick < frame.tick);
                self.pending_inputs.insert(idx, frame);
            }
        } else {
            self.pending_inputs.push_back(frame);
        }

        // Cap queue length to prevent memory exhaustion
        while self.pending_inputs.len() > 16 {
            self.pending_inputs.pop_front();
        }
    }

    /// Triggers an authoritative teleport to a new world position.
    pub fn teleport(&mut self, new_pos: DVec3) -> u32 {
        self.teleport_id_counter = self.teleport_id_counter.wrapping_add(1).max(1);
        let tp_id = self.teleport_id_counter;
        self.awaiting_teleport = Some(tp_id);
        self.position = new_pos;
        self.move_state.pos = new_pos;
        self.move_state.vel = Vec3::ZERO;
        self.pending_inputs.clear();
        tp_id
    }

    /// Returns `true` if subscriptions should be re-evaluated.
    #[must_use]
    pub fn should_recompute_subscriptions(&self) -> bool {
        let curr = self.player_chunk();
        if curr != self.last_subscription_chunk {
            return true;
        }
        (self.yaw - self.last_subscription_yaw).abs() >= 30.0
    }

    /// Recomputes desired chunk and LOD subscriptions and returns items to be unloaded.
    #[allow(clippy::cast_possible_wrap)]
    pub fn recompute_subscriptions(&mut self) -> (Vec<ChunkPos>, Vec<LodNodeKey>) {
        let player_chunk = self.player_chunk();
        let look_dir = self.look_direction();
        self.last_subscription_chunk = player_chunk;
        self.last_subscription_yaw = self.yaw;

        let r = self.view_distance as i32;
        let ry = self.vertical_view_distance as i32;
        let r_sq = r * r;

        // 1. Enqueue newly needed chunks in view cone
        for dz in -r..=r {
            for dx in -r..=r {
                if dx * dx + dz * dz > r_sq {
                    continue;
                }
                for dy in -ry..=ry {
                    let target = ChunkPos::new(
                        player_chunk.x() + dx,
                        player_chunk.y() + dy,
                        player_chunk.z() + dz,
                    );

                    if !self.sent_chunks.contains(&target) && self.queued_set.insert(target) {
                        let priority = compute_chunk_priority(player_chunk, look_dir, target);
                        self.queued_chunks.push(QueuedChunk {
                            pos: target,
                            priority,
                        });
                    }
                }
            }
        }

        // 2. Identify and remove chunks outside hysteresis radius (r + 2, ry + 4)
        let evict_r = r + 2;
        let evict_r_sq = evict_r * evict_r;
        let evict_ry = ry + 4;

        let mut to_unload = Vec::new();
        self.sent_chunks.retain(|pos| {
            let dx = pos.x() - player_chunk.x();
            let dz = pos.z() - player_chunk.z();
            let dy = (pos.y() - player_chunk.y()).abs();

            if dx * dx + dz * dz > evict_r_sq || dy > evict_ry {
                to_unload.push(*pos);
                false
            } else {
                true
            }
        });

        // 3. Enqueue desired far-field LOD clipmap nodes
        let cam_pos = Vec3::new(
            self.position.x as f32,
            self.position.y as f32,
            self.position.z as f32,
        );
        let desired_lod_nodes = self.clipmap.compute_desired_nodes(cam_pos);

        for key in &desired_lod_nodes {
            if !self.sent_lod_nodes.contains(key) && self.queued_lod_set.insert(*key) {
                self.queued_lod_nodes.push(*key);
            }
        }

        // 4. Identify and remove LOD nodes outside desired clipmap rings
        let mut lod_unloads = Vec::new();
        self.sent_lod_nodes.retain(|key| {
            if desired_lod_nodes.contains(key) {
                true
            } else {
                lod_unloads.push(*key);
                false
            }
        });

        (to_unload, lod_unloads)
    }

    /// Pops the highest-priority chunk scheduled for delivery.
    pub fn pop_next_chunk(&mut self) -> Option<ChunkPos> {
        while let Some(queued) = self.queued_chunks.pop() {
            if self.queued_set.remove(&queued.pos) && !self.sent_chunks.contains(&queued.pos) {
                return Some(queued.pos);
            }
        }
        None
    }

    /// Marks a chunk as delivered and active on the client.
    pub fn mark_chunk_sent(&mut self, pos: ChunkPos) {
        self.sent_chunks.insert(pos);
    }

    /// Pops the next LOD node scheduled for delivery.
    pub fn pop_next_lod_node(&mut self) -> Option<LodNodeKey> {
        while let Some(key) = self.queued_lod_nodes.pop() {
            self.queued_lod_set.remove(&key);
            if !self.sent_lod_nodes.contains(&key) {
                return Some(key);
            }
        }
        None
    }

    /// Marks an LOD node as delivered and active on the client.
    pub fn mark_lod_node_sent(&mut self, key: LodNodeKey) {
        self.sent_lod_nodes.insert(key);
    }

    /// Clears active chunk and LOD subscriptions (e.g. during world transfer), returning unloads.
    pub fn clear_subscriptions(&mut self) -> (Vec<ChunkPos>, Vec<LodNodeKey>) {
        let unloads: Vec<ChunkPos> = self.sent_chunks.drain().collect();
        let lod_unloads: Vec<LodNodeKey> = self.sent_lod_nodes.drain().collect();
        self.queued_chunks.clear();
        self.queued_set.clear();
        self.queued_lod_nodes.clear();
        self.queued_lod_set.clear();
        self.last_subscription_chunk = ChunkPos::new(i32::MAX, i32::MAX, i32::MAX);
        (unloads, lod_unloads)
    }
}
