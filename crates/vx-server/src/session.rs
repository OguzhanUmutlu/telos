//! Client player session management, state machine, and chunk delivery queues.

use glam::{DVec3, Vec3};
use hashbrown::HashSet;
use std::collections::BinaryHeap;
use vx_core::coords::{BlockPos, ChunkPos};
use vx_lod::coords::LodNodeKey;
use vx_lod::selection::{LodClipmap, LodClipmapConfig};
use vx_net::Connection;
use vx_protocol::messages::{C2sMessage, ConnectionPhase, S2cMessage};

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
    /// ECS Entity representing the player in vx-sim.
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
    /// Player display username.
    pub username: String,
    /// Tick count when client last sent a chat or command message.
    pub last_chat_tick: u64,
    /// Number of chat messages sent in current burst window.
    pub chat_burst_count: u32,
}

impl PlayerSession {
    /// Creates a new `PlayerSession` starting in the `Hello` phase.
    #[must_use]
    pub fn new(
        session_id: u64,
        entity_id: u32,
        connection: Box<dyn Connection<S2cMessage, C2sMessage>>,
        config: &ServerConfig,
    ) -> Self {
        let spawn_pos = DVec3::new(128.0, 45.0, 160.0);
        #[allow(clippy::cast_possible_wrap)]
        let clipmap_config = LodClipmapConfig {
            r0: (config.view_distance as f32) * 32.0,
            max_level: config.max_lod_level,
            hysteresis: 0.12,
            vertical_span: config.vertical_view_distance as i32,
        };

        Self {
            session_id,
            entity_id,
            phase: ConnectionPhase::Hello,
            connection,
            position: spawn_pos,
            yaw: -90.0,
            pitch: 0.0,
            on_ground: false,
            view_distance: config.view_distance,
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
            username: format!("Player{session_id}"),
            last_chat_tick: 0,
            chat_burst_count: 0,
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
}
