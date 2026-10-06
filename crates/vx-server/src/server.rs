//! Authoritative game server ticking at 20 TPS with session management and chunk streaming.

use glam::DVec3;
use hashbrown::HashMap;
use std::sync::Arc;
use tracing::{debug, info};
use vx_core::coords::{BlockPos, Face};
use vx_net::{Connection, Lane, Payload};
use vx_protocol::messages::{
    BlockActionKind, C2sBlockAction, C2sMessage, ChunkPayload, ConnectionPhase, LodPayload,
    S2cBlockActionAck, S2cBlockUpdate, S2cChunkData, S2cChunkUnload, S2cHelloReply, S2cJoinGame,
    S2cLodNodeData, S2cLodNodeUnload, S2cLoginSuccess, S2cMessage, S2cUniformChunk,
};
use vx_voxel::registry::BlockRegistry;
use vx_voxel::state::BlockStateId;
use vx_voxel::storage::Blocks;

use crate::config::ServerConfig;
use crate::session::PlayerSession;
use crate::world::ServerWorld;

/// Top-level authoritative server orchestrating worlds, simulation, and client streaming.
pub struct Server {
    config: ServerConfig,
    world: ServerWorld,
    sessions: HashMap<u64, PlayerSession>,
    next_session_id: u64,
    next_entity_id: u32,
    tick_count: u64,
}

impl Server {
    /// Creates a new `Server` instance with procedural worldgen seed and configuration.
    #[must_use]
    pub fn new(seed: u64, config: ServerConfig) -> Self {
        let registry = BlockRegistry::standard();
        let world = if let Some(dir) = &config.save_directory {
            match ServerWorld::with_storage(seed, registry.clone(), dir) {
                Ok(w) => w,
                Err(err) => {
                    tracing::error!(
                        "Failed to initialize storage at {dir:?}: {err}; falling back to memory"
                    );
                    ServerWorld::new(seed, registry)
                }
            }
        } else {
            ServerWorld::new(seed, registry)
        };

        Self {
            config,
            world,
            sessions: HashMap::new(),
            next_session_id: 1,
            next_entity_id: 1,
            tick_count: 0,
        }
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

    /// Advances the server simulation by one tick (20 TPS fixed).
    #[allow(clippy::too_many_lines)]
    pub fn tick(&mut self) {
        self.tick_count += 1;

        // 1. Process inbound network packets across all sessions
        let mut disconnected = Vec::new();
        let mut block_actions: Vec<(u64, C2sBlockAction)> = Vec::new();

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
                        C2sMessage::Disconnect(_) => {
                            disconnected.push(*session_id);
                        }
                        _ => {}
                    },
                }
            }
        }

        for id in disconnected {
            self.sessions.remove(&id);
            info!(session_id = id, "Session disconnected");
        }

        // 2. Process block actions (authoritative validation & simulation)
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

        // 3. Process active chunk subscriptions and deliveries
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

        // 4. Periodic autosave
        if self.config.autosave_interval_ticks > 0
            && self
                .tick_count
                .is_multiple_of(u64::from(self.config.autosave_interval_ticks))
            && let Err(err) = self.world.save_dirty_chunks()
        {
            tracing::error!("Autosave failed: {err}");
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
