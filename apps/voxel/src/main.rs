//! Client application executable for the voxel engine.

pub mod camera;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use camera::{Camera, FlyController, Frustum};
use clap::Parser;
use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use mimalloc::MiMalloc;
use tracing::info;
use vx_assets::{AnimatedTextureInfo, ResourcePackStack, TextureArrayBuilder};
use vx_core::{
    BlockPos, FixedTimestep, RaycastHit, TelemetryConfig, coords::ChunkPos, init_telemetry,
    raycast_voxels,
};
use vx_gpu::{
    DepthBuffer, GpuBuffer, GpuContext, GpuTextureArray, GraphicsPipeline, ShaderModule,
    TextureMipRegion, vk,
};
use vx_net::{Connection, Lane, MemoryConnection, Payload};
use vx_protocol::bounded::BoundedString;
use vx_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sClientSettings, C2sConfigAck, C2sHello,
    C2sLoginStart, C2sMessage, C2sPlayerPosition, ChunkPayload, ConnectionPhase, S2cMessage,
};
use vx_voxel::chunk::{Chunk, ChunkSnapshot};
use vx_voxel::coords::LocalIdx;
use vx_voxel::light::ChunkLight;
use vx_voxel::registry::BlockRegistry;
use vx_voxel::shape::BlockShape;
use vx_voxel::state::BlockStateId;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Voxel engine client.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log level filter.
    #[arg(long, default_value = "info")]
    log: String,

    /// Procedural world generation seed.
    #[arg(short, long, default_value_t = 0x5EED_C0DE_1234_5678)]
    seed: u64,

    /// Horizontal chunk view distance radius.
    #[arg(short, long, default_value_t = 8)]
    view_distance: u32,

    /// Enable Vulkan validation layers.
    #[arg(long, default_value_t = false)]
    validation: bool,

    /// World save directory for singleplayer server.
    #[arg(short, long, default_value = "worlds/default")]
    world_dir: PathBuf,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChunkPushConstants {
    view_proj: glam::Mat4,
    chunk_pos: [i32; 3],
    pattern_offset: u32,
    quad_buffer_address: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LodPushConstants {
    view_proj: glam::Mat4,
    node_pos: [i32; 3],
    level: u32,
    quad_count: u32,
    buffer_address: u64,
    camera_pos: Vec3,
    max_distance: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HighlightPushConstants {
    view_proj: glam::Mat4,
    min_bound: [f32; 4],
    max_bound: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TranslucentPushConstants {
    view_proj: glam::Mat4,
    chunk_pos: [i32; 3],
    pattern_offset: u32,
    quad_buffer_address: u64,
    frame_tick: u32,
    water_base_layer: u32,
    water_frame_count: u32,
    _pad: [u32; 3],
}

const HOTBAR_ITEMS: [(&str, BlockStateId); 9] = [
    ("Stone", BlockStateId::new(1)),
    ("Dirt", BlockStateId::new(2)),
    ("Grass", BlockStateId::new(3)),
    ("Oak Planks", BlockStateId::new(7)),
    ("Stone Slab", BlockStateId::new(10)),
    ("Oak Stairs", BlockStateId::new(11)),
    ("Oak Leaves", BlockStateId::new(8)),
    ("Glass", BlockStateId::new(9)),
    ("Water", BlockStateId::new(6)),
];

struct GpuMeshLayer {
    buffer: GpuBuffer,
    quad_count: u32,
    pattern_offset: u32,
}

struct GpuChunkMesh {
    pos: [i32; 3],
    min_aabb: Vec3,
    max_aabb: Vec3,
    opaque: Option<GpuMeshLayer>,
    t1_opaque: Option<GpuMeshLayer>,
    cutout: Option<GpuMeshLayer>,
    translucent: Option<GpuMeshLayer>,
}

impl GpuChunkMesh {
    fn destroy(mut self, ctx: &GpuContext) {
        let device = ctx.device().raw();
        let allocator = ctx.allocator();
        if let Some(mut layer) = self.opaque.take() {
            layer.buffer.destroy(device, allocator);
        }
        if let Some(mut layer) = self.t1_opaque.take() {
            layer.buffer.destroy(device, allocator);
        }
        if let Some(mut layer) = self.cutout.take() {
            layer.buffer.destroy(device, allocator);
        }
        if let Some(mut layer) = self.translucent.take() {
            layer.buffer.destroy(device, allocator);
        }
    }
}

fn upload_t0_layer(
    ctx: &GpuContext,
    mesh: &vx_mesh::mesh::T0Mesh,
    label: &'static str,
) -> Option<GpuMeshLayer> {
    if mesh.is_empty() {
        return None;
    }
    let mut buffer_data = Vec::new();
    mesh.write_to_u32_buffer(&mut buffer_data);
    match ctx.create_buffer_with_data(label, &buffer_data, vk::BufferUsageFlags::empty()) {
        Ok(buffer) => Some(GpuMeshLayer {
            buffer,
            #[allow(clippy::cast_possible_truncation)]
            quad_count: mesh.quads.len() as u32,
            #[allow(clippy::cast_possible_truncation)]
            pattern_offset: mesh.quads.len() as u32,
        }),
        Err(err) => {
            tracing::error!("Failed to create GPU buffer for {label}: {err}");
            None
        }
    }
}

fn upload_t1_layer(
    ctx: &GpuContext,
    mesh: &vx_mesh::t1::T1Mesh,
    label: &'static str,
) -> Option<GpuMeshLayer> {
    if mesh.is_empty() {
        return None;
    }
    let mut buffer_data = Vec::new();
    mesh.write_to_u32_buffer(&mut buffer_data);
    match ctx.create_buffer_with_data(label, &buffer_data, vk::BufferUsageFlags::empty()) {
        Ok(buffer) => Some(GpuMeshLayer {
            buffer,
            #[allow(clippy::cast_possible_truncation)]
            quad_count: mesh.quads.len() as u32,
            #[allow(clippy::cast_possible_truncation)]
            pattern_offset: mesh.quads.len() as u32,
        }),
        Err(err) => {
            tracing::error!("Failed to create GPU buffer for {label}: {err}");
            None
        }
    }
}

struct GpuLodMesh {
    node_pos: [i32; 3],
    level: u8,
    buffer: GpuBuffer,
    quad_count: u32,
    min_aabb: Vec3,
    max_aabb: Vec3,
}

struct App {
    validation: bool,
    view_distance: u32,
    window: Option<Window>,
    gpu_context: Option<GpuContext>,
    depth_buffer: Option<DepthBuffer>,
    texture_array: Option<GpuTextureArray>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_set: Option<vk::DescriptorSet>,
    pipeline: Option<GraphicsPipeline>,
    vert_shader: Option<ShaderModule>,
    frag_shader: Option<ShaderModule>,

    t1_pipeline: Option<GraphicsPipeline>,
    t1_vert_shader: Option<ShaderModule>,
    t1_frag_shader: Option<ShaderModule>,

    cutout_pipeline: Option<GraphicsPipeline>,
    cutout_vert_shader: Option<ShaderModule>,
    cutout_frag_shader: Option<ShaderModule>,

    translucent_pipeline: Option<GraphicsPipeline>,
    translucent_vert_shader: Option<ShaderModule>,
    translucent_frag_shader: Option<ShaderModule>,

    water_anim_info: AnimatedTextureInfo,
    frame_tick: u32,

    lod_pipeline: Option<GraphicsPipeline>,
    lod_vert_shader: Option<ShaderModule>,
    lod_frag_shader: Option<ShaderModule>,

    highlight_pipeline: Option<GraphicsPipeline>,
    highlight_vert_shader: Option<ShaderModule>,
    highlight_frag_shader: Option<ShaderModule>,

    targeted_block: Option<RaycastHit>,
    selected_hotbar_slot: usize,
    selected_block_state: BlockStateId,
    action_sequence: u32,

    server_running: Arc<AtomicBool>,
    server_handle: Option<std::thread::JoinHandle<()>>,
    client_conn: Box<dyn Connection<C2sMessage, S2cMessage>>,
    client_phase: ConnectionPhase,

    block_registry: BlockRegistry,
    chunks: HashMap<ChunkPos, Arc<ChunkSnapshot>>,
    chunk_meshes: HashMap<ChunkPos, GpuChunkMesh>,
    dirty_chunks: HashSet<ChunkPos>,

    lod_meshes: HashMap<vx_lod::coords::LodNodeKey, GpuLodMesh>,
    pending_lod_uploads: HashMap<vx_lod::coords::LodNodeKey, (u32, Vec<u32>)>,

    camera: Camera,
    controller: FlyController,
    last_sent_pos: Vec3,
    last_sent_time: Instant,
    last_frame_time: Instant,
    last_fps_time: Instant,
    frame_counter: u32,
    visible_chunks_last: usize,
    visible_lod_nodes_last: usize,
}

impl App {
    fn new(validation: bool, seed: u64, view_distance: u32, world_dir: PathBuf) -> Self {
        let spawn_pos = Vec3::new(128.0, 45.0, 160.0);
        let mut camera = Camera::new(spawn_pos);
        camera.pitch = -0.3;

        let server_running = Arc::new(AtomicBool::new(true));
        let running_clone = server_running.clone();

        let (server_conn, client_conn) = MemoryConnection::pair_default();
        let server_conn: Box<dyn Connection<S2cMessage, C2sMessage>> = Box::new(server_conn);
        let client_conn: Box<dyn Connection<C2sMessage, S2cMessage>> = Box::new(client_conn);

        let server_handle = std::thread::Builder::new()
            .name("voxel-server".into())
            .spawn(move || {
                let config = vx_server::ServerConfig {
                    tps: 20,
                    view_distance,
                    vertical_view_distance: 2,
                    chunks_per_tick_per_player: 16,
                    save_directory: Some(world_dir),
                    ..Default::default()
                };
                let mut server = vx_server::Server::new(seed, config);
                server.add_connection(server_conn);

                let mut timestep = FixedTimestep::new(20);
                while running_clone.load(Ordering::Relaxed) {
                    timestep.advance(|_| {
                        server.tick();
                    });
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("Failed to spawn background singleplayer server thread");

        // Send initial Hello handshake
        let _ = client_conn.send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        );

        Self {
            validation,
            view_distance,
            window: None,
            gpu_context: None,
            depth_buffer: None,
            texture_array: None,
            descriptor_pool: None,
            descriptor_set_layout: None,
            descriptor_set: None,
            pipeline: None,
            vert_shader: None,
            frag_shader: None,

            t1_pipeline: None,
            t1_vert_shader: None,
            t1_frag_shader: None,

            cutout_pipeline: None,
            cutout_vert_shader: None,
            cutout_frag_shader: None,

            translucent_pipeline: None,
            translucent_vert_shader: None,
            translucent_frag_shader: None,

            water_anim_info: AnimatedTextureInfo {
                base_layer: 9,
                frame_count: 32,
                frame_time: 2,
            },
            frame_tick: 0,

            lod_pipeline: None,
            lod_vert_shader: None,
            lod_frag_shader: None,

            highlight_pipeline: None,
            highlight_vert_shader: None,
            highlight_frag_shader: None,

            targeted_block: None,
            selected_hotbar_slot: 0,
            selected_block_state: HOTBAR_ITEMS[0].1,
            action_sequence: 0,

            server_running,
            server_handle: Some(server_handle),
            client_conn,
            client_phase: ConnectionPhase::Hello,

            block_registry: BlockRegistry::standard(),
            chunks: HashMap::new(),
            chunk_meshes: HashMap::new(),
            dirty_chunks: HashSet::new(),

            lod_meshes: HashMap::new(),
            pending_lod_uploads: HashMap::new(),

            camera,
            controller: FlyController::default(),
            last_sent_pos: spawn_pos,
            last_sent_time: Instant::now(),
            last_frame_time: Instant::now(),
            last_fps_time: Instant::now(),
            frame_counter: 0,
            visible_chunks_last: 0,
            visible_lod_nodes_last: 0,
        }
    }

    fn select_hotbar_slot(&mut self, slot: usize) {
        if slot < HOTBAR_ITEMS.len() {
            self.selected_hotbar_slot = slot;
            self.selected_block_state = HOTBAR_ITEMS[slot].1;
            info!(
                slot = slot + 1,
                item = HOTBAR_ITEMS[slot].0,
                state_id = self.selected_block_state.as_u32(),
                "Selected hotbar slot"
            );
        }
    }

    fn apply_block_update(&mut self, pos: BlockPos, state_id: BlockStateId) {
        let (chunk_pos, local_idx) = vx_voxel::coords::split_block_pos(pos);
        if let Some(snap) = self.chunks.get(&chunk_pos) {
            let old_state = snap.blocks().get(local_idx);
            if old_state == state_id {
                return;
            }

            let registry = &self.block_registry;
            let mut chunk = Chunk::from_blocks(chunk_pos, snap.blocks().clone(), |s| {
                registry
                    .flags(s)
                    .contains(vx_voxel::state::StateFlags::OPAQUE_FULL)
            });
            if let Some(light) = snap.light() {
                chunk.set_light(Some(light.clone()));
            }
            let old_flags = registry.flags(old_state);
            let new_flags = registry.flags(state_id);
            chunk.set(local_idx, state_id, old_flags, new_flags, 1);
            self.chunks.insert(chunk_pos, chunk.publish_snapshot());
            self.mark_dirty_with_neighbors(chunk_pos);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn poll_network(&mut self) {
        while let Ok(Some(incoming)) = self.client_conn.try_recv() {
            let Some(msg) = incoming.into_msg() else {
                continue;
            };

            match self.client_phase {
                ConnectionPhase::Hello => {
                    if let S2cMessage::HelloReply(reply) = msg {
                        info!(
                            protocol = reply.protocol,
                            "Server accepted Hello, logging in..."
                        );
                        let login = C2sMessage::LoginStart(C2sLoginStart {
                            username: BoundedString::new("Player").unwrap(),
                            mode: AuthMode::Offline,
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(login));
                        self.client_phase = ConnectionPhase::Login;
                    }
                }
                ConnectionPhase::Login => {
                    if let S2cMessage::LoginSuccess(succ) = msg {
                        info!(
                            username = succ.username.as_str(),
                            "Login successful, configuring client..."
                        );
                        let settings = C2sMessage::ClientSettings(C2sClientSettings {
                            view_distance: self.view_distance as u16,
                            simulation_distance: self.view_distance as u16,
                            locale: BoundedString::new("en_US").unwrap(),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(settings));
                        let ack = C2sMessage::ConfigAck(C2sConfigAck);
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(ack));
                        self.client_phase = ConnectionPhase::Config;
                    }
                }
                ConnectionPhase::Config => {
                    if let S2cMessage::JoinGame(join) = msg {
                        info!(
                            entity_id = join.entity_id,
                            view_distance = join.view_distance,
                            spawn_x = join.spawn_x,
                            spawn_y = join.spawn_y,
                            spawn_z = join.spawn_z,
                            "Joined world, starting chunk streaming"
                        );
                        self.client_phase = ConnectionPhase::Play;
                    }
                }
                ConnectionPhase::Play => match msg {
                    S2cMessage::UniformChunk(uniform) => {
                        let pos = ChunkPos::new(uniform.chunk_x, uniform.chunk_y, uniform.chunk_z);
                        let is_opaque = self
                            .block_registry
                            .flags(uniform.block_state)
                            .contains(vx_voxel::state::StateFlags::OPAQUE_FULL);
                        let light = Some(ChunkLight::new_uniform(
                            uniform.sky_light,
                            uniform.block_light,
                        ));
                        let snap = Arc::new(ChunkSnapshot::new_uniform(
                            pos,
                            uniform.block_state,
                            is_opaque,
                            light,
                        ));

                        self.chunks.insert(pos, snap);
                        self.mark_dirty_with_neighbors(pos);
                    }
                    S2cMessage::ChunkData(data) => {
                        let pos = ChunkPos::new(data.chunk_x, data.chunk_y, data.chunk_z);
                        let snap = match data.payload {
                            ChunkPayload::Snapshot(s) => s,
                            ChunkPayload::Wire(bytes) => {
                                let mut cursor = bytes.as_slice();
                                match vx_protocol::messages::decode_chunk_snapshot(pos, &mut cursor)
                                {
                                    Ok(s) => Arc::new(s),
                                    Err(err) => {
                                        tracing::error!("Failed to decode wire chunk: {err}");
                                        continue;
                                    }
                                }
                            }
                        };
                        self.chunks.insert(pos, snap);
                        self.mark_dirty_with_neighbors(pos);
                    }
                    S2cMessage::ChunkUnload(unload) => {
                        let pos = ChunkPos::new(unload.chunk_x, unload.chunk_y, unload.chunk_z);
                        self.chunks.remove(&pos);
                        if let (Some(mesh), Some(ctx)) =
                            (self.chunk_meshes.remove(&pos), &self.gpu_context)
                        {
                            mesh.destroy(ctx);
                        }
                        self.mark_dirty_neighbors(pos);
                    }
                    S2cMessage::LodNodeData(lod) => {
                        let key = vx_lod::coords::LodNodeKey::new(
                            lod.level, lod.node_x, lod.node_y, lod.node_z,
                        );
                        let words = lod.payload.to_words();
                        if !words.is_empty() {
                            self.pending_lod_uploads
                                .insert(key, (lod.quad_count, words));
                        } else if let (Some(mut mesh), Some(ctx)) =
                            (self.lod_meshes.remove(&key), &self.gpu_context)
                        {
                            mesh.buffer.destroy(ctx.device().raw(), ctx.allocator());
                        }
                    }
                    S2cMessage::LodNodeUnload(unload) => {
                        let key = vx_lod::coords::LodNodeKey::new(
                            unload.level,
                            unload.node_x,
                            unload.node_y,
                            unload.node_z,
                        );
                        self.pending_lod_uploads.remove(&key);
                        if let (Some(mut mesh), Some(ctx)) =
                            (self.lod_meshes.remove(&key), &self.gpu_context)
                        {
                            mesh.buffer.destroy(ctx.device().raw(), ctx.allocator());
                        }
                    }
                    S2cMessage::BlockUpdate(upd) => {
                        let pos = BlockPos::new(upd.x, upd.y, upd.z);
                        self.apply_block_update(pos, upd.state_id);
                    }
                    S2cMessage::BlockActionAck(ack) => {
                        tracing::trace!(
                            sequence = ack.sequence,
                            "Server acknowledged block action"
                        );
                    }
                    _ => {}
                },
            }
        }
    }

    fn send_player_position_if_needed(&mut self) {
        if self.client_phase != ConnectionPhase::Play {
            return;
        }

        let moved = self.camera.position.distance_squared(self.last_sent_pos) > 0.05;
        let time_elapsed = self.last_sent_time.elapsed() >= Duration::from_millis(50);

        if moved || time_elapsed {
            let msg = C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: f64::from(self.camera.position.x),
                y: f64::from(self.camera.position.y),
                z: f64::from(self.camera.position.z),
                yaw: self.camera.yaw.to_degrees(),
                pitch: self.camera.pitch.to_degrees(),
                on_ground: true,
            });
            let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
            self.last_sent_pos = self.camera.position;
            self.last_sent_time = Instant::now();
        }
    }

    fn mark_dirty_with_neighbors(&mut self, pos: ChunkPos) {
        self.dirty_chunks.insert(pos);
        self.mark_dirty_neighbors(pos);
    }

    fn mark_dirty_neighbors(&mut self, pos: ChunkPos) {
        let offsets = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];
        for (dx, dy, dz) in offsets {
            let neighbor = ChunkPos::new(pos.x() + dx, pos.y() + dy, pos.z() + dz);
            if self.chunks.contains_key(&neighbor) {
                self.dirty_chunks.insert(neighbor);
            }
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::similar_names
    )]
    fn rebuild_dirty_meshes(&mut self, budget: usize) {
        let Some(gpu_context) = &self.gpu_context else {
            return;
        };

        let cam_cx = (self.camera.position.x / 32.0).floor() as i32;
        let cam_cy = (self.camera.position.y / 32.0).floor() as i32;
        let cam_cz = (self.camera.position.z / 32.0).floor() as i32;

        let mut processed = 0;
        while processed < budget {
            // Prioritize dirty chunks closest to the player's view
            let Some(&pos) = self.dirty_chunks.iter().min_by_key(|p| {
                let dx = p.x() - cam_cx;
                let dy = p.y() - cam_cy;
                let dz = p.z() - cam_cz;
                dx * dx + dy * dy * 4 + dz * dz
            }) else {
                break;
            };

            self.dirty_chunks.remove(&pos);
            processed += 1;

            let Some(chunk) = self.chunks.get(&pos).cloned() else {
                if let Some(mesh) = self.chunk_meshes.remove(&pos) {
                    mesh.destroy(gpu_context);
                }
                continue;
            };

            // Fast path: uniform air requires no mesh
            if chunk.blocks().is_uniform()
                && chunk.blocks().get(LocalIdx::from_coords_unchecked(0, 0, 0)) == BlockStateId::AIR
            {
                if let Some(mesh) = self.chunk_meshes.remove(&pos) {
                    mesh.destroy(gpu_context);
                }
                continue;
            }

            let pos_x = self
                .chunks
                .get(&ChunkPos::new(pos.x() + 1, pos.y(), pos.z()))
                .map(std::convert::AsRef::as_ref);
            let neg_x = self
                .chunks
                .get(&ChunkPos::new(pos.x() - 1, pos.y(), pos.z()))
                .map(std::convert::AsRef::as_ref);
            let pos_y = self
                .chunks
                .get(&ChunkPos::new(pos.x(), pos.y() + 1, pos.z()))
                .map(std::convert::AsRef::as_ref);
            let neg_y = self
                .chunks
                .get(&ChunkPos::new(pos.x(), pos.y() - 1, pos.z()))
                .map(std::convert::AsRef::as_ref);
            let pos_z = self
                .chunks
                .get(&ChunkPos::new(pos.x(), pos.y(), pos.z() + 1))
                .map(std::convert::AsRef::as_ref);
            let neg_z = self
                .chunks
                .get(&ChunkPos::new(pos.x(), pos.y(), pos.z() - 1))
                .map(std::convert::AsRef::as_ref);

            let neighbors = [pos_x, neg_x, pos_y, neg_y, pos_z, neg_z];
            let layers =
                vx_mesh::mesher::mesh_chunk_multilayers(&chunk, &neighbors, &self.block_registry);

            let opaque = upload_t0_layer(gpu_context, &layers.opaque, "chunk_mesh_opaque");
            let t1_opaque = upload_t1_layer(gpu_context, &layers.t1_opaque, "chunk_mesh_t1");
            let cutout = upload_t0_layer(gpu_context, &layers.cutout, "chunk_mesh_cutout");
            let translucent =
                upload_t0_layer(gpu_context, &layers.translucent, "chunk_mesh_translucent");

            if opaque.is_none() && t1_opaque.is_none() && cutout.is_none() && translucent.is_none()
            {
                if let Some(old_mesh) = self.chunk_meshes.remove(&pos) {
                    old_mesh.destroy(gpu_context);
                }
            } else {
                let (cx, cy, cz) = (pos.x(), pos.y(), pos.z());
                let min_aabb = Vec3::new((cx * 32) as f32, (cy * 32) as f32, (cz * 32) as f32);
                let max_aabb = Vec3::new(
                    ((cx + 1) * 32) as f32,
                    ((cy + 1) * 32) as f32,
                    ((cz + 1) * 32) as f32,
                );

                let new_mesh = GpuChunkMesh {
                    pos: [cx * 32, cy * 32, cz * 32],
                    min_aabb,
                    max_aabb,
                    opaque,
                    t1_opaque,
                    cutout,
                    translucent,
                };

                if let Some(old_mesh) = self.chunk_meshes.insert(pos, new_mesh) {
                    old_mesh.destroy(gpu_context);
                }
            }
        }
    }

    fn upload_pending_lod_meshes(&mut self) {
        let Some(gpu_context) = &mut self.gpu_context else {
            return;
        };
        if self.pending_lod_uploads.is_empty() {
            return;
        }

        let device = gpu_context.device().raw();
        let allocator = gpu_context.allocator();

        let pending: Vec<_> = self.pending_lod_uploads.drain().collect();
        for (key, (quad_count, words)) in pending {
            if quad_count == 0 || words.is_empty() {
                if let Some(mut old_mesh) = self.lod_meshes.remove(&key) {
                    old_mesh.buffer.destroy(device, allocator);
                }
                continue;
            }

            match gpu_context.create_buffer_with_data(
                "lod_mesh",
                &words,
                vk::BufferUsageFlags::empty(),
            ) {
                Ok(buffer) => {
                    let shift = 5 + i32::from(key.level);
                    let node_size = (1 << shift) as f32;
                    let min_aabb = Vec3::new(
                        (key.x << shift) as f32,
                        (key.y << shift) as f32,
                        (key.z << shift) as f32,
                    );
                    let max_aabb = min_aabb + Vec3::splat(node_size);

                    let new_mesh = GpuLodMesh {
                        node_pos: [key.x, key.y, key.z],
                        level: key.level,
                        buffer,
                        quad_count,
                        min_aabb,
                        max_aabb,
                    };

                    if let Some(mut old_mesh) = self.lod_meshes.insert(key, new_mesh) {
                        old_mesh.buffer.destroy(device, allocator);
                    }
                }
                Err(err) => {
                    tracing::error!("Failed to create GPU buffer for LOD node {key:?}: {err}");
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn render(&mut self) {
        self.poll_network();
        self.send_player_position_if_needed();
        self.rebuild_dirty_meshes(16);
        self.upload_pending_lod_meshes();

        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;

        self.controller.update(&mut self.camera, dt);

        // Voxel DDA Raycast for block aiming & selection
        let origin = self.camera.position;
        let forward = self.camera.forward();
        let max_reach = 5.0;
        let is_solid = |pos: BlockPos| -> bool {
            let (chunk_pos, local_idx) = vx_voxel::coords::split_block_pos(pos);
            if let Some(snap) = self.chunks.get(&chunk_pos) {
                let state = snap.blocks().get(local_idx);
                !self
                    .block_registry
                    .flags(state)
                    .contains(vx_voxel::state::StateFlags::AIR)
            } else {
                false
            }
        };
        self.targeted_block = raycast_voxels(origin, forward, max_reach, is_solid);

        self.frame_counter += 1;
        self.frame_tick = self.frame_tick.wrapping_add(1);
        if now.duration_since(self.last_fps_time) >= Duration::from_secs(1) {
            let fps = self.frame_counter;
            self.frame_counter = 0;
            self.last_fps_time = now;
            info!(
                fps,
                visible = self.visible_chunks_last,
                visible_lods = self.visible_lod_nodes_last,
                loaded_chunks = self.chunks.len(),
                lod_meshes = self.lod_meshes.len(),
                gpu_meshes = self.chunk_meshes.len(),
                dirty_queue = self.dirty_chunks.len(),
                pos = ?self.camera.position,
                "Performance metrics"
            );
        }

        let (Some(gpu_context), Some(pipeline), Some(depth_buffer), Some(descriptor_set)) = (
            &mut self.gpu_context,
            &self.pipeline,
            &mut self.depth_buffer,
            self.descriptor_set,
        ) else {
            return;
        };

        let frame_data = match gpu_context.begin_frame() {
            Ok(Some(data)) => data,
            Ok(None) => return,
            Err(err) => {
                tracing::error!("begin_frame failed: {err}");
                return;
            }
        };

        let (cmd, image_index) = frame_data;
        let swapchain_extent = gpu_context.extent();
        let image_view = gpu_context.swapchain().image_view(image_index as usize);

        #[allow(clippy::cast_precision_loss)]
        let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        let frustum = Frustum::from_view_proj(&view_proj);

        let device = gpu_context.device().raw();

        // SAFETY: Recording render barriers and drawing commands into active command buffer
        unsafe {
            // Transition depth image to DEPTH_ATTACHMENT_OPTIMAL
            let depth_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(depth_buffer.raw())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let depth_barriers = [depth_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&depth_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.52, 0.73, 0.95, 1.0], // Sky blue clear color
                    },
                });

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_buffer.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);

            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.raw());

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                0,
                &[descriptor_set],
                &[],
            );

            // Negative-height viewport: Vulkan clip space (+Y down) maps CCW to CCW
            #[allow(clippy::cast_precision_loss)]
            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(swapchain_extent.height as f32)
                .width(swapchain_extent.width as f32)
                .height(-(swapchain_extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Pass 1: Opaque T0 Chunks
            let mut visible_chunks = 0;

            for mesh in self.chunk_meshes.values() {
                if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                    continue;
                }

                visible_chunks += 1;

                if let Some(layer) = &mesh.opaque {
                    let pc = ChunkPushConstants {
                        view_proj,
                        chunk_pos: mesh.pos,
                        pattern_offset: layer.pattern_offset,
                        quad_buffer_address: layer.buffer.device_address(),
                    };

                    let pc_bytes = std::slice::from_raw_parts(
                        std::ptr::from_ref(&pc).cast::<u8>(),
                        size_of::<ChunkPushConstants>(),
                    );

                    device.cmd_push_constants(
                        cmd,
                        pipeline.layout(),
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        pc_bytes,
                    );

                    device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                }
            }

            self.visible_chunks_last = visible_chunks;

            // Pass 2: Opaque T1 Sub-Cubes (Slabs & Stairs)
            if let Some(t1_pipeline) = &self.t1_pipeline {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, t1_pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    t1_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );

                for mesh in self.chunk_meshes.values() {
                    if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                        continue;
                    }

                    if let Some(layer) = &mesh.t1_opaque {
                        let pc = ChunkPushConstants {
                            view_proj,
                            chunk_pos: mesh.pos,
                            pattern_offset: layer.pattern_offset,
                            quad_buffer_address: layer.buffer.device_address(),
                        };

                        let pc_bytes = std::slice::from_raw_parts(
                            std::ptr::from_ref(&pc).cast::<u8>(),
                            size_of::<ChunkPushConstants>(),
                        );

                        device.cmd_push_constants(
                            cmd,
                            t1_pipeline.layout(),
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            pc_bytes,
                        );

                        device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                    }
                }
            }

            // Pass 3: Cutout T0 (Leaves & Glass with alpha discard)
            if let Some(cutout_pipeline) = &self.cutout_pipeline {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    cutout_pipeline.raw(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    cutout_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );

                for mesh in self.chunk_meshes.values() {
                    if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                        continue;
                    }

                    if let Some(layer) = &mesh.cutout {
                        let pc = ChunkPushConstants {
                            view_proj,
                            chunk_pos: mesh.pos,
                            pattern_offset: layer.pattern_offset,
                            quad_buffer_address: layer.buffer.device_address(),
                        };

                        let pc_bytes = std::slice::from_raw_parts(
                            std::ptr::from_ref(&pc).cast::<u8>(),
                            size_of::<ChunkPushConstants>(),
                        );

                        device.cmd_push_constants(
                            cmd,
                            cutout_pipeline.layout(),
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            pc_bytes,
                        );

                        device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                    }
                }
            }

            // Pass 4: Far-field LOD draws
            if let Some(lod_pipeline) = &self.lod_pipeline {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, lod_pipeline.raw());
                let mut visible_lods = 0;
                let max_dist = (self.view_distance as f32 * 32.0) * 4.0;

                for mesh in self.lod_meshes.values() {
                    if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                        continue;
                    }

                    visible_lods += 1;

                    let pc = LodPushConstants {
                        view_proj,
                        node_pos: mesh.node_pos,
                        level: u32::from(mesh.level),
                        quad_count: mesh.quad_count,
                        buffer_address: mesh.buffer.device_address(),
                        camera_pos: self.camera.position,
                        max_distance: max_dist,
                    };

                    let pc_bytes = std::slice::from_raw_parts(
                        std::ptr::from_ref(&pc).cast::<u8>(),
                        size_of::<LodPushConstants>(),
                    );

                    device.cmd_push_constants(
                        cmd,
                        lod_pipeline.layout(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        pc_bytes,
                    );

                    device.cmd_draw(cmd, mesh.quad_count * 6, 1, 0, 0);
                }

                self.visible_lod_nodes_last = visible_lods;
            }

            // Pass 5: Translucent (Water with alpha blending and animated frames)
            if let Some(trans_pipeline) = &self.translucent_pipeline {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    trans_pipeline.raw(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    trans_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );

                for mesh in self.chunk_meshes.values() {
                    if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                        continue;
                    }

                    if let Some(layer) = &mesh.translucent {
                        let pc = TranslucentPushConstants {
                            view_proj,
                            chunk_pos: mesh.pos,
                            pattern_offset: layer.pattern_offset,
                            quad_buffer_address: layer.buffer.device_address(),
                            frame_tick: self.frame_tick,
                            water_base_layer: self.water_anim_info.base_layer,
                            water_frame_count: self.water_anim_info.frame_count,
                            _pad: [0; 3],
                        };

                        let pc_bytes = std::slice::from_raw_parts(
                            std::ptr::from_ref(&pc).cast::<u8>(),
                            size_of::<TranslucentPushConstants>(),
                        );

                        device.cmd_push_constants(
                            cmd,
                            trans_pipeline.layout(),
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            pc_bytes,
                        );

                        device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                    }
                }
            }

            // Pass 6: Block selection wireframe highlight (snapped to sub-cube bounds)
            if let (Some(hit), Some(highlight_pipeline)) =
                (self.targeted_block, &self.highlight_pipeline)
            {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    highlight_pipeline.raw(),
                );
                #[allow(clippy::cast_precision_loss)]
                let (bx, by, bz) = (hit.pos.x() as f32, hit.pos.y() as f32, hit.pos.z() as f32);

                let (chunk_pos, local_idx) = vx_voxel::coords::split_block_pos(hit.pos);
                let boxes = if let Some(snap) = self.chunks.get(&chunk_pos) {
                    let state = snap.blocks().get(local_idx);
                    match self.block_registry.shape(state) {
                        BlockShape::Boxes(b) if !b.is_empty() => b.clone(),
                        _ => vec![vx_voxel::shape::SubBox::FULL_CUBE],
                    }
                } else {
                    vec![vx_voxel::shape::SubBox::FULL_CUBE]
                };

                for b in boxes {
                    #[allow(clippy::cast_precision_loss)]
                    let min_bound = [
                        bx + f32::from(b.min[0]) / 16.0 - 0.002,
                        by + f32::from(b.min[1]) / 16.0 - 0.002,
                        bz + f32::from(b.min[2]) / 16.0 - 0.002,
                        0.0,
                    ];
                    #[allow(clippy::cast_precision_loss)]
                    let max_bound = [
                        bx + f32::from(b.max[0]) / 16.0 + 0.002,
                        by + f32::from(b.max[1]) / 16.0 + 0.002,
                        bz + f32::from(b.max[2]) / 16.0 + 0.002,
                        0.0,
                    ];

                    let pc = HighlightPushConstants {
                        view_proj,
                        min_bound,
                        max_bound,
                        color: [0.05, 0.05, 0.05, 0.75],
                    };
                    let pc_bytes = std::slice::from_raw_parts(
                        std::ptr::from_ref(&pc).cast::<u8>(),
                        size_of::<HighlightPushConstants>(),
                    );
                    device.cmd_push_constants(
                        cmd,
                        highlight_pipeline.layout(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        pc_bytes,
                    );
                    device.cmd_draw(cmd, 24, 1, 0, 0);
                }
            }

            device.cmd_end_rendering(cmd);
        }

        if let Err(err) = gpu_context.end_frame(image_index) {
            tracing::error!("end_frame failed: {err}");
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::too_many_lines)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Voxel Engine - Client (Dynamic Chunk Streaming)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = match event_loop.create_window(attributes) {
            Ok(w) => w,
            Err(err) => {
                tracing::error!("Failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        let gpu_context = match GpuContext::new(&window, size.width, size.height, self.validation) {
            Ok(ctx) => ctx,
            Err(err) => {
                tracing::error!("Failed to initialize GpuContext: {err}");
                event_loop.exit();
                return;
            }
        };

        let depth_buffer = match gpu_context.create_depth_buffer() {
            Ok(d) => d,
            Err(err) => {
                tracing::error!("Failed to create DepthBuffer: {err}");
                event_loop.exit();
                return;
            }
        };

        let (texture_array, water_anim_info) = match load_and_upload_textures(&gpu_context) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!("Failed to load and upload texture array: {err}");
                event_loop.exit();
                return;
            }
        };
        self.water_anim_info = water_anim_info;

        // Create Descriptor Set Layout for binding 0 (sampler2DArray)
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = match unsafe {
            gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&layout_info, None)
        } {
            Ok(l) => l,
            Err(err) => {
                tracing::error!("Failed to create descriptor set layout: {err}");
                event_loop.exit();
                return;
            }
        };

        // Create Descriptor Pool
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = match unsafe {
            gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&pool_info, None)
        } {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create descriptor pool: {err}");
                event_loop.exit();
                return;
            }
        };

        // Allocate and write descriptor set
        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_set = match unsafe {
            gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&alloc_info)
        } {
            Ok(sets) => sets[0],
            Err(err) => {
                tracing::error!("Failed to allocate descriptor set: {err}");
                event_loop.exit();
                return;
            }
        };

        let image_info = vk::DescriptorImageInfo::default()
            .sampler(texture_array.sampler())
            .image_view(texture_array.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let image_infos = [image_info];

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        let descriptor_writes = [descriptor_write];
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&descriptor_writes, &[]);
        }

        // Load SPIR-V bytecode generated by build script / xtask
        let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.vert.spv"));
        let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv"));

        let vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), vert_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), frag_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<ChunkPushConstants>() as u32);

        let pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            vert_module.raw(),
            frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[descriptor_set_layout],
            &[push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load T1 SPIR-V bytecode
        let t1_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_t1.vert.spv"));
        let t1_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_t1.frag.spv"));

        let t1_vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), t1_vert_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create T1 vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let t1_frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), t1_frag_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create T1 fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let t1_pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            t1_vert_module.raw(),
            t1_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[descriptor_set_layout],
            &[push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic T1 graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load Cutout SPIR-V bytecode
        let cutout_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/cutout.vert.spv"));
        let cutout_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/cutout.frag.spv"));

        let cutout_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), cutout_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create cutout vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let cutout_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), cutout_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create cutout fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let cutout_pipeline = match GraphicsPipeline::create_dynamic_cutout(
            gpu_context.device().raw(),
            cutout_vert_module.raw(),
            cutout_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[descriptor_set_layout],
            &[push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic cutout graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load Translucent SPIR-V bytecode
        let trans_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/translucent.vert.spv"));
        let trans_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/translucent.frag.spv"));

        let trans_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), trans_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create translucent vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let trans_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), trans_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create translucent fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        #[allow(clippy::cast_possible_truncation)]
        let trans_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<TranslucentPushConstants>() as u32);

        let trans_pipeline = match GraphicsPipeline::create_dynamic_translucent(
            gpu_context.device().raw(),
            trans_vert_module.raw(),
            trans_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[descriptor_set_layout],
            &[trans_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic translucent graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load LOD SPIR-V bytecode
        let lod_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/lod.vert.spv"));
        let lod_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/lod.frag.spv"));

        let lod_vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), lod_vert_spv)
        {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create LOD vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let lod_frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), lod_frag_spv)
        {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create LOD fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let lod_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<LodPushConstants>() as u32);

        let lod_pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            lod_vert_module.raw(),
            lod_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[],
            &[lod_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic LOD graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load highlight SPIR-V bytecode
        let highlight_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/highlight.vert.spv"));
        let highlight_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/highlight.frag.spv"));

        let highlight_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), highlight_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create highlight vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let highlight_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), highlight_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create highlight fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        #[allow(clippy::cast_possible_truncation)]
        let highlight_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<HighlightPushConstants>() as u32);

        let highlight_pipeline = match GraphicsPipeline::create_dynamic_lines(
            gpu_context.device().raw(),
            highlight_vert_module.raw(),
            highlight_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            &[highlight_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic highlight graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        info!("Renderer and window loop successfully initialized, singleplayer streaming active");

        self.pipeline = Some(pipeline);
        self.vert_shader = Some(vert_module);
        self.frag_shader = Some(frag_module);

        self.t1_pipeline = Some(t1_pipeline);
        self.t1_vert_shader = Some(t1_vert_module);
        self.t1_frag_shader = Some(t1_frag_module);

        self.cutout_pipeline = Some(cutout_pipeline);
        self.cutout_vert_shader = Some(cutout_vert_module);
        self.cutout_frag_shader = Some(cutout_frag_module);

        self.translucent_pipeline = Some(trans_pipeline);
        self.translucent_vert_shader = Some(trans_vert_module);
        self.translucent_frag_shader = Some(trans_frag_module);
        self.lod_pipeline = Some(lod_pipeline);
        self.lod_vert_shader = Some(lod_vert_module);
        self.lod_frag_shader = Some(lod_frag_module);
        self.highlight_pipeline = Some(highlight_pipeline);
        self.highlight_vert_shader = Some(highlight_vert_module);
        self.highlight_frag_shader = Some(highlight_frag_module);
        self.depth_buffer = Some(depth_buffer);
        self.texture_array = Some(texture_array);
        self.descriptor_pool = Some(descriptor_pool);
        self.descriptor_set_layout = Some(descriptor_set_layout);
        self.descriptor_set = Some(descriptor_set);
        self.gpu_context = Some(gpu_context);
        self.window = Some(window);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.controller.on_mouse_move(&mut self.camera, dx, dy);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };
        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested, exiting application");
                event_loop.exit();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if self.controller.mouse_captured {
                    match button {
                        MouseButton::Left => {
                            if let Some(hit) = self.targeted_block {
                                self.action_sequence += 1;
                                let msg = C2sMessage::BlockAction(C2sBlockAction {
                                    sequence: self.action_sequence,
                                    action: BlockActionKind::Break,
                                    x: hit.pos.x(),
                                    y: hit.pos.y(),
                                    z: hit.pos.z(),
                                    input_tick: self.frame_counter,
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
                                self.apply_block_update(hit.pos, BlockStateId::AIR);
                            }
                        }
                        MouseButton::Right => {
                            if let Some(hit) = self.targeted_block {
                                let norm = hit.face.normal_ivec();
                                let place_pos = BlockPos::new(
                                    hit.pos.x() + norm.x,
                                    hit.pos.y() + norm.y,
                                    hit.pos.z() + norm.z,
                                );
                                self.action_sequence += 1;
                                let msg = C2sMessage::BlockAction(C2sBlockAction {
                                    sequence: self.action_sequence,
                                    action: BlockActionKind::Place {
                                        state_id: self.selected_block_state,
                                        hit_face: hit.face as u8,
                                    },
                                    x: hit.pos.x(),
                                    y: hit.pos.y(),
                                    z: hit.pos.z(),
                                    input_tick: self.frame_counter,
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
                                self.apply_block_update(place_pos, self.selected_block_state);
                            }
                        }
                        _ => {}
                    }
                } else if button == MouseButton::Left {
                    self.controller.mouse_captured = true;
                    let _ = window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .or_else(|_| {
                            window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        });
                    window.set_cursor_visible(false);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                match code {
                    KeyCode::KeyW => self.controller.forward = pressed,
                    KeyCode::KeyS => self.controller.backward = pressed,
                    KeyCode::KeyA => self.controller.left = pressed,
                    KeyCode::KeyD => self.controller.right = pressed,
                    KeyCode::Space => self.controller.up = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.controller.down = pressed,
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.controller.sprint = pressed;
                    }
                    KeyCode::Digit1 if pressed => self.select_hotbar_slot(0),
                    KeyCode::Digit2 if pressed => self.select_hotbar_slot(1),
                    KeyCode::Digit3 if pressed => self.select_hotbar_slot(2),
                    KeyCode::Digit4 if pressed => self.select_hotbar_slot(3),
                    KeyCode::Digit5 if pressed => self.select_hotbar_slot(4),
                    KeyCode::Digit6 if pressed => self.select_hotbar_slot(5),
                    KeyCode::Digit7 if pressed => self.select_hotbar_slot(6),
                    KeyCode::Digit8 if pressed => self.select_hotbar_slot(7),
                    KeyCode::Digit9 if pressed => self.select_hotbar_slot(8),
                    KeyCode::Escape if pressed => {
                        if self.controller.mouse_captured {
                            self.controller.mouse_captured = false;
                            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        } else {
                            info!("Escape pressed, exiting application");
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0
                    && physical_size.height > 0
                    && let Some(gpu_context) = &mut self.gpu_context
                {
                    if let Err(err) = gpu_context.resize(physical_size.width, physical_size.height)
                    {
                        tracing::error!("Swapchain resize failed: {err}");
                    }
                    if let Some(mut old_depth) = self.depth_buffer.take() {
                        old_depth.destroy(gpu_context.device().raw(), gpu_context.allocator());
                    }
                    match gpu_context.create_depth_buffer() {
                        Ok(d) => self.depth_buffer = Some(d),
                        Err(err) => tracing::error!("Failed to recreate depth buffer: {err}"),
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.server_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.server_handle.take() {
            let _ = handle.join();
        }

        if let Some(gpu_context) = &mut self.gpu_context {
            let _ = gpu_context.wait_idle();

            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(mut pipeline) = self.t1_pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.t1_vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.t1_frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(mut pipeline) = self.cutout_pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.cutout_vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.cutout_frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(mut pipeline) = self.translucent_pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.translucent_vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.translucent_frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(mut pipeline) = self.lod_pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.lod_vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.lod_frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(mut pipeline) = self.highlight_pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.highlight_vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.highlight_frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }

            if let Some(pool) = self.descriptor_pool.take() {
                // SAFETY: Destroying descriptor pool on valid device
                unsafe {
                    gpu_context
                        .device()
                        .raw()
                        .destroy_descriptor_pool(pool, None);
                }
            }

            if let Some(layout) = self.descriptor_set_layout.take() {
                // SAFETY: Destroying descriptor set layout on valid device
                unsafe {
                    gpu_context
                        .device()
                        .raw()
                        .destroy_descriptor_set_layout(layout, None);
                }
            }

            if let Some(mut tex) = self.texture_array.take() {
                tex.destroy(gpu_context.device().raw(), gpu_context.allocator());
            }

            for (_, mesh) in self.chunk_meshes.drain() {
                mesh.destroy(gpu_context);
            }

            for (_, mut mesh) in self.lod_meshes.drain() {
                mesh.buffer
                    .destroy(gpu_context.device().raw(), gpu_context.allocator());
            }

            if let Some(mut depth) = self.depth_buffer.take() {
                depth.destroy(gpu_context.device().raw(), gpu_context.allocator());
            }
        }
    }
}

fn load_and_upload_textures(
    gpu_context: &GpuContext,
) -> Result<(GpuTextureArray, AnimatedTextureInfo)> {
    let mut stack = ResourcePackStack::new();
    stack.add_root("dev-assets/faithful-32x");
    stack.add_root("dev-assets/classic-26.2");
    stack.add_root("assets/voxel");

    let is_faithful = stack
        .find_block_texture("stone")
        .is_some_and(|p| p.to_string_lossy().contains("faithful"));
    let target_res = if is_faithful { 32 } else { 16 };

    info!(
        target_resolution = target_res,
        pack = if is_faithful {
            "Faithful 32x"
        } else {
            "Default 16x"
        },
        "Loading and baking block textures into 2D texture array"
    );

    let mut builder = TextureArrayBuilder::new(target_res);
    builder.insert("stone", stack.load_block_texture("stone")?);
    builder.insert("dirt", stack.load_block_texture("dirt")?);
    builder.insert(
        "grass_block_top",
        stack.load_block_texture("grass_block_top")?,
    );
    builder.insert(
        "grass_block_side",
        stack.load_block_texture("grass_block_side")?,
    );
    builder.insert("bedrock", stack.load_block_texture("bedrock")?);
    builder.insert("sand", stack.load_block_texture("sand")?);
    builder.insert("oak_planks", stack.load_block_texture("oak_planks")?);
    builder.insert("oak_leaves", stack.load_block_texture("oak_leaves")?);
    builder.insert("glass", stack.load_block_texture("glass")?);

    let water_frames = stack.load_animated_block_texture("water_still")?;
    let water_anim = builder.insert_animated("water_still", water_frames, 2);

    let baked = builder.bake();
    info!(
        resolution = baked.resolution,
        layers = baked.layer_count,
        mips = baked.mip_levels,
        "Texture array baked with full mip chains"
    );

    let regions: Vec<TextureMipRegion> = baked
        .copy_regions
        .iter()
        .map(|r| TextureMipRegion {
            buffer_offset: r.buffer_offset,
            layer: r.layer,
            mip_level: r.mip_level,
            width: r.width,
            height: r.height,
        })
        .collect();

    let texture_array = gpu_context.create_texture_array(
        baked.resolution,
        baked.layer_count,
        baked.mip_levels,
        &baked.pixel_data,
        &regions,
    )?;

    Ok((texture_array, water_anim))
}

fn main() -> Result<()> {
    let args = Args::parse();

    init_telemetry(&TelemetryConfig {
        app_name: "voxel-client",
        default_filter: args.log.into(),
        ansi_colors: true,
    });

    info!(
        version = env!("CARGO_PKG_VERSION"),
        seed = args.seed,
        view_distance = args.view_distance,
        world_dir = ?args.world_dir,
        "Starting Voxel client"
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(
        args.validation,
        args.seed,
        args.view_distance,
        args.world_dir,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}
