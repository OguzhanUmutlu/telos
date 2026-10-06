//! Client application executable for the voxel engine.

pub mod camera;
pub mod entity_client;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use camera::{Camera, FlyController};
use clap::Parser;
use entity_client::{ClientEntityStore, EntityPushConstants, EntityVertexGpu};
use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use mimalloc::MiMalloc;
use tracing::info;
use vx_assets::{AnimatedTextureInfo, ResourcePackStack, TextureArrayBuilder};
use vx_content::FrozenRegistries;
use vx_core::{
    BlockPos, FixedTimestep, RaycastHit, TelemetryConfig, coords::ChunkPos, init_telemetry,
    raycast_voxels,
};
use vx_gpu::{
    ComputePipeline, DepthBuffer, GpuBuffer, GpuContext, GpuTexture2d, GpuTextureArray,
    GraphicsPipeline, HiZPyramid, MemoryLocation, ShaderModule, TextureMipRegion, ash, vk,
};
use vx_net::{Connection, Lane, MemoryConnection, Payload};
use vx_protocol::bounded::BoundedString;
use vx_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sClientSettings, C2sConfigAck, C2sHello,
    C2sInteractEntity, C2sInventoryClick, C2sLoginStart, C2sMessage, C2sPlayerCommand,
    C2sPlayerPosition, ChunkPayload, ConnectionPhase, PlayerCommandKind, S2cMessage,
};
use vx_ui::{
    BitmapFont, HudState, UiLayers, UiQuad, UiSlotItem, compute_gui_scale, render_hud,
    render_inventory_screen, slot_at_pos,
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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkCullCandidate {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    active_mask: u32,

    max_x: f32,
    max_y: f32,
    max_z: f32,
    chunk_id: u32,

    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    opaque_pattern_offset: u32,

    opaque_quad_address: u64,
    opaque_quad_count: u32,
    t1_pattern_offset: u32,

    t1_quad_address: u64,
    t1_quad_count: u32,
    cutout_pattern_offset: u32,

    cutout_quad_address: u64,
    cutout_quad_count: u32,
    translucent_pattern_offset: u32,

    translucent_quad_address: u64,
    translucent_quad_count: u32,
    _pad: u32,
}
const _: () = assert!(size_of::<ChunkCullCandidate>() == 112);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LodCullCandidate {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    lod_id: u32,

    max_x: f32,
    max_y: f32,
    max_z: f32,
    level: u32,

    node_x: i32,
    node_y: i32,
    node_z: i32,
    quad_count: u32,

    quad_address: u64,
    _pad0: u32,
    _pad1: u32,
}
const _: () = assert!(size_of::<LodCullCandidate>() == 64);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullPointers {
    candidate_buffer: u64,
    visibility_buffer: u64,
    count_buffer: u64,

    opaque_cmd: u64,
    opaque_draw: u64,

    t1_cmd: u64,
    t1_draw: u64,

    cutout_cmd: u64,
    cutout_draw: u64,

    translucent_cmd: u64,
    translucent_draw: u64,
}
const _: () = assert!(size_of::<CullPointers>() == 88);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(clippy::struct_field_names)]
struct LodPointers {
    candidate_buffer: u64,
    visibility_buffer: u64,
    count_buffer: u64,
    cmd_buffer: u64,
    draw_buffer: u64,
}
const _: () = assert!(size_of::<LodPointers>() == 40);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullPushConstants {
    view_proj: [f32; 16],
    camera_pos: [f32; 3],
    cull_phase: u32,
    hiz_width: u32,
    hiz_height: u32,
    candidate_count: u32,
    is_first_frame: u32,
    pointers_address: u64,
}
const _: () = assert!(size_of::<CullPushConstants>() == 104);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HiZPushConstants {
    in_extent: [u32; 2],
    out_extent: [u32; 2],
    is_mip0: u32,
    _pad: u32,
}
const _: () = assert!(size_of::<HiZPushConstants>() == 24);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainMdiPushConstants {
    view_proj: [f32; 16],
    draw_info_buffer_address: u64,
}
const _: () = assert!(size_of::<TerrainMdiPushConstants>() == 72);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TranslucentMdiPushConstants {
    view_proj: [f32; 16],
    draw_info_buffer_address: u64,
    frame_tick: u32,
    water_base_layer: u32,
    water_frame_count: u32,
    _pad: u32,
}
const _: () = assert!(size_of::<TranslucentMdiPushConstants>() == 88);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LodMdiPushConstants {
    view_proj: [f32; 16],
    camera_pos: [f32; 3],
    max_distance: f32,
    draw_info_buffer_address: u64,
}
const _: () = assert!(size_of::<LodMdiPushConstants>() == 88);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HighlightPushConstants {
    view_proj: [f32; 16],
    min_bound: [f32; 4],
    max_bound: [f32; 4],
    color: [f32; 4],
}
const _: () = assert!(size_of::<HighlightPushConstants>() == 112);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiPushConstants {
    viewport_size: [f32; 2],
    quad_buffer_address: u64,
}
const _: () = assert!(size_of::<UiPushConstants>() == 16);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyPushConstants {
    inv_view_proj: [f32; 16],
    sun_dir: [f32; 3],
    time_of_day: f32,
    moon_dir: [f32; 3],
    moon_phase: u32,
    camera_pos: [f32; 3],
    rain_level: f32,
    thunder_level: f32,
    lightning_flash: f32,
    _pad0: f32,
    _pad1: f32,
}
const _: () = assert!(size_of::<SkyPushConstants>() == 128);

/// GPU representation of an atmospheric precipitation particle (rain streak or snowflake).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WeatherParticleGpu {
    /// World position of the particle center.
    pub pos: [f32; 3],
    /// Billboard quad width.
    pub size_x: f32,
    /// Billboard quad height.
    pub size_y: f32,
    /// Vertical UV animation scroll offset.
    pub uv_anim: f32,
    /// Texture array layer index (0: Rain, 1: Snow).
    pub layer: u32,
    /// RGBA8 tint and opacity color.
    pub color: u32,
}
const _: () = assert!(size_of::<WeatherParticleGpu>() == 32);

/// Push constants for the dynamic weather precipitation particle pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WeatherPushConstants {
    /// Combined view projection matrix.
    pub view_proj: [f32; 16],
    /// Camera right basis vector.
    pub camera_right: [f32; 3],
    /// Alignment padding.
    pub pad0: f32,
    /// Camera up basis vector.
    pub camera_up: [f32; 3],
    /// Alignment padding.
    pub pad1: f32,
    /// 64-bit device address of the weather particle buffer.
    pub particle_buffer_address: u64,
    /// Alignment padding.
    pub pad2: [u32; 2],
}
const _: () = assert!(size_of::<WeatherPushConstants>() == 112);

/// 16x16 RGBA8 Dynamic Lighting Lookup Table (1024 bytes).
#[derive(Clone, Copy)]
pub struct LightmapLut {
    /// Raw RGBA8 texel data for the 16x16 lightmap (X: block light, Y: sky light).
    pub data: [u8; 1024],
}

impl Default for LightmapLut {
    fn default() -> Self {
        let mut lut = Self { data: [0; 1024] };
        lut.update(1.0, 0.0, 0.0, 0.0, 0.0);
        lut
    }
}

impl LightmapLut {
    /// Recomputes the 16x16 lighting lookup table for the given sun elevation, time, and atmospheric weather.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn update(
        &mut self,
        sun_elev: f32,
        time_ticks: f32,
        rain_level: f32,
        thunder_level: f32,
        lightning_flash: f32,
    ) {
        let daylight = vx_core::time::daylight_factor(sun_elev);
        let sunset = vx_core::time::sunset_factor(sun_elev);

        // Bounded torch flicker +-3.5% (strictly within +-4%)
        let flicker = 1.0 + (time_ticks * 0.05).sin() * 0.025 + (time_ticks * 0.13).cos() * 0.01;
        let day_col = Vec3::new(1.0, 1.0, 1.0).lerp(Vec3::new(1.0, 0.55, 0.25), sunset);
        let mut sky_ambient = Vec3::new(0.05, 0.07, 0.11).lerp(day_col, daylight);

        // Weather overcast dimming: rain dims skylight up to 30%, thunder dims further up to 35%
        sky_ambient *= (1.0 - 0.30 * rain_level) * (1.0 - 0.35 * thunder_level);

        let torch_base = Vec3::new(1.0, 0.82, 0.58) * flicker;
        let mut ambient = Vec3::splat(0.035);

        // Lightning flash surge: floods world with bright blue-white atmospheric flash
        if lightning_flash > 0.001 {
            let flash_boost = Vec3::new(0.70, 0.78, 1.0) * (lightning_flash * 0.70);
            ambient += flash_boost;
            sky_ambient = sky_ambient.max(flash_boost);
        }

        for sky in 0..16 {
            let s_norm = sky as f32 / 15.0;
            let s_curve = s_norm * s_norm;
            let sky_comp = sky_ambient * s_curve;

            for block in 0..16 {
                let b_norm = block as f32 / 15.0;
                let b_curve = b_norm * b_norm;
                let block_comp = torch_base * b_curve;

                let color = (ambient + block_comp + sky_comp).clamp(Vec3::ZERO, Vec3::ONE);
                let idx = (sky * 16 + block) * 4;
                self.data[idx] = (color.x * 255.0).round() as u8;
                self.data[idx + 1] = (color.y * 255.0).round() as u8;
                self.data[idx + 2] = (color.z * 255.0).round() as u8;
                self.data[idx + 3] = 255;
            }
        }
    }
}

const MAX_CHUNK_CANDIDATES: usize = 4096;
const MAX_LOD_CANDIDATES: usize = 4096;

struct SlotAllocator<K: std::hash::Hash + Eq> {
    free_slots: Vec<u32>,
    allocated: HashMap<K, u32>,
    next_fresh: u32,
    capacity: u32,
}

impl<K: std::hash::Hash + Eq + Copy> SlotAllocator<K> {
    fn new(capacity: u32) -> Self {
        Self {
            free_slots: Vec::new(),
            allocated: HashMap::new(),
            next_fresh: 0,
            capacity,
        }
    }

    fn get_or_allocate(&mut self, key: K) -> Option<u32> {
        if let Some(&id) = self.allocated.get(&key) {
            return Some(id);
        }
        let id = if let Some(id) = self.free_slots.pop() {
            id
        } else if self.next_fresh < self.capacity {
            let id = self.next_fresh;
            self.next_fresh += 1;
            id
        } else {
            return None;
        };
        self.allocated.insert(key, id);
        Some(id)
    }

    fn release(&mut self, key: &K) -> Option<u32> {
        if let Some(id) = self.allocated.remove(key) {
            self.free_slots.push(id);
            Some(id)
        } else {
            None
        }
    }
}

struct MdiBuffers {
    chunk_candidates: GpuBuffer,
    chunk_visibility: GpuBuffer,
    chunk_counts_early: GpuBuffer,
    chunk_counts_late: GpuBuffer,

    opaque_cmd_early: GpuBuffer,
    opaque_draw_early: GpuBuffer,
    t1_cmd_early: GpuBuffer,
    t1_draw_early: GpuBuffer,
    cutout_cmd_early: GpuBuffer,
    cutout_draw_early: GpuBuffer,

    opaque_cmd_late: GpuBuffer,
    opaque_draw_late: GpuBuffer,
    t1_cmd_late: GpuBuffer,
    t1_draw_late: GpuBuffer,
    cutout_cmd_late: GpuBuffer,
    cutout_draw_late: GpuBuffer,
    translucent_cmd_late: GpuBuffer,
    translucent_draw_late: GpuBuffer,

    cull_ptrs_early: GpuBuffer,
    cull_ptrs_late: GpuBuffer,

    lod_candidates: GpuBuffer,
    lod_visibility: GpuBuffer,
    lod_counts_early: GpuBuffer,
    lod_counts_late: GpuBuffer,

    lod_cmd_early: GpuBuffer,
    lod_draw_early: GpuBuffer,
    lod_cmd_late: GpuBuffer,
    lod_draw_late: GpuBuffer,

    lod_ptrs_early: GpuBuffer,
    lod_ptrs_late: GpuBuffer,
}

impl MdiBuffers {
    #[allow(clippy::too_many_lines)]
    fn new(ctx: &GpuContext) -> Result<Self, vx_gpu::GpuError> {
        let device = ctx.device().raw();
        let allocator = ctx.allocator();

        let chunk_cand_size =
            (MAX_CHUNK_CANDIDATES * size_of::<ChunkCullCandidate>()) as vk::DeviceSize;
        let chunk_candidates = GpuBuffer::new(
            device,
            allocator,
            "chunk_candidates",
            chunk_cand_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;

        let vis_size = (MAX_CHUNK_CANDIDATES * size_of::<u32>()) as vk::DeviceSize;
        let chunk_visibility = GpuBuffer::new(
            device,
            allocator,
            "chunk_visibility",
            vis_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        let count_size = (4 * size_of::<u32>()) as vk::DeviceSize;
        let count_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;

        let chunk_counts_early = GpuBuffer::new(
            device,
            allocator,
            "chunk_counts_early",
            count_size,
            count_usage,
            MemoryLocation::GpuOnly,
        )?;
        let chunk_counts_late = GpuBuffer::new(
            device,
            allocator,
            "chunk_counts_late",
            count_size,
            count_usage,
            MemoryLocation::GpuOnly,
        )?;

        let cmd_size = (MAX_CHUNK_CANDIDATES * 16) as vk::DeviceSize;
        let cmd_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let draw_size = (MAX_CHUNK_CANDIDATES * 32) as vk::DeviceSize;
        let draw_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let opaque_cmd_early = GpuBuffer::new(
            device,
            allocator,
            "opaque_cmd_early",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let opaque_draw_early = GpuBuffer::new(
            device,
            allocator,
            "opaque_draw_early",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let t1_cmd_early = GpuBuffer::new(
            device,
            allocator,
            "t1_cmd_early",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let t1_draw_early = GpuBuffer::new(
            device,
            allocator,
            "t1_draw_early",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let cutout_cmd_early = GpuBuffer::new(
            device,
            allocator,
            "cutout_cmd_early",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let cutout_draw_early = GpuBuffer::new(
            device,
            allocator,
            "cutout_draw_early",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;

        let opaque_cmd_late = GpuBuffer::new(
            device,
            allocator,
            "opaque_cmd_late",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let opaque_draw_late = GpuBuffer::new(
            device,
            allocator,
            "opaque_draw_late",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let t1_cmd_late = GpuBuffer::new(
            device,
            allocator,
            "t1_cmd_late",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let t1_draw_late = GpuBuffer::new(
            device,
            allocator,
            "t1_draw_late",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let cutout_cmd_late = GpuBuffer::new(
            device,
            allocator,
            "cutout_cmd_late",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let cutout_draw_late = GpuBuffer::new(
            device,
            allocator,
            "cutout_draw_late",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let translucent_cmd_late = GpuBuffer::new(
            device,
            allocator,
            "translucent_cmd_late",
            cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let translucent_draw_late = GpuBuffer::new(
            device,
            allocator,
            "translucent_draw_late",
            draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;

        let ptrs_size = size_of::<CullPointers>() as vk::DeviceSize;
        let mut cull_ptrs_early = GpuBuffer::new(
            device,
            allocator,
            "cull_ptrs_early",
            ptrs_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        let mut cull_ptrs_late = GpuBuffer::new(
            device,
            allocator,
            "cull_ptrs_late",
            ptrs_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;

        let early_ptrs = CullPointers {
            candidate_buffer: chunk_candidates.device_address(),
            visibility_buffer: chunk_visibility.device_address(),
            count_buffer: chunk_counts_early.device_address(),
            opaque_cmd: opaque_cmd_early.device_address(),
            opaque_draw: opaque_draw_early.device_address(),
            t1_cmd: t1_cmd_early.device_address(),
            t1_draw: t1_draw_early.device_address(),
            cutout_cmd: cutout_cmd_early.device_address(),
            cutout_draw: cutout_draw_early.device_address(),
            translucent_cmd: 0,
            translucent_draw: 0,
        };
        cull_ptrs_early.write_bytes(bytemuck::bytes_of(&early_ptrs))?;

        let late_ptrs = CullPointers {
            candidate_buffer: chunk_candidates.device_address(),
            visibility_buffer: chunk_visibility.device_address(),
            count_buffer: chunk_counts_late.device_address(),
            opaque_cmd: opaque_cmd_late.device_address(),
            opaque_draw: opaque_draw_late.device_address(),
            t1_cmd: t1_cmd_late.device_address(),
            t1_draw: t1_draw_late.device_address(),
            cutout_cmd: cutout_cmd_late.device_address(),
            cutout_draw: cutout_draw_late.device_address(),
            translucent_cmd: translucent_cmd_late.device_address(),
            translucent_draw: translucent_draw_late.device_address(),
        };
        cull_ptrs_late.write_bytes(bytemuck::bytes_of(&late_ptrs))?;

        // LOD buffers
        let lod_cand_size = (MAX_LOD_CANDIDATES * size_of::<LodCullCandidate>()) as vk::DeviceSize;
        let lod_candidates = GpuBuffer::new(
            device,
            allocator,
            "lod_candidates",
            lod_cand_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;

        let lod_vis_size = (MAX_LOD_CANDIDATES * size_of::<u32>()) as vk::DeviceSize;
        let lod_visibility = GpuBuffer::new(
            device,
            allocator,
            "lod_visibility",
            lod_vis_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        let lod_counts_early = GpuBuffer::new(
            device,
            allocator,
            "lod_counts_early",
            count_size,
            count_usage,
            MemoryLocation::GpuOnly,
        )?;
        let lod_counts_late = GpuBuffer::new(
            device,
            allocator,
            "lod_counts_late",
            count_size,
            count_usage,
            MemoryLocation::GpuOnly,
        )?;

        let lod_cmd_size = (MAX_LOD_CANDIDATES * 16) as vk::DeviceSize;
        let lod_draw_size = (MAX_LOD_CANDIDATES * 32) as vk::DeviceSize;

        let lod_cmd_early = GpuBuffer::new(
            device,
            allocator,
            "lod_cmd_early",
            lod_cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let lod_draw_early = GpuBuffer::new(
            device,
            allocator,
            "lod_draw_early",
            lod_draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;
        let lod_cmd_late = GpuBuffer::new(
            device,
            allocator,
            "lod_cmd_late",
            lod_cmd_size,
            cmd_usage,
            MemoryLocation::GpuOnly,
        )?;
        let lod_draw_late = GpuBuffer::new(
            device,
            allocator,
            "lod_draw_late",
            lod_draw_size,
            draw_usage,
            MemoryLocation::GpuOnly,
        )?;

        let lod_ptrs_size = size_of::<LodPointers>() as vk::DeviceSize;
        let mut lod_ptrs_early = GpuBuffer::new(
            device,
            allocator,
            "lod_ptrs_early",
            lod_ptrs_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        let mut lod_ptrs_late = GpuBuffer::new(
            device,
            allocator,
            "lod_ptrs_late",
            lod_ptrs_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;

        let early_l_ptrs = LodPointers {
            candidate_buffer: lod_candidates.device_address(),
            visibility_buffer: lod_visibility.device_address(),
            count_buffer: lod_counts_early.device_address(),
            cmd_buffer: lod_cmd_early.device_address(),
            draw_buffer: lod_draw_early.device_address(),
        };
        lod_ptrs_early.write_bytes(bytemuck::bytes_of(&early_l_ptrs))?;

        let late_l_ptrs = LodPointers {
            candidate_buffer: lod_candidates.device_address(),
            visibility_buffer: lod_visibility.device_address(),
            count_buffer: lod_counts_late.device_address(),
            cmd_buffer: lod_cmd_late.device_address(),
            draw_buffer: lod_draw_late.device_address(),
        };
        lod_ptrs_late.write_bytes(bytemuck::bytes_of(&late_l_ptrs))?;

        Ok(Self {
            chunk_candidates,
            chunk_visibility,
            chunk_counts_early,
            chunk_counts_late,
            opaque_cmd_early,
            opaque_draw_early,
            t1_cmd_early,
            t1_draw_early,
            cutout_cmd_early,
            cutout_draw_early,
            opaque_cmd_late,
            opaque_draw_late,
            t1_cmd_late,
            t1_draw_late,
            cutout_cmd_late,
            cutout_draw_late,
            translucent_cmd_late,
            translucent_draw_late,
            cull_ptrs_early,
            cull_ptrs_late,
            lod_candidates,
            lod_visibility,
            lod_counts_early,
            lod_counts_late,
            lod_cmd_early,
            lod_draw_early,
            lod_cmd_late,
            lod_draw_late,
            lod_ptrs_early,
            lod_ptrs_late,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &vx_gpu::GpuAllocator) {
        self.chunk_candidates.destroy(device, allocator);
        self.chunk_visibility.destroy(device, allocator);
        self.chunk_counts_early.destroy(device, allocator);
        self.chunk_counts_late.destroy(device, allocator);
        self.opaque_cmd_early.destroy(device, allocator);
        self.opaque_draw_early.destroy(device, allocator);
        self.t1_cmd_early.destroy(device, allocator);
        self.t1_draw_early.destroy(device, allocator);
        self.cutout_cmd_early.destroy(device, allocator);
        self.cutout_draw_early.destroy(device, allocator);
        self.opaque_cmd_late.destroy(device, allocator);
        self.opaque_draw_late.destroy(device, allocator);
        self.t1_cmd_late.destroy(device, allocator);
        self.t1_draw_late.destroy(device, allocator);
        self.cutout_cmd_late.destroy(device, allocator);
        self.cutout_draw_late.destroy(device, allocator);
        self.translucent_cmd_late.destroy(device, allocator);
        self.translucent_draw_late.destroy(device, allocator);
        self.cull_ptrs_early.destroy(device, allocator);
        self.cull_ptrs_late.destroy(device, allocator);
        self.lod_candidates.destroy(device, allocator);
        self.lod_visibility.destroy(device, allocator);
        self.lod_counts_early.destroy(device, allocator);
        self.lod_counts_late.destroy(device, allocator);
        self.lod_cmd_early.destroy(device, allocator);
        self.lod_draw_early.destroy(device, allocator);
        self.lod_cmd_late.destroy(device, allocator);
        self.lod_draw_late.destroy(device, allocator);
        self.lod_ptrs_early.destroy(device, allocator);
        self.lod_ptrs_late.destroy(device, allocator);
    }
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

    registries: Arc<FrozenRegistries>,
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

    // MDI & Compute Culling
    mdi_buffers: Option<MdiBuffers>,
    chunk_slots: SlotAllocator<ChunkPos>,
    lod_slots: SlotAllocator<vx_lod::coords::LodNodeKey>,

    hiz_pyramid: Option<HiZPyramid>,
    point_clamp_sampler: vk::Sampler,
    cull_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    cull_descriptor_set: Option<vk::DescriptorSet>,
    hiz_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    hiz_descriptor_pool: Option<vk::DescriptorPool>,
    hiz_descriptor_sets: Vec<vk::DescriptorSet>,

    hiz_pipeline: Option<ComputePipeline>,
    hiz_comp_shader: Option<ShaderModule>,
    cull_chunks_pipeline: Option<ComputePipeline>,
    cull_chunks_comp_shader: Option<ShaderModule>,
    cull_lod_pipeline: Option<ComputePipeline>,
    cull_lod_comp_shader: Option<ShaderModule>,

    ui_pipeline: Option<GraphicsPipeline>,
    ui_vert_shader: Option<ShaderModule>,
    ui_frag_shader: Option<ShaderModule>,
    ui_texture_array: Option<GpuTextureArray>,
    ui_descriptor_pool: Option<vk::DescriptorPool>,
    ui_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    ui_descriptor_set: Option<vk::DescriptorSet>,
    ui_buffer: Option<GpuBuffer>,
    ui_font: Option<BitmapFont>,
    ui_layers: UiLayers,
    hud_state: HudState,
    inventory_sim: vx_sim::Inventory,
    inventory_open: bool,
    mouse_cursor_pos: [f32; 2],
    inventory_hovered_slot: Option<usize>,
    shift_held: bool,

    // Day/Night & Celestial Rendering
    client_time_of_day: f32,
    client_world_age: u64,
    lightmap_lut: LightmapLut,
    lightmap_texture: Option<GpuTexture2d>,
    lightmap_staging_buffers: Vec<GpuBuffer>,
    sky_pipeline: Option<GraphicsPipeline>,
    sky_vert_shader: Option<ShaderModule>,
    sky_frag_shader: Option<ShaderModule>,
    celestial_texture: Option<GpuTextureArray>,
    sky_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    sky_descriptor_pool: Option<vk::DescriptorPool>,
    sky_descriptor_set: Option<vk::DescriptorSet>,

    // Weather Simulation & Precipitation Particles
    weather_rain_level: f32,
    weather_thunder_level: f32,
    weather_lightning_flash: f32,
    weather_pipeline: Option<GraphicsPipeline>,
    weather_vert_shader: Option<ShaderModule>,
    weather_frag_shader: Option<ShaderModule>,
    weather_texture: Option<GpuTextureArray>,
    weather_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    weather_descriptor_pool: Option<vk::DescriptorPool>,
    weather_descriptor_set: Option<vk::DescriptorSet>,
    weather_buffer: Option<GpuBuffer>,
    weather_particles_count: u32,
    world_seed: u64,

    // Entities & Mob Rendering
    entity_store: ClientEntityStore,
    entity_pipeline: Option<GraphicsPipeline>,
    entity_vert_shader: Option<ShaderModule>,
    entity_frag_shader: Option<ShaderModule>,
    entity_textures: Option<GpuTextureArray>,
    entity_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    entity_descriptor_pool: Option<vk::DescriptorPool>,
    entity_descriptor_set: Option<vk::DescriptorSet>,
    entity_buffer: Option<GpuBuffer>,
    entity_vertex_count: u32,
    mob_spawn_type_cycle: u8,
}

impl App {
    #[allow(clippy::too_many_lines)]
    fn new(validation: bool, seed: u64, view_distance: u32, world_dir: PathBuf) -> Self {
        let spawn_pos = Vec3::new(128.0, 45.0, 160.0);
        let mut camera = Camera::new(spawn_pos);
        camera.pitch = -0.3;

        let registries = Arc::new(FrozenRegistries::new_default());
        let server_registries = Arc::clone(&registries);

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
                let mut server =
                    vx_server::Server::with_registries(seed, config, server_registries);
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

            registries: Arc::clone(&registries),
            block_registry: registries.block_registry().clone(),
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

            mdi_buffers: None,
            chunk_slots: SlotAllocator::new(MAX_CHUNK_CANDIDATES as u32),
            lod_slots: SlotAllocator::new(MAX_LOD_CANDIDATES as u32),

            hiz_pyramid: None,
            point_clamp_sampler: vk::Sampler::null(),
            cull_descriptor_set_layout: None,
            cull_descriptor_set: None,
            hiz_descriptor_set_layout: None,
            hiz_descriptor_pool: None,
            hiz_descriptor_sets: Vec::new(),

            hiz_pipeline: None,
            hiz_comp_shader: None,
            cull_chunks_pipeline: None,
            cull_chunks_comp_shader: None,
            cull_lod_pipeline: None,
            cull_lod_comp_shader: None,

            ui_pipeline: None,
            ui_vert_shader: None,
            ui_frag_shader: None,
            ui_texture_array: None,
            ui_descriptor_pool: None,
            ui_descriptor_set_layout: None,
            ui_descriptor_set: None,
            ui_buffer: None,
            ui_font: None,
            ui_layers: UiLayers::default(),
            hud_state: HudState::default(),
            inventory_sim: vx_sim::Inventory::default(),
            inventory_open: false,
            mouse_cursor_pos: [0.0, 0.0],
            inventory_hovered_slot: None,
            shift_held: false,

            client_time_of_day: 6000.0,
            client_world_age: 0,
            lightmap_lut: LightmapLut::default(),
            lightmap_texture: None,
            lightmap_staging_buffers: Vec::new(),
            sky_pipeline: None,
            sky_vert_shader: None,
            sky_frag_shader: None,
            celestial_texture: None,
            sky_descriptor_set_layout: None,
            sky_descriptor_pool: None,
            sky_descriptor_set: None,

            weather_rain_level: 0.0,
            weather_thunder_level: 0.0,
            weather_lightning_flash: 0.0,
            weather_pipeline: None,
            weather_vert_shader: None,
            weather_frag_shader: None,
            weather_texture: None,
            weather_descriptor_set_layout: None,
            weather_descriptor_pool: None,
            weather_descriptor_set: None,
            weather_buffer: None,
            weather_particles_count: 0,
            world_seed: seed,

            entity_store: ClientEntityStore::new(),
            entity_pipeline: None,
            entity_vert_shader: None,
            entity_frag_shader: None,
            entity_textures: None,
            entity_descriptor_set_layout: None,
            entity_descriptor_pool: None,
            entity_descriptor_set: None,
            entity_buffer: None,
            entity_vertex_count: 0,
            mob_spawn_type_cycle: 1,
        }
    }

    fn select_hotbar_slot(&mut self, slot: usize) {
        if slot < 9 {
            self.selected_hotbar_slot = slot;
            self.hud_state.selected_slot = slot;
            self.inventory_sim.selected_slot = slot;
            let item = self.inventory_sim.slots[slot].item;
            if item > 0 {
                self.selected_block_state = BlockStateId::new(item);
            } else if slot < HOTBAR_ITEMS.len() {
                self.selected_block_state = HOTBAR_ITEMS[slot].1;
            }
            let item_name = self
                .registries
                .item_registry()
                .get_by_id(item)
                .map_or_else(|| vx_sim::item_name(item), |def| def.name.as_str());
            info!(
                slot = slot + 1,
                item = item_name,
                state_id = self.selected_block_state.as_u32(),
                "Selected hotbar slot"
            );
        }
    }

    fn handle_inventory_swap_hotbar(&mut self, hotbar_idx: usize) {
        if let Some(hovered) = self.inventory_hovered_slot {
            let predicted_carried = self.inventory_sim.carried;
            let click_msg = C2sMessage::InventoryClick(C2sInventoryClick {
                slot: hovered as u16,
                button: hotbar_idx as u8,
                mode: 2, // SwapHotbar
                predicted_carried_item: predicted_carried.item,
                predicted_carried_count: predicted_carried.count,
            });
            let _ = self
                .client_conn
                .send(Lane::Control, Payload::Msg(click_msg));
            self.inventory_sim.selected_slot = hotbar_idx;
            let _ = vx_sim::inventory_click(
                &mut self.inventory_sim,
                hovered,
                if hotbar_idx == 0 {
                    vx_sim::ClickButton::Right
                } else {
                    vx_sim::ClickButton::Left
                },
                vx_sim::ClickMode::SwapHotbar,
            );
            self.selected_block_state = BlockStateId::new(self.inventory_sim.selected_item().item);
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
                            "Login successful, entering configuration phase..."
                        );
                        self.client_phase = ConnectionPhase::Config;
                    }
                }
                ConnectionPhase::Config => match msg {
                    S2cMessage::RegistryData(data) => {
                        info!(
                            registry_id = data.registry_id.as_str(),
                            entries = data.entries.len(),
                            "Received server registry data"
                        );
                    }
                    S2cMessage::ConfigDone(_) => {
                        info!("Server configuration complete, acknowledging config...");
                        let settings = C2sMessage::ClientSettings(C2sClientSettings {
                            view_distance: self.view_distance as u16,
                            simulation_distance: self.view_distance as u16,
                            locale: BoundedString::new("en_US").unwrap(),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(settings));
                        let ack = C2sMessage::ConfigAck(C2sConfigAck);
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(ack));
                    }
                    S2cMessage::JoinGame(join) => {
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
                    other => {
                        tracing::debug!("Ignored packet in Config phase: {other:?}");
                    }
                },
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
                        self.chunk_slots.release(&pos);
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
                        if words.is_empty() {
                            self.lod_slots.release(&key);
                            if let (Some(mut mesh), Some(ctx)) =
                                (self.lod_meshes.remove(&key), &self.gpu_context)
                            {
                                mesh.buffer.destroy(ctx.device().raw(), ctx.allocator());
                            }
                        } else {
                            self.pending_lod_uploads
                                .insert(key, (lod.quad_count, words));
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
                        self.lod_slots.release(&key);
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
                    S2cMessage::UpdateTime(update) => {
                        self.client_world_age = update.world_age;
                        let server_time = (update.time_of_day % vx_core::time::DAY_TICKS) as f32;
                        self.client_time_of_day = server_time;
                    }
                    S2cMessage::UpdateStats(stats) => {
                        self.hud_state.health = stats.health;
                        self.hud_state.max_health = stats.max_health;
                        self.hud_state.food = stats.food;
                        self.hud_state.saturation = stats.saturation;
                        self.hud_state.xp_level = stats.xp_level;
                        self.hud_state.xp_progress = stats.xp_progress;
                    }
                    S2cMessage::InventoryBulk(bulk) => {
                        for (i, slot) in bulk.slots.iter().enumerate() {
                            if i < self.inventory_sim.slots.len() {
                                self.inventory_sim.slots[i] =
                                    vx_sim::ItemStack::new(slot.item, slot.count);
                            }
                        }
                        self.inventory_sim.carried =
                            vx_sim::ItemStack::new(bulk.carried.item, bulk.carried.count);
                        let selected_item = self.inventory_sim.selected_item();
                        if selected_item.count > 0 {
                            self.selected_block_state = BlockStateId::new(selected_item.item);
                        }
                    }
                    S2cMessage::InventorySlot(slot_msg) => {
                        let idx = slot_msg.slot as usize;
                        if idx < self.inventory_sim.slots.len() {
                            self.inventory_sim.slots[idx] =
                                vx_sim::ItemStack::new(slot_msg.item, slot_msg.count);
                            if idx == self.selected_hotbar_slot && slot_msg.count > 0 {
                                self.selected_block_state = BlockStateId::new(slot_msg.item);
                            }
                        }
                    }
                    S2cMessage::UpdateWeather(weather) => {
                        self.weather_rain_level = weather.rain_level;
                        self.weather_thunder_level = weather.thunder_level;
                        if weather.lightning_flash > 0 {
                            self.weather_lightning_flash = 1.0;
                        }
                    }
                    S2cMessage::SpawnEntity(spawn) => {
                        self.entity_store.on_spawn(spawn);
                    }
                    S2cMessage::DespawnEntity(despawn) => {
                        self.entity_store.on_despawn(despawn.net_ids.as_slice());
                    }
                    S2cMessage::EntityMove(m) => {
                        self.entity_store.on_move(m);
                    }
                    S2cMessage::EntityStatus(s) => {
                        self.entity_store.on_status(s.net_id, s.status);
                    }
                    _ => {}
                },
            }
        }
    }

    fn get_sky_light_at(&self, pos: BlockPos) -> u8 {
        let (cpos, lpos) = vx_voxel::coords::split_block_pos(pos);
        if let Some(snap) = self.chunks.get(&cpos)
            && let Some(light) = snap.light()
        {
            return light.get_sky(lpos.as_usize());
        }
        15
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::similar_names
    )]
    fn generate_weather_particles(&mut self, _dt: f32) {
        let rain_factor =
            (self.weather_rain_level * 0.85 + self.weather_thunder_level * 0.15).clamp(0.0, 1.0);
        if rain_factor <= 0.01 {
            self.weather_particles_count = 0;
            return;
        }

        let max_particles: usize = 640;
        let active_count = ((max_particles as f32) * rain_factor).round() as usize;
        let cam_pos = self.camera.position;
        let time = (self.frame_tick as f32) * 0.05;

        let mut particles: Vec<WeatherParticleGpu> = Vec::with_capacity(active_count);

        let radius = 14.0_f32;
        let height_span = 20.0_f32;
        let min_rel_y = -8.0_f32;

        for i in 0..active_count {
            let seed_i = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let h0 = ((seed_i ^ (seed_i >> 16)) & 0xFFFF) as f32 / 65535.0;
            let h1 = (((seed_i >> 16) ^ (seed_i >> 32)) & 0xFFFF) as f32 / 65535.0;
            let h2 = (((seed_i >> 32) ^ (seed_i >> 48)) & 0xFFFF) as f32 / 65535.0;

            let base_rx = (h0 * 2.0 - 1.0) * radius;
            let base_rz = (h1 * 2.0 - 1.0) * radius;
            let base_ry = h2 * height_span;

            let px = cam_pos.x + base_rx;
            let pz = cam_pos.z + base_rz;
            let climate = vx_worldgen::ClimatePoint::sample(self.world_seed, px, pz);
            let is_dry = climate.humidity < -0.35 && climate.temperature > 0.5;

            let test_y = cam_pos.y + min_rel_y + base_ry;
            let precip_kind =
                vx_sim::weather::precipitation_at(climate.temperature, test_y, is_dry);
            if precip_kind == vx_sim::weather::PrecipitationKind::None {
                continue;
            }

            let is_snow = precip_kind == vx_sim::weather::PrecipitationKind::Snow;

            let (rel_x, rel_y, rel_z, uv_anim, size_x, size_y, layer, alpha) = if is_snow {
                let fall_speed = 2.2_f32;
                let flutter_x = (time * 2.0 + h0 * std::f32::consts::TAU).sin() * 0.4;
                let flutter_z = (time * 1.5 + h1 * std::f32::consts::TAU).cos() * 0.4;

                let y_off = (base_ry - time * fall_speed).rem_euclid(height_span);
                let rel_y = min_rel_y + y_off;
                let rel_x = base_rx + flutter_x;
                let rel_z = base_rz + flutter_z;

                let uv_anim = (h0 * 4.0).floor() * 0.25;
                let size = 0.16 + h2 * 0.08;
                let alpha = (0.75 * rain_factor).min(0.9);
                (rel_x, rel_y, rel_z, uv_anim, size, size, 1u32, alpha)
            } else {
                let fall_speed = 18.0_f32;
                let y_off = (base_ry - time * fall_speed).rem_euclid(height_span);
                let rel_y = min_rel_y + y_off;
                let wind_drift_x = -time * 1.2;
                let wind_drift_z = time * 0.6;
                let rel_x = ((base_rx + wind_drift_x + radius).rem_euclid(radius * 2.0)) - radius;
                let rel_z = ((base_rz + wind_drift_z + radius).rem_euclid(radius * 2.0)) - radius;

                let uv_anim = (time * 8.0 + h0).fract();
                let streak_w = 0.05 + h0 * 0.03;
                let streak_h = 0.75 + h1 * 0.35;
                let alpha = (0.65 * rain_factor).min(0.85);
                (
                    rel_x, rel_y, rel_z, uv_anim, streak_w, streak_h, 0u32, alpha,
                )
            };

            let world_x = cam_pos.x + rel_x;
            let world_y = cam_pos.y + rel_y;
            let world_z = cam_pos.z + rel_z;

            let bx = world_x.floor() as i32;
            let by = world_y.floor() as i32;
            let bz = world_z.floor() as i32;
            let sky_light = self.get_sky_light_at(vx_core::BlockPos::new(bx, by, bz));
            if sky_light < 15 {
                continue;
            }

            let dist_sq = rel_x * rel_x + rel_z * rel_z;
            let edge_fade = (1.0 - (dist_sq / (radius * radius))).clamp(0.0, 1.0);
            let final_alpha = (alpha * edge_fade * 255.0) as u32;
            if final_alpha < 5 {
                continue;
            }

            let color = 0x00FF_FFFF | (final_alpha << 24);

            particles.push(WeatherParticleGpu {
                pos: [world_x, world_y, world_z],
                size_x,
                size_y,
                uv_anim,
                layer,
                color,
            });
        }

        self.weather_particles_count = particles.len() as u32;
        if self.weather_particles_count > 0
            && let Some(buf) = &mut self.weather_buffer
        {
            let _ = buf.write_bytes(bytemuck::cast_slice(&particles));
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
                self.chunk_slots.release(&pos);
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
                self.lod_slots.release(&key);
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

    fn setup_hiz_and_culling(&mut self) -> Result<()> {
        let Some(gpu_context) = &self.gpu_context else {
            return Ok(());
        };
        let device = gpu_context.device().raw();
        let extent = gpu_context.extent();

        // 1. Point clamp sampler for downsampling depth buffer and mips
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(16.0);
        let point_clamp_sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        self.point_clamp_sampler = point_clamp_sampler;

        // 2. Cull descriptor set layout (binding 0: combined image sampler for Hi-Z pyramid)
        let cull_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let cull_bindings = [cull_binding];
        let cull_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&cull_bindings);
        let cull_layout = unsafe { device.create_descriptor_set_layout(&cull_layout_info, None)? };
        self.cull_descriptor_set_layout = Some(cull_layout);

        // 3. Hi-Z descriptor set layout (binding 0: sampler2D in_depth, binding 1: image2D out_mip)
        let hiz_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let hiz_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&hiz_bindings);
        let hiz_layout = unsafe { device.create_descriptor_set_layout(&hiz_layout_info, None)? };
        self.hiz_descriptor_set_layout = Some(hiz_layout);

        // 4. Descriptor pool for Hi-Z and culling
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(48),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(48),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(48)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&pool_sizes);
        let hiz_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        self.hiz_descriptor_pool = Some(hiz_pool);

        // 5. Compute pipelines
        let hiz_spv = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_generate.comp.spv"));
        let hiz_module = ShaderModule::from_spv(device, hiz_spv)?;
        let hiz_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<HiZPushConstants>() as u32);
        let hiz_pipeline =
            ComputePipeline::new(device, hiz_module.raw(), &[hiz_layout], &[hiz_pc_range])?;
        self.hiz_comp_shader = Some(hiz_module);
        self.hiz_pipeline = Some(hiz_pipeline);

        let cull_spv = include_bytes!(concat!(env!("OUT_DIR"), "/cull_chunks.comp.spv"));
        let cull_module = ShaderModule::from_spv(device, cull_spv)?;
        let cull_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<CullPushConstants>() as u32);
        let cull_pipeline =
            ComputePipeline::new(device, cull_module.raw(), &[cull_layout], &[cull_pc_range])?;
        self.cull_chunks_comp_shader = Some(cull_module);
        self.cull_chunks_pipeline = Some(cull_pipeline);

        let cull_lod_spv = include_bytes!(concat!(env!("OUT_DIR"), "/cull_lod.comp.spv"));
        let cull_lod_module = ShaderModule::from_spv(device, cull_lod_spv)?;
        let cull_lod_pipeline = ComputePipeline::new(
            device,
            cull_lod_module.raw(),
            &[cull_layout],
            &[cull_pc_range],
        )?;
        self.cull_lod_comp_shader = Some(cull_lod_module);
        self.cull_lod_pipeline = Some(cull_lod_pipeline);

        // 6. Hi-Z Pyramid allocation & descriptor sets
        self.recreate_hiz_resources(extent)?;

        Ok(())
    }

    fn recreate_hiz_resources(&mut self, extent: vk::Extent2D) -> Result<()> {
        let Some(gpu_context) = &self.gpu_context else {
            return Ok(());
        };
        let Some(depth_buffer) = &self.depth_buffer else {
            return Ok(());
        };
        let Some(hiz_pool) = self.hiz_descriptor_pool else {
            return Ok(());
        };
        let Some(cull_layout) = self.cull_descriptor_set_layout else {
            return Ok(());
        };
        let Some(hiz_layout) = self.hiz_descriptor_set_layout else {
            return Ok(());
        };

        let device = gpu_context.device().raw();
        let allocator = gpu_context.allocator();

        // Destroy previous pyramid if any
        if let Some(mut old_hiz) = self.hiz_pyramid.take() {
            old_hiz.destroy(device, allocator);
        }

        // Reset descriptor pool
        unsafe {
            device.reset_descriptor_pool(hiz_pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        self.hiz_descriptor_sets.clear();

        // Create new Hi-Z pyramid
        let hiz_pyramid = HiZPyramid::new(device, allocator, extent)?;
        let mip_levels = hiz_pyramid.mip_levels();

        // Allocate cull descriptor set
        let cull_layouts = [cull_layout];
        let cull_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(hiz_pool)
            .set_layouts(&cull_layouts);
        let cull_set = unsafe { device.allocate_descriptor_sets(&cull_alloc_info)?[0] };

        let cull_image_info = vk::DescriptorImageInfo::default()
            .sampler(hiz_pyramid.sampler())
            .image_view(hiz_pyramid.full_view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let cull_image_infos = [cull_image_info];
        let cull_write = vk::WriteDescriptorSet::default()
            .dst_set(cull_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&cull_image_infos);
        let cull_writes = [cull_write];
        unsafe {
            device.update_descriptor_sets(&cull_writes, &[]);
        }
        self.cull_descriptor_set = Some(cull_set);

        // Allocate Hi-Z descriptor sets for each mip level
        let hiz_layouts = vec![hiz_layout; mip_levels as usize];
        let hiz_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(hiz_pool)
            .set_layouts(&hiz_layouts);
        let hiz_sets = unsafe { device.allocate_descriptor_sets(&hiz_alloc_info)? };

        for (k, &set) in hiz_sets.iter().enumerate().take(mip_levels as usize) {
            let in_image_info = if k == 0 {
                vk::DescriptorImageInfo::default()
                    .sampler(self.point_clamp_sampler)
                    .image_view(depth_buffer.view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            } else {
                vk::DescriptorImageInfo::default()
                    .sampler(self.point_clamp_sampler)
                    .image_view(hiz_pyramid.mip_view(k - 1))
                    .image_layout(vk::ImageLayout::GENERAL)
            };

            let out_image_info = vk::DescriptorImageInfo::default()
                .image_view(hiz_pyramid.mip_view(k))
                .image_layout(vk::ImageLayout::GENERAL);

            let in_image_infos = [in_image_info];
            let out_image_infos = [out_image_info];

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&in_image_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_image_infos),
            ];

            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }

        self.hiz_descriptor_sets = hiz_sets;
        self.hiz_pyramid = Some(hiz_pyramid);

        Ok(())
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

        // Advance smooth client time of day (20 ticks per second)
        self.client_time_of_day =
            (self.client_time_of_day + dt * 20.0) % (vx_core::time::DAY_TICKS as f32);

        let sun_angle = vx_core::time::sun_angle(
            self.client_time_of_day as u64,
            self.client_time_of_day.fract(),
        );
        let sun_dir = vx_core::time::sun_direction(sun_angle);
        let moon_dir = vx_core::time::moon_direction(sun_angle);
        let day_number = self.client_world_age / vx_core::time::DAY_TICKS;
        let moon_phase = vx_core::time::moon_phase(day_number);

        self.lightmap_lut.update(
            sun_dir.y,
            self.client_time_of_day,
            self.weather_rain_level,
            self.weather_thunder_level,
            self.weather_lightning_flash,
        );

        if self.weather_lightning_flash > 0.0 {
            self.weather_lightning_flash = (self.weather_lightning_flash - dt * 3.5).max(0.0);
        }

        self.generate_weather_particles(dt);

        // Entity simulation update & mesh generation (Phase 21)
        self.entity_store.update(dt);
        #[allow(clippy::cast_possible_truncation)]
        {
            self.hud_state.entities_rendered = self.entity_store.count() as u32;
        }

        let mut entity_vertices = Vec::new();
        {
            #[allow(clippy::cast_possible_truncation)]
            let get_light = |pos: glam::DVec3| -> (u8, u8) {
                let bpos = BlockPos::new(
                    pos.x.floor() as i32,
                    pos.y.floor() as i32,
                    pos.z.floor() as i32,
                );
                let (cpos, lpos) = vx_voxel::coords::split_block_pos(bpos);
                if let Some(snap) = self.chunks.get(&cpos)
                    && let Some(light) = snap.light()
                {
                    (
                        light.get_sky(lpos.as_usize()),
                        light.get_block(lpos.as_usize()),
                    )
                } else {
                    (15, 0)
                }
            };
            self.entity_store
                .build_mesh(get_light, &mut entity_vertices);
        }

        if entity_vertices.len() > 16384 {
            entity_vertices.truncate(16384);
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            self.entity_vertex_count = entity_vertices.len() as u32;
        }
        if self.entity_vertex_count > 0
            && let Some(buf) = &mut self.entity_buffer
        {
            let _ = buf.write_bytes(bytemuck::cast_slice(&entity_vertices));
        }

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
                visible_chunks = self.visible_chunks_last,
                visible_lods = self.visible_lod_nodes_last,
                loaded_chunks = self.chunks.len(),
                lod_meshes = self.lod_meshes.len(),
                gpu_meshes = self.chunk_meshes.len(),
                dirty_queue = self.dirty_chunks.len(),
                pos = ?self.camera.position,
                "Performance metrics"
            );
        }

        let (
            Some(gpu_context),
            Some(pipeline),
            Some(depth_buffer),
            Some(descriptor_set),
            Some(mdi),
            Some(hiz_pyramid),
            Some(cull_desc_set),
            Some(cull_chunks_pipe),
            Some(cull_lod_pipe),
            Some(hiz_pipe),
        ) = (
            &mut self.gpu_context,
            &self.pipeline,
            &mut self.depth_buffer,
            self.descriptor_set,
            &mut self.mdi_buffers,
            &self.hiz_pyramid,
            self.cull_descriptor_set,
            &self.cull_chunks_pipeline,
            &self.cull_lod_pipeline,
            &self.hiz_pipeline,
        )
        else {
            return;
        };

        // 1. Gather Chunk Cull Candidates
        let mut chunk_candidates =
            Vec::with_capacity(self.chunk_meshes.len().min(MAX_CHUNK_CANDIDATES));
        for (&pos, mesh) in &self.chunk_meshes {
            if chunk_candidates.len() >= MAX_CHUNK_CANDIDATES {
                break;
            }
            let Some(chunk_id) = self.chunk_slots.get_or_allocate(pos) else {
                continue;
            };

            let mut active_mask = 0u32;
            let mut opaque_quad_address = 0u64;
            let mut opaque_quad_count = 0u32;
            let mut opaque_pattern_offset = 0u32;
            if let Some(l) = &mesh.opaque {
                active_mask |= 1;
                opaque_quad_address = l.buffer.device_address();
                opaque_quad_count = l.quad_count;
                opaque_pattern_offset = l.pattern_offset;
            }

            let mut t1_quad_address = 0u64;
            let mut t1_quad_count = 0u32;
            let mut t1_pattern_offset = 0u32;
            if let Some(l) = &mesh.t1_opaque {
                active_mask |= 2;
                t1_quad_address = l.buffer.device_address();
                t1_quad_count = l.quad_count;
                t1_pattern_offset = l.pattern_offset;
            }

            let mut cutout_quad_address = 0u64;
            let mut cutout_quad_count = 0u32;
            let mut cutout_pattern_offset = 0u32;
            if let Some(l) = &mesh.cutout {
                active_mask |= 4;
                cutout_quad_address = l.buffer.device_address();
                cutout_quad_count = l.quad_count;
                cutout_pattern_offset = l.pattern_offset;
            }

            let mut translucent_quad_address = 0u64;
            let mut translucent_quad_count = 0u32;
            let mut translucent_pattern_offset = 0u32;
            if let Some(l) = &mesh.translucent {
                active_mask |= 8;
                translucent_quad_address = l.buffer.device_address();
                translucent_quad_count = l.quad_count;
                translucent_pattern_offset = l.pattern_offset;
            }

            if active_mask == 0 {
                continue;
            }

            chunk_candidates.push(ChunkCullCandidate {
                min_x: mesh.min_aabb.x,
                min_y: mesh.min_aabb.y,
                min_z: mesh.min_aabb.z,
                active_mask,
                max_x: mesh.max_aabb.x,
                max_y: mesh.max_aabb.y,
                max_z: mesh.max_aabb.z,
                chunk_id,
                chunk_x: mesh.pos[0],
                chunk_y: mesh.pos[1],
                chunk_z: mesh.pos[2],
                opaque_pattern_offset,
                opaque_quad_address,
                opaque_quad_count,
                t1_pattern_offset,
                t1_quad_address,
                t1_quad_count,
                cutout_pattern_offset,
                cutout_quad_address,
                cutout_quad_count,
                translucent_pattern_offset,
                translucent_quad_address,
                translucent_quad_count,
                _pad: 0,
            });
        }

        // 2. Gather LOD Cull Candidates
        let mut lod_candidates = Vec::with_capacity(self.lod_meshes.len().min(MAX_LOD_CANDIDATES));
        for (&key, mesh) in &self.lod_meshes {
            if lod_candidates.len() >= MAX_LOD_CANDIDATES {
                break;
            }
            if mesh.quad_count == 0 {
                continue;
            }
            let Some(lod_id) = self.lod_slots.get_or_allocate(key) else {
                continue;
            };

            lod_candidates.push(LodCullCandidate {
                min_x: mesh.min_aabb.x,
                min_y: mesh.min_aabb.y,
                min_z: mesh.min_aabb.z,
                lod_id,
                max_x: mesh.max_aabb.x,
                max_y: mesh.max_aabb.y,
                max_z: mesh.max_aabb.z,
                level: u32::from(mesh.level),
                node_x: mesh.node_pos[0],
                node_y: mesh.node_pos[1],
                node_z: mesh.node_pos[2],
                quad_count: mesh.quad_count,
                quad_address: mesh.buffer.device_address(),
                _pad0: 0,
                _pad1: 0,
            });
        }

        // 3. Upload Candidates to Host-Visible GPU Buffers
        if !chunk_candidates.is_empty() {
            let _ = mdi
                .chunk_candidates
                .write_bytes(bytemuck::cast_slice(&chunk_candidates));
        }
        if !lod_candidates.is_empty() {
            let _ = mdi
                .lod_candidates
                .write_bytes(bytemuck::cast_slice(&lod_candidates));
        }

        self.visible_chunks_last = chunk_candidates.len();
        self.visible_lod_nodes_last = lod_candidates.len();

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
        let view_proj_mat = self.camera.view_proj_matrix(aspect);
        let view_proj = view_proj_mat.to_cols_array();
        let inv_view_proj = view_proj_mat.inverse().to_cols_array();

        let staging_idx = if self.lightmap_staging_buffers.is_empty() {
            0
        } else {
            (self.frame_counter as usize) % self.lightmap_staging_buffers.len()
        };
        if let Some(staging_buf) = self.lightmap_staging_buffers.get_mut(staging_idx) {
            let _ = staging_buf.write_bytes(&self.lightmap_lut.data);
        }

        let device = gpu_context.device().raw();
        let is_first_frame = u32::from(self.frame_counter <= 1);
        let hiz_extent = hiz_pyramid.extent();
        let hiz_levels = hiz_pyramid.mip_levels();

        // SAFETY: Recording barriers, compute culling dispatches, and MDI render passes
        unsafe {
            // First frame: transition Hi-Z image from UNDEFINED to SHADER_READ_ONLY_OPTIMAL
            if self.frame_counter == 1 {
                let init_barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(hiz_pyramid.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: hiz_levels,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let init_barriers = [init_barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&init_barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            // Upload dynamic Lightmap LUT to lightmap_texture
            if let Some(lightmap_tex) = &self.lightmap_texture
                && let Some(staging_buf) = self.lightmap_staging_buffers.get(staging_idx)
            {
                let to_transfer = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(if self.frame_counter <= 1 {
                        vk::PipelineStageFlags2::TOP_OF_PIPE
                    } else {
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                    })
                    .src_access_mask(if self.frame_counter <= 1 {
                        vk::AccessFlags2::NONE
                    } else {
                        vk::AccessFlags2::SHADER_SAMPLED_READ
                    })
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(if self.frame_counter <= 1 {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    })
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(lightmap_tex.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let to_transfer_barriers = [to_transfer];
                let dep_info =
                    vk::DependencyInfo::default().image_memory_barriers(&to_transfer_barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                let copy_region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    });

                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging_buf.raw(),
                    lightmap_tex.image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                );

                let to_read = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(lightmap_tex.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let to_read_barriers = [to_read];
                let dep_info =
                    vk::DependencyInfo::default().image_memory_barriers(&to_read_barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            // Memory barrier for uploaded candidates: HOST_WRITE -> COMPUTE_SHADER
            let upload_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::HOST)
                .src_access_mask(vk::AccessFlags2::HOST_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
            let upload_barriers = [upload_barrier];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&upload_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // Clear Early Count Buffers
            device.cmd_fill_buffer(cmd, mdi.chunk_counts_early.raw(), 0, 16, 0);
            device.cmd_fill_buffer(cmd, mdi.lod_counts_early.raw(), 0, 16, 0);

            // Barrier: TRANSFER_WRITE -> COMPUTE_SHADER
            let count_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
            let count_barriers = [count_barrier];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&count_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // ==========================================
            // PHASE 1: EARLY COMPUTE CULL
            // ==========================================
            let cull_pc = CullPushConstants {
                view_proj,
                camera_pos: [
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                ],
                cull_phase: 0,
                hiz_width: hiz_extent.width,
                hiz_height: hiz_extent.height,
                candidate_count: chunk_candidates.len() as u32,
                is_first_frame,
                pointers_address: mdi.cull_ptrs_early.device_address(),
            };

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull_chunks_pipe.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cull_chunks_pipe.layout(),
                0,
                &[cull_desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                cull_chunks_pipe.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&cull_pc),
            );
            if !chunk_candidates.is_empty() {
                let groups = (chunk_candidates.len() as u32).div_ceil(64);
                device.cmd_dispatch(cmd, groups, 1, 1);
            }

            // Early Cull LOD
            let lod_cull_pc = CullPushConstants {
                view_proj,
                camera_pos: [
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                ],
                cull_phase: 0,
                hiz_width: hiz_extent.width,
                hiz_height: hiz_extent.height,
                candidate_count: lod_candidates.len() as u32,
                is_first_frame,
                pointers_address: mdi.lod_ptrs_early.device_address(),
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull_lod_pipe.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cull_lod_pipe.layout(),
                0,
                &[cull_desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                cull_lod_pipe.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&lod_cull_pc),
            );
            if !lod_candidates.is_empty() {
                let groups = (lod_candidates.len() as u32).div_ceil(64);
                device.cmd_dispatch(cmd, groups, 1, 1);
            }

            // Barrier: Phase 1 Compute Write -> Indirect Command Read + Vertex Shader Read (BDA)
            let early_render_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::VERTEX_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
                );

            let depth_init_barrier = vk::ImageMemoryBarrier2::default()
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

            let early_render_barriers = [early_render_barrier];
            let depth_init_barriers = [depth_init_barrier];
            let dep_info = vk::DependencyInfo::default()
                .memory_barriers(&early_render_barriers)
                .image_memory_barriers(&depth_init_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // ==========================================
            // PHASE 1: EARLY RASTERIZATION PASSES
            // ==========================================
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
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0, // Reversed-Z: clear to 0.0!
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

            // Early Pass 1: Opaque T0
            let t0_pc = TerrainMdiPushConstants {
                view_proj,
                draw_info_buffer_address: mdi.opaque_draw_early.device_address(),
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pipeline.layout(),
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&t0_pc),
            );
            device.cmd_draw_indirect_count(
                cmd,
                mdi.opaque_cmd_early.raw(),
                0,
                mdi.chunk_counts_early.raw(),
                0,
                MAX_CHUNK_CANDIDATES as u32,
                16,
            );

            // Early Pass 2: Opaque T1
            if let Some(t1_pipeline) = &self.t1_pipeline {
                let t1_pc = TerrainMdiPushConstants {
                    view_proj,
                    draw_info_buffer_address: mdi.t1_draw_early.device_address(),
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, t1_pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    t1_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    t1_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&t1_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.t1_cmd_early.raw(),
                    0,
                    mdi.chunk_counts_early.raw(),
                    4,
                    MAX_CHUNK_CANDIDATES as u32,
                    16,
                );
            }

            // Early Pass 3: Cutout
            if let Some(cutout_pipeline) = &self.cutout_pipeline {
                let cutout_pc = TerrainMdiPushConstants {
                    view_proj,
                    draw_info_buffer_address: mdi.cutout_draw_early.device_address(),
                };
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
                device.cmd_push_constants(
                    cmd,
                    cutout_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&cutout_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.cutout_cmd_early.raw(),
                    0,
                    mdi.chunk_counts_early.raw(),
                    8,
                    MAX_CHUNK_CANDIDATES as u32,
                    16,
                );
            }

            // Early Pass 4: LOD
            if let Some(lod_pipeline) = &self.lod_pipeline {
                let max_dist = (self.view_distance as f32 * 32.0) * 4.0;
                let lod_pc = LodMdiPushConstants {
                    view_proj,
                    camera_pos: [
                        self.camera.position.x,
                        self.camera.position.y,
                        self.camera.position.z,
                    ],
                    max_distance: max_dist,
                    draw_info_buffer_address: mdi.lod_draw_early.device_address(),
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, lod_pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    lod_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    lod_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&lod_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.lod_cmd_early.raw(),
                    0,
                    mdi.lod_counts_early.raw(),
                    0,
                    MAX_LOD_CANDIDATES as u32,
                    16,
                );
            }

            device.cmd_end_rendering(cmd);

            // ==========================================
            // HI-Z PYRAMID DOWNSAMPLING
            // ==========================================
            // Transition depth buffer: DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let depth_to_sample_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(depth_buffer.raw())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Transition Hi-Z: SHADER_READ_ONLY_OPTIMAL -> GENERAL
            let hiz_to_general_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(hiz_pyramid.image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: hiz_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let hiz_barriers = [depth_to_sample_barrier, hiz_to_general_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&hiz_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, hiz_pipe.raw());

            // Mip 0 Dispatch: depth buffer -> Hi-Z level 0
            let mip0_pc = HiZPushConstants {
                in_extent: [swapchain_extent.width, swapchain_extent.height],
                out_extent: [hiz_extent.width, hiz_extent.height],
                is_mip0: 1,
                _pad: 0,
            };
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                hiz_pipe.layout(),
                0,
                &[self.hiz_descriptor_sets[0]],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                hiz_pipe.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&mip0_pc),
            );
            let gx0 = hiz_extent.width.div_ceil(16);
            let gy0 = hiz_extent.height.div_ceil(16);
            device.cmd_dispatch(cmd, gx0, gy0, 1);

            // Subsequent Mip Dispatches (1 .. hiz_levels - 1)
            let mut prev_w = hiz_extent.width;
            let mut prev_h = hiz_extent.height;

            for k in 1..hiz_levels as usize {
                let cur_w = (prev_w >> 1).max(1);
                let cur_h = (prev_h >> 1).max(1);

                // Memory barrier: STORAGE_WRITE (mip k-1) -> SAMPLED_READ (for mip k)
                let mip_barrier = vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
                let mip_barriers = [mip_barrier];
                let dep_info = vk::DependencyInfo::default().memory_barriers(&mip_barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                let sub_mip_pc = HiZPushConstants {
                    in_extent: [prev_w, prev_h],
                    out_extent: [cur_w, cur_h],
                    is_mip0: 0,
                    _pad: 0,
                };
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    hiz_pipe.layout(),
                    0,
                    &[self.hiz_descriptor_sets[k]],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    hiz_pipe.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&sub_mip_pc),
                );
                let gx = cur_w.div_ceil(16);
                let gy = cur_h.div_ceil(16);
                device.cmd_dispatch(cmd, gx, gy, 1);

                prev_w = cur_w;
                prev_h = cur_h;
            }

            // Transition Hi-Z: GENERAL -> SHADER_READ_ONLY_OPTIMAL
            let hiz_to_sample_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(hiz_pyramid.image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: hiz_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let hiz_to_sample_barriers = [hiz_to_sample_barrier];
            let dep_info =
                vk::DependencyInfo::default().image_memory_barriers(&hiz_to_sample_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // ==========================================
            // PHASE 2: LATE COMPUTE CULL
            // ==========================================
            // Clear Late Count Buffers
            device.cmd_fill_buffer(cmd, mdi.chunk_counts_late.raw(), 0, 16, 0);
            device.cmd_fill_buffer(cmd, mdi.lod_counts_late.raw(), 0, 16, 0);

            let count_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
            let count_barriers = [count_barrier];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&count_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            let late_cull_pc = CullPushConstants {
                view_proj,
                camera_pos: [
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                ],
                cull_phase: 1, // Late Phase!
                hiz_width: hiz_extent.width,
                hiz_height: hiz_extent.height,
                candidate_count: chunk_candidates.len() as u32,
                is_first_frame,
                pointers_address: mdi.cull_ptrs_late.device_address(),
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull_chunks_pipe.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cull_chunks_pipe.layout(),
                0,
                &[cull_desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                cull_chunks_pipe.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&late_cull_pc),
            );
            if !chunk_candidates.is_empty() {
                let groups = (chunk_candidates.len() as u32).div_ceil(64);
                device.cmd_dispatch(cmd, groups, 1, 1);
            }

            // Late Cull LOD
            let late_lod_cull_pc = CullPushConstants {
                view_proj,
                camera_pos: [
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                ],
                cull_phase: 1, // Late Phase!
                hiz_width: hiz_extent.width,
                hiz_height: hiz_extent.height,
                candidate_count: lod_candidates.len() as u32,
                is_first_frame,
                pointers_address: mdi.lod_ptrs_late.device_address(),
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull_lod_pipe.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cull_lod_pipe.layout(),
                0,
                &[cull_desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                cull_lod_pipe.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&late_lod_cull_pc),
            );
            if !lod_candidates.is_empty() {
                let groups = (lod_candidates.len() as u32).div_ceil(64);
                device.cmd_dispatch(cmd, groups, 1, 1);
            }

            // Barrier: Phase 2 Compute Write -> Indirect Command Read + Vertex Shader Read (BDA)
            let late_render_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::VERTEX_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
                );

            // Transition depth buffer back to DEPTH_ATTACHMENT_OPTIMAL for Late pass
            let depth_back_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(depth_buffer.raw())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let late_render_barriers = [late_render_barrier];
            let depth_back_barriers = [depth_back_barrier];
            let dep_info = vk::DependencyInfo::default()
                .memory_barriers(&late_render_barriers)
                .image_memory_barriers(&depth_back_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // ==========================================
            // PHASE 2: LATE RASTERIZATION PASSES
            // ==========================================
            let late_color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD) // Preserve Early Pass color!
                .store_op(vk::AttachmentStoreOp::STORE);

            let late_depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_buffer.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD) // Preserve Early Pass depth!
                .store_op(vk::AttachmentStoreOp::DONT_CARE);

            let late_color_attachments = [late_color_attachment];
            let late_rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .layer_count(1)
                .color_attachments(&late_color_attachments)
                .depth_attachment(&late_depth_attachment);

            device.cmd_begin_rendering(cmd, &late_rendering_info);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Late Pass 1: Opaque T0
            let late_t0_pc = TerrainMdiPushConstants {
                view_proj,
                draw_info_buffer_address: mdi.opaque_draw_late.device_address(),
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.raw());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pipeline.layout(),
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&late_t0_pc),
            );
            device.cmd_draw_indirect_count(
                cmd,
                mdi.opaque_cmd_late.raw(),
                0,
                mdi.chunk_counts_late.raw(),
                0,
                MAX_CHUNK_CANDIDATES as u32,
                16,
            );

            // Late Pass 2: Opaque T1
            if let Some(t1_pipeline) = &self.t1_pipeline {
                let late_t1_pc = TerrainMdiPushConstants {
                    view_proj,
                    draw_info_buffer_address: mdi.t1_draw_late.device_address(),
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, t1_pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    t1_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    t1_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&late_t1_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.t1_cmd_late.raw(),
                    0,
                    mdi.chunk_counts_late.raw(),
                    4,
                    MAX_CHUNK_CANDIDATES as u32,
                    16,
                );
            }

            // Late Pass 3: Cutout
            if let Some(cutout_pipeline) = &self.cutout_pipeline {
                let late_cutout_pc = TerrainMdiPushConstants {
                    view_proj,
                    draw_info_buffer_address: mdi.cutout_draw_late.device_address(),
                };
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
                device.cmd_push_constants(
                    cmd,
                    cutout_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&late_cutout_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.cutout_cmd_late.raw(),
                    0,
                    mdi.chunk_counts_late.raw(),
                    8,
                    MAX_CHUNK_CANDIDATES as u32,
                    16,
                );
            }

            // Late Pass 4: LOD
            if let Some(lod_pipeline) = &self.lod_pipeline {
                let max_dist = (self.view_distance as f32 * 32.0) * 4.0;
                let late_lod_pc = LodMdiPushConstants {
                    view_proj,
                    camera_pos: [
                        self.camera.position.x,
                        self.camera.position.y,
                        self.camera.position.z,
                    ],
                    max_distance: max_dist,
                    draw_info_buffer_address: mdi.lod_draw_late.device_address(),
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, lod_pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    lod_pipeline.layout(),
                    0,
                    &[descriptor_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    lod_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&late_lod_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.lod_cmd_late.raw(),
                    0,
                    mdi.lod_counts_late.raw(),
                    0,
                    MAX_LOD_CANDIDATES as u32,
                    16,
                );
            }

            // Pass 12: Celestial Sky Pass (Fullscreen reversed-Z triangle at z = 0.0)
            if let (Some(sky_pipe), Some(sky_set)) = (&self.sky_pipeline, self.sky_descriptor_set) {
                let sky_pc = SkyPushConstants {
                    inv_view_proj,
                    sun_dir: sun_dir.to_array(),
                    time_of_day: self.client_time_of_day,
                    moon_dir: moon_dir.to_array(),
                    moon_phase,
                    camera_pos: [
                        self.camera.position.x,
                        self.camera.position.y,
                        self.camera.position.z,
                    ],
                    rain_level: self.weather_rain_level,
                    thunder_level: self.weather_thunder_level,
                    lightning_flash: self.weather_lightning_flash,
                    _pad0: 0.0,
                    _pad1: 0.0,
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sky_pipe.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sky_pipe.layout(),
                    0,
                    &[sky_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    sky_pipe.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&sky_pc),
                );
                device.cmd_draw(cmd, 3, 1, 0, 0);
            }

            // Late Pass 5: Translucent (Water)
            if let Some(trans_pipeline) = &self.translucent_pipeline {
                let trans_pc = TranslucentMdiPushConstants {
                    view_proj,
                    draw_info_buffer_address: mdi.translucent_draw_late.device_address(),
                    frame_tick: self.frame_tick,
                    water_base_layer: self.water_anim_info.base_layer,
                    water_frame_count: self.water_anim_info.frame_count,
                    _pad: 0,
                };
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
                device.cmd_push_constants(
                    cmd,
                    trans_pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&trans_pc),
                );
                device.cmd_draw_indirect_count(
                    cmd,
                    mdi.translucent_cmd_late.raw(),
                    0,
                    mdi.chunk_counts_late.raw(),
                    12,
                    MAX_CHUNK_CANDIDATES as u32,
                    16,
                );
            }

            // Pass 14: Dynamic Entities (Mobs: Zombie, Pig, Cow)
            if self.entity_vertex_count > 0
                && let (Some(entity_pipe), Some(entity_set), Some(entity_buf)) = (
                    &self.entity_pipeline,
                    self.entity_descriptor_set,
                    &self.entity_buffer,
                )
            {
                let entity_pc = EntityPushConstants {
                    view_proj,
                    vertex_buffer_address: entity_buf.device_address(),
                    pad: [0, 0],
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, entity_pipe.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    entity_pipe.layout(),
                    0,
                    &[entity_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    entity_pipe.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&entity_pc),
                );
                device.cmd_draw(cmd, self.entity_vertex_count, 1, 0, 0);
            }

            // Pass 17: Atmospheric Precipitation Particles (Rain streaks & fluttering Snow)
            if self.weather_particles_count > 0
                && let (Some(weather_pipe), Some(weather_set), Some(weather_buf)) = (
                    &self.weather_pipeline,
                    self.weather_descriptor_set,
                    &self.weather_buffer,
                )
            {
                let weather_pc = WeatherPushConstants {
                    view_proj,
                    camera_right: self.camera.right().to_array(),
                    pad0: 0.0,
                    camera_up: self.camera.up().to_array(),
                    pad1: 0.0,
                    particle_buffer_address: weather_buf.device_address(),
                    pad2: [0, 0],
                };
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, weather_pipe.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    weather_pipe.layout(),
                    0,
                    &[weather_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    weather_pipe.layout(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&weather_pc),
                );
                device.cmd_draw(cmd, 6, self.weather_particles_count, 0, 0);
            }

            // Pass 6: Block selection wireframe highlight
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
                    device.cmd_push_constants(
                        cmd,
                        highlight_pipeline.layout(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&pc),
                    );
                    device.cmd_draw(cmd, 24, 1, 0, 0);
                }
            }

            device.cmd_end_rendering(cmd);

            // ==========================================
            // PHASE 3: RETAINED GUI & HUD PASS (Phase 16)
            // ==========================================
            if let (Some(ui_pipeline), Some(ui_descriptor_set), Some(font)) =
                (&self.ui_pipeline, self.ui_descriptor_set, &self.ui_font)
            {
                let gui_scale = compute_gui_scale(swapchain_extent.width, swapchain_extent.height);

                // Update HUD dynamic state
                self.hud_state.selected_slot = self.selected_hotbar_slot;
                self.hud_state.player_pos = [
                    f64::from(self.camera.position.x),
                    f64::from(self.camera.position.y),
                    f64::from(self.camera.position.z),
                ];
                let (cpos, _) = vx_voxel::coords::split_block_pos(BlockPos::new(
                    self.camera.position.x.floor() as i32,
                    self.camera.position.y.floor() as i32,
                    self.camera.position.z.floor() as i32,
                ));
                self.hud_state.chunk_pos = [cpos.x(), cpos.y(), cpos.z()];

                let yaw_deg = self.camera.yaw.to_degrees();
                let pitch_deg = self.camera.pitch.to_degrees();
                self.hud_state.yaw = yaw_deg;
                self.hud_state.pitch = pitch_deg;

                let normalized_yaw = (yaw_deg % 360.0 + 360.0) % 360.0;
                self.hud_state.facing = if (45.0..135.0).contains(&normalized_yaw) {
                    "East (+X)".to_string()
                } else if (135.0..225.0).contains(&normalized_yaw) {
                    "South (+Z)".to_string()
                } else if (225.0..315.0).contains(&normalized_yaw) {
                    "West (-X)".to_string()
                } else {
                    "North (-Z)".to_string()
                };

                self.hud_state.frame_time_ms = (dt * 1000.0).max(0.01);
                self.hud_state.chunks_rendered = self.visible_chunks_last as u32;
                self.hud_state.lod_nodes_rendered = self.visible_lod_nodes_last as u32;
                self.hud_state.time_of_day = self.client_time_of_day as u64;
                self.hud_state.day_number = day_number;
                self.hud_state.moon_phase_name = match moon_phase {
                    0 => "Full Moon",
                    1 => "Waning Gibbous",
                    2 => "Third Quarter",
                    3 => "Waning Crescent",
                    4 => "New Moon",
                    5 => "Waxing Crescent",
                    6 => "First Quarter",
                    7 => "Waxing Gibbous",
                    _ => "Unknown",
                }
                .to_string();

                let weather_kind_name = if self.weather_thunder_level > 0.05 {
                    "Thunder"
                } else if self.weather_rain_level > 0.05 {
                    "Rain"
                } else {
                    "Clear"
                };
                self.hud_state.weather_name = weather_kind_name.to_string();
                self.hud_state.weather_rain_level = self.weather_rain_level;
                self.hud_state.weather_thunder_level = self.weather_thunder_level;

                let climate = vx_worldgen::ClimatePoint::sample(
                    self.world_seed,
                    self.camera.position.x,
                    self.camera.position.z,
                );
                let is_dry = climate.humidity < -0.35 && climate.temperature > 0.5;
                let local_precip =
                    if self.weather_rain_level <= 0.01 && self.weather_thunder_level <= 0.01 {
                        "None"
                    } else {
                        match vx_sim::weather::precipitation_at(
                            climate.temperature,
                            self.camera.position.y,
                            is_dry,
                        ) {
                            vx_sim::weather::PrecipitationKind::None => "None (Dry)",
                            vx_sim::weather::PrecipitationKind::Rain => "Rain",
                            vx_sim::weather::PrecipitationKind::Snow => "Snow",
                        }
                    };
                self.hud_state.local_precipitation = local_precip.to_string();

                let mut ui_quads = Vec::with_capacity(512);
                render_hud(
                    &self.hud_state,
                    font,
                    &self.ui_layers,
                    swapchain_extent.width,
                    swapchain_extent.height,
                    gui_scale,
                    &mut ui_quads,
                );

                if self.inventory_open {
                    let mut ui_slots = [UiSlotItem::EMPTY; vx_ui::INVENTORY_SLOT_COUNT];
                    for (i, slot) in self
                        .inventory_sim
                        .slots
                        .iter()
                        .enumerate()
                        .take(vx_ui::INVENTORY_SLOT_COUNT)
                    {
                        ui_slots[i] = UiSlotItem::new(slot.item, slot.count);
                    }
                    let ui_carried = UiSlotItem::new(
                        self.inventory_sim.carried.item,
                        self.inventory_sim.carried.count,
                    );
                    let item_lookup = |id: u32| {
                        if let Some(def) = self.registries.item_registry().get_by_id(id) {
                            def.name.as_str()
                        } else {
                            vx_sim::item_name(id)
                        }
                    };
                    render_inventory_screen(
                        &ui_slots,
                        ui_carried,
                        self.inventory_hovered_slot,
                        swapchain_extent.width,
                        swapchain_extent.height,
                        gui_scale,
                        font,
                        &self.ui_layers,
                        item_lookup,
                        self.mouse_cursor_pos,
                        &mut ui_quads,
                    );
                }

                if !ui_quads.is_empty() {
                    let required_bytes =
                        (ui_quads.len() * std::mem::size_of::<UiQuad>()) as vk::DeviceSize;
                    if let Some(ui_buf) = &mut self.ui_buffer {
                        let allocator = gpu_context.allocator();
                        if ui_buf.size() < required_bytes {
                            ui_buf.destroy(device, allocator);
                            if let Ok(new_buf) = GpuBuffer::new(
                                device,
                                allocator,
                                "ui_quad_buffer",
                                required_bytes.max(65536),
                                vk::BufferUsageFlags::STORAGE_BUFFER
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                                MemoryLocation::CpuToGpu,
                            ) {
                                *ui_buf = new_buf;
                            }
                        }
                        if let Err(err) = ui_buf.write_bytes(bytemuck::cast_slice(&ui_quads)) {
                            tracing::error!("Failed to write UI quad buffer: {err}");
                        }

                        let ui_color_attachment = vk::RenderingAttachmentInfo::default()
                            .image_view(image_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE);
                        let ui_color_attachments = [ui_color_attachment];
                        let ui_rendering_info = vk::RenderingInfo::default()
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: swapchain_extent,
                            })
                            .layer_count(1)
                            .color_attachments(&ui_color_attachments);

                        device.cmd_begin_rendering(cmd, &ui_rendering_info);

                        let ui_viewport = vk::Viewport::default()
                            .x(0.0)
                            .y(0.0)
                            .width(swapchain_extent.width as f32)
                            .height(swapchain_extent.height as f32)
                            .min_depth(0.0)
                            .max_depth(1.0);
                        let ui_scissor = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: swapchain_extent,
                        };
                        device.cmd_set_viewport(cmd, 0, &[ui_viewport]);
                        device.cmd_set_scissor(cmd, 0, &[ui_scissor]);

                        let ui_pc = UiPushConstants {
                            viewport_size: [
                                swapchain_extent.width as f32,
                                swapchain_extent.height as f32,
                            ],
                            quad_buffer_address: ui_buf.device_address(),
                        };
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            ui_pipeline.raw(),
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            ui_pipeline.layout(),
                            0,
                            &[ui_descriptor_set],
                            &[],
                        );
                        device.cmd_push_constants(
                            cmd,
                            ui_pipeline.layout(),
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&ui_pc),
                        );
                        device.cmd_draw(cmd, 6, ui_quads.len() as u32, 0, 0);

                        device.cmd_end_rendering(cmd);
                    }
                }
            }
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

        let lightmap_texture = match GpuTexture2d::new_empty(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            16,
            16,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::Filter::LINEAR,
        ) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!("Failed to create lightmap texture: {err}");
                event_loop.exit();
                return;
            }
        };

        let mut lightmap_staging_buffers = Vec::with_capacity(2);
        let staging_names = ["lightmap_staging_0", "lightmap_staging_1"];
        for name in staging_names {
            let buf = match GpuBuffer::new(
                gpu_context.device().raw(),
                gpu_context.allocator(),
                name,
                1024,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            ) {
                Ok(b) => b,
                Err(err) => {
                    tracing::error!("Failed to create lightmap staging buffer: {err}");
                    event_loop.exit();
                    return;
                }
            };
            lightmap_staging_buffers.push(buf);
        }

        // Create Descriptor Set Layout for binding 0 (sampler2DArray) and binding 1 (sampler2D lightmap)
        let tex_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let lightmap_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [tex_binding, lightmap_binding];
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
            .descriptor_count(2);
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

        let tex_image_info = [vk::DescriptorImageInfo::default()
            .sampler(texture_array.sampler())
            .image_view(texture_array.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let lightmap_image_info = [vk::DescriptorImageInfo::default()
            .sampler(lightmap_texture.sampler())
            .image_view(lightmap_texture.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tex_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&lightmap_image_info),
        ];
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
            .size(size_of::<TerrainMdiPushConstants>() as u32);

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
            .size(size_of::<TranslucentMdiPushConstants>() as u32);

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
            .size(size_of::<LodMdiPushConstants>() as u32);

        let lod_pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            lod_vert_module.raw(),
            lod_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[descriptor_set_layout],
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

        // ---------------------------------------------------------------------
        // Initialize Retained GUI & HUD Resources (Phase 16)
        // ---------------------------------------------------------------------
        let (ui_texture_array, ui_font) = match load_and_upload_ui_textures(&gpu_context) {
            Ok(res) => res,
            Err(err) => {
                tracing::warn!("Failed to load UI textures: {err}; using procedural fallback");
                let fallback_bytes = vec![255u8; 256 * 256 * 4 * 5];
                let fallback_regions: Vec<TextureMipRegion> = (0..5)
                    .map(|layer| TextureMipRegion {
                        buffer_offset: u64::from(layer * 256 * 256 * 4),
                        layer,
                        mip_level: 0,
                        width: 256,
                        height: 256,
                    })
                    .collect();
                let fallback = match gpu_context.create_texture_array(
                    256,
                    5,
                    1,
                    &fallback_bytes,
                    &fallback_regions,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("Failed to create fallback UI texture array: {e}");
                        event_loop.exit();
                        return;
                    }
                };
                (fallback, BitmapFont::new_fallback(3))
            }
        };

        // Query GPU device name for F3 overlay
        let gpu_name = unsafe {
            let props = gpu_context
                .instance()
                .raw()
                .get_physical_device_properties(gpu_context.device().physical_device());
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        self.hud_state.gpu_name = gpu_name;

        // Create UI descriptor set layout (binding 0: sampler2DArray)
        let ui_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let ui_bindings = [ui_binding];
        let ui_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&ui_bindings);
        let ui_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&ui_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create UI descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        // Create UI descriptor pool and allocate descriptor set
        let ui_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let ui_pool_sizes = [ui_pool_size];
        let ui_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&ui_pool_sizes);
        let ui_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&ui_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create UI descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let ui_set_layouts = [ui_descriptor_set_layout];
        let ui_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(ui_descriptor_pool)
            .set_layouts(&ui_set_layouts);
        let ui_descriptor_sets = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&ui_alloc_info)
            {
                Ok(s) => s,
                Err(err) => {
                    tracing::error!("Failed to allocate UI descriptor sets: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };
        let ui_descriptor_set = ui_descriptor_sets[0];

        // Bind UI texture array to descriptor set
        let ui_image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ui_texture_array.view())
            .sampler(ui_texture_array.sampler());
        let ui_image_infos = [ui_image_info];
        let ui_write = vk::WriteDescriptorSet::default()
            .dst_set(ui_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&ui_image_infos);
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&[ui_write], &[]);
        }

        // Load UI shaders
        let ui_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/ui.vert.spv"));
        let ui_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/ui.frag.spv"));
        let ui_vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), ui_vert_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create UI vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };
        let ui_frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), ui_frag_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create UI fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let ui_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<UiPushConstants>() as u32);

        let ui_pipeline = match GraphicsPipeline::create_dynamic_ui(
            gpu_context.device().raw(),
            ui_vert_module.raw(),
            ui_frag_module.raw(),
            gpu_context.swapchain().format(),
            &[ui_descriptor_set_layout],
            &[ui_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create UI graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        let initial_ui_buf_size = 4096 * std::mem::size_of::<UiQuad>() as vk::DeviceSize;
        let ui_buffer = match GpuBuffer::new(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            "ui_quad_buffer",
            initial_ui_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        ) {
            Ok(b) => b,
            Err(err) => {
                tracing::error!("Failed to create UI quad buffer: {err}");
                event_loop.exit();
                return;
            }
        };

        self.ui_texture_array = Some(ui_texture_array);
        self.ui_font = Some(ui_font);
        self.ui_descriptor_set_layout = Some(ui_descriptor_set_layout);
        self.ui_descriptor_pool = Some(ui_descriptor_pool);
        self.ui_descriptor_set = Some(ui_descriptor_set);
        self.ui_vert_shader = Some(ui_vert_module);
        self.ui_frag_shader = Some(ui_frag_module);
        self.ui_pipeline = Some(ui_pipeline);
        self.ui_buffer = Some(ui_buffer);

        // ---------------------------------------------------------------------
        // Initialize Celestial & Sky Pass Resources (Phase 17)
        // ---------------------------------------------------------------------
        let celestial_texture = match load_and_upload_celestial_textures(&gpu_context) {
            Ok(tex) => tex,
            Err(err) => {
                tracing::error!("Failed to load celestial textures: {err}");
                event_loop.exit();
                return;
            }
        };

        let sky_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let sky_bindings = [sky_binding];
        let sky_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&sky_bindings);
        let sky_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&sky_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create sky descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let sky_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let sky_pool_sizes = [sky_pool_size];
        let sky_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sky_pool_sizes);
        let sky_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&sky_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create sky descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let sky_set_layouts = [sky_descriptor_set_layout];
        let sky_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(sky_descriptor_pool)
            .set_layouts(&sky_set_layouts);
        let sky_descriptor_set = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&sky_alloc_info)
            {
                Ok(sets) => sets[0],
                Err(err) => {
                    tracing::error!("Failed to allocate sky descriptor set: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let celestial_image_info = [vk::DescriptorImageInfo::default()
            .sampler(celestial_texture.sampler())
            .image_view(celestial_texture.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sky_descriptor_write = [vk::WriteDescriptorSet::default()
            .dst_set(sky_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&celestial_image_info)];
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&sky_descriptor_write, &[]);
        }

        let sky_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv"));
        let sky_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"));
        let sky_vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), sky_vert_spv)
        {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create sky vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };
        let sky_frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), sky_frag_spv)
        {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create sky fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let sky_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<SkyPushConstants>() as u32);

        let sky_pipeline = match GraphicsPipeline::create_dynamic_sky(
            gpu_context.device().raw(),
            sky_vert_module.raw(),
            sky_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            &[sky_descriptor_set_layout],
            &[sky_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create sky graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        // ---------------------------------------------------------------------
        // Initialize Atmospheric Weather & Precipitation Resources (Phase 20)
        // ---------------------------------------------------------------------
        let weather_texture = match load_and_upload_weather_textures(&gpu_context) {
            Ok(tex) => tex,
            Err(err) => {
                tracing::error!("Failed to load weather textures: {err}");
                event_loop.exit();
                return;
            }
        };

        let weather_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let weather_bindings = [weather_binding];
        let weather_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&weather_bindings);
        let weather_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&weather_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create weather descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let weather_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let weather_pool_sizes = [weather_pool_size];
        let weather_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&weather_pool_sizes);
        let weather_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&weather_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create weather descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let weather_set_layouts = [weather_descriptor_set_layout];
        let weather_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(weather_descriptor_pool)
            .set_layouts(&weather_set_layouts);
        let weather_descriptor_set = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&weather_alloc_info)
            {
                Ok(sets) => sets[0],
                Err(err) => {
                    tracing::error!("Failed to allocate weather descriptor set: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let weather_image_info = [vk::DescriptorImageInfo::default()
            .sampler(weather_texture.sampler())
            .image_view(weather_texture.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let weather_descriptor_write = [vk::WriteDescriptorSet::default()
            .dst_set(weather_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&weather_image_info)];
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&weather_descriptor_write, &[]);
        }

        let weather_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/weather.vert.spv"));
        let weather_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/weather.frag.spv"));
        let weather_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), weather_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create weather vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };
        let weather_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), weather_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create weather fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        #[allow(clippy::cast_possible_truncation)]
        let weather_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<WeatherPushConstants>() as u32);

        let weather_pipeline = match GraphicsPipeline::create_dynamic_translucent(
            gpu_context.device().raw(),
            weather_vert_module.raw(),
            weather_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[weather_descriptor_set_layout],
            &[weather_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create weather graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        let weather_buffer = match GpuBuffer::new(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            "weather_particle_buffer",
            (1024 * size_of::<WeatherParticleGpu>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        ) {
            Ok(buf) => buf,
            Err(err) => {
                tracing::error!("Failed to allocate weather particle buffer: {err}");
                event_loop.exit();
                return;
            }
        };

        // ---------------------------------------------------------------------
        // Initialize Dynamic Entity & Mob Rendering Resources (Phase 21)
        // ---------------------------------------------------------------------
        let entity_textures = match load_and_upload_entity_textures(&gpu_context) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!("Failed to load entity textures: {err}");
                event_loop.exit();
                return;
            }
        };

        let entity_binding_tex = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let entity_binding_lightmap = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let entity_bindings = [entity_binding_tex, entity_binding_lightmap];
        let entity_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&entity_bindings);
        let entity_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&entity_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create entity descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let entity_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2);
        let entity_pool_sizes = [entity_pool_size];
        let entity_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&entity_pool_sizes);
        let entity_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&entity_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create entity descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let entity_set_layouts = [entity_descriptor_set_layout];
        let entity_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(entity_descriptor_pool)
            .set_layouts(&entity_set_layouts);
        let entity_descriptor_set = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&entity_alloc_info)
            {
                Ok(sets) => sets[0],
                Err(err) => {
                    tracing::error!("Failed to allocate entity descriptor set: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let entity_image_info = [vk::DescriptorImageInfo::default()
            .sampler(entity_textures.sampler())
            .image_view(entity_textures.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let entity_lightmap_info = [vk::DescriptorImageInfo::default()
            .sampler(lightmap_texture.sampler())
            .image_view(lightmap_texture.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let entity_descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(entity_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&entity_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(entity_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&entity_lightmap_info),
        ];
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&entity_descriptor_writes, &[]);
        }

        let entity_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/entity.vert.spv"));
        let entity_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/entity.frag.spv"));
        let entity_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), entity_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create entity vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };
        let entity_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), entity_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create entity fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        #[allow(clippy::cast_possible_truncation)]
        let entity_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<EntityPushConstants>() as u32);

        let entity_pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            entity_vert_module.raw(),
            entity_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[entity_descriptor_set_layout],
            &[entity_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create entity graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        let entity_buffer = match GpuBuffer::new(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            "entity_vertex_buffer",
            (16384 * size_of::<EntityVertexGpu>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        ) {
            Ok(buf) => buf,
            Err(err) => {
                tracing::error!("Failed to allocate entity vertex buffer: {err}");
                event_loop.exit();
                return;
            }
        };

        self.entity_textures = Some(entity_textures);
        self.entity_descriptor_set_layout = Some(entity_descriptor_set_layout);
        self.entity_descriptor_pool = Some(entity_descriptor_pool);
        self.entity_descriptor_set = Some(entity_descriptor_set);
        self.entity_vert_shader = Some(entity_vert_module);
        self.entity_frag_shader = Some(entity_frag_module);
        self.entity_pipeline = Some(entity_pipeline);
        self.entity_buffer = Some(entity_buffer);

        self.celestial_texture = Some(celestial_texture);
        self.sky_descriptor_set_layout = Some(sky_descriptor_set_layout);
        self.sky_descriptor_pool = Some(sky_descriptor_pool);
        self.sky_descriptor_set = Some(sky_descriptor_set);
        self.sky_vert_shader = Some(sky_vert_module);
        self.sky_frag_shader = Some(sky_frag_module);
        self.sky_pipeline = Some(sky_pipeline);
        self.lightmap_texture = Some(lightmap_texture);
        self.lightmap_staging_buffers = lightmap_staging_buffers;

        self.weather_texture = Some(weather_texture);
        self.weather_descriptor_set_layout = Some(weather_descriptor_set_layout);
        self.weather_descriptor_pool = Some(weather_descriptor_pool);
        self.weather_descriptor_set = Some(weather_descriptor_set);
        self.weather_vert_shader = Some(weather_vert_module);
        self.weather_frag_shader = Some(weather_frag_module);
        self.weather_pipeline = Some(weather_pipeline);
        self.weather_buffer = Some(weather_buffer);

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

        let mdi_buffers = match MdiBuffers::new(&gpu_context) {
            Ok(b) => b,
            Err(err) => {
                tracing::error!("Failed to create MDI buffers: {err}");
                event_loop.exit();
                return;
            }
        };
        self.mdi_buffers = Some(mdi_buffers);

        self.gpu_context = Some(gpu_context);
        self.window = Some(window);

        if let Err(err) = self.setup_hiz_and_culling() {
            tracing::error!("Failed to setup Hi-Z and compute culling: {err}");
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if self.inventory_open {
            return;
        }
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_cursor_pos = [position.x as f32, position.y as f32];
                if self.inventory_open {
                    let win_size = window.inner_size();
                    let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                    self.inventory_hovered_slot = slot_at_pos(
                        self.mouse_cursor_pos,
                        win_size.width,
                        win_size.height,
                        gui_scale,
                    );
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if self.inventory_open {
                    let click_btn = match button {
                        MouseButton::Left => 0u8,
                        MouseButton::Right => 1u8,
                        _ => 255u8,
                    };
                    if click_btn <= 1
                        && let Some(hovered) = self.inventory_hovered_slot
                    {
                        let mode = u8::from(self.shift_held); // 1 = QuickMove, 0 = Pickup
                        let predicted_carried = self.inventory_sim.carried;
                        let click_msg = C2sMessage::InventoryClick(C2sInventoryClick {
                            slot: hovered as u16,
                            button: click_btn,
                            mode,
                            predicted_carried_item: predicted_carried.item,
                            predicted_carried_count: predicted_carried.count,
                        });
                        let _ = self
                            .client_conn
                            .send(Lane::Control, Payload::Msg(click_msg));

                        let btn_sim = if click_btn == 0 {
                            vx_sim::ClickButton::Left
                        } else {
                            vx_sim::ClickButton::Right
                        };
                        let mode_sim = if mode == 1 {
                            vx_sim::ClickMode::QuickMove
                        } else {
                            vx_sim::ClickMode::Pickup
                        };
                        let _ = vx_sim::inventory_click(
                            &mut self.inventory_sim,
                            hovered,
                            btn_sim,
                            mode_sim,
                        );
                        self.selected_block_state =
                            BlockStateId::new(self.inventory_sim.selected_item().item);
                    }
                    return;
                }
                if self.controller.mouse_captured {
                    match button {
                        MouseButton::Left => {
                            // Check combat raycast against entities first
                            let origin = self.camera.position;
                            let forward = self.camera.forward();
                            if let Some((target_net_id, _dist)) = self.entity_store.raycast(
                                glam::DVec3::new(
                                    f64::from(origin.x),
                                    f64::from(origin.y),
                                    f64::from(origin.z),
                                ),
                                forward,
                                3.5,
                            ) {
                                let interact_msg = C2sMessage::InteractEntity(C2sInteractEntity {
                                    target_net_id,
                                    action: 1, // Attack
                                });
                                let _ = self
                                    .client_conn
                                    .send(Lane::Control, Payload::Msg(interact_msg));
                                return;
                            }

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
                if code == KeyCode::ShiftLeft || code == KeyCode::ShiftRight {
                    self.shift_held = pressed;
                }

                if self.inventory_open {
                    match code {
                        KeyCode::KeyE if pressed => {
                            self.inventory_open = false;
                            self.controller.mouse_captured = true;
                            let _ = window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                .or_else(|_| {
                                    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                });
                            window.set_cursor_visible(false);
                            self.inventory_hovered_slot = None;
                        }
                        KeyCode::Escape if pressed => {
                            self.inventory_open = false;
                            self.controller.mouse_captured = true;
                            let _ = window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                .or_else(|_| {
                                    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                });
                            window.set_cursor_visible(false);
                            self.inventory_hovered_slot = None;
                        }
                        KeyCode::Digit1 if pressed => self.handle_inventory_swap_hotbar(0),
                        KeyCode::Digit2 if pressed => self.handle_inventory_swap_hotbar(1),
                        KeyCode::Digit3 if pressed => self.handle_inventory_swap_hotbar(2),
                        KeyCode::Digit4 if pressed => self.handle_inventory_swap_hotbar(3),
                        KeyCode::Digit5 if pressed => self.handle_inventory_swap_hotbar(4),
                        KeyCode::Digit6 if pressed => self.handle_inventory_swap_hotbar(5),
                        KeyCode::Digit7 if pressed => self.handle_inventory_swap_hotbar(6),
                        KeyCode::Digit8 if pressed => self.handle_inventory_swap_hotbar(7),
                        KeyCode::Digit9 if pressed => self.handle_inventory_swap_hotbar(8),
                        _ => {}
                    }
                    return;
                }

                match code {
                    KeyCode::KeyE if pressed => {
                        self.inventory_open = true;
                        self.controller.mouse_captured = false;
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        self.controller.forward = false;
                        self.controller.backward = false;
                        self.controller.left = false;
                        self.controller.right = false;
                        self.controller.up = false;
                        self.controller.down = false;
                        let win_size = window.inner_size();
                        let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                        self.inventory_hovered_slot = slot_at_pos(
                            self.mouse_cursor_pos,
                            win_size.width,
                            win_size.height,
                            gui_scale,
                        );
                    }
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
                    KeyCode::F3 if pressed => {
                        self.hud_state.f3_open = !self.hud_state.f3_open;
                        info!(f3_open = self.hud_state.f3_open, "Toggled F3 debug overlay");
                    }
                    KeyCode::F7 if pressed => {
                        let next_kind =
                            if self.weather_rain_level < 0.1 && self.weather_thunder_level < 0.1 {
                                1u8 // Rain
                            } else if self.weather_thunder_level < 0.1 {
                                2u8 // Thunder
                            } else {
                                0u8 // Clear
                            };
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::SetWeather(next_kind),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!(next_weather = next_kind, "Sent toggle weather command");
                    }
                    KeyCode::F8 if pressed => {
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::TriggerLightning,
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!("Sent trigger lightning command");
                    }
                    KeyCode::F9 if pressed => {
                        let mob_type = self.mob_spawn_type_cycle;
                        self.mob_spawn_type_cycle = match self.mob_spawn_type_cycle {
                            1 => 2, // Zombie -> Pig
                            2 => 3, // Pig -> Cow
                            _ => 1, // Cow -> Zombie
                        };
                        let origin = self.camera.position;
                        let forward = self.camera.forward();
                        let spawn_pos = origin + forward * 3.0;
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::SpawnMob {
                                mob_type,
                                x: f64::from(spawn_pos.x),
                                y: f64::from(spawn_pos.y),
                                z: f64::from(spawn_pos.z),
                            },
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!(
                            mob_type,
                            next_cycle = self.mob_spawn_type_cycle,
                            "Sent spawn mob command (F9)"
                        );
                    }
                    KeyCode::F10 if pressed => {
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::ClearMobs,
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!("Sent clear mobs command (F10)");
                    }
                    KeyCode::KeyK if pressed => {
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::Damage(2.0),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!("Sent test damage command (2.0 HP)");
                    }
                    KeyCode::KeyJ if pressed => {
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::AddXp(50),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!("Sent test add XP command (50 XP)");
                    }
                    KeyCode::KeyL if pressed => {
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::SetFood(6),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!("Sent test set food command (6 food)");
                    }
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
            WindowEvent::MouseWheel { delta, .. } => {
                if self.inventory_open {
                    return;
                }
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    #[allow(clippy::cast_possible_truncation)]
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                if scroll > 0.0 {
                    let new_slot = (self.selected_hotbar_slot + 8) % 9;
                    self.select_hotbar_slot(new_slot);
                } else if scroll < 0.0 {
                    let new_slot = (self.selected_hotbar_slot + 1) % 9;
                    self.select_hotbar_slot(new_slot);
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
                    let extent = vk::Extent2D {
                        width: physical_size.width,
                        height: physical_size.height,
                    };
                    if let Err(err) = self.recreate_hiz_resources(extent) {
                        tracing::error!("Failed to recreate Hi-Z resources on resize: {err}");
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

    #[allow(clippy::too_many_lines)]
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.server_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.server_handle.take() {
            let _ = handle.join();
        }

        if let Some(gpu_context) = &mut self.gpu_context {
            let _ = gpu_context.wait_idle();
            let device = gpu_context.device().raw();
            let allocator = gpu_context.allocator();

            if let Some(mut mdi) = self.mdi_buffers.take() {
                mdi.destroy(device, allocator);
            }

            if let Some(mut pipeline) = self.hiz_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.hiz_comp_shader.take() {
                vert.destroy(device);
            }

            if let Some(mut pipeline) = self.cull_chunks_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.cull_chunks_comp_shader.take() {
                vert.destroy(device);
            }

            if let Some(mut pipeline) = self.cull_lod_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.cull_lod_comp_shader.take() {
                vert.destroy(device);
            }

            if let Some(mut hiz) = self.hiz_pyramid.take() {
                hiz.destroy(device, allocator);
            }

            // SAFETY: Destroying sampler and descriptor pools/layouts on valid Vulkan device
            unsafe {
                if self.point_clamp_sampler != vk::Sampler::null() {
                    device.destroy_sampler(self.point_clamp_sampler, None);
                    self.point_clamp_sampler = vk::Sampler::null();
                }
                if let Some(pool) = self.hiz_descriptor_pool.take() {
                    device.destroy_descriptor_pool(pool, None);
                }
                if let Some(layout) = self.hiz_descriptor_set_layout.take() {
                    device.destroy_descriptor_set_layout(layout, None);
                }
                if let Some(layout) = self.cull_descriptor_set_layout.take() {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }

            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.t1_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.t1_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.t1_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.cutout_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.cutout_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.cutout_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.translucent_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.translucent_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.translucent_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.lod_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.lod_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.lod_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.highlight_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.highlight_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.highlight_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(mut pipeline) = self.ui_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.ui_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.ui_frag_shader.take() {
                frag.destroy(device);
            }

            if let Some(pool) = self.ui_descriptor_pool.take() {
                // SAFETY: Destroying UI descriptor pool on valid device
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }

            if let Some(layout) = self.ui_descriptor_set_layout.take() {
                // SAFETY: Destroying UI descriptor set layout on valid device
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }

            if let Some(mut tex) = self.ui_texture_array.take() {
                tex.destroy(device, allocator);
            }

            if let Some(mut buf) = self.ui_buffer.take() {
                buf.destroy(device, allocator);
            }

            if let Some(pool) = self.descriptor_pool.take() {
                // SAFETY: Destroying descriptor pool on valid device
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }

            if let Some(layout) = self.descriptor_set_layout.take() {
                // SAFETY: Destroying descriptor set layout on valid device
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }

            if let Some(mut tex) = self.texture_array.take() {
                tex.destroy(device, allocator);
            }

            for (_, mesh) in self.chunk_meshes.drain() {
                mesh.destroy(gpu_context);
            }

            for (_, mut mesh) in self.lod_meshes.drain() {
                mesh.buffer.destroy(device, allocator);
            }

            if let Some(mut depth) = self.depth_buffer.take() {
                depth.destroy(device, allocator);
            }

            if let Some(mut pipeline) = self.sky_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.sky_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.sky_frag_shader.take() {
                frag.destroy(device);
            }
            if let Some(pool) = self.sky_descriptor_pool.take() {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            if let Some(layout) = self.sky_descriptor_set_layout.take() {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
            if let Some(mut tex) = self.celestial_texture.take() {
                tex.destroy(device, allocator);
            }
            if let Some(mut tex) = self.lightmap_texture.take() {
                tex.destroy(device, allocator);
            }
            for mut buf in self.lightmap_staging_buffers.drain(..) {
                buf.destroy(device, allocator);
            }

            if let Some(mut pipeline) = self.weather_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.weather_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.weather_frag_shader.take() {
                frag.destroy(device);
            }
            if let Some(pool) = self.weather_descriptor_pool.take() {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            if let Some(layout) = self.weather_descriptor_set_layout.take() {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
            if let Some(mut tex) = self.weather_texture.take() {
                tex.destroy(device, allocator);
            }
            if let Some(mut buf) = self.weather_buffer.take() {
                buf.destroy(device, allocator);
            }

            if let Some(mut pipeline) = self.entity_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.entity_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.entity_frag_shader.take() {
                frag.destroy(device);
            }
            if let Some(pool) = self.entity_descriptor_pool.take() {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            if let Some(layout) = self.entity_descriptor_set_layout.take() {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
            if let Some(mut tex) = self.entity_textures.take() {
                tex.destroy(device, allocator);
            }
            if let Some(mut buf) = self.entity_buffer.take() {
                buf.destroy(device, allocator);
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

#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
fn load_and_upload_ui_textures(gpu_context: &GpuContext) -> Result<(GpuTextureArray, BitmapFont)> {
    const UI_RES: u32 = 256;

    let mut stack = ResourcePackStack::new();
    stack.add_root("dev-assets/faithful-32x");
    stack.add_root("dev-assets/classic-26.2");
    stack.add_root("assets/voxel");

    let mut pixel_data = vec![0u8; (UI_RES * UI_RES * 4 * 7) as usize];

    // Helper to copy a sub-image into a 256x256 layer at specified offset
    let copy_to_layer_at = |dest: &mut [u8],
                            layer: usize,
                            x_offset: usize,
                            y_offset: usize,
                            img: &vx_assets::RgbaImage| {
        let layer_offset = layer * (UI_RES * UI_RES * 4) as usize;
        let w = (img.width as usize).min(UI_RES as usize - x_offset);
        let h = (img.height as usize).min(UI_RES as usize - y_offset);
        for y in 0..h {
            let src_start = (y * img.width as usize) * 4;
            let src_end = src_start + w * 4;
            let dst_start = layer_offset + ((y_offset + y) * UI_RES as usize + x_offset) * 4;
            dest[dst_start..dst_start + w * 4].copy_from_slice(&img.data[src_start..src_end]);
        }
    };

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &vx_assets::RgbaImage| {
        copy_to_layer_at(dest, layer, 0, 0, img);
    };

    // Layer 0: Hotbar (182x22)
    let hotbar_img = stack.load_gui_sprite("hud/hotbar").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(182, 22);
        for i in (0..img.data.len()).step_by(4) {
            img.data[i] = 40;
            img.data[i + 1] = 40;
            img.data[i + 2] = 40;
            img.data[i + 3] = 200;
        }
        img
    });
    copy_to_layer(&mut pixel_data, 0, &hotbar_img);

    // Layer 1: Hotbar Selection (24x23)
    let selection_img = stack
        .load_gui_sprite("hud/hotbar_selection")
        .unwrap_or_else(|_| {
            let mut img = vx_assets::RgbaImage::new(24, 23);
            for y in 0..23 {
                for x in 0..24 {
                    let idx = ((y * 24 + x) * 4) as usize;
                    let is_border = x == 0 || x == 23 || y == 0 || y == 22;
                    if is_border {
                        img.data[idx] = 255;
                        img.data[idx + 1] = 255;
                        img.data[idx + 2] = 255;
                        img.data[idx + 3] = 255;
                    }
                }
            }
            img
        });
    copy_to_layer(&mut pixel_data, 1, &selection_img);

    // Layer 2: Crosshair (15x15)
    let crosshair_img = stack.load_gui_sprite("hud/crosshair").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(15, 15);
        for i in 0..15 {
            let h_idx = ((7 * 15 + i) * 4) as usize;
            img.data[h_idx] = 255;
            img.data[h_idx + 1] = 255;
            img.data[h_idx + 2] = 255;
            img.data[h_idx + 3] = 255;

            let v_idx = ((i * 15 + 7) * 4) as usize;
            img.data[v_idx] = 255;
            img.data[v_idx + 1] = 255;
            img.data[v_idx + 2] = 255;
            img.data[v_idx + 3] = 255;
        }
        img
    });
    copy_to_layer(&mut pixel_data, 2, &crosshair_img);

    // Layer 3: Ascii Font (128x128 -> scale 2x to 256x256)
    let ascii_img = stack.load_font_texture("ascii").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(128, 128);
        for (i, b) in img.data.iter_mut().enumerate() {
            if i % 4 == 3 {
                *b = 255;
            } else {
                *b = 200;
            }
        }
        img
    });

    // Scale ascii 2x into Layer 3
    let font = {
        let mut ascii_256 = vec![0u8; (256 * 256 * 4) as usize];
        let scale = (256 / ascii_img.width).max(1);
        for y in 0..ascii_img.height {
            for x in 0..ascii_img.width {
                let src_idx = ((y * ascii_img.width + x) * 4) as usize;
                let r = ascii_img.data[src_idx];
                let g = ascii_img.data[src_idx + 1];
                let b = ascii_img.data[src_idx + 2];
                let a = ascii_img.data[src_idx + 3];

                for dy in 0..scale {
                    for dx in 0..scale {
                        let dst_x = x * scale + dx;
                        let dst_y = y * scale + dy;
                        if dst_x < 256 && dst_y < 256 {
                            let dst_idx = ((dst_y * 256 + dst_x) * 4) as usize;
                            ascii_256[dst_idx] = r;
                            ascii_256[dst_idx + 1] = g;
                            ascii_256[dst_idx + 2] = b;
                            ascii_256[dst_idx + 3] = a;
                        }
                    }
                }
            }
        }

        let layer3_offset = 3 * (UI_RES * UI_RES * 4) as usize;
        pixel_data[layer3_offset..layer3_offset + (256 * 256 * 4) as usize]
            .copy_from_slice(&ascii_256);

        BitmapFont::from_rgba(&ascii_256, 256, 256, 3)
    };

    // Layer 4: Survival Icons & Bars
    let heart_container_img = stack
        .load_gui_sprite("hud/heart/container")
        .unwrap_or_else(|_| {
            let mut img = vx_assets::RgbaImage::new(9, 9);
            for y in 0..9 {
                for x in 0..9 {
                    let idx = ((y * 9 + x) * 4) as usize;
                    if x == 0 || x == 8 || y == 0 || y == 8 {
                        img.data[idx] = 60;
                        img.data[idx + 1] = 60;
                        img.data[idx + 2] = 60;
                        img.data[idx + 3] = 255;
                    }
                }
            }
            img
        });
    copy_to_layer_at(&mut pixel_data, 4, 0, 0, &heart_container_img);

    let heart_full_img = stack.load_gui_sprite("hud/heart/full").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(9, 9);
        for y in 1..8 {
            for x in 1..8 {
                let idx = ((y * 9 + x) * 4) as usize;
                img.data[idx] = 230;
                img.data[idx + 1] = 30;
                img.data[idx + 2] = 30;
                img.data[idx + 3] = 255;
            }
        }
        img
    });
    copy_to_layer_at(&mut pixel_data, 4, 16, 0, &heart_full_img);

    let heart_half_img = stack.load_gui_sprite("hud/heart/half").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(9, 9);
        for y in 1..8 {
            for x in 1..5 {
                let idx = ((y * 9 + x) * 4) as usize;
                img.data[idx] = 230;
                img.data[idx + 1] = 30;
                img.data[idx + 2] = 30;
                img.data[idx + 3] = 255;
            }
        }
        img
    });
    copy_to_layer_at(&mut pixel_data, 4, 32, 0, &heart_half_img);

    let food_empty_img = stack.load_gui_sprite("hud/food_empty").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let idx = ((y * 9 + x) * 4) as usize;
                if x == 0 || x == 8 || y == 0 || y == 8 {
                    img.data[idx] = 70;
                    img.data[idx + 1] = 50;
                    img.data[idx + 2] = 30;
                    img.data[idx + 3] = 255;
                }
            }
        }
        img
    });
    copy_to_layer_at(&mut pixel_data, 4, 48, 0, &food_empty_img);

    let food_full_img = stack.load_gui_sprite("hud/food_full").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(9, 9);
        for y in 1..8 {
            for x in 1..8 {
                let idx = ((y * 9 + x) * 4) as usize;
                img.data[idx] = 180;
                img.data[idx + 1] = 100;
                img.data[idx + 2] = 40;
                img.data[idx + 3] = 255;
            }
        }
        img
    });
    copy_to_layer_at(&mut pixel_data, 4, 64, 0, &food_full_img);

    let food_half_img = stack.load_gui_sprite("hud/food_half").unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(9, 9);
        for y in 1..8 {
            for x in 1..5 {
                let idx = ((y * 9 + x) * 4) as usize;
                img.data[idx] = 180;
                img.data[idx + 1] = 100;
                img.data[idx + 2] = 40;
                img.data[idx + 3] = 255;
            }
        }
        img
    });
    copy_to_layer_at(&mut pixel_data, 4, 80, 0, &food_half_img);

    let xp_bar_bg_img = stack
        .load_gui_sprite("hud/experience_bar_background")
        .unwrap_or_else(|_| {
            let mut img = vx_assets::RgbaImage::new(182, 5);
            for i in (0..img.data.len()).step_by(4) {
                img.data[i] = 30;
                img.data[i + 1] = 30;
                img.data[i + 2] = 30;
                img.data[i + 3] = 200;
            }
            img
        });
    copy_to_layer_at(&mut pixel_data, 4, 0, 16, &xp_bar_bg_img);

    let xp_bar_progress_img = stack
        .load_gui_sprite("hud/experience_bar_progress")
        .unwrap_or_else(|_| {
            let mut img = vx_assets::RgbaImage::new(182, 5);
            for i in (0..img.data.len()).step_by(4) {
                img.data[i] = 120;
                img.data[i + 1] = 230;
                img.data[i + 2] = 30;
                img.data[i + 3] = 255;
            }
            img
        });
    copy_to_layer_at(&mut pixel_data, 4, 0, 24, &xp_bar_progress_img);

    // Layer 5: Inventory Container Background (176x166)
    let inv_bg_img = stack
        .find_texture("textures/gui/container/inventory.png")
        .and_then(|p| vx_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = vx_assets::RgbaImage::new(176, 166);
            for y in 0..166 {
                for x in 0..176 {
                    let idx = ((y * 176 + x) * 4) as usize;
                    let is_border = x == 0 || x == 175 || y == 0 || y == 165;
                    let color = if is_border { 40 } else { 198 };
                    img.data[idx] = color;
                    img.data[idx + 1] = color;
                    img.data[idx + 2] = color;
                    img.data[idx + 3] = 255;
                }
            }
            img
        });
    copy_to_layer_at(&mut pixel_data, 5, 0, 0, &inv_bg_img);

    // Layer 6: Item Icons Atlas (256x256 holding 16x16 icons for items 1..=19)
    let item_textures: [(u32, &str, [u8; 4]); 19] = [
        (1, "textures/block/stone.png", [128, 128, 128, 255]),
        (2, "textures/block/dirt.png", [134, 96, 67, 255]),
        (3, "textures/block/grass_block_side.png", [90, 160, 60, 255]),
        (4, "textures/block/cobblestone.png", [100, 100, 100, 255]),
        (5, "textures/block/oak_log.png", [103, 82, 49, 255]),
        (6, "textures/block/water_still.png", [64, 100, 200, 255]),
        (7, "textures/block/oak_planks.png", [162, 130, 78, 255]),
        (8, "textures/block/oak_leaves.png", [50, 120, 40, 255]),
        (9, "textures/block/glass.png", [200, 220, 240, 180]),
        (10, "textures/block/stone.png", [120, 120, 120, 255]), // Slab
        (11, "textures/block/oak_planks.png", [162, 130, 78, 255]), // Stairs
        (12, "textures/block/sand.png", [219, 207, 156, 255]),
        (13, "textures/item/stick.png", [140, 100, 50, 255]),
        (
            14,
            "textures/block/crafting_table_front.png",
            [170, 120, 70, 255],
        ),
        (15, "textures/block/torch.png", [255, 200, 50, 255]),
        (16, "textures/item/iron_helmet.png", [220, 220, 220, 255]),
        (
            17,
            "textures/item/iron_chestplate.png",
            [220, 220, 220, 255],
        ),
        (18, "textures/item/iron_leggings.png", [200, 200, 200, 255]),
        (19, "textures/item/iron_boots.png", [180, 180, 180, 255]),
    ];

    let copy_icon =
        |dest: &mut [u8], x_offset: usize, y_offset: usize, img: &vx_assets::RgbaImage| {
            let layer_offset = 6 * (UI_RES * UI_RES * 4) as usize;
            for y in 0..16usize {
                for x in 0..16usize {
                    let src_x = (x * img.width as usize) / 16;
                    let src_y = (y * img.height as usize) / 16;
                    let src_idx = (src_y * img.width as usize + src_x) * 4;
                    let dst_idx =
                        layer_offset + ((y_offset + y) * UI_RES as usize + (x_offset + x)) * 4;
                    if src_idx + 4 <= img.data.len() && dst_idx + 4 <= dest.len() {
                        dest[dst_idx..dst_idx + 4].copy_from_slice(&img.data[src_idx..src_idx + 4]);
                    }
                }
            }
        };

    for (item_id, rel_path, fallback_color) in item_textures {
        let idx = (item_id - 1) as usize;
        let col = idx % 16;
        let row = idx / 16;
        let x_offset = col * 16;
        let y_offset = row * 16;

        let icon_img = stack
            .find_texture(rel_path)
            .and_then(|p| vx_assets::RgbaImage::from_file(&p).ok())
            .unwrap_or_else(|| {
                let mut img = vx_assets::RgbaImage::new(16, 16);
                for y in 0..16 {
                    for x in 0..16 {
                        let i = ((y * 16 + x) * 4) as usize;
                        let is_edge = x == 0 || x == 15 || y == 0 || y == 15;
                        img.data[i] = if is_edge {
                            fallback_color[0].saturating_sub(40)
                        } else {
                            fallback_color[0]
                        };
                        img.data[i + 1] = if is_edge {
                            fallback_color[1].saturating_sub(40)
                        } else {
                            fallback_color[1]
                        };
                        img.data[i + 2] = if is_edge {
                            fallback_color[2].saturating_sub(40)
                        } else {
                            fallback_color[2]
                        };
                        img.data[i + 3] = fallback_color[3];
                    }
                }
                img
            });
        copy_icon(&mut pixel_data, x_offset, y_offset, &icon_img);
    }

    let regions: Vec<TextureMipRegion> = (0..7)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * UI_RES * UI_RES * 4),
            layer,
            mip_level: 0,
            width: UI_RES,
            height: UI_RES,
        })
        .collect();

    let texture_array = gpu_context.create_texture_array(UI_RES, 7, 1, &pixel_data, &regions)?;

    info!(
        "UI texture array loaded (7 layers, 256x256, font baked, survival icons, inventory background, item icons)"
    );

    Ok((texture_array, font))
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn load_and_upload_celestial_textures(gpu_context: &GpuContext) -> Result<GpuTextureArray> {
    const CELESTIAL_RES: u32 = 32;
    const LAYER_COUNT: u32 = 9;

    let mut pixel_data = vec![0u8; (CELESTIAL_RES * CELESTIAL_RES * 4 * LAYER_COUNT) as usize];

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &vx_assets::RgbaImage| {
        let layer_offset = layer * (CELESTIAL_RES * CELESTIAL_RES * 4) as usize;
        let w = img.width.min(CELESTIAL_RES);
        let h = img.height.min(CELESTIAL_RES);
        for y in 0..h {
            let src_start = ((y * img.width) * 4) as usize;
            let src_end = src_start + (w * 4) as usize;
            let dst_start = layer_offset + ((y * CELESTIAL_RES) * 4) as usize;
            dest[dst_start..dst_start + (w * 4) as usize]
                .copy_from_slice(&img.data[src_start..src_end]);
        }
    };

    // Layer 0: Sun
    let sun_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/environment/celestial/sun.png",
    );
    let sun_img = vx_assets::RgbaImage::from_file(sun_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(CELESTIAL_RES, CELESTIAL_RES);
        for y in 0..CELESTIAL_RES {
            for x in 0..CELESTIAL_RES {
                let idx = ((y * CELESTIAL_RES + x) * 4) as usize;
                let dx = x as f32 - 15.5;
                let dy = y as f32 - 15.5;
                if dx * dx + dy * dy <= 14.0 * 14.0 {
                    img.data[idx] = 255;
                    img.data[idx + 1] = 250;
                    img.data[idx + 2] = 220;
                    img.data[idx + 3] = 255;
                }
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 0, &sun_img);

    // Layers 1..=8: Moon phases
    let moon_files = [
        "full_moon",
        "waning_gibbous",
        "third_quarter",
        "waning_crescent",
        "new_moon",
        "waxing_crescent",
        "first_quarter",
        "waxing_gibbous",
    ];

    for (phase_idx, name) in moon_files.iter().enumerate() {
        let moon_path = format!(
            "dev-assets/classic-26.2/assets/classic/textures/environment/celestial/moon/{name}.png"
        );
        let moon_img = vx_assets::RgbaImage::from_file(std::path::Path::new(&moon_path))
            .unwrap_or_else(|_| {
                let mut img = vx_assets::RgbaImage::new(CELESTIAL_RES, CELESTIAL_RES);
                for y in 0..CELESTIAL_RES {
                    for x in 0..CELESTIAL_RES {
                        let idx = ((y * CELESTIAL_RES + x) * 4) as usize;
                        let dx = x as f32 - 15.5;
                        let dy = y as f32 - 15.5;
                        if dx * dx + dy * dy <= 12.0 * 12.0 && phase_idx != 4 {
                            img.data[idx] = 230;
                            img.data[idx + 1] = 235;
                            img.data[idx + 2] = 245;
                            img.data[idx + 3] = 255;
                        }
                    }
                }
                img
            });
        copy_to_layer(&mut pixel_data, 1 + phase_idx, &moon_img);
    }

    let regions: Vec<TextureMipRegion> = (0..LAYER_COUNT)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * CELESTIAL_RES * CELESTIAL_RES * 4),
            layer,
            mip_level: 0,
            width: CELESTIAL_RES,
            height: CELESTIAL_RES,
        })
        .collect();

    let texture_array =
        gpu_context.create_texture_array(CELESTIAL_RES, LAYER_COUNT, 1, &pixel_data, &regions)?;

    info!("Celestial texture array loaded (9 layers, 32x32)");

    Ok(texture_array)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn load_and_upload_weather_textures(gpu_context: &GpuContext) -> Result<GpuTextureArray> {
    const WEATHER_W: u32 = 64;
    const WEATHER_H: u32 = 256;
    const LAYER_COUNT: u32 = 2;

    let mut pixel_data = vec![0u8; (WEATHER_W * WEATHER_H * 4 * LAYER_COUNT) as usize];

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &vx_assets::RgbaImage| {
        let layer_offset = layer * (WEATHER_W * WEATHER_H * 4) as usize;
        let w = img.width.min(WEATHER_W);
        let h = img.height.min(WEATHER_H);
        for y in 0..h {
            let src_start = ((y * img.width) * 4) as usize;
            let src_end = src_start + (w * 4) as usize;
            let dst_start = layer_offset + ((y * WEATHER_W) * 4) as usize;
            dest[dst_start..dst_start + (w * 4) as usize]
                .copy_from_slice(&img.data[src_start..src_end]);
        }
    };

    // Layer 0: Rain
    let rain_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/environment/rain.png",
    );
    let rain_img = vx_assets::RgbaImage::from_file_exact(rain_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(WEATHER_W, WEATHER_H);
        for y in 0..WEATHER_H {
            for x in 0..WEATHER_W {
                let idx = ((y * WEATHER_W + x) * 4) as usize;
                let is_streak = (x % 8 == (y / 4) % 8) && (y % 16 < 12);
                if is_streak {
                    img.data[idx] = 160;
                    img.data[idx + 1] = 180;
                    img.data[idx + 2] = 255;
                    img.data[idx + 3] = 200;
                }
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 0, &rain_img);

    // Layer 1: Snow
    let snow_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/environment/snow.png",
    );
    let snow_img = vx_assets::RgbaImage::from_file_exact(snow_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(WEATHER_W, WEATHER_H);
        for y in 0..WEATHER_H {
            for x in 0..WEATHER_W {
                let idx = ((y * WEATHER_W + x) * 4) as usize;
                let is_flake = (x % 16 == 8) && (y % 16 == 8);
                if is_flake {
                    img.data[idx] = 255;
                    img.data[idx + 1] = 255;
                    img.data[idx + 2] = 255;
                    img.data[idx + 3] = 240;
                }
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 1, &snow_img);

    let copy_regions = [
        TextureMipRegion {
            buffer_offset: 0,
            layer: 0,
            mip_level: 0,
            width: WEATHER_W,
            height: WEATHER_H,
        },
        TextureMipRegion {
            buffer_offset: u64::from(WEATHER_W * WEATHER_H * 4),
            layer: 1,
            mip_level: 0,
            width: WEATHER_W,
            height: WEATHER_H,
        },
    ];

    let texture_array = gpu_context.create_texture_array_2d(
        WEATHER_W,
        WEATHER_H,
        LAYER_COUNT,
        1,
        &pixel_data,
        &copy_regions,
    )?;

    info!("Weather precipitation texture array loaded (2 layers: rain, snow, 64x256)");

    Ok(texture_array)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn load_and_upload_entity_textures(gpu_context: &GpuContext) -> Result<GpuTextureArray> {
    const ENTITY_RES: u32 = 64;
    const LAYER_COUNT: u32 = 3;

    let mut pixel_data = vec![0u8; (ENTITY_RES * ENTITY_RES * 4 * LAYER_COUNT) as usize];

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &vx_assets::RgbaImage| {
        let layer_offset = layer * (ENTITY_RES * ENTITY_RES * 4) as usize;
        let w = img.width.min(ENTITY_RES);
        let h = img.height.min(ENTITY_RES);
        for y in 0..h {
            let src_start = ((y * img.width) * 4) as usize;
            let src_end = src_start + (w * 4) as usize;
            let dst_start = layer_offset + ((y * ENTITY_RES) * 4) as usize;
            dest[dst_start..dst_start + (w * 4) as usize]
                .copy_from_slice(&img.data[src_start..src_end]);
        }
    };

    // Layer 0: Zombie
    let zombie_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/entity/zombie/zombie.png",
    );
    let zombie_img = vx_assets::RgbaImage::from_file_exact(zombie_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
        for y in 0..ENTITY_RES {
            for x in 0..ENTITY_RES {
                let idx = ((y * ENTITY_RES + x) * 4) as usize;
                if y < 32 {
                    // Head & torso
                    img.data[idx] = 60;
                    img.data[idx + 1] = 140;
                    img.data[idx + 2] = 60;
                    img.data[idx + 3] = 255;
                } else {
                    // Legs / pants
                    img.data[idx] = 40;
                    img.data[idx + 1] = 50;
                    img.data[idx + 2] = 160;
                    img.data[idx + 3] = 255;
                }
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 0, &zombie_img);

    // Layer 1: Pig
    let pig_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/entity/pig/pig_temperate.png",
    );
    let pig_img = vx_assets::RgbaImage::from_file_exact(pig_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
        for y in 0..ENTITY_RES {
            for x in 0..ENTITY_RES {
                let idx = ((y * ENTITY_RES + x) * 4) as usize;
                img.data[idx] = 240;
                img.data[idx + 1] = 160;
                img.data[idx + 2] = 160;
                img.data[idx + 3] = 255;
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 1, &pig_img);

    // Layer 2: Cow
    let cow_path = std::path::Path::new(
        "dev-assets/classic-26.2/assets/classic/textures/entity/cow/cow_temperate.png",
    );
    let cow_img = vx_assets::RgbaImage::from_file_exact(cow_path).unwrap_or_else(|_| {
        let mut img = vx_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
        for y in 0..ENTITY_RES {
            for x in 0..ENTITY_RES {
                let idx = ((y * ENTITY_RES + x) * 4) as usize;
                let is_spot = ((x / 8) + (y / 8)) % 2 == 0;
                if is_spot {
                    img.data[idx] = 80;
                    img.data[idx + 1] = 50;
                    img.data[idx + 2] = 40;
                    img.data[idx + 3] = 255;
                } else {
                    img.data[idx] = 230;
                    img.data[idx + 1] = 230;
                    img.data[idx + 2] = 230;
                    img.data[idx + 3] = 255;
                }
            }
        }
        img
    });
    copy_to_layer(&mut pixel_data, 2, &cow_img);

    let regions: Vec<TextureMipRegion> = (0..LAYER_COUNT)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * ENTITY_RES * ENTITY_RES * 4),
            layer,
            mip_level: 0,
            width: ENTITY_RES,
            height: ENTITY_RES,
        })
        .collect();

    let texture_array =
        gpu_context.create_texture_array(ENTITY_RES, LAYER_COUNT, 1, &pixel_data, &regions)?;

    info!("Entity texture array loaded (3 layers: Zombie, Pig, Cow, 64x64)");

    Ok(texture_array)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lightmap_lut_dark_cave() {
        let mut lut = LightmapLut::default();
        lut.update(1.0, 0.0, 0.0, 0.0, 0.0); // Noon, clear

        // Texel (0, 0) should be pitch dark ambient floor (<= 0.05, i.e. <= 13/255)
        let r = f32::from(lut.data[0]) / 255.0;
        let g = f32::from(lut.data[1]) / 255.0;
        let b = f32::from(lut.data[2]) / 255.0;
        assert!(r <= 0.05, "Texel (0,0) red too high: {r}");
        assert!(g <= 0.05, "Texel (0,0) green too high: {g}");
        assert!(b <= 0.05, "Texel (0,0) blue too high: {b}");
    }

    #[test]
    fn test_lightmap_lut_bright_torchlight() {
        let mut lut = LightmapLut::default();
        lut.update(0.0, 0.0, 0.0, 0.0, 0.0); // Sunset, clear

        // Texel (15, 0) should be bright torchlight (>= 0.8)
        let idx = 15 * 4;
        let r = f32::from(lut.data[idx]) / 255.0;
        let g = f32::from(lut.data[idx + 1]) / 255.0;
        assert!(r >= 0.8, "Texel (15,0) torch red too low: {r}");
        assert!(g >= 0.7, "Texel (15,0) torch green too low: {g}");
    }

    #[test]
    fn test_lightmap_lut_day_night_skylight() {
        let mut lut = LightmapLut::default();

        // Noon: texel (0, 15) should be bright day (>= 0.9)
        lut.update(1.0, 0.0, 0.0, 0.0, 0.0);
        let idx = (15 * 16) * 4;
        let r_day = f32::from(lut.data[idx]) / 255.0;
        let g_day = f32::from(lut.data[idx + 1]) / 255.0;
        let b_day = f32::from(lut.data[idx + 2]) / 255.0;
        assert!(r_day >= 0.9, "Texel (0,15) day red too low: {r_day}");
        assert!(g_day >= 0.9, "Texel (0,15) day green too low: {g_day}");
        assert!(b_day >= 0.9, "Texel (0,15) day blue too low: {b_day}");

        // Midnight: texel (0, 15) should be dark night (<= 0.15)
        lut.update(-1.0, 0.0, 0.0, 0.0, 0.0);
        let r_night = f32::from(lut.data[idx]) / 255.0;
        let g_night = f32::from(lut.data[idx + 1]) / 255.0;
        let b_night = f32::from(lut.data[idx + 2]) / 255.0;
        assert!(
            r_night <= 0.15,
            "Texel (0,15) night red too high: {r_night}"
        );
        assert!(
            g_night <= 0.15,
            "Texel (0,15) night green too high: {g_night}"
        );
        assert!(
            b_night <= 0.15,
            "Texel (0,15) night blue too high: {b_night}"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_lightmap_lut_torch_flicker_bounded() {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        let mut lut = LightmapLut::default();
        let idx = 15 * 4;

        for tick in 0..1000 {
            lut.update(0.0, tick as f32, 0.0, 0.0, 0.0);
            let val = f32::from(lut.data[idx]) / 255.0;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        let avg = f32::midpoint(min_val, max_val);
        let dev = (max_val - min_val) / avg;
        // Total deviation range <= 8% (+-4%)
        assert!(
            dev <= 0.08,
            "Torch flicker deviation exceeded 8%: {dev} (min: {min_val}, max: {max_val})"
        );
    }

    #[test]
    fn test_lightmap_lut_weather_dimming() {
        let mut lut_clear = LightmapLut::default();
        let mut lut_storm = LightmapLut::default();

        lut_clear.update(1.0, 0.0, 0.0, 0.0, 0.0);
        lut_storm.update(1.0, 0.0, 1.0, 1.0, 0.0); // Rain + Thunder

        let idx = (15 * 16) * 4;
        let val_clear = f32::from(lut_clear.data[idx]) / 255.0;
        let val_storm = f32::from(lut_storm.data[idx]) / 255.0;

        assert!(
            val_storm < val_clear * 0.65,
            "Storm skylight ({val_storm}) was not significantly dimmed relative to clear ({val_clear})"
        );
    }

    #[test]
    fn test_lightmap_lut_lightning_flash() {
        let mut lut_night = LightmapLut::default();
        let mut lut_flash = LightmapLut::default();

        lut_night.update(-1.0, 0.0, 1.0, 1.0, 0.0); // Night storm, no flash
        lut_flash.update(-1.0, 0.0, 1.0, 1.0, 1.0); // Night storm with full lightning flash

        // Ambient floor at (0, 0) should surge brightly during flash
        let amb_night = f32::from(lut_night.data[0]) / 255.0;
        let amb_flash = f32::from(lut_flash.data[0]) / 255.0;

        assert!(
            amb_flash >= 0.50,
            "Lightning flash ambient ({amb_flash}) did not surge brightly"
        );
        assert!(
            amb_flash > amb_night * 10.0,
            "Flash ({amb_flash}) should be far brighter than dark night ambient ({amb_night})"
        );
    }
}
