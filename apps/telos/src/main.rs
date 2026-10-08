//! Client application executable for the voxel engine.

pub mod camera;
pub mod entity_client;
pub mod opengl_renderer;
pub mod physics;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::physics::{
    Aabb, GameMode, PlayerCapabilities, PlayerInputState, PlayerPhysicsController,
};
use anyhow::Result;
use camera::{Camera, FlyController};
use clap::Parser;
use entity_client::{ClientEntityStore, EntityPushConstants, EntityVertexGpu};
use glam::{DVec3, Vec3};
use hashbrown::{HashMap, HashSet};
use mimalloc::MiMalloc;
use telos_assets::{AnimatedTextureInfo, ResourcePackStack, TextureArrayBuilder};
use telos_content::FrozenRegistries;
use telos_core::{
    AppDirs, BlockPos, FixedTimestep, LanguageCatalog, RaycastHit, TelemetryConfig,
    coords::ChunkPos, detect_system_locale, init_telemetry, profile::PlayerCredentials,
    raycast_voxels,
};
use telos_gpu::{
    ComputePipeline, DepthBuffer, GpuBuffer, GpuContext, GpuTexture2d, GpuTextureArray,
    GraphicsPipeline, HiZPyramid, MemoryLocation, PostCompositePushConstants,
    PostProcessFrameGraph, ShaderModule, SsaoPushConstants, TextureMipRegion, ash, vk,
};
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::{BoundedString, BoundedVec};
use telos_protocol::messages::{
    AuthMode, BlockActionKind, C2sBlockAction, C2sChatMessage, C2sClientSettings,
    C2sCloseContainer, C2sCommandSuggest, C2sConfigAck, C2sHello, C2sInteractEntity,
    C2sInventoryClick, C2sLoginProof, C2sLoginStart, C2sMessage, C2sModalFormResponse,
    C2sPlayerCommand, C2sPlayerInput, C2sTeleportAck, ChunkPayload, ConnectionPhase, InputFrame,
    NetworkEffect, ParticleEffectKind, PlayerCommandKind, S2cMessage, S2cParticleEvent,
    S2cPlayerMovementAck, input_buttons,
};
use telos_sim::particle::{ParticleGpu, ParticleSystem};
use telos_sim::{
    MoveMode, MoveState, PredictionBuffer, VisualSmoothing, quantize_pitch, quantize_yaw,
};
use telos_ui::menu::{
    AdvancementsScreen, MainMenuAction, MainMenuScreen, ModalFormAction, ModalFormScreen,
    PauseMenuAction, PauseMenuScreen, SettingsScreen, UiAdvancementCategory, UiAdvancementFrame,
    UiAdvancementNode, WorldCreateAction, WorldCreateWizard, WorldSelectAction, WorldSelectScreen,
};
use telos_ui::settings::GameSettings;
use telos_ui::{
    BitmapFont, ChatHudState, DUAL_ANVIL_SLOT_COUNT, DUAL_CONTAINER_SLOT_COUNT,
    DUAL_CRAFTING_TABLE_SLOT_COUNT, DUAL_FURNACE_SLOT_COUNT, GuiStyleSheet, HudState,
    ServerListAction, ServerListScreen, ToastState, UiLayers, UiQuad, UiSlotItem,
    anvil_slot_at_pos_styled, chest_slot_at_pos_styled, compute_gui_scale,
    crafting_table_slot_at_pos_styled, furnace_slot_at_pos_styled, inventory_slot_at_pos_styled,
    render_anvil_container_styled, render_chat_hud, render_chest_container_styled,
    render_crafting_table_container_styled, render_furnace_container_styled, render_hud_styled,
    render_inventory_screen_styled, snap_to_physical,
};
use telos_ui::{InputKey, KeyAction};

/// Top-level application screen state.
#[derive(Debug, Clone, PartialEq)]
pub enum AppScreen {
    /// Title main menu screen.
    MainMenu,
    /// Singleplayer world selection screen.
    WorldSelect,
    /// Interactive world creation wizard.
    WorldCreate,
    /// Video, audio, and controls settings screen.
    Settings {
        /// Whether returning from settings resumes the pause menu.
        return_to_pause: bool,
    },
    /// Multiplayer server browser and direct connect screen.
    Multiplayer,
    /// Active 3D gameplay view.
    InGame,
}

/// Active container window state on the client (e.g. chest, furnace).
#[derive(Debug, Clone)]
pub struct ClientContainerState {
    /// Server window ID.
    pub window_id: u8,
    /// Container kind (0 = Chest, 1 = Furnace).
    pub container_kind: u8,
    /// Container display title (e.g. "Chest", "Furnace").
    pub title: String,
    /// Container block coordinate X.
    pub x: i32,
    /// Container block coordinate Y.
    pub y: i32,
    /// Container block coordinate Z.
    pub z: i32,
    /// Container item slot contents (27 slots for chest, 3 slots for furnace).
    pub slots: [UiSlotItem; 27],
    /// Container properties (e.g. furnace burn/cook timers).
    pub properties: [i16; 4],
}
use telos_core::ident::Identifier;
use telos_voxel::chunk::{Chunk, ChunkSnapshot};
use telos_voxel::coords::LocalIdx;
use telos_voxel::light::ChunkLight;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::shape::{BlockShape, SubBox};
use telos_voxel::state::{BlockStateId, StateFlags};
use tracing::{info, warn};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Telos client.
#[derive(Parser, Debug, Clone)]
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

    /// World save directory for singleplayer server (defaults to platform saves dir).
    #[arg(short, long)]
    world_dir: Option<PathBuf>,

    /// Run in portable mode with all data, configuration, and caches isolated locally.
    #[arg(long)]
    portable: bool,

    /// Custom root application data directory override.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Custom configuration directory override.
    #[arg(long)]
    config_dir: Option<PathBuf>,

    /// Custom cache directory override (LOD clipmaps, shaders).
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Vertical chunk view distance radius.
    #[arg(long, default_value_t = 12)]
    vertical_view_distance: u32,

    /// Initial camera position as X,Y,Z (e.g. "208,175,191").
    #[arg(long)]
    pos: Option<String>,

    /// Initial camera yaw in degrees.
    #[arg(long, allow_hyphen_values = true)]
    yaw: Option<f32>,

    /// Initial camera pitch in degrees.
    #[arg(long, allow_hyphen_values = true)]
    pitch: Option<f32>,

    /// Path to write a screenshot PNG to (and exit after target frames).
    #[arg(long)]
    screenshot: Option<PathBuf>,

    /// Number of frames to render before taking screenshot (default: 30).
    #[arg(long, default_value_t = 30)]
    frames: u32,

    /// Disable Hi-Z GPU occlusion culling.
    #[arg(long, default_value_t = false)]
    no_cull: bool,

    /// Rendering backend selection ('vulkan', 'opengl', or 'auto').
    #[arg(long, default_value = "auto")]
    backend: String,
}

fn key_code_to_input_key(code: KeyCode) -> InputKey {
    match code {
        KeyCode::KeyA => InputKey::KeyA,
        KeyCode::KeyB => InputKey::KeyB,
        KeyCode::KeyC => InputKey::KeyC,
        KeyCode::KeyD => InputKey::KeyD,
        KeyCode::KeyE => InputKey::KeyE,
        KeyCode::KeyF => InputKey::KeyF,
        KeyCode::KeyG => InputKey::KeyG,
        KeyCode::KeyH => InputKey::KeyH,
        KeyCode::KeyI => InputKey::KeyI,
        KeyCode::KeyJ => InputKey::KeyJ,
        KeyCode::KeyK => InputKey::KeyK,
        KeyCode::KeyL => InputKey::KeyL,
        KeyCode::KeyM => InputKey::KeyM,
        KeyCode::KeyN => InputKey::KeyN,
        KeyCode::KeyO => InputKey::KeyO,
        KeyCode::KeyP => InputKey::KeyP,
        KeyCode::KeyQ => InputKey::KeyQ,
        KeyCode::KeyR => InputKey::KeyR,
        KeyCode::KeyS => InputKey::KeyS,
        KeyCode::KeyT => InputKey::KeyT,
        KeyCode::KeyU => InputKey::KeyU,
        KeyCode::KeyV => InputKey::KeyV,
        KeyCode::KeyW => InputKey::KeyW,
        KeyCode::KeyX => InputKey::KeyX,
        KeyCode::KeyY => InputKey::KeyY,
        KeyCode::KeyZ => InputKey::KeyZ,
        KeyCode::Digit0 => InputKey::Digit0,
        KeyCode::Digit1 => InputKey::Digit1,
        KeyCode::Digit2 => InputKey::Digit2,
        KeyCode::Digit3 => InputKey::Digit3,
        KeyCode::Digit4 => InputKey::Digit4,
        KeyCode::Digit5 => InputKey::Digit5,
        KeyCode::Digit6 => InputKey::Digit6,
        KeyCode::Digit7 => InputKey::Digit7,
        KeyCode::Digit8 => InputKey::Digit8,
        KeyCode::Digit9 => InputKey::Digit9,
        KeyCode::Space => InputKey::Space,
        KeyCode::ShiftLeft => InputKey::ShiftLeft,
        KeyCode::ShiftRight => InputKey::ShiftRight,
        KeyCode::ControlLeft => InputKey::ControlLeft,
        KeyCode::ControlRight => InputKey::ControlRight,
        KeyCode::AltLeft => InputKey::AltLeft,
        KeyCode::AltRight => InputKey::AltRight,
        KeyCode::Escape => InputKey::Escape,
        KeyCode::Enter => InputKey::Enter,
        KeyCode::Tab => InputKey::Tab,
        KeyCode::Backspace => InputKey::Backspace,
        KeyCode::ArrowUp => InputKey::ArrowUp,
        KeyCode::ArrowDown => InputKey::ArrowDown,
        KeyCode::ArrowLeft => InputKey::ArrowLeft,
        KeyCode::ArrowRight => InputKey::ArrowRight,
        KeyCode::Slash => InputKey::Slash,
        KeyCode::Backslash => InputKey::Backslash,
        KeyCode::Minus => InputKey::Minus,
        KeyCode::Equal => InputKey::Equal,
        KeyCode::Semicolon => InputKey::Semicolon,
        KeyCode::Quote => InputKey::Quote,
        KeyCode::F1 => InputKey::F1,
        KeyCode::F2 => InputKey::F2,
        KeyCode::F3 => InputKey::F3,
        KeyCode::F4 => InputKey::F4,
        KeyCode::F5 => InputKey::F5,
        KeyCode::F6 => InputKey::F6,
        KeyCode::F7 => InputKey::F7,
        KeyCode::F8 => InputKey::F8,
        KeyCode::F9 => InputKey::F9,
        KeyCode::F10 => InputKey::F10,
        KeyCode::F11 => InputKey::F11,
        KeyCode::F12 => InputKey::F12,
        _ => InputKey::None,
    }
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

/// Collection of animated texture layer descriptors used across GPU pipelines.
#[derive(Debug, Clone, Copy)]
pub struct AnimatedTextures {
    /// Still water texture sequence info.
    pub water_still: AnimatedTextureInfo,
    /// Flowing water texture sequence info.
    pub water_flow: AnimatedTextureInfo,
    /// Still lava texture sequence info.
    pub lava_still: AnimatedTextureInfo,
    /// Flowing lava texture sequence info.
    pub lava_flow: AnimatedTextureInfo,
    /// Fire animation sequence info.
    pub fire: AnimatedTextureInfo,
    /// Nether portal animation sequence info.
    pub nether_portal: AnimatedTextureInfo,
}

impl Default for AnimatedTextures {
    fn default() -> Self {
        Self {
            water_still: AnimatedTextureInfo {
                base_layer: 9,
                frame_count: 32,
                frame_time: 2,
            },
            water_flow: AnimatedTextureInfo {
                base_layer: 9,
                frame_count: 32,
                frame_time: 2,
            },
            lava_still: AnimatedTextureInfo {
                base_layer: 0,
                frame_count: 1,
                frame_time: 2,
            },
            lava_flow: AnimatedTextureInfo {
                base_layer: 0,
                frame_count: 1,
                frame_time: 2,
            },
            fire: AnimatedTextureInfo {
                base_layer: 0,
                frame_count: 1,
                frame_time: 1,
            },
            nether_portal: AnimatedTextureInfo {
                base_layer: 0,
                frame_count: 1,
                frame_time: 1,
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TranslucentMdiPushConstants {
    view_proj: [f32; 16],
    draw_info_buffer_address: u64,
    frame_tick: u32,
    water_base_layer: u32,
    camera_pos: [f32; 4], // xyz: camera pos, w: sim_dist_meters
    water_frame_count: u32,
    water_flow_base_layer: u32,
    water_flow_frame_count: u32,
    lava_base_layer: u32,
    lava_frame_count: u32,
    fire_base_layer: u32,
    fire_frame_count: u32,
    _pad: u32,
}
const _: () = assert!(size_of::<TranslucentMdiPushConstants>() == 128);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct T2PushConstants {
    view_proj: [f32; 16],
    quad_buffer_address: u64,
    chunk_x: i32,
    chunk_y: i32,
    camera_pos: [f32; 4], // xyz: camera pos, w: sim_dist_meters
    chunk_z: i32,
    frame_tick_flags: u32,
    water_base_layer: u32,
    water_frame_count: u32,
    lava_base_layer: u32,
    lava_frame_count: u32,
    fire_base_layer: u32,
    fire_frame_count: u32,
}
const _: () = assert!(size_of::<T2PushConstants>() == 128);

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
    wind_time: f32,
    cloud_coverage: f32,
}
const _: () = assert!(size_of::<SkyPushConstants>() == 128);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PanoramaPushConstants {
    inv_view_proj: [f32; 16],
}
const _: () = assert!(size_of::<PanoramaPushConstants>() == 64);

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

/// Push constants for the visual particle billboard quad pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticlePushConstants {
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
    /// 64-bit device address of the particle buffer.
    pub particle_buffer_address: u64,
    /// Alignment padding.
    pub pad2: [u32; 2],
}
const _: () = assert!(size_of::<ParticlePushConstants>() == 112);

/// Resolves the terrain texture array layer and average tint color for a block state.
fn block_state_to_particle_layer(state: telos_voxel::state::BlockStateId) -> (u32, [u8; 4]) {
    match state.0 {
        1 => (0, [128, 128, 128, 255]),             // stone
        2 => (1, [134, 96, 67, 255]),               // dirt
        3 => (3, [120, 180, 80, 255]),              // grass
        4 => (4, [60, 60, 60, 255]),                // bedrock
        5 => (5, [219, 207, 156, 255]),             // sand
        6 => (24, [64, 100, 200, 200]),             // water
        7 | 11 | 63 => (6, [162, 130, 78, 255]),    // oak planks, stairs & chest
        8 => (7, [60, 140, 50, 255]),               // oak leaves
        9 => (8, [200, 220, 240, 180]),             // glass
        10 => (0, [120, 120, 120, 255]),            // stone slab
        15 => (11, [255, 200, 50, 255]),            // torch
        19 | 20 => (20, [200, 20, 20, 255]),        // wire
        21 => (17, [220, 30, 30, 255]),             // redstone_block
        22 | 23 => (23, [140, 110, 70, 255]),       // lever
        24 => (18, [90, 55, 30, 255]),              // lamp
        25 => (19, [255, 210, 100, 255]),           // lamp lit
        26 | 27 | 30 => (21, [180, 180, 180, 255]), // repeater / diode
        28 => (15, [255, 60, 60, 255]),             // inverter
        29 => (16, [100, 30, 30, 255]),             // inverter off
        _ => (0, [180, 180, 180, 255]),
    }
}

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
        let daylight = telos_core::time::daylight_factor(sun_elev);
        let sunset = telos_core::time::sunset_factor(sun_elev);

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
    fn new(ctx: &GpuContext) -> Result<Self, telos_gpu::GpuError> {
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

    fn destroy(&mut self, device: &ash::Device, allocator: &telos_gpu::GpuAllocator) {
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
    ("Logic Wire", BlockStateId::new(19)),
    ("Power Block", BlockStateId::new(21)),
    ("Lever", BlockStateId::new(22)),
    ("Logic Lamp", BlockStateId::new(24)),
    ("Repeater", BlockStateId::new(26)),
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
    t2_cutout: Option<GpuMeshLayer>,
    t2_translucent: Option<GpuMeshLayer>,
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
        if let Some(mut layer) = self.t2_cutout.take() {
            layer.buffer.destroy(device, allocator);
        }
        if let Some(mut layer) = self.t2_translucent.take() {
            layer.buffer.destroy(device, allocator);
        }
    }
}

fn upload_t0_layer(
    ctx: &GpuContext,
    mesh: &telos_mesh::mesh::T0Mesh,
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
    mesh: &telos_mesh::t1::T1Mesh,
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

fn upload_t2_layer(
    ctx: &GpuContext,
    mesh: &telos_mesh::t2::T2Mesh,
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

#[allow(clippy::struct_excessive_bools)]
struct App {
    #[allow(dead_code)]
    dirs: AppDirs,
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

    t2_cutout_pipeline: Option<GraphicsPipeline>,
    t2_translucent_pipeline: Option<GraphicsPipeline>,
    t2_vert_shader: Option<ShaderModule>,
    t2_frag_shader: Option<ShaderModule>,

    anim_textures: AnimatedTextures,
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
    custom_item_names: HashMap<u32, String>,
    chunks: HashMap<ChunkPos, Arc<ChunkSnapshot>>,
    chunk_meshes: HashMap<ChunkPos, GpuChunkMesh>,
    dirty_chunks: HashSet<ChunkPos>,

    lod_meshes: HashMap<telos_lod::coords::LodNodeKey, GpuLodMesh>,
    pending_lod_uploads: HashMap<telos_lod::coords::LodNodeKey, (u32, Vec<u32>)>,

    camera: Camera,
    controller: FlyController,
    last_frame_time: Instant,
    last_fps_time: Instant,
    frame_counter: u32,
    visible_chunks_last: usize,
    visible_lod_nodes_last: usize,

    // Prediction & Movement Reconciliation
    prediction: PredictionBuffer,
    smoothing: VisualSmoothing,
    client_tick: u32,
    sim_time_acc: f32,
    last_acked_server_tick: u32,

    // MDI & Compute Culling
    mdi_buffers: Option<MdiBuffers>,
    chunk_slots: SlotAllocator<ChunkPos>,
    lod_slots: SlotAllocator<telos_lod::coords::LodNodeKey>,

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
    gui_style: GuiStyleSheet,
    hud_state: HudState,
    inventory_sim: telos_sim::Inventory,
    fuel_registry: telos_sim::FuelRegistry,
    smelting_registry: telos_sim::SmeltingRegistry,
    recipe_registry: telos_sim::RecipeRegistry,
    inventory_open: bool,
    container_state: Option<ClientContainerState>,
    container_hovered_slot: Option<usize>,
    mouse_cursor_pos: [f32; 2],
    inventory_hovered_slot: Option<usize>,
    shift_held: bool,
    player_hurt_timer: f32,
    bow_charging: bool,
    bow_charge_ticks: u16,

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

    // 3D Title Screen Rotating Panorama
    panorama_texture: Option<GpuTextureArray>,
    panorama_pipeline: Option<GraphicsPipeline>,
    panorama_frag_shader: Option<ShaderModule>,
    panorama_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    panorama_descriptor_pool: Option<vk::DescriptorPool>,
    panorama_descriptor_set: Option<vk::DescriptorSet>,

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

    // Visual Particles (Phase 35)
    particle_pipeline: Option<GraphicsPipeline>,
    particle_vert_shader: Option<ShaderModule>,
    particle_frag_shader: Option<ShaderModule>,
    particle_texture: Option<GpuTextureArray>,
    particle_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    particle_descriptor_pool: Option<vk::DescriptorPool>,
    particle_descriptor_set: Option<vk::DescriptorSet>,
    particle_buffer: Option<GpuBuffer>,
    particle_system: ParticleSystem,
    active_particles_count: u32,
    particle_seed: u64,

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
    start_time: Instant,
    chat_state: ChatHudState,
    lan_listener: Option<telos_net::LanDiscoveryListener>,
    suggest_query_id: u32,
    args: Args,
    total_frames: u32,
    manual_screenshot_requested: bool,

    // Audio Engine & Acoustics (Phase 27)
    audio: telos_audio::AudioEngine,
    audio_step_dist: f32,
    last_audio_pos: Vec3,

    // Menu System, Settings & Physics (Phase 33)
    current_screen: AppScreen,
    main_menu: MainMenuScreen,
    world_select: WorldSelectScreen,
    world_create: WorldCreateWizard,
    settings_screen: SettingsScreen,
    pause_menu: PauseMenuScreen,
    is_paused: bool,
    server_list: ServerListScreen,
    server_listener_tx: Option<std::sync::mpsc::Sender<telos_net::QuicListener>>,
    lan_opened_port: Option<u16>,
    last_invite_link: Option<String>,
    advancements_screen: AdvancementsScreen,
    advancements_open: bool,
    toast_queue: VecDeque<ToastState>,
    active_modal_form: Option<ModalFormScreen>,
    game_settings: GameSettings,
    settings_path: PathBuf,
    worlds_dir: PathBuf,
    physics: PlayerPhysicsController,
    last_frame_start: Instant,

    // Internationalization Engine (Phase 34)
    catalog: LanguageCatalog,

    // Active status effects on client player (Phase 38)
    client_effects: Vec<NetworkEffect>,

    // Sandboxed JavaScript runtime for resource pack particle/sound modulation (Phase 40)
    js_particle_hook: Option<telos_mod::JsParticleHook>,

    // Extensible post-processing framework & custom shader pipeline (Phase 41)
    postprocess: Option<PostProcessFrameGraph>,

    // Active rendering backend and OpenGL fallback resources (Phase 44)
    active_backend: telos_gpu::rhi::RenderBackendType,
    gl_context: Option<telos_gpu::opengl::GlContext>,
    gl_renderer: Option<opengl_renderer::OpenGlRenderer>,

    // Cryptographic player profile & credentials (Phase 71)
    credentials: PlayerCredentials,
    player_uuid: [u8; 16],
}

impl App {
    #[allow(clippy::too_many_lines)]
    fn new(args: Args) -> Self {
        let spawn_pos = if let Some(ref pos_str) = args.pos {
            let parts: Vec<f32> = pos_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if parts.len() == 3 {
                Vec3::new(parts[0], parts[1], parts[2])
            } else {
                let climate = telos_worldgen::ClimatePoint::sample(args.seed, 0.0, 0.0);
                let h = telos_worldgen::density::base_terrain_height(args.seed, 0.0, 0.0, &climate);
                let y = (h.max(1.0).ceil() + 2.0).max(25.0);
                Vec3::new(0.5, y, 0.5)
            }
        } else {
            let climate = telos_worldgen::ClimatePoint::sample(args.seed, 0.0, 0.0);
            let h = telos_worldgen::density::base_terrain_height(args.seed, 0.0, 0.0, &climate);
            let y = (h.max(1.0).ceil() + 2.0).max(25.0);
            Vec3::new(0.5, y, 0.5)
        };
        let mut camera = Camera::new(spawn_pos);
        if let Some(yaw_deg) = args.yaw {
            camera.yaw = yaw_deg.to_radians();
        }
        if let Some(pitch_deg) = args.pitch {
            camera.pitch = pitch_deg.to_radians();
        } else {
            camera.pitch = -0.15;
        }

        let registries = Arc::new(FrozenRegistries::new_default());
        let server_registries = Arc::clone(&registries);

        let app_dirs = if args.portable {
            AppDirs::portable(".")
        } else if let Some(ref data) = args.data_dir {
            let mut dirs = AppDirs::from_data_dir(data);
            if let Some(ref cfg) = args.config_dir {
                dirs = dirs.with_config_dir(cfg);
            }
            if let Some(ref cache) = args.cache_dir {
                dirs = dirs.with_cache_dir(cache);
            }
            dirs
        } else {
            let mut dirs = AppDirs::standard_with_local_fallback();
            if let Some(ref cfg) = args.config_dir {
                dirs = dirs.with_config_dir(cfg);
            }
            if let Some(ref cache) = args.cache_dir {
                dirs = dirs.with_cache_dir(cache);
            }
            dirs
        };

        let _ = app_dirs.ensure_dirs_exist();

        let is_direct_ingame = args.screenshot.is_some() || args.pos.is_some();
        let current_screen = if is_direct_ingame {
            AppScreen::InGame
        } else {
            AppScreen::MainMenu
        };

        let settings_path = app_dirs.config_dir().join("settings.toml");
        let mut game_settings = GameSettings::load_or_create(&settings_path);

        let creds_path = app_dirs.config_dir().join("credentials.json");
        let credentials = PlayerCredentials::load_or_create_at(&creds_path, "Player")
            .unwrap_or_else(|_| PlayerCredentials::new("Player").expect("Player credentials"));
        let player_uuid = credentials.uuid().to_bytes();

        let worlds_dir = if std::path::Path::new("worlds").is_dir() {
            PathBuf::from("worlds")
        } else {
            app_dirs.data_dir().join("worlds")
        };
        let _ = std::fs::create_dir_all(&worlds_dir);

        let view_distance = args.view_distance;
        let vertical_view_distance = args.vertical_view_distance;
        let default_save_dir = args.world_dir.clone().unwrap_or_else(|| {
            if std::path::Path::new("worlds").is_dir() {
                PathBuf::from("worlds/default")
            } else {
                app_dirs.world_save_dir("default")
            }
        });
        let seed = args.seed;
        let simulation_distance = game_settings.video.simulation_distance;

        let server_running = Arc::new(AtomicBool::new(is_direct_ingame));
        let running_clone = server_running.clone();

        let (server_conn, client_conn) = MemoryConnection::pair_default();
        let server_conn: Box<dyn Connection<S2cMessage, C2sMessage>> = Box::new(server_conn);
        let client_conn: Box<dyn Connection<C2sMessage, S2cMessage>> = Box::new(client_conn);

        let server_handle = if is_direct_ingame {
            let s_registries = Arc::clone(&server_registries);
            let handle = std::thread::Builder::new()
                .name("telos-server".into())
                .spawn(move || {
                    let config = telos_server::ServerConfig {
                        tps: 20,
                        view_distance,
                        vertical_view_distance,
                        simulation_distance,
                        chunks_per_tick_per_player: 16,
                        save_directory: Some(default_save_dir),
                        ..Default::default()
                    };
                    let mut server =
                        telos_server::Server::with_registries(seed, config, s_registries);
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

            Some(handle)
        } else {
            None
        };

        let d_spawn = DVec3::new(
            f64::from(spawn_pos.x),
            f64::from(spawn_pos.y),
            f64::from(spawn_pos.z),
        );
        let prediction = PredictionBuffer::new(MoveState::new(
            d_spawn,
            camera.yaw.to_degrees(),
            camera.pitch.to_degrees(),
            true,
        ));
        let smoothing = VisualSmoothing::default();

        let controller = FlyController {
            mouse_captured: is_direct_ingame && args.screenshot.is_none(),
            ..Default::default()
        };

        let initial_mode = if args.screenshot.is_some() {
            GameMode::Creative
        } else {
            match game_settings.gameplay.game_mode.to_lowercase().as_str() {
                "survival" => GameMode::Survival,
                _ => GameMode::Creative,
            }
        };
        let mut physics = PlayerPhysicsController::new(d_spawn, initial_mode);
        if args.screenshot.is_some() {
            physics.flying = true;
        }
        physics.set_pos_from_eye(camera.position);

        let mut pack_stack = ResourcePackStack::new();
        mount_asset_roots(&mut pack_stack);
        let mut catalog = LanguageCatalog::with_default_embedded();
        pack_stack.populate_language_catalog(&mut catalog);
        let gui_style = pack_stack.load_gui_style_sheet();

        if game_settings.gameplay.language.is_empty() {
            let sys = detect_system_locale();
            if catalog.has_locale(&sys) {
                game_settings.gameplay.language = sys;
            } else {
                game_settings.gameplay.language = "en_us".to_string();
            }
        }
        catalog.set_active_locale(&game_settings.gameplay.language);

        let js_particle_hook = if game_settings.gameplay.allow_pack_scripts {
            let scripts = pack_stack.load_scripts();
            if scripts.is_empty() {
                None
            } else {
                let mut combined = String::new();
                for (name, src) in scripts {
                    tracing::info!(script = %name, "Loading resource pack JavaScript hook");
                    combined.push_str(&src);
                    combined.push('\n');
                }
                match telos_mod::JsParticleHook::new(&combined) {
                    Ok(hook) => {
                        tracing::info!(
                            "Compiled resource pack JavaScript particle/sound hook successfully"
                        );
                        Some(hook)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to initialize resource pack JavaScript hook");
                        None
                    }
                }
            }
        } else {
            None
        };

        let main_menu = MainMenuScreen::new();
        let mut world_select = WorldSelectScreen::new();
        world_select.scan_worlds(&worlds_dir);
        let world_create = WorldCreateWizard::new();
        let settings_screen = SettingsScreen::new(game_settings.clone());
        let pause_menu = PauseMenuScreen::new();

        let mut advancements_screen = AdvancementsScreen::new();
        for adv in telos_sim::AdvancementRegistry::standard().all() {
            let category = match adv.category {
                telos_sim::AdvancementCategory::Story => UiAdvancementCategory::Story,
                _ => UiAdvancementCategory::Adventure,
            };
            let frame = match adv.frame {
                telos_sim::AdvancementFrame::Task => UiAdvancementFrame::Task,
                telos_sim::AdvancementFrame::Goal => UiAdvancementFrame::Goal,
                telos_sim::AdvancementFrame::Challenge => UiAdvancementFrame::Challenge,
            };
            advancements_screen.nodes.push(UiAdvancementNode {
                id: adv.id.to_string(),
                parent_id: adv.parent.map(String::from),
                category,
                title: adv.title.to_string(),
                description: adv.description.to_string(),
                icon_item: adv.icon_item,
                frame,
                x: adv.x,
                y: adv.y,
                completed: false,
            });
        }

        let mut initial_chunks = HashMap::new();
        let mut initial_dirty = HashSet::new();
        if !is_direct_ingame {
            let world_gen =
                telos_worldgen::WorldGenerator::new(args.seed, registries.block_registry());
            for cx in -2..=2 {
                for cz in -2..=2 {
                    for cy in -1..=1 {
                        let cpos = ChunkPos::new(cx, cy, cz);
                        let mut chunk = world_gen.generate_chunk(cpos);
                        let snap = chunk.publish_snapshot();
                        initial_chunks.insert(cpos, snap);
                        initial_dirty.insert(cpos);
                    }
                }
            }
        }

        Self {
            validation: args.validation,
            view_distance: args.view_distance,
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

            t2_cutout_pipeline: None,
            t2_translucent_pipeline: None,
            t2_vert_shader: None,
            t2_frag_shader: None,

            anim_textures: AnimatedTextures::default(),
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
            server_handle,
            client_conn,
            client_phase: ConnectionPhase::Hello,

            registries: Arc::clone(&registries),
            block_registry: registries.block_registry().clone(),
            custom_item_names: HashMap::new(),
            chunks: initial_chunks,
            chunk_meshes: HashMap::new(),
            dirty_chunks: initial_dirty,

            lod_meshes: HashMap::new(),
            pending_lod_uploads: HashMap::new(),

            camera,
            controller,
            last_frame_time: Instant::now(),
            last_fps_time: Instant::now(),
            frame_counter: 0,
            visible_chunks_last: 0,
            visible_lod_nodes_last: 0,

            prediction,
            smoothing,
            client_tick: 0,
            sim_time_acc: 0.0,
            last_acked_server_tick: 0,

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
            gui_style,
            hud_state: HudState::default(),
            inventory_sim: telos_sim::Inventory::default(),
            fuel_registry: telos_sim::FuelRegistry::standard(),
            smelting_registry: telos_sim::SmeltingRegistry::standard(),
            recipe_registry: telos_sim::RecipeRegistry::standard(),
            inventory_open: false,
            container_state: None,
            container_hovered_slot: None,
            mouse_cursor_pos: [0.0, 0.0],
            inventory_hovered_slot: None,
            shift_held: false,
            player_hurt_timer: 0.0,
            bow_charging: false,
            bow_charge_ticks: 0,

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

            panorama_texture: None,
            panorama_pipeline: None,
            panorama_frag_shader: None,
            panorama_descriptor_set_layout: None,
            panorama_descriptor_pool: None,
            panorama_descriptor_set: None,

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

            particle_pipeline: None,
            particle_vert_shader: None,
            particle_frag_shader: None,
            particle_texture: None,
            particle_descriptor_set_layout: None,
            particle_descriptor_pool: None,
            particle_descriptor_set: None,
            particle_buffer: None,
            particle_system: ParticleSystem::new(4096),
            active_particles_count: 0,
            particle_seed: 0xCAFE_BABE_1234_5678,

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
            start_time: Instant::now(),
            chat_state: ChatHudState::new(),
            lan_listener: telos_net::LanDiscoveryListener::new().ok(),
            suggest_query_id: 0,
            dirs: app_dirs,
            args,
            total_frames: 0,
            manual_screenshot_requested: false,

            audio: telos_audio::AudioEngine::new(),
            audio_step_dist: 0.0,
            last_audio_pos: spawn_pos,

            current_screen,
            main_menu,
            world_select,
            world_create,
            settings_screen,
            pause_menu,
            is_paused: false,
            server_list: ServerListScreen::new(),
            server_listener_tx: None,
            lan_opened_port: None,
            last_invite_link: None,
            advancements_screen,
            advancements_open: false,
            toast_queue: VecDeque::new(),
            active_modal_form: None,
            game_settings,
            settings_path,
            worlds_dir,
            physics,
            last_frame_start: Instant::now(),
            catalog,
            client_effects: Vec::new(),
            js_particle_hook,
            postprocess: None,

            active_backend: telos_gpu::rhi::RenderBackendType::Vulkan,
            gl_context: None,
            gl_renderer: None,
            credentials,
            player_uuid,
        }
    }

    fn start_singleplayer_server(&mut self, world_dir: PathBuf, seed: u64, generator: &str) {
        self.stop_singleplayer_server();

        self.chunks.clear();
        self.chunk_meshes.clear();
        self.dirty_chunks.clear();
        self.lod_meshes.clear();
        self.pending_lod_uploads.clear();
        self.entity_store = ClientEntityStore::new();
        self.client_effects.clear();
        self.advancements_open = false;
        self.hud_state.active_toast = None;
        self.toast_queue.clear();
        for node in &mut self.advancements_screen.nodes {
            node.completed = false;
        }

        let gen_kind = match generator.to_lowercase().as_str() {
            "flat" => telos_worldgen::GeneratorKind::Flat,
            "void" => telos_worldgen::GeneratorKind::Void,
            _ => telos_worldgen::GeneratorKind::Standard,
        };

        let world_cfg = telos_server::WorldConfig {
            name: "overworld".to_string(),
            seed,
            generator: gen_kind,
            save_directory: Some(world_dir.clone()),
            ..Default::default()
        };

        let config = telos_server::ServerConfig {
            tps: 20,
            view_distance: self.game_settings.video.view_distance,
            vertical_view_distance: self.game_settings.video.vertical_view_distance,
            simulation_distance: self.game_settings.video.simulation_distance,
            chunks_per_tick_per_player: 16,
            save_directory: Some(world_dir),
            worlds: vec![world_cfg],
            ..Default::default()
        };

        let (server_conn, client_conn) = MemoryConnection::pair_default();
        let server_conn: Box<dyn Connection<S2cMessage, C2sMessage>> = Box::new(server_conn);
        self.client_conn = Box::new(client_conn);

        let (listener_tx, listener_rx) = std::sync::mpsc::channel::<telos_net::QuicListener>();
        self.server_listener_tx = Some(listener_tx);
        self.lan_opened_port = None;
        self.last_invite_link = None;

        let server_running = Arc::new(AtomicBool::new(true));
        self.server_running = server_running.clone();
        let running_clone = server_running;
        let server_registries = Arc::clone(&self.registries);

        let server_handle = std::thread::Builder::new()
            .name("telos-server".into())
            .spawn(move || {
                let mut server =
                    telos_server::Server::with_registries(seed, config, server_registries);
                server.add_connection(server_conn);

                let mut timestep = FixedTimestep::new(20);
                while running_clone.load(Ordering::Relaxed) {
                    if let Ok(listener) = listener_rx.try_recv() {
                        let port = listener.local_addr().port();
                        let beacon = telos_net::LanBeacon::new(port, &server.config().motd);
                        server.lan_emitter = telos_net::LanBeaconEmitter::spawn(beacon, port).ok();
                        server.listener = Some(listener);
                        info!(port, "Server opened to LAN & P2P with active QUIC listener");
                    }
                    timestep.advance(|_| {
                        server.tick();
                    });
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("Failed to spawn singleplayer server thread");

        self.server_handle = Some(server_handle);

        let _ = self.client_conn.send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        );

        let initial_mode = match self
            .game_settings
            .gameplay
            .game_mode
            .to_lowercase()
            .as_str()
        {
            "survival" => GameMode::Survival,
            _ => GameMode::Creative,
        };
        self.physics = PlayerPhysicsController::new(DVec3::new(128.0, 45.0, 160.0), initial_mode);
        self.camera.position = self.physics.eye_pos();
        self.client_phase = ConnectionPhase::Hello;
        self.is_paused = false;
        info!(seed, generator, "Started singleplayer server session");
    }

    fn stop_singleplayer_server(&mut self) {
        if let Ok(message) = BoundedString::new("Quit to title") {
            let _ = self.client_conn.send(
                Lane::Control,
                Payload::Msg(C2sMessage::Disconnect(
                    telos_protocol::messages::Disconnect {
                        reason: telos_protocol::messages::DisconnectReason::Normal,
                        message,
                    },
                )),
            );
        }
        self.server_running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.server_handle.take() {
            let _ = handle.join();
        }
        self.server_listener_tx = None;
        self.lan_opened_port = None;
        self.last_invite_link = None;
        self.client_phase = ConnectionPhase::Hello;
        info!("Stopped singleplayer server session");
    }

    fn connect_to_remote_server(&mut self, target: &str) {
        info!(
            target,
            "Connecting to remote multiplayer server / invite ticket"
        );
        self.stop_singleplayer_server();

        match telos_net::ticket::connect_to_target_sync(target) {
            Ok(conn) => {
                self.client_conn = conn;
                self.chunks.clear();
                self.chunk_meshes.clear();
                self.dirty_chunks.clear();
                self.lod_meshes.clear();
                self.pending_lod_uploads.clear();
                self.entity_store = ClientEntityStore::new();
                self.client_effects.clear();
                self.advancements_open = false;
                self.hud_state.active_toast = None;
                self.toast_queue.clear();
                for node in &mut self.advancements_screen.nodes {
                    node.completed = false;
                }

                let _ = self.client_conn.send(
                    Lane::Control,
                    Payload::Msg(C2sMessage::Hello(C2sHello {
                        protocol: 1,
                        build: BoundedString::new("0.1.0").unwrap(),
                        features: 0,
                    })),
                );

                let initial_mode = match self
                    .game_settings
                    .gameplay
                    .game_mode
                    .to_lowercase()
                    .as_str()
                {
                    "survival" => GameMode::Survival,
                    _ => GameMode::Creative,
                };
                self.physics =
                    PlayerPhysicsController::new(DVec3::new(128.0, 45.0, 160.0), initial_mode);
                self.camera.position = self.physics.eye_pos();
                self.client_phase = ConnectionPhase::Hello;
                self.is_paused = false;
                self.current_screen = AppScreen::InGame;
                self.controller.mouse_captured = true;
                self.set_cursor_captured(true);

                info!("Connected to remote server successfully");
            }
            Err(e) => {
                warn!("Failed to connect to {}: {}", target, e);
                let toast = ToastState {
                    id: "connect_err".to_string(),
                    title: format!("Connect Failed: {e}"),
                    icon_item: 0,
                    frame: 2,
                    elapsed_secs: 0.0,
                    duration_secs: 5.0,
                };
                self.hud_state.active_toast = Some(toast);
            }
        }
    }

    fn open_singleplayer_to_lan_and_invite(&mut self) {
        if let Some(link) = &self.last_invite_link {
            info!(link, "Invite link already generated");
            let toast = ToastState {
                id: "lan".to_string(),
                title: "Invite Link Ready in Chat".to_string(),
                icon_item: 0,
                frame: 1,
                elapsed_secs: 0.0,
                duration_secs: 5.0,
            };
            self.hud_state.active_toast = Some(toast);
            let now_secs = self.start_time.elapsed().as_secs_f64();
            self.chat_state.add_system_message(
                format!("Invite link: {link}"),
                UiQuad::rgba(85, 255, 85, 255),
                now_secs,
            );
            return;
        }

        let Some(listener_tx) = &self.server_listener_tx else {
            warn!("Cannot open to LAN: server listener channel unavailable");
            return;
        };

        // Attempt standard port 25565 first, fallback to ephemeral port 0
        let listener = telos_net::QuicListener::bind("0.0.0.0:25565".parse().unwrap())
            .or_else(|_| telos_net::QuicListener::bind("0.0.0.0:0".parse().unwrap()));

        let Ok(listener) = listener else {
            warn!("Failed to bind QUIC listener on 0.0.0.0");
            return;
        };

        let bound_port = listener.local_addr().port();
        let cert_spki = listener.cert_spki_sha256();
        let _ = listener_tx.send(listener);
        self.lan_opened_port = Some(bound_port);

        // Port mapper discovery (UPnP, NAT-PMP, STUN)
        let rt = telos_net::quic::runtime();
        let mapping = rt.block_on(async {
            let mut mapper = telos_net::nat::PortMapper::new();
            mapper
                .map_port(bound_port, 7200, Duration::from_millis(1500))
                .await
        });

        let signing_key = [
            0x54, 0x45, 0x4c, 0x4f, 0x53, 0x5f, 0x4c, 0x41, 0x4e, 0x5f, 0x53, 0x45, 0x43, 0x52,
            0x45, 0x54, 0x20, 0x32, 0x30, 0x32, 0x36, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b,
        ];

        let ticket = telos_net::ticket::ConnectionTicket::create(
            "Telos LAN Game",
            &mapping,
            cert_spki,
            [0u8; 16],
            86400,
            &signing_key,
        );

        let link = ticket
            .to_invite_link()
            .unwrap_or_else(|_| format!("127.0.0.1:{bound_port}"));
        self.last_invite_link = Some(link.clone());

        info!(bound_port, link, "Opened game session to LAN and P2P");
        let toast = ToastState {
            id: "lan".to_string(),
            title: format!("Opened on Port {bound_port}! Link in Chat"),
            icon_item: 0,
            frame: 1,
            elapsed_secs: 0.0,
            duration_secs: 5.0,
        };
        self.hud_state.active_toast = Some(toast);

        let now_secs = self.start_time.elapsed().as_secs_f64();
        self.chat_state.add_system_message(
            format!("Game opened on port {bound_port}! Invite Link: {link}"),
            UiQuad::rgba(85, 255, 85, 255),
            now_secs,
        );
    }

    fn set_cursor_captured(&self, captured: bool) {
        if let Some(window) = &self.window {
            if captured {
                let _ = window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                window.set_cursor_visible(false);
            } else {
                window.set_cursor_visible(true);
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
            }
        }
    }

    fn window_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.window.as_ref().map_or_else(
            || winit::dpi::PhysicalSize::new(1280, 720),
            winit::window::Window::inner_size,
        )
    }

    fn request_command_suggestions(&mut self) {
        if !self.chat_state.input_buffer.starts_with('/') {
            return;
        }
        self.suggest_query_id = self.suggest_query_id.wrapping_add(1);
        if let Ok(bounded) = BoundedString::new(self.chat_state.input_buffer.clone()) {
            #[allow(clippy::cast_possible_truncation)]
            let req = C2sMessage::CommandSuggest(C2sCommandSuggest {
                id: self.suggest_query_id,
                command: bounded,
                cursor: self.chat_state.cursor_pos as u32,
            });
            let _ = self.client_conn.send(Lane::Control, Payload::Msg(req));
        }
    }

    fn update_player_hurt_tilt(&mut self, dt: f32) {
        if self.player_hurt_timer > 0.0 {
            self.player_hurt_timer = (self.player_hurt_timer - dt).max(0.0);
            let progress = (self.player_hurt_timer / 0.4).clamp(0.0, 1.0);
            let tilt_deg = (progress * std::f32::consts::PI).sin() * 12.0;
            self.camera.roll = tilt_deg.to_radians();
        } else {
            self.camera.roll = 0.0;
        }
    }

    fn update_toast(&mut self, dt: f32) {
        if let Some(toast) = &mut self.hud_state.active_toast {
            toast.elapsed_secs += dt;
            if toast.elapsed_secs >= toast.duration_secs {
                self.hud_state.active_toast = self.toast_queue.pop_front();
            }
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
                .map(|def| def.name.as_str())
                .or_else(|| self.custom_item_names.get(&item).map(String::as_str))
                .unwrap_or_else(|| telos_sim::item_name(item));
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
            let _ = telos_sim::inventory_click_with_registry(
                &mut self.inventory_sim,
                hovered,
                if hotbar_idx == 0 {
                    telos_sim::ClickButton::Right
                } else {
                    telos_sim::ClickButton::Left
                },
                telos_sim::ClickMode::SwapHotbar,
                &self.recipe_registry,
            );
            self.selected_block_state = BlockStateId::new(self.inventory_sim.selected_item().item);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn handle_container_swap_hotbar(&mut self, hotbar_idx: usize) {
        if let Some(hovered) = self.container_hovered_slot {
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
            if let Some(ref mut cont) = self.container_state {
                if cont.container_kind == 2 {
                    let mut ct_inv = telos_sim::CraftingTableInventory::new();
                    ct_inv.result =
                        telos_sim::ItemStack::new(cont.slots[0].item, cont.slots[0].count);
                    for (i, slot) in cont.slots[1..10].iter().enumerate() {
                        ct_inv.grid[i] = telos_sim::ItemStack::new(slot.item, slot.count);
                    }
                    let _ = telos_sim::crafting_table_container_click(
                        &mut ct_inv,
                        &mut self.inventory_sim,
                        hovered,
                        telos_sim::ClickButton::Left,
                        telos_sim::ClickMode::SwapHotbar,
                        &self.recipe_registry,
                    );
                    cont.slots[0] = UiSlotItem::new(ct_inv.result.item, ct_inv.result.count);
                    for (i, slot) in ct_inv.grid.iter().enumerate() {
                        cont.slots[1 + i] = UiSlotItem::new(slot.item, slot.count);
                    }
                } else if cont.container_kind == 1 {
                    let mut furnace_inv = telos_sim::FurnaceInventory::default();
                    for (i, slot) in cont.slots.iter().take(3).enumerate() {
                        furnace_inv.slots[i] = telos_sim::ItemStack::new(slot.item, slot.count);
                    }
                    let _ = telos_sim::furnace_container_click(
                        &mut furnace_inv,
                        &mut self.inventory_sim,
                        hovered,
                        telos_sim::ClickButton::Left,
                        telos_sim::ClickMode::SwapHotbar,
                        &self.fuel_registry,
                        &self.smelting_registry,
                    );
                    for (i, slot) in furnace_inv.slots.iter().enumerate() {
                        cont.slots[i] = UiSlotItem::new(slot.item, slot.count);
                    }
                } else if cont.container_kind == 3 {
                    let mut anvil_inv = telos_sim::AnvilInventory::new();
                    anvil_inv.left = telos_sim::ItemStack::new_enchanted(
                        cont.slots[0].item,
                        cont.slots[0].count,
                        telos_sim::CompactEnchantments(cont.slots[0].enchantments),
                    );
                    anvil_inv.right = telos_sim::ItemStack::new_enchanted(
                        cont.slots[1].item,
                        cont.slots[1].count,
                        telos_sim::CompactEnchantments(cont.slots[1].enchantments),
                    );
                    anvil_inv.result = telos_sim::ItemStack::new_enchanted(
                        cont.slots[2].item,
                        cont.slots[2].count,
                        telos_sim::CompactEnchantments(cont.slots[2].enchantments),
                    );
                    anvil_inv.level_cost = cont.properties[0].max(0) as u32;
                    let _ = telos_sim::anvil_container_click(
                        &mut anvil_inv,
                        &mut self.inventory_sim,
                        self.hud_state.xp_level,
                        hovered,
                        telos_sim::ClickButton::Left,
                        telos_sim::ClickMode::SwapHotbar,
                    );
                    cont.slots[0] = UiSlotItem::new_enchanted(
                        anvil_inv.left.item,
                        anvil_inv.left.count,
                        anvil_inv.left.enchantments.0,
                    );
                    cont.slots[1] = UiSlotItem::new_enchanted(
                        anvil_inv.right.item,
                        anvil_inv.right.count,
                        anvil_inv.right.enchantments.0,
                    );
                    cont.slots[2] = UiSlotItem::new_enchanted(
                        anvil_inv.result.item,
                        anvil_inv.result.count,
                        anvil_inv.result.enchantments.0,
                    );
                    cont.properties[0] = anvil_inv.level_cost as i16;
                } else {
                    let mut chest_inv = telos_sim::ChestInventory::default();
                    for (i, slot) in cont.slots.iter().enumerate() {
                        chest_inv.slots[i] = telos_sim::ItemStack::new(slot.item, slot.count);
                    }
                    let _ = telos_sim::container_click(
                        &mut chest_inv,
                        &mut self.inventory_sim,
                        hovered,
                        telos_sim::ClickButton::Left,
                        telos_sim::ClickMode::SwapHotbar,
                    );
                    for (i, slot) in chest_inv.slots.iter().enumerate() {
                        cont.slots[i] = UiSlotItem::new(slot.item, slot.count);
                    }
                }
            }
            self.selected_block_state = BlockStateId::new(self.inventory_sim.selected_item().item);
        }
    }

    fn close_active_container(&mut self) {
        if let Some(cont) = self.container_state.take() {
            let close_msg = C2sMessage::CloseContainer(C2sCloseContainer {
                window_id: cont.window_id,
            });
            let _ = self
                .client_conn
                .send(Lane::Control, Payload::Msg(close_msg));
            self.container_hovered_slot = None;
            self.inventory_sim.return_carried();
            if cont.container_kind == 2 {
                for slot in &cont.slots[1..10] {
                    if !slot.is_empty() {
                        let mut stack = telos_sim::ItemStack::new(slot.item, slot.count);
                        self.inventory_sim.insert_into_storage_or_hotbar(&mut stack);
                    }
                }
            } else if cont.container_kind == 3 {
                for slot in &cont.slots[0..2] {
                    if !slot.is_empty() {
                        let mut stack = telos_sim::ItemStack::new_enchanted(
                            slot.item,
                            slot.count,
                            telos_sim::CompactEnchantments(slot.enchantments),
                        );
                        self.inventory_sim.insert_into_storage_or_hotbar(&mut stack);
                    }
                }
            }
            if self.current_screen == AppScreen::InGame
                && !self.is_paused
                && !self.chat_state.is_open
            {
                self.controller.mouse_captured = true;
                self.set_cursor_captured(true);
            }
        }
    }

    fn recompute_active_effects(&mut self) {
        let mut speed_mult = 1.0f32;
        let mut jump_boost = 0.0f32;
        for eff in &self.client_effects {
            if eff.effect_id == 1 {
                speed_mult += 0.20 * (f32::from(eff.amplifier) + 1.0);
            } else if eff.effect_id == 2 {
                speed_mult *= (1.0 - 0.15 * (f32::from(eff.amplifier) + 1.0)).max(0.1);
            } else if eff.effect_id == 14 {
                jump_boost += 1.5 * (f32::from(eff.amplifier) + 1.0);
            }
        }
        self.physics.speed_multiplier = speed_mult;
        self.physics.jump_boost = jump_boost;

        self.hud_state.active_effects = self
            .client_effects
            .iter()
            .map(|e| {
                let (name_base, is_beneficial) =
                    telos_sim::StatusEffectRegistry::get_by_id(e.effect_id).map_or_else(
                        || ("Effect", true),
                        |def| (def.display_name, def.is_beneficial),
                    );
                let name = if e.amplifier == 0 {
                    name_base.to_string()
                } else {
                    let roman = match e.amplifier {
                        1 => "II",
                        2 => "III",
                        3 => "IV",
                        4 => "V",
                        _ => "+",
                    };
                    format!("{name_base} {roman}")
                };
                telos_ui::HudEffectDisplay {
                    effect_id: e.effect_id,
                    name,
                    amplifier: e.amplifier,
                    duration_ticks: e.duration_ticks,
                    color: e.particle_color,
                    ambient: e.ambient,
                    is_beneficial,
                }
            })
            .collect();
    }

    fn apply_block_update(&mut self, pos: BlockPos, state_id: BlockStateId) {
        let (chunk_pos, local_idx) = telos_voxel::coords::split_block_pos(pos);
        if let Some(snap) = self.chunks.get(&chunk_pos) {
            let old_state = snap.blocks().get(local_idx);
            if old_state == state_id {
                return;
            }

            let registry = &self.block_registry;
            let mut chunk = Chunk::from_blocks(chunk_pos, snap.blocks().clone(), |s| {
                registry
                    .flags(s)
                    .contains(telos_voxel::state::StateFlags::OPAQUE_FULL)
            });
            if let Some(light) = snap.light() {
                chunk.set_light(Some(light.clone()));
            }
            let old_flags = registry.flags(old_state);
            let new_flags = registry.flags(state_id);
            chunk.set(local_idx, state_id, old_flags, new_flags, 1);
            self.chunks.insert(chunk_pos, chunk.publish_snapshot());
            self.mark_dirty_with_neighbors(chunk_pos);
            tracing::info!(?pos, ?state_id, "Applied block update on client");
        } else {
            tracing::warn!(
                ?pos,
                ?chunk_pos,
                "apply_block_update: chunk not loaded on client!"
            );
        }
    }

    fn get_block_at(&self, pos: BlockPos) -> BlockStateId {
        let (chunk_pos, local_idx) = telos_voxel::coords::split_block_pos(pos);
        if let Some(snap) = self.chunks.get(&chunk_pos) {
            snap.blocks().get(local_idx)
        } else {
            BlockStateId::AIR
        }
    }

    fn resolve_block_sound_material(&self, id: BlockStateId) -> &'static str {
        if let Some(ident) = self.block_registry.identifier(id) {
            let path = ident.path();
            if path.contains("grass")
                || path.contains("leaves")
                || path.contains("plant")
                || path.contains("flower")
                || path.contains("dandelion")
                || path.contains("poppy")
            {
                "grass"
            } else if path.contains("wood")
                || path.contains("log")
                || path.contains("plank")
                || path.contains("door")
                || path.contains("fence")
                || path.contains("stair")
                || path.contains("chest")
            {
                "wood"
            } else if path.contains("dirt") || path.contains("gravel") {
                "dirt"
            } else if path.contains("sand") {
                "sand"
            } else if path.contains("water") {
                "water"
            } else {
                "stone"
            }
        } else {
            "stone"
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
                        let username = self.credentials.username.clone();
                        let login = C2sMessage::LoginStart(C2sLoginStart {
                            username: BoundedString::new(username)
                                .unwrap_or_else(|_| BoundedString::new("Player").unwrap()),
                            mode: AuthMode::Keyed,
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(login));
                        self.client_phase = ConnectionPhase::Login;
                    }
                }
                ConnectionPhase::Login => match msg {
                    S2cMessage::LoginChallenge(challenge) => {
                        info!(
                            "Received cryptographic login challenge from server, generating proof..."
                        );
                        let mut payload =
                            Vec::with_capacity(15 + 32 + self.credentials.username.len());
                        payload.extend_from_slice(b"telos-login-v1:");
                        payload.extend_from_slice(&challenge.challenge_nonce);
                        payload.extend_from_slice(self.credentials.username.as_bytes());

                        match self.credentials.keypair().map(|kp| kp.sign(&payload)) {
                            Ok(sig) => {
                                let proof = C2sMessage::LoginProof(C2sLoginProof {
                                    public_key: self.credentials.public_key(),
                                    signature: sig.to_bytes(),
                                    certificate_data: None,
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(proof));
                            }
                            Err(err) => {
                                tracing::error!(error = ?err, "Failed to derive keypair for login");
                            }
                        }
                    }
                    S2cMessage::LoginSuccess(succ) => {
                        info!(
                            username = succ.username.as_str(),
                            "Login successful, entering configuration phase..."
                        );
                        self.player_uuid = succ.player_uuid;
                        self.client_phase = ConnectionPhase::Config;
                    }
                    _ => {}
                },
                ConnectionPhase::Config => match msg {
                    S2cMessage::RegistryData(data) => {
                        info!(
                            registry_id = data.registry_id.as_str(),
                            entries = data.entries.len(),
                            "Received server registry data"
                        );
                    }
                    S2cMessage::ContentManifest(manifest) => {
                        info!(
                            custom_blocks = manifest.custom_blocks.len(),
                            custom_items = manifest.custom_items.len(),
                            "Received server content manifest, registering dynamic content..."
                        );
                        for wire_block in manifest.custom_blocks.iter() {
                            if let Ok(ident) = wire_block.identifier.as_str().parse::<Identifier>()
                            {
                                let shape = match wire_block.shape_kind {
                                    1 => {
                                        BlockShape::Boxes(vec![SubBox::new([0, 0, 0], [16, 8, 16])])
                                    }
                                    2 => BlockShape::Boxes(vec![
                                        SubBox::new([0, 0, 0], [16, 8, 16]),
                                        SubBox::new([0, 8, 8], [16, 16, 16]),
                                    ]),
                                    3 => BlockShape::Cross,
                                    _ => BlockShape::Cube,
                                };
                                let flags = match wire_block.shape_kind {
                                    0 => StateFlags::OPAQUE_CUBE,
                                    _ => StateFlags::empty(),
                                };
                                let state_id = if let Some(block) = self.block_registry.get(&ident)
                                {
                                    let existing = block.default_state();
                                    self.block_registry
                                        .set_light_emission(existing, wire_block.light_emission);
                                    existing
                                } else {
                                    self.block_registry.register_with_properties(
                                        ident.clone(),
                                        flags,
                                        shape,
                                        wire_block.light_emission,
                                    )
                                };
                                info!(
                                    identifier = wire_block.identifier.as_str(),
                                    state_id = state_id.as_u32(),
                                    light_emission = wire_block.light_emission,
                                    "Registered server custom block"
                                );

                                if let Some(gl) = &mut self.gl_renderer {
                                    let tint = [
                                        f32::from(wire_block.base_color[0]) / 255.0,
                                        f32::from(wire_block.base_color[1]) / 255.0,
                                        f32::from(wire_block.base_color[2]) / 255.0,
                                        f32::from(wire_block.base_color[3]) / 255.0,
                                    ];
                                    gl.register_custom_material(
                                        state_id.as_u32() as u16,
                                        0.0,
                                        tint,
                                    );
                                }
                            }
                        }

                        for wire_item in manifest.custom_items.iter() {
                            self.custom_item_names
                                .insert(wire_item.item_id, wire_item.name.to_string());
                            info!(
                                item_id = wire_item.item_id,
                                identifier = wire_item.identifier.as_str(),
                                name = wire_item.name.as_str(),
                                "Registered server custom item"
                            );
                        }
                    }
                    S2cMessage::RecipeManifest(manifest) => {
                        info!(
                            shaped = manifest.shaped_recipes.len(),
                            shapeless = manifest.shapeless_recipes.len(),
                            smelting = manifest.smelting_recipes.len(),
                            fuels = manifest.fuels.len(),
                            "Received server recipe manifest, registering dynamic recipes..."
                        );
                        for r in manifest.shaped_recipes.iter() {
                            let mut shaped = telos_sim::ShapedRecipe::new_mirrored(
                                r.width as usize,
                                r.height as usize,
                                r.pattern.to_vec(),
                                telos_sim::ItemStack::new(r.result_item, r.result_count),
                                r.mirrored,
                            );
                            if r.remainder_item > 0 {
                                shaped.remainder_item = Some(r.remainder_item);
                            }
                            self.recipe_registry.register_shaped(shaped);
                        }
                        for r in manifest.shapeless_recipes.iter() {
                            let mut shapeless = telos_sim::ShapelessRecipe::new(
                                r.ingredients.to_vec(),
                                telos_sim::ItemStack::new(r.result_item, r.result_count),
                            );
                            if r.remainder_item > 0 {
                                shapeless.remainder_item = Some(r.remainder_item);
                            }
                            self.recipe_registry.register_shapeless(shapeless);
                        }
                        for r in manifest.smelting_recipes.iter() {
                            self.smelting_registry.register(telos_sim::SmeltingRecipe {
                                input_item: r.input_item,
                                output_item: r.output_item,
                                output_count: r.output_count,
                                cook_duration: r.cook_duration,
                                experience: r.experience,
                            });
                        }
                        for f in manifest.fuels.iter() {
                            self.fuel_registry
                                .register(f.item_id, f.burn_duration_ticks);
                        }
                    }
                    S2cMessage::ConfigDone(_) => {
                        info!("Server configuration complete, acknowledging config...");
                        #[allow(clippy::cast_possible_truncation)]
                        let settings = C2sMessage::ClientSettings(C2sClientSettings {
                            view_distance: self.view_distance as u16,
                            simulation_distance: self.game_settings.video.simulation_distance
                                as u16,
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
                        let spawn_dvec = DVec3::new(join.spawn_x, join.spawn_y, join.spawn_z);
                        self.prediction.reset_to(MoveState::new(
                            spawn_dvec,
                            self.camera.yaw.to_degrees(),
                            self.camera.pitch.to_degrees(),
                            true,
                        ));
                        self.smoothing.reset();
                        self.physics.pos = spawn_dvec;
                        self.physics.vel = Vec3::ZERO;
                        self.physics.on_ground = false;
                        self.camera.position = self.physics.eye_pos();
                        self.client_phase = ConnectionPhase::Play;
                    }
                    other => {
                        tracing::debug!("Ignored packet in Config phase: {other:?}");
                    }
                },
                ConnectionPhase::Play => match msg {
                    S2cMessage::PlayerMovementAck(ack) => {
                        self.handle_movement_ack(ack);
                    }
                    S2cMessage::UniformChunk(uniform) => {
                        let pos = ChunkPos::new(uniform.chunk_x, uniform.chunk_y, uniform.chunk_z);
                        let is_opaque = self
                            .block_registry
                            .flags(uniform.block_state)
                            .contains(telos_voxel::state::StateFlags::OPAQUE_FULL);
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
                                match telos_protocol::messages::decode_chunk_snapshot(
                                    pos,
                                    &mut cursor,
                                ) {
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
                        if let Some(renderer) = &mut self.gl_renderer {
                            renderer.remove_chunk_mesh(&pos);
                        }
                        self.mark_dirty_neighbors(pos);
                    }
                    S2cMessage::LodNodeData(lod) => {
                        let key = telos_lod::coords::LodNodeKey::new(
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
                        let key = telos_lod::coords::LodNodeKey::new(
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
                        let server_time = (update.time_of_day % telos_core::time::DAY_TICKS) as f32;
                        self.client_time_of_day = server_time;
                    }
                    S2cMessage::UpdateStats(stats) => {
                        if stats.health < self.hud_state.health {
                            self.player_hurt_timer = 0.4;
                            self.audio.play_procedural_hurt(self.camera.position, 1.0);
                        }
                        self.hud_state.health = stats.health;
                        self.hud_state.max_health = stats.max_health;
                        self.hud_state.food = stats.food;
                        self.hud_state.saturation = stats.saturation;
                        self.hud_state.xp_level = stats.xp_level;
                        self.hud_state.xp_progress = stats.xp_progress;
                    }
                    S2cMessage::UpdateEffects(eff_msg) => {
                        self.client_effects.clone_from(&eff_msg.effects);
                        self.recompute_active_effects();
                    }
                    S2cMessage::EntityEffect(eff) => {
                        if eff.entity_id == 0 {
                            let particle_color =
                                telos_sim::StatusEffectRegistry::get_by_id(eff.effect_id)
                                    .map_or([255, 255, 255], |def| def.particle_color);
                            if let Some(existing) = self
                                .client_effects
                                .iter_mut()
                                .find(|e| e.effect_id == eff.effect_id)
                            {
                                existing.amplifier = eff.amplifier;
                                existing.duration_ticks = eff.duration_ticks;
                                existing.ambient = eff.ambient;
                                existing.particle_color = particle_color;
                            } else {
                                self.client_effects.push(NetworkEffect {
                                    effect_id: eff.effect_id,
                                    amplifier: eff.amplifier,
                                    duration_ticks: eff.duration_ticks,
                                    ambient: eff.ambient,
                                    particle_color,
                                });
                            }
                            self.recompute_active_effects();
                        }
                    }
                    S2cMessage::RemoveEntityEffect(rem) => {
                        if rem.entity_id == 0 {
                            self.client_effects.retain(|e| e.effect_id != rem.effect_id);
                            self.recompute_active_effects();
                        }
                    }
                    S2cMessage::InventoryBulk(bulk) => {
                        let had_items = self
                            .inventory_sim
                            .slots
                            .iter()
                            .map(|s| s.count)
                            .sum::<u16>();
                        for (i, slot) in bulk.slots.iter().enumerate() {
                            if i < self.inventory_sim.slots.len() {
                                self.inventory_sim.slots[i] = telos_sim::ItemStack::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    telos_sim::CompactEnchantments(slot.enchantments),
                                );
                            }
                        }
                        self.inventory_sim.carried = telos_sim::ItemStack::new_enchanted(
                            bulk.carried.item,
                            bulk.carried.count,
                            telos_sim::CompactEnchantments(bulk.carried.enchantments),
                        );
                        let now_items = self
                            .inventory_sim
                            .slots
                            .iter()
                            .map(|s| s.count)
                            .sum::<u16>();
                        if now_items > had_items && self.current_screen == AppScreen::InGame {
                            self.audio.play_item_pickup(self.camera.position);
                        }
                        let selected_item = self.inventory_sim.selected_item();
                        if selected_item.count > 0 {
                            self.selected_block_state = BlockStateId::new(selected_item.item);
                        }
                    }
                    S2cMessage::InventorySlot(slot_msg) => {
                        let idx = slot_msg.slot as usize;
                        if idx < self.inventory_sim.slots.len() {
                            let old_count = self.inventory_sim.slots[idx].count;
                            let old_item = self.inventory_sim.slots[idx].item;
                            self.inventory_sim.slots[idx] = telos_sim::ItemStack::new_enchanted(
                                slot_msg.item,
                                slot_msg.count,
                                telos_sim::CompactEnchantments(slot_msg.enchantments),
                            );
                            if (slot_msg.count > old_count
                                || (old_count == 0
                                    && slot_msg.count > 0
                                    && old_item != slot_msg.item))
                                && self.current_screen == AppScreen::InGame
                            {
                                self.audio.play_item_pickup(self.camera.position);
                            }
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
                            let lightning_pos = self.camera.position + self.camera.forward() * 45.0;
                            self.audio.play_thunder(lightning_pos);
                        }
                    }
                    S2cMessage::SpawnEntity(spawn) => {
                        self.entity_store.on_spawn(spawn);
                    }
                    S2cMessage::SpawnItem(spawn) => {
                        self.entity_store.on_spawn_item(spawn);
                    }
                    S2cMessage::SpawnArrow(spawn) => {
                        self.entity_store.on_spawn_arrow(spawn);
                    }
                    S2cMessage::DespawnEntity(despawn) => {
                        self.entity_store.on_despawn(despawn.net_ids.as_slice());
                    }
                    S2cMessage::EntityMove(m) => {
                        if m.on_ground
                            && let Some(ent) = self.entity_store.get(m.net_id)
                            && ent.entity_type == telos_sim::EntityType::Arrow
                        {
                            self.audio.play_arrow_hit(ent.pos.as_vec3(), false);
                        }
                        self.entity_store.on_move(m);
                    }
                    S2cMessage::EntityStatus(s) => {
                        self.entity_store.on_status(s.net_id, s.status);
                        if s.status == 2
                            && let Some(ent) = self.entity_store.get(s.net_id)
                        {
                            self.audio.play_procedural_hurt(ent.pos.as_vec3(), 1.0);
                        }
                    }
                    S2cMessage::ChatMessage(chat) => {
                        let now_secs = self.start_time.elapsed().as_secs_f64();
                        self.chat_state.add_message(
                            chat.sender.as_str(),
                            chat.message.as_str(),
                            now_secs,
                        );
                    }
                    S2cMessage::CommandSuggestions(sug) => {
                        let matches: Vec<String> =
                            sug.matches.iter().map(|s| s.as_str().to_string()).collect();
                        self.chat_state.set_suggestions(
                            sug.start as usize,
                            sug.length as usize,
                            matches,
                        );
                    }
                    S2cMessage::ParticleEvent(ev) => {
                        if let Some(hook) = &self.js_particle_hook {
                            let type_str = match ev.effect {
                                ParticleEffectKind::BlockBreak => "block_break",
                                ParticleEffectKind::BlockPlace => "block_place",
                                ParticleEffectKind::Footstep => "footstep",
                                ParticleEffectKind::Crit => "crit",
                                ParticleEffectKind::Heart => "heart",
                                ParticleEffectKind::Smoke => "smoke",
                                ParticleEffectKind::Flame => "flame",
                                ParticleEffectKind::Explosion => "explosion",
                            };
                            let p = hook.on_particle_spawn(
                                type_str,
                                telos_mod::JsParticleParams {
                                    pos: [ev.x, ev.y, ev.z],
                                    velocity: [0.0, 0.0, 0.0],
                                    color_tint: 0xFFFF_FFFF,
                                    scale: 1.0,
                                    lifetime: f32::from(ev.count),
                                },
                            );
                            let mut mod_ev = ev;
                            mod_ev.x = p.pos[0];
                            mod_ev.y = p.pos[1];
                            mod_ev.z = p.pos[2];
                            mod_ev.count = (p.lifetime as u16).clamp(1, 128);
                            self.particle_system.spawn_from_event(&mod_ev, |id| {
                                block_state_to_particle_layer(telos_voxel::state::BlockStateId(id))
                            });
                        } else {
                            self.particle_system.spawn_from_event(&ev, |id| {
                                block_state_to_particle_layer(telos_voxel::state::BlockStateId(id))
                            });
                        }
                    }
                    S2cMessage::OpenContainer(open) => {
                        let mut slots = [UiSlotItem::EMPTY; 27];
                        for (i, slot) in open.slots.iter().enumerate().take(27) {
                            slots[i] =
                                UiSlotItem::new_enchanted(slot.item, slot.count, slot.enchantments);
                        }
                        if let Some(existing) = &mut self.container_state {
                            if existing.window_id == open.window_id {
                                existing.container_kind = open.container_kind;
                                existing.title = open.title.as_str().to_string();
                                existing.slots = slots;
                            } else {
                                self.container_state = Some(ClientContainerState {
                                    window_id: open.window_id,
                                    container_kind: open.container_kind,
                                    title: open.title.as_str().to_string(),
                                    x: open.x,
                                    y: open.y,
                                    z: open.z,
                                    slots,
                                    properties: [0; 4],
                                });
                            }
                        } else {
                            self.container_state = Some(ClientContainerState {
                                window_id: open.window_id,
                                container_kind: open.container_kind,
                                title: open.title.as_str().to_string(),
                                x: open.x,
                                y: open.y,
                                z: open.z,
                                slots,
                                properties: [0; 4],
                            });
                            self.inventory_open = false;
                            self.controller.mouse_captured = false;
                            self.set_cursor_captured(false);
                            self.controller.forward = false;
                            self.controller.backward = false;
                            self.controller.left = false;
                            self.controller.right = false;
                            self.controller.up = false;
                            self.controller.down = false;
                        }
                    }
                    S2cMessage::ContainerProperty(prop) => {
                        if let Some(cont) = &mut self.container_state
                            && cont.window_id == prop.window_id
                            && (prop.property_id as usize) < cont.properties.len()
                        {
                            cont.properties[prop.property_id as usize] = prop.value;
                        }
                    }
                    S2cMessage::CloseContainer(close) => {
                        if self
                            .container_state
                            .as_ref()
                            .is_some_and(|c| c.window_id == close.window_id)
                        {
                            self.container_state = None;
                            self.container_hovered_slot = None;
                            self.inventory_sim.return_carried();
                            if self.current_screen == AppScreen::InGame
                                && !self.is_paused
                                && !self.chat_state.is_open
                            {
                                self.controller.mouse_captured = true;
                                self.set_cursor_captured(true);
                            }
                        }
                    }
                    S2cMessage::BlockEvent(evt) if evt.action == 1 => {
                        let sound_pos = glam::Vec3::new(
                            evt.x as f32 + 0.5,
                            evt.y as f32 + 0.5,
                            evt.z as f32 + 0.5,
                        );
                        if evt.param == 1 {
                            self.audio.play_chest_open(sound_pos);
                        } else if evt.param == 0 {
                            self.audio.play_chest_close(sound_pos);
                        }
                    }
                    S2cMessage::ModalFormRequest(req) => match req.parse_form() {
                        Ok(form_data) => {
                            let mut form = ModalFormScreen::new(req.form_id, form_data);
                            let win_size = self.window_size();
                            let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                            let width_gui = win_size.width as f32 / gui_scale as f32;
                            let height_gui = win_size.height as f32 / gui_scale as f32;
                            form.update_layout(width_gui, height_gui);
                            self.active_modal_form = Some(form);
                            self.controller.mouse_captured = false;
                            self.controller.forward = false;
                            self.controller.backward = false;
                            self.controller.left = false;
                            self.controller.right = false;
                            self.controller.up = false;
                            self.controller.down = false;
                            self.controller.sprint = false;
                            self.set_cursor_captured(false);
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Failed to parse S2cModalFormRequest {}: {err}",
                                req.form_id
                            );
                            let cancel_msg =
                                C2sMessage::ModalFormResponse(C2sModalFormResponse::cancelled(
                                    req.form_id,
                                    telos_core::form::FormCancelReason::UserClosed,
                                ));
                            let _ = self
                                .client_conn
                                .send(Lane::Control, Payload::Msg(cancel_msg));
                        }
                    },
                    S2cMessage::GameMode(gm_msg) => {
                        let caps = PlayerCapabilities::from_wire(
                            gm_msg.game_mode,
                            gm_msg.flags,
                            gm_msg.fly_speed,
                            gm_msg.walk_speed,
                            gm_msg.reach_distance,
                        );
                        self.physics.set_capabilities(caps);
                        let mode_key = self.physics.game_mode.translation_key();
                        self.hud_state.game_mode = self.catalog.translate(mode_key).to_string();
                        self.hud_state.is_flying = self.physics.flying;
                    }
                    S2cMessage::AdvancementUpdate(update) => {
                        if update.reset_all {
                            for node in &mut self.advancements_screen.nodes {
                                node.completed = false;
                            }
                        }
                        for wire in update.advancements.as_slice() {
                            if let Some(node) = self
                                .advancements_screen
                                .nodes
                                .iter_mut()
                                .find(|n| n.id == wire.id.as_str())
                            {
                                node.completed = wire.completed;
                            }
                        }
                    }
                    S2cMessage::AdvancementToast(toast) => {
                        self.audio.play_advancement_chime();
                        let new_toast = ToastState {
                            id: toast.id.to_string(),
                            title: toast.title.to_string(),
                            icon_item: toast.icon_item,
                            frame: toast.frame,
                            elapsed_secs: 0.0,
                            duration_secs: 5.0,
                        };
                        if self.hud_state.active_toast.is_none() {
                            self.hud_state.active_toast = Some(new_toast);
                        } else {
                            self.toast_queue.push_back(new_toast);
                        }
                    }
                    _ => {}
                },
            }
        }

        if let Some(listener) = &mut self.lan_listener {
            listener.poll();
            for s in listener.active_servers() {
                let addr = format!("{}:{}", s.sender_addr.ip(), s.beacon.quic_port);
                self.server_list
                    .add_or_update_lan_server(telos_ui::ServerEntry::new(
                        &s.beacon.motd,
                        addr,
                        &s.beacon.motd,
                        s.beacon.current_players,
                        s.beacon.max_players,
                        true,
                    ));
            }
        }
    }

    fn get_sky_light_at(&self, pos: BlockPos) -> u8 {
        let (cpos, lpos) = telos_voxel::coords::split_block_pos(pos);
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
            let climate = telos_worldgen::ClimatePoint::sample(self.world_seed, px, pz);
            let is_dry = climate.humidity < -0.35 && climate.temperature > 0.5;

            let test_y = cam_pos.y + min_rel_y + base_ry;
            let precip_kind =
                telos_sim::weather::precipitation_at(climate.temperature, test_y, is_dry);
            if precip_kind == telos_sim::weather::PrecipitationKind::None {
                continue;
            }

            let is_snow = precip_kind == telos_sim::weather::PrecipitationKind::Snow;

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
            let sky_light = self.get_sky_light_at(telos_core::BlockPos::new(bx, by, bz));
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

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::too_many_lines
    )]
    fn update_particles(&mut self, dt: f32) {
        // Dev/screenshot showcase burst: trigger a visual particle burst when in screenshot mode in-game
        let should_spawn_burst = self.current_screen == AppScreen::InGame
            && self.args.screenshot.is_some()
            && self.total_frames == 20;

        if should_spawn_burst {
            let forward = self.camera.forward();
            let burst_center = self.camera.position + forward * 2.5;
            self.particle_system.spawn_block_break(
                burst_center - Vec3::new(0.6, 0.0, 0.0),
                0,
                [130, 130, 130, 255],
                32,
            );
            self.particle_system.spawn_block_place(
                burst_center + Vec3::new(0.6, 0.0, 0.0),
                3,
                [110, 175, 75, 255],
                16,
            );
            self.particle_system
                .spawn_torch_flame(burst_center + Vec3::new(0.0, -0.3, 0.0));
            self.particle_system
                .spawn_torch_smoke(burst_center + Vec3::new(0.0, -0.3, 0.0));
            self.particle_system
                .spawn_crit(burst_center + Vec3::new(0.0, 0.3, 0.0), 16);
            let heart_ev = S2cParticleEvent {
                x: burst_center.x,
                y: burst_center.y + 0.6,
                z: burst_center.z,
                effect: ParticleEffectKind::Heart,
                count: 4,
                speed: 1.0,
                block_state_id: 0,
            };
            self.particle_system
                .spawn_from_event(&heart_ev, |_| (0, [255, 100, 120, 255]));
        }

        if self.total_frames == 35 && self.args.screenshot.is_some() {
            // Build a demonstration circuit platform right in front of camera for visual verification
            let cx = 128;
            let cy = 44;
            let cz = 156;

            tracing::info!(
                cx,
                cy,
                cz,
                "Placing logic circuit showcase in front of camera"
            );

            for dx in -5..=5 {
                for dz in -2..=2 {
                    self.apply_block_update(
                        BlockPos::new(cx + dx, cy - 1, cz + dz),
                        BlockStateId::new(1),
                    );
                    self.apply_block_update(BlockPos::new(cx + dx, cy, cz + dz), BlockStateId::AIR);
                    self.apply_block_update(
                        BlockPos::new(cx + dx, cy + 1, cz + dz),
                        BlockStateId::AIR,
                    );
                }
            }
            self.apply_block_update(BlockPos::new(cx - 4, cy, cz), BlockStateId::new(28)); // Logic Inverter
            self.apply_block_update(BlockPos::new(cx - 3, cy, cz), BlockStateId::new(21)); // Power block
            self.apply_block_update(BlockPos::new(cx - 2, cy, cz), BlockStateId::new(20)); // Powered wire
            self.apply_block_update(BlockPos::new(cx - 1, cy, cz), BlockStateId::new(25)); // Lamp lit
            self.apply_block_update(BlockPos::new(cx, cy, cz), BlockStateId::new(22)); // Lever
            self.apply_block_update(BlockPos::new(cx + 1, cy, cz), BlockStateId::new(19)); // Wire (unpowered)
            self.apply_block_update(BlockPos::new(cx + 2, cy, cz), BlockStateId::new(26)); // Repeater
            self.apply_block_update(BlockPos::new(cx + 3, cy, cz), BlockStateId::new(24)); // Lamp unlit
            self.apply_block_update(BlockPos::new(cx + 4, cy, cz), BlockStateId::new(30)); // Diode
        }

        // Ambient torch particles: sample a few positions around camera
        let cam_pos = self.camera.position;
        let cx = cam_pos.x.floor() as i32;
        let cy = cam_pos.y.floor() as i32;
        let cz = cam_pos.z.floor() as i32;
        for _ in 0..8 {
            self.particle_seed = self
                .particle_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let rx = ((self.particle_seed >> 16) & 0x1F) as i32 - 16;
            let ry = ((self.particle_seed >> 24) & 0x0F) as i32 - 8;
            let rz = ((self.particle_seed >> 32) & 0x1F) as i32 - 16;
            let pos = BlockPos::new(cx + rx, cy + ry, cz + rz);
            let state = self.get_block_at(pos);
            if state.0 == 15 {
                let tpos = Vec3::new(
                    pos.x() as f32 + 0.5,
                    pos.y() as f32 + 0.5,
                    pos.z() as f32 + 0.5,
                );
                if (self.particle_seed & 1) == 0 {
                    self.particle_system.spawn_torch_flame(tpos);
                } else {
                    self.particle_system.spawn_torch_smoke(tpos);
                }
            }
        }

        // Ambient status effect swirl particles around player
        if !self.client_effects.is_empty() && self.total_frames.is_multiple_of(4) {
            let eye = self.physics.eye_pos();
            let p_pos = Vec3::new(eye.x, eye.y - 0.8, eye.z);
            for eff in &self.client_effects {
                self.particle_system
                    .spawn_status_effect_swirl(p_pos, eff.particle_color);
            }
        }

        // Physics tick with solid block collision check
        let chunks = &self.chunks;
        let is_solid = |bx: i32, by: i32, bz: i32| -> bool {
            let (chunk_pos, local_idx) =
                telos_voxel::coords::split_block_pos(BlockPos::new(bx, by, bz));
            if let Some(snap) = chunks.get(&chunk_pos) {
                let state = snap.blocks().get(local_idx);
                state.0 != 0 && state.0 != 6
            } else {
                false
            }
        };
        self.particle_system.tick(dt, is_solid);

        // Extract active particles to GPU format and upload to device-addressable buffer
        let mut gpu_particles = Vec::new();
        self.particle_system
            .extract_gpu_particles(&mut gpu_particles);
        self.active_particles_count = gpu_particles.len() as u32;

        if self.active_particles_count > 0
            && let Some(buf) = &mut self.particle_buffer
        {
            let _ = buf.write_bytes(bytemuck::cast_slice(&gpu_particles));
            if self.total_frames.is_multiple_of(20) {
                tracing::info!(
                    count = self.active_particles_count,
                    frame = self.total_frames,
                    "Active particles simulated and uploaded to GPU"
                );
            }
        }
    }

    fn tick_movement_prediction(&mut self) {
        if self.client_phase != ConnectionPhase::Play {
            return;
        }

        self.client_tick = self.client_tick.wrapping_add(1).max(1);

        let mut buttons = 0u16;
        if self.controller.mouse_captured {
            if self.controller.forward {
                buttons |= input_buttons::FORWARD;
            }
            if self.controller.backward {
                buttons |= input_buttons::BACK;
            }
            if self.controller.left {
                buttons |= input_buttons::LEFT;
            }
            if self.controller.right {
                buttons |= input_buttons::RIGHT;
            }
            if self.controller.sprint {
                buttons |= input_buttons::SPRINT;
            }
            if self.controller.up {
                buttons |= input_buttons::FLY_UP | input_buttons::JUMP;
            }
            if self.controller.down {
                buttons |= input_buttons::FLY_DOWN | input_buttons::SNEAK;
            }
        }

        let frame = InputFrame {
            tick: self.client_tick,
            buttons,
            yaw: quantize_yaw(self.camera.yaw.to_degrees()),
            pitch: quantize_pitch(self.camera.pitch.to_degrees()),
            hotbar: self.selected_hotbar_slot as u8,
        };

        let mode = if self.physics.flying || self.physics.capabilities.noclip {
            MoveMode::NoClipFly
        } else {
            MoveMode::Walk
        };

        self.prediction.push_and_predict(frame, mode, 0.05);

        let unacked = self.prediction.unacked_frames(self.last_acked_server_tick);
        if let Ok(bounded_frames) = BoundedVec::new(unacked) {
            let input_msg = C2sMessage::PlayerInput(C2sPlayerInput {
                last_server_tick_ack: self.last_acked_server_tick,
                frames: bounded_frames,
            });
            let _ = self
                .client_conn
                .send(Lane::Control, Payload::Msg(input_msg));
        }
    }

    fn handle_movement_ack(&mut self, ack: S2cPlayerMovementAck) {
        self.last_acked_server_tick = ack.server_tick;

        if ack.teleport_id != 0 {
            let target_pos = DVec3::new(ack.x, ack.y, ack.z);
            let vel = Vec3::new(ack.vx, ack.vy, ack.vz);
            self.prediction.reset_to(MoveState {
                pos: target_pos,
                vel,
                yaw: ack.yaw,
                pitch: ack.pitch,
                on_ground: ack.on_ground,
                flying: ack.flying,
            });
            self.smoothing.reset();
            #[allow(clippy::cast_possible_truncation)]
            {
                self.camera.position = Vec3::new(
                    target_pos.x as f32,
                    target_pos.y as f32,
                    target_pos.z as f32,
                );
            }

            let tp_ack = C2sMessage::TeleportAck(C2sTeleportAck {
                teleport_id: ack.teleport_id,
            });
            let _ = self.client_conn.send(Lane::Control, Payload::Msg(tp_ack));
        } else {
            let mode = if self.physics.flying || self.physics.capabilities.noclip {
                MoveMode::NoClipFly
            } else {
                MoveMode::Walk
            };
            if let Some(reconciliation) = self.prediction.reconcile(&ack, mode, 0.05) {
                self.smoothing
                    .add_error(reconciliation.error, reconciliation.is_teleport);
            }
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
        clippy::similar_names,
        clippy::too_many_lines
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
            let layers = telos_mesh::mesher::mesh_chunk_multilayers(
                &chunk,
                &neighbors,
                &self.block_registry,
            );

            let opaque = upload_t0_layer(gpu_context, &layers.opaque, "chunk_mesh_opaque");
            let t1_opaque = upload_t1_layer(gpu_context, &layers.t1_opaque, "chunk_mesh_t1");
            let cutout = upload_t0_layer(gpu_context, &layers.cutout, "chunk_mesh_cutout");
            let translucent =
                upload_t0_layer(gpu_context, &layers.translucent, "chunk_mesh_translucent");
            let t2_cutout = upload_t2_layer(gpu_context, &layers.t2_cutout, "chunk_mesh_t2_cutout");
            let t2_translucent = upload_t2_layer(
                gpu_context,
                &layers.t2_translucent,
                "chunk_mesh_t2_translucent",
            );

            if opaque.is_none()
                && t1_opaque.is_none()
                && cutout.is_none()
                && translucent.is_none()
                && t2_cutout.is_none()
                && t2_translucent.is_none()
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
                    t2_cutout,
                    t2_translucent,
                };

                if let Some(old_mesh) = self.chunk_meshes.insert(pos, new_mesh) {
                    old_mesh.destroy(gpu_context);
                }
            }
        }
    }

    #[allow(clippy::similar_names)]
    fn rebuild_dirty_meshes_gl(&mut self, budget: usize) {
        let Some(gl_renderer) = &mut self.gl_renderer else {
            return;
        };

        let cam_cx = (self.camera.position.x / 32.0).floor() as i32;
        let cam_cy = (self.camera.position.y / 32.0).floor() as i32;
        let cam_cz = (self.camera.position.z / 32.0).floor() as i32;

        let mut processed = 0;
        while processed < budget {
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
                gl_renderer.remove_chunk_mesh(&pos);
                continue;
            };

            if chunk.blocks().is_uniform()
                && chunk.blocks().get(LocalIdx::from_coords_unchecked(0, 0, 0)) == BlockStateId::AIR
            {
                gl_renderer.remove_chunk_mesh(&pos);
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
            let layers = telos_mesh::mesher::mesh_chunk_multilayers(
                &chunk,
                &neighbors,
                &self.block_registry,
            );

            gl_renderer.update_chunk_mesh(pos, &layers);
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
    fn render(&mut self) -> bool {
        if self.active_backend == telos_gpu::rhi::RenderBackendType::OpenGl {
            return self.render_opengl();
        }

        let fps_limit = self.game_settings.video.fps_limit;
        if fps_limit > 0 {
            let target_frame_dur = Duration::from_secs_f64(1.0 / f64::from(fps_limit));
            let elapsed = self.last_frame_start.elapsed();
            if elapsed < target_frame_dur {
                std::thread::sleep(target_frame_dur.checked_sub(elapsed).unwrap());
            }
        }
        self.last_frame_start = Instant::now();

        if self.current_screen == AppScreen::InGame {
            self.poll_network();
            self.rebuild_dirty_meshes(16);
            self.upload_pending_lod_meshes();
        }

        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        self.update_player_hurt_tilt(dt);
        self.update_toast(dt);

        if self.current_screen == AppScreen::InGame {
            if !self.is_paused {
                let input_state = PlayerInputState {
                    forward: self.controller.forward && self.controller.mouse_captured,
                    backward: self.controller.backward && self.controller.mouse_captured,
                    left: self.controller.left && self.controller.mouse_captured,
                    right: self.controller.right && self.controller.mouse_captured,
                    jump: self.controller.up && self.controller.mouse_captured,
                    sneak: self.controller.down && self.controller.mouse_captured,
                    sprint: self.controller.sprint && self.controller.mouse_captured,
                };
                let reg = &self.block_registry;
                let chunks = &self.chunks;
                self.physics.update(
                    dt,
                    self.camera.yaw.to_degrees(),
                    &input_state,
                    |bx, by, bz, out| {
                        let (cpos, lidx) =
                            telos_voxel::coords::split_block_pos(BlockPos::new(bx, by, bz));
                        let block = if let Some(snap) = chunks.get(&cpos) {
                            snap.blocks().get(lidx)
                        } else {
                            BlockStateId::AIR
                        };
                        if block == BlockStateId::AIR {
                            return;
                        }
                        match reg.shape(block) {
                            BlockShape::Cube => {
                                out.push(Aabb::from_block(bx, by, bz));
                            }
                            BlockShape::Boxes(boxes) => {
                                for b in boxes {
                                    out.push(Aabb::from_sub_box(bx, by, bz, b.min, b.max));
                                }
                            }
                            _ => {}
                        }
                    },
                );
                self.camera.position = self.physics.eye_pos();
                let mode_key = self.physics.game_mode.translation_key();
                self.hud_state.game_mode = self.catalog.translate(mode_key).to_string();
                self.hud_state.is_flying = self.physics.flying;

                self.sim_time_acc += dt;
                while self.sim_time_acc >= 0.05 {
                    self.sim_time_acc -= 0.05;
                    self.tick_movement_prediction();

                    if !self.client_effects.is_empty() {
                        let mut changed = false;
                        for eff in &mut self.client_effects {
                            if eff.duration_ticks > 0 {
                                eff.duration_ticks -= 1;
                                changed = true;
                            }
                        }
                        let prev_len = self.client_effects.len();
                        self.client_effects.retain(|e| e.duration_ticks != 0);
                        if changed || self.client_effects.len() != prev_len {
                            self.recompute_active_effects();
                        }
                    }

                    if let Some(ref cont) = self.container_state {
                        let chest_center = Vec3::new(
                            cont.x as f32 + 0.5,
                            cont.y as f32 + 0.5,
                            cont.z as f32 + 0.5,
                        );
                        if (self.camera.position - chest_center).length_squared() > 5.0 * 5.0 {
                            self.close_active_container();
                        }
                    }

                    if self.bow_charging {
                        if self.inventory_sim.selected_item().item == telos_sim::ITEM_BOW
                            && !self.inventory_open
                            && !self.advancements_open
                            && self.container_state.is_none()
                            && !self.is_paused
                        {
                            self.bow_charge_ticks = self.bow_charge_ticks.saturating_add(1);
                        } else {
                            self.bow_charging = false;
                            self.bow_charge_ticks = 0;
                        }
                    }
                }

                let base_fov = self.game_settings.video.fov.to_radians();
                if self.bow_charging && self.bow_charge_ticks >= telos_sim::BOW_MIN_CHARGE_TICKS {
                    #[allow(clippy::cast_precision_loss)]
                    let charge_ratio = ((f32::from(self.bow_charge_ticks)
                        - f32::from(telos_sim::BOW_MIN_CHARGE_TICKS))
                        / (f32::from(telos_sim::BOW_FULL_CHARGE_TICKS)
                            - f32::from(telos_sim::BOW_MIN_CHARGE_TICKS)))
                    .clamp(0.0, 1.0);
                    let zoom_mult = 1.0 - 0.15 * charge_ratio;
                    self.camera.fov_y = base_fov * zoom_mult;
                } else {
                    self.camera.fov_y = base_fov;
                }
            }
        } else if self.current_screen == AppScreen::MainMenu {
            self.camera.yaw += 0.05 * dt;
        }

        // Update spatial audio listener
        self.audio.set_listener(
            self.camera.position,
            self.camera.forward(),
            self.camera.up(),
        );

        // Track player movement for footsteps
        let delta_audio = self.camera.position - self.last_audio_pos;
        let horiz_dist = (delta_audio.x * delta_audio.x + delta_audio.z * delta_audio.z).sqrt();
        self.last_audio_pos = self.camera.position;

        let foot_pos = BlockPos::new(
            self.camera.position.x.floor() as i32,
            (self.camera.position.y - 1.6).floor() as i32,
            self.camera.position.z.floor() as i32,
        );
        let ground_block = self.get_block_at(foot_pos);
        if ground_block != BlockStateId::AIR && horiz_dist > 1e-4 && horiz_dist < 2.5 {
            self.audio_step_dist += horiz_dist;
            if self.audio_step_dist >= 1.65 {
                self.audio_step_dist = 0.0;
                let sound_mat = self.resolve_block_sound_material(ground_block);
                let pitch = 0.95
                    + (self.camera.position.x.abs() * 7.13 + self.camera.position.z.abs() * 11.37)
                        .fract()
                        * 0.15;
                self.audio.play_procedural_step(
                    sound_mat,
                    self.camera.position - Vec3::new(0.0, 1.5, 0.0),
                    pitch,
                );
                let (_layer, tint) = block_state_to_particle_layer(ground_block);
                self.particle_system
                    .spawn_footstep(self.camera.position - Vec3::new(0.0, 1.5, 0.0), tint);
            }
        }

        // Ambient rain loop volume modulation & player cleanup
        let rain_intensity = self
            .weather_rain_level
            .max(self.weather_thunder_level * 0.85);
        self.audio.update_ambient_rain(rain_intensity);
        self.audio.cleanup_finished_players();

        // Advance smooth client time of day (20 ticks per second)
        self.client_time_of_day =
            (self.client_time_of_day + dt * 20.0) % (telos_core::time::DAY_TICKS as f32);

        let sun_angle = telos_core::time::sun_angle(
            self.client_time_of_day as u64,
            self.client_time_of_day.fract(),
        );
        let sun_dir = telos_core::time::sun_direction(sun_angle);
        let moon_dir = telos_core::time::moon_direction(sun_angle);
        let day_number = self.client_world_age / telos_core::time::DAY_TICKS;
        let moon_phase = telos_core::time::moon_phase(day_number);

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
        self.update_particles(dt);

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
                let (cpos, lpos) = telos_voxel::coords::split_block_pos(bpos);
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
        let max_reach = self.physics.capabilities.reach_distance;
        let is_solid = |pos: BlockPos| -> bool {
            let (chunk_pos, local_idx) = telos_voxel::coords::split_block_pos(pos);
            if let Some(snap) = self.chunks.get(&chunk_pos) {
                let state = snap.blocks().get(local_idx);
                !self
                    .block_registry
                    .flags(state)
                    .contains(telos_voxel::state::StateFlags::AIR)
            } else {
                false
            }
        };
        self.targeted_block = if max_reach > 0.0 {
            raycast_voxels(origin, forward, max_reach, is_solid)
        } else {
            None
        };

        self.total_frames += 1;
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
            return false;
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
            Ok(None) => return false,
            Err(err) => {
                tracing::error!("begin_frame failed: {err}");
                return false;
            }
        };

        let (cmd, image_index) = frame_data;
        let swapchain_extent = gpu_context.extent();
        let image_view = gpu_context.swapchain().image_view(image_index as usize);
        let swapchain_image = gpu_context.swapchain().image(image_index as usize);

        let use_postprocess =
            self.game_settings.video.post_processing && self.postprocess.is_some();
        let target_color_view = if use_postprocess {
            self.postprocess.as_ref().unwrap().scene_color_view()
        } else {
            image_view
        };

        #[allow(clippy::cast_precision_loss)]
        let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;
        let view_proj_mat = self.camera.view_proj_matrix(aspect);
        let view_proj = view_proj_mat.to_cols_array();
        let inv_view_proj = view_proj_mat.inverse().to_cols_array();
        #[allow(clippy::cast_precision_loss)]
        let sim_dist_meters = (self.game_settings.video.simulation_distance as f32) * 32.0;

        let staging_idx = if self.lightmap_staging_buffers.is_empty() {
            0
        } else {
            (self.frame_counter as usize) % self.lightmap_staging_buffers.len()
        };
        if let Some(staging_buf) = self.lightmap_staging_buffers.get_mut(staging_idx) {
            let _ = staging_buf.write_bytes(&self.lightmap_lut.data);
        }

        let device = gpu_context.device().raw();
        let is_first_frame = u32::from(self.frame_counter <= 1 || self.args.no_cull);
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

                if let Some(pp) = &self.postprocess {
                    let pp_init_barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .image(pp.scene_color_image())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    let dep = vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&pp_init_barrier));
                    device.cmd_pipeline_barrier2(cmd, &dep);
                }
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
                .image_view(target_color_view)
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

            if self.current_screen == AppScreen::InGame {
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t1_pipeline.raw(),
                    );
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        lod_pipeline.raw(),
                    );
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
                .image_view(target_color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD) // Preserve Early Pass color!
                .store_op(vk::AttachmentStoreOp::STORE);

            let late_depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_buffer.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD) // Preserve Early Pass depth!
                .store_op(vk::AttachmentStoreOp::STORE);

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

            if self.current_screen == AppScreen::InGame {
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t1_pipeline.raw(),
                    );
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

                // Late Pass 3b: Cutout T2 (Flora & torches: poppy, dandelion, torch)
                if let Some(t2_cutout_pipeline) = &self.t2_cutout_pipeline {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t2_cutout_pipeline.raw(),
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t2_cutout_pipeline.layout(),
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    let max_dist_sq = (self.view_distance as f32 * 32.0).powi(2);
                    for mesh in self.chunk_meshes.values() {
                        if let Some(layer) = &mesh.t2_cutout
                            && layer.quad_count > 0
                        {
                            let center = (mesh.min_aabb + mesh.max_aabb) * 0.5;
                            if center.distance_squared(self.camera.position) > max_dist_sq {
                                continue;
                            }
                            let pc = T2PushConstants {
                                view_proj,
                                quad_buffer_address: layer.buffer.device_address(),
                                chunk_x: mesh.pos[0],
                                chunk_y: mesh.pos[1],
                                camera_pos: [
                                    self.camera.position.x,
                                    self.camera.position.y,
                                    self.camera.position.z,
                                    sim_dist_meters,
                                ],
                                chunk_z: mesh.pos[2],
                                frame_tick_flags: self.frame_tick & 0x7FFF_FFFF,
                                water_base_layer: self.anim_textures.water_still.base_layer,
                                water_frame_count: self.anim_textures.water_still.frame_count,
                                lava_base_layer: self.anim_textures.lava_still.base_layer,
                                lava_frame_count: self.anim_textures.lava_still.frame_count,
                                fire_base_layer: self.anim_textures.fire.base_layer,
                                fire_frame_count: self.anim_textures.fire.frame_count,
                            };
                            device.cmd_push_constants(
                                cmd,
                                t2_cutout_pipeline.layout(),
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&pc),
                            );
                            device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                        }
                    }
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        lod_pipeline.raw(),
                    );
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
            }

            // Pass 12: Celestial Sky Pass (or 3D Rotating Panorama Skybox for Menus)
            if self.current_screen != AppScreen::InGame {
                if let (Some(pano_pipe), Some(pano_set)) =
                    (&self.panorama_pipeline, self.panorama_descriptor_set)
                {
                    let yaw = (self.frame_counter as f32 * 0.0015) % std::f32::consts::TAU;
                    let fov = 85.0_f32.to_radians();
                    let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;
                    let mut pano_cam = Camera::new(Vec3::ZERO);
                    pano_cam.yaw = yaw;
                    pano_cam.pitch = 0.0;
                    pano_cam.fov_y = fov;
                    let inv_view_proj = pano_cam.view_proj_matrix(aspect).inverse();
                    let pano_pc = PanoramaPushConstants {
                        inv_view_proj: inv_view_proj.to_cols_array(),
                    };

                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pano_pipe.raw());
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pano_pipe.layout(),
                        0,
                        &[pano_set],
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        pano_pipe.layout(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&pano_pc),
                    );
                    device.cmd_draw(cmd, 3, 1, 0, 0);
                }
            } else if let (Some(sky_pipe), Some(sky_set)) =
                (&self.sky_pipeline, self.sky_descriptor_set)
            {
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
                    wind_time: self.start_time.elapsed().as_secs_f32(),
                    cloud_coverage: if self.game_settings.video.volumetric_clouds {
                        1.0
                    } else {
                        0.0
                    },
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

            if self.current_screen == AppScreen::InGame {
                // Late Pass 5: Translucent (Water)
                if let Some(trans_pipeline) = &self.translucent_pipeline {
                    let trans_pc = TranslucentMdiPushConstants {
                        view_proj,
                        draw_info_buffer_address: mdi.translucent_draw_late.device_address(),
                        frame_tick: self.frame_tick,
                        water_base_layer: self.anim_textures.water_still.base_layer,
                        camera_pos: [
                            self.camera.position.x,
                            self.camera.position.y,
                            self.camera.position.z,
                            sim_dist_meters,
                        ],
                        water_frame_count: self.anim_textures.water_still.frame_count,
                        water_flow_base_layer: self.anim_textures.water_flow.base_layer,
                        water_flow_frame_count: self.anim_textures.water_flow.frame_count,
                        lava_base_layer: self.anim_textures.lava_still.base_layer,
                        lava_frame_count: self.anim_textures.lava_still.frame_count,
                        fire_base_layer: self.anim_textures.fire.base_layer,
                        fire_frame_count: self.anim_textures.fire.frame_count,
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

                // Late Pass 5b: Translucent T2 (Sloped fluids & waterlogging)
                if let Some(t2_trans_pipeline) = &self.t2_translucent_pipeline {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t2_trans_pipeline.raw(),
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        t2_trans_pipeline.layout(),
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    let max_dist_sq = (self.view_distance as f32 * 32.0).powi(2);
                    for mesh in self.chunk_meshes.values() {
                        if let Some(layer) = &mesh.t2_translucent
                            && layer.quad_count > 0
                        {
                            let center = (mesh.min_aabb + mesh.max_aabb) * 0.5;
                            if center.distance_squared(self.camera.position) > max_dist_sq {
                                continue;
                            }
                            let pc = T2PushConstants {
                                view_proj,
                                quad_buffer_address: layer.buffer.device_address(),
                                chunk_x: mesh.pos[0],
                                chunk_y: mesh.pos[1],
                                camera_pos: [
                                    self.camera.position.x,
                                    self.camera.position.y,
                                    self.camera.position.z,
                                    sim_dist_meters,
                                ],
                                chunk_z: mesh.pos[2],
                                frame_tick_flags: (self.frame_tick & 0x7FFF_FFFF) | 0x8000_0000,
                                water_base_layer: self.anim_textures.water_still.base_layer,
                                water_frame_count: self.anim_textures.water_still.frame_count,
                                lava_base_layer: self.anim_textures.lava_still.base_layer,
                                lava_frame_count: self.anim_textures.lava_still.frame_count,
                                fire_base_layer: self.anim_textures.fire.base_layer,
                                fire_frame_count: self.anim_textures.fire.frame_count,
                            };
                            device.cmd_push_constants(
                                cmd,
                                t2_trans_pipeline.layout(),
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&pc),
                            );
                            device.cmd_draw(cmd, layer.quad_count * 6, 1, 0, 0);
                        }
                    }
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        entity_pipe.raw(),
                    );
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
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        weather_pipe.raw(),
                    );
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

                // Pass 17a: Visual Billboard Particles (Block debris, dust, torch smoke/flame, sparks, hearts)
                if self.active_particles_count > 0
                    && let (Some(particle_pipe), Some(particle_set), Some(particle_buf)) = (
                        &self.particle_pipeline,
                        self.particle_descriptor_set,
                        &self.particle_buffer,
                    )
                {
                    let particle_pc = ParticlePushConstants {
                        view_proj,
                        camera_right: self.camera.right().to_array(),
                        pad0: 0.0,
                        camera_up: self.camera.up().to_array(),
                        pad1: 0.0,
                        particle_buffer_address: particle_buf.device_address(),
                        pad2: [0, 0],
                    };
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        particle_pipe.raw(),
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        particle_pipe.layout(),
                        0,
                        &[particle_set],
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        particle_pipe.layout(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&particle_pc),
                    );
                    device.cmd_draw(cmd, 6, self.active_particles_count, 0, 0);
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

                    let (chunk_pos, local_idx) = telos_voxel::coords::split_block_pos(hit.pos);
                    let boxes = if let Some(snap) = self.chunks.get(&chunk_pos) {
                        let state = snap.blocks().get(local_idx);
                        match self.block_registry.shape(state) {
                            BlockShape::Boxes(b) if !b.is_empty() => b.clone(),
                            _ => vec![telos_voxel::shape::SubBox::FULL_CUBE],
                        }
                    } else {
                        vec![telos_voxel::shape::SubBox::FULL_CUBE]
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
            }

            device.cmd_end_rendering(cmd);

            // ==========================================
            // PHASE 2.5: POST-PROCESSING (SSAO, FOG, TONEMAP)
            // ==========================================
            if use_postprocess {
                let pp = self.postprocess.as_ref().unwrap();

                let proj_mat = self.camera.projection_matrix(aspect);
                let inv_proj_mat = proj_mat.inverse();
                let screen_size = [
                    swapchain_extent.width as f32,
                    swapchain_extent.height as f32,
                ];

                let ssao_pc = SsaoPushConstants {
                    inv_proj: inv_proj_mat.to_cols_array(),
                    proj: proj_mat.to_cols_array(),
                    screen_size,
                    radius: 0.75,
                    bias: 0.025,
                };

                let sun_angle = (self.client_time_of_day / 24000.0) * std::f32::consts::TAU;
                let sun_dir = [sun_angle.sin(), sun_angle.cos(), 0.2];

                let mut flags = 0u32;
                if self.current_screen == AppScreen::InGame {
                    if self.game_settings.video.ssao {
                        flags |= 1;
                    }
                    if self.game_settings.video.volumetric_fog {
                        flags |= 2;
                    }
                }
                if self.game_settings.video.tonemapping {
                    flags |= 4;
                }
                flags |= 8; // Vignette

                let composite_pc = PostCompositePushConstants {
                    inv_view_proj,
                    cam_pos: [
                        self.camera.position.x,
                        self.camera.position.y,
                        self.camera.position.z,
                    ],
                    time_of_day: self.client_time_of_day,
                    sun_dir,
                    view_distance: self.game_settings.video.view_distance as f32,
                    fog_density: 1.0,
                    flags,
                    screen_size,
                };

                pp.record_postprocess(
                    device,
                    cmd,
                    swapchain_image,
                    image_view,
                    depth_buffer.raw(),
                    &ssao_pc,
                    &composite_pc,
                );
            }

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
                let (cpos, _) = telos_voxel::coords::split_block_pos(BlockPos::new(
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

                let climate = telos_worldgen::ClimatePoint::sample(
                    self.world_seed,
                    self.camera.position.x,
                    self.camera.position.z,
                );
                let is_dry = climate.humidity < -0.35 && climate.temperature > 0.5;
                let local_precip =
                    if self.weather_rain_level <= 0.01 && self.weather_thunder_level <= 0.01 {
                        "None"
                    } else {
                        match telos_sim::weather::precipitation_at(
                            climate.temperature,
                            self.camera.position.y,
                            is_dry,
                        ) {
                            telos_sim::weather::PrecipitationKind::None => "None (Dry)",
                            telos_sim::weather::PrecipitationKind::Rain => "Rain",
                            telos_sim::weather::PrecipitationKind::Snow => "Snow",
                        }
                    };
                self.hud_state.local_precipitation = local_precip.to_string();
                #[allow(clippy::cast_possible_truncation)]
                {
                    self.hud_state.audio_channels = self.audio.active_players_count() as u32;
                }

                let mut ui_quads = Vec::with_capacity(512);
                let width_gui = swapchain_extent.width as f32 / gui_scale as f32;
                let height_gui = swapchain_extent.height as f32 / gui_scale as f32;

                match &mut self.current_screen {
                    AppScreen::MainMenu => {
                        self.main_menu
                            .update_layout_i18n(width_gui, height_gui, &self.catalog);
                        self.main_menu.render(
                            font,
                            width_gui,
                            height_gui,
                            gui_scale,
                            u64::from(self.frame_counter),
                            &mut ui_quads,
                        );
                    }
                    AppScreen::WorldSelect => {
                        self.world_select
                            .update_layout_i18n(width_gui, height_gui, &self.catalog);
                        self.world_select.render(
                            font,
                            width_gui,
                            height_gui,
                            gui_scale,
                            &mut ui_quads,
                        );
                    }
                    AppScreen::WorldCreate => {
                        self.world_create
                            .update_layout_i18n(width_gui, height_gui, &self.catalog);
                        self.world_create.render(
                            font,
                            width_gui,
                            height_gui,
                            gui_scale,
                            u64::from(self.frame_counter),
                            &mut ui_quads,
                        );
                    }
                    AppScreen::Settings { .. } => {
                        self.settings_screen.update_layout_i18n(
                            width_gui,
                            height_gui,
                            &self.catalog,
                        );
                        self.settings_screen.render(
                            font,
                            width_gui,
                            height_gui,
                            gui_scale,
                            &mut ui_quads,
                        );
                    }
                    AppScreen::Multiplayer => {
                        self.server_list.update_layout(width_gui, height_gui);
                        telos_ui::render_server_list(
                            &self.server_list,
                            font,
                            swapchain_extent.width,
                            swapchain_extent.height,
                            gui_scale,
                            &mut ui_quads,
                        );
                    }
                    AppScreen::InGame => {
                        if self.is_paused {
                            self.pause_menu.update_layout_i18n(
                                width_gui,
                                height_gui,
                                &self.catalog,
                            );
                            self.pause_menu.render(
                                font,
                                width_gui,
                                height_gui,
                                gui_scale,
                                &mut ui_quads,
                            );
                        } else {
                            render_hud_styled(
                                &self.hud_state,
                                font,
                                &self.ui_layers,
                                swapchain_extent.width,
                                swapchain_extent.height,
                                gui_scale,
                                &self.gui_style.hud,
                                &mut ui_quads,
                            );

                            if let Some(ref cont) = self.container_state {
                                let ui_carried = UiSlotItem::new_enchanted(
                                    self.inventory_sim.carried.item,
                                    self.inventory_sim.carried.count,
                                    self.inventory_sim.carried.enchantments.0,
                                );
                                let item_lookup = |id: u32| {
                                    if let Some(def) = self.registries.item_registry().get_by_id(id)
                                    {
                                        def.name.as_str()
                                    } else if let Some(name) = self.custom_item_names.get(&id) {
                                        name.as_str()
                                    } else {
                                        telos_sim::item_name(id)
                                    }
                                };
                                if cont.container_kind == 3 {
                                    let mut ui_slots = [UiSlotItem::EMPTY; DUAL_ANVIL_SLOT_COUNT];
                                    ui_slots[..3].copy_from_slice(&cont.slots[..3]);
                                    for (i, slot) in
                                        self.inventory_sim.slots[9..36].iter().enumerate()
                                    {
                                        ui_slots[3 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    for (i, slot) in
                                        self.inventory_sim.slots[0..9].iter().enumerate()
                                    {
                                        ui_slots[30 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    render_anvil_container_styled(
                                        &ui_slots,
                                        ui_carried,
                                        self.container_hovered_slot,
                                        cont.properties[0].max(0) as u32,
                                        self.hud_state.xp_level,
                                        swapchain_extent.width,
                                        swapchain_extent.height,
                                        gui_scale,
                                        font,
                                        &self.ui_layers,
                                        &cont.title,
                                        item_lookup,
                                        self.mouse_cursor_pos,
                                        &self.gui_style.anvil,
                                        &mut ui_quads,
                                    );
                                } else if cont.container_kind == 2 {
                                    let mut ui_slots =
                                        [UiSlotItem::EMPTY; DUAL_CRAFTING_TABLE_SLOT_COUNT];
                                    ui_slots[..10].copy_from_slice(&cont.slots[..10]);
                                    for (i, slot) in
                                        self.inventory_sim.slots[9..36].iter().enumerate()
                                    {
                                        ui_slots[10 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    for (i, slot) in
                                        self.inventory_sim.slots[0..9].iter().enumerate()
                                    {
                                        ui_slots[37 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    render_crafting_table_container_styled(
                                        &ui_slots,
                                        ui_carried,
                                        self.container_hovered_slot,
                                        swapchain_extent.width,
                                        swapchain_extent.height,
                                        gui_scale,
                                        font,
                                        &self.ui_layers,
                                        &cont.title,
                                        item_lookup,
                                        self.mouse_cursor_pos,
                                        &self.gui_style.crafting_table,
                                        &mut ui_quads,
                                    );
                                } else if cont.container_kind == 1 {
                                    let mut ui_slots = [UiSlotItem::EMPTY; DUAL_FURNACE_SLOT_COUNT];
                                    ui_slots[..3].copy_from_slice(&cont.slots[..3]);
                                    for (i, slot) in
                                        self.inventory_sim.slots[9..36].iter().enumerate()
                                    {
                                        ui_slots[3 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    for (i, slot) in
                                        self.inventory_sim.slots[0..9].iter().enumerate()
                                    {
                                        ui_slots[30 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    render_furnace_container_styled(
                                        &ui_slots,
                                        ui_carried,
                                        self.container_hovered_slot,
                                        cont.properties[0],
                                        cont.properties[1],
                                        cont.properties[2],
                                        cont.properties[3],
                                        swapchain_extent.width,
                                        swapchain_extent.height,
                                        gui_scale,
                                        font,
                                        &self.ui_layers,
                                        &cont.title,
                                        item_lookup,
                                        self.mouse_cursor_pos,
                                        &self.gui_style.furnace,
                                        &mut ui_quads,
                                    );
                                } else {
                                    let mut ui_slots =
                                        [UiSlotItem::EMPTY; DUAL_CONTAINER_SLOT_COUNT];
                                    ui_slots[..27].copy_from_slice(&cont.slots);
                                    for (i, slot) in
                                        self.inventory_sim.slots[9..36].iter().enumerate()
                                    {
                                        ui_slots[27 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    for (i, slot) in
                                        self.inventory_sim.slots[0..9].iter().enumerate()
                                    {
                                        ui_slots[54 + i] = UiSlotItem::new_enchanted(
                                            slot.item,
                                            slot.count,
                                            slot.enchantments.0,
                                        );
                                    }
                                    render_chest_container_styled(
                                        &ui_slots,
                                        ui_carried,
                                        self.container_hovered_slot,
                                        swapchain_extent.width,
                                        swapchain_extent.height,
                                        gui_scale,
                                        font,
                                        &self.ui_layers,
                                        &cont.title,
                                        item_lookup,
                                        self.mouse_cursor_pos,
                                        &self.gui_style.chest,
                                        &mut ui_quads,
                                    );
                                }
                            } else if self.inventory_open {
                                let mut ui_slots =
                                    [UiSlotItem::EMPTY; telos_ui::INVENTORY_SLOT_COUNT];
                                for (i, slot) in self
                                    .inventory_sim
                                    .slots
                                    .iter()
                                    .enumerate()
                                    .take(telos_ui::INVENTORY_SLOT_COUNT)
                                {
                                    ui_slots[i] = UiSlotItem::new_enchanted(
                                        slot.item,
                                        slot.count,
                                        slot.enchantments.0,
                                    );
                                }
                                let ui_carried = UiSlotItem::new_enchanted(
                                    self.inventory_sim.carried.item,
                                    self.inventory_sim.carried.count,
                                    self.inventory_sim.carried.enchantments.0,
                                );
                                let item_lookup = |id: u32| {
                                    if let Some(def) = self.registries.item_registry().get_by_id(id)
                                    {
                                        def.name.as_str()
                                    } else if let Some(name) = self.custom_item_names.get(&id) {
                                        name.as_str()
                                    } else {
                                        telos_sim::item_name(id)
                                    }
                                };
                                render_inventory_screen_styled(
                                    &ui_slots,
                                    ui_carried,
                                    self.inventory_hovered_slot,
                                    swapchain_extent.width,
                                    swapchain_extent.height,
                                    gui_scale,
                                    font,
                                    &self.ui_layers,
                                    &self.gui_style.inventory,
                                    item_lookup,
                                    self.mouse_cursor_pos,
                                    &mut ui_quads,
                                );
                            }

                            if self.advancements_open {
                                self.advancements_screen.render(
                                    &self.ui_layers,
                                    font,
                                    width_gui,
                                    height_gui,
                                    gui_scale,
                                    &mut ui_quads,
                                );
                            }

                            let now_secs = self.start_time.elapsed().as_secs_f64();
                            render_chat_hud(
                                &self.chat_state,
                                font,
                                swapchain_extent.width,
                                swapchain_extent.height,
                                gui_scale,
                                now_secs,
                                &mut ui_quads,
                            );
                        }
                    }
                }

                if let Some(form) = &mut self.active_modal_form {
                    form.update_layout(width_gui, height_gui);
                    let full_w = snap_to_physical(width_gui, gui_scale) as u16;
                    let full_h = snap_to_physical(height_gui, gui_scale) as u16;
                    ui_quads.push(UiQuad::solid(
                        [0, 0],
                        [full_w, full_h],
                        UiQuad::rgba(0, 0, 0, 160),
                    ));
                    form.render(
                        font,
                        gui_scale,
                        u64::from(self.frame_counter),
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

        let mut should_exit = false;

        let take_cli_screenshot =
            self.args.screenshot.is_some() && self.total_frames >= self.args.frames;
        let take_manual_screenshot = self.manual_screenshot_requested;

        if take_cli_screenshot || take_manual_screenshot {
            self.manual_screenshot_requested = false;
            let target_path = if let Some(ref p) = self.args.screenshot {
                p.clone()
            } else {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                PathBuf::from(format!("dev-assets/screenshots/screenshot_{timestamp}.png"))
            };

            if let Some(parent) = target_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            match gpu_context.capture_screenshot(image_index) {
                Ok((width, height, pixels)) => {
                    if let Some(img) = image::RgbaImage::from_raw(width, height, pixels) {
                        match img.save(&target_path) {
                            Ok(()) => {
                                info!(
                                    path = %target_path.display(),
                                    width,
                                    height,
                                    frame = self.total_frames,
                                    "Saved screenshot successfully"
                                );
                            }
                            Err(e) => {
                                tracing::error!("Failed to save screenshot image: {e}");
                            }
                        }
                    } else {
                        tracing::error!("Failed to create RgbaImage from raw screenshot buffer");
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to capture GPU screenshot: {e}");
                }
            }

            if take_cli_screenshot {
                should_exit = true;
            }
        }

        should_exit
    }

    fn update_hud_state(&mut self, dt: f32, day_number: u64, moon_phase: u32) {
        self.hud_state.selected_slot = self.selected_hotbar_slot;
        self.hud_state.player_pos = [
            f64::from(self.camera.position.x),
            f64::from(self.camera.position.y),
            f64::from(self.camera.position.z),
        ];
        let (cpos, _) = telos_voxel::coords::split_block_pos(BlockPos::new(
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

        let climate = telos_worldgen::ClimatePoint::sample(
            self.world_seed,
            self.camera.position.x,
            self.camera.position.z,
        );
        let is_dry = climate.humidity < -0.35 && climate.temperature > 0.5;
        let local_precip = if self.weather_rain_level <= 0.01 && self.weather_thunder_level <= 0.01
        {
            "None"
        } else {
            match telos_sim::weather::precipitation_at(
                climate.temperature,
                self.camera.position.y,
                is_dry,
            ) {
                telos_sim::weather::PrecipitationKind::None => "None (Dry)",
                telos_sim::weather::PrecipitationKind::Rain => "Rain",
                telos_sim::weather::PrecipitationKind::Snow => "Snow",
            }
        };
        self.hud_state.local_precipitation = local_precip.to_string();
        #[allow(clippy::cast_possible_truncation)]
        {
            self.hud_state.audio_channels = self.audio.active_players_count() as u32;
        }
    }

    #[allow(clippy::too_many_lines)]
    fn build_ui_quads(&mut self, width: u32, height: u32) -> Vec<UiQuad> {
        let Some(font) = &self.ui_font else {
            return Vec::new();
        };

        let gui_scale = compute_gui_scale(width, height);
        let mut ui_quads = Vec::with_capacity(512);
        let width_gui = width as f32 / gui_scale as f32;
        let height_gui = height as f32 / gui_scale as f32;

        match &mut self.current_screen {
            AppScreen::MainMenu => {
                self.main_menu
                    .update_layout_i18n(width_gui, height_gui, &self.catalog);
                self.main_menu.render(
                    font,
                    width_gui,
                    height_gui,
                    gui_scale,
                    u64::from(self.frame_counter),
                    &mut ui_quads,
                );
            }
            AppScreen::WorldSelect => {
                self.world_select
                    .update_layout_i18n(width_gui, height_gui, &self.catalog);
                self.world_select
                    .render(font, width_gui, height_gui, gui_scale, &mut ui_quads);
            }
            AppScreen::WorldCreate => {
                self.world_create
                    .update_layout_i18n(width_gui, height_gui, &self.catalog);
                self.world_create.render(
                    font,
                    width_gui,
                    height_gui,
                    gui_scale,
                    u64::from(self.frame_counter),
                    &mut ui_quads,
                );
            }
            AppScreen::Settings { .. } => {
                self.settings_screen
                    .update_layout_i18n(width_gui, height_gui, &self.catalog);
                self.settings_screen
                    .render(font, width_gui, height_gui, gui_scale, &mut ui_quads);
            }
            AppScreen::Multiplayer => {
                self.server_list.update_layout(width_gui, height_gui);
                telos_ui::render_server_list(
                    &self.server_list,
                    font,
                    width,
                    height,
                    gui_scale,
                    &mut ui_quads,
                );
            }
            AppScreen::InGame => {
                if self.is_paused {
                    self.pause_menu
                        .update_layout_i18n(width_gui, height_gui, &self.catalog);
                    self.pause_menu
                        .render(font, width_gui, height_gui, gui_scale, &mut ui_quads);
                } else {
                    render_hud_styled(
                        &self.hud_state,
                        font,
                        &self.ui_layers,
                        width,
                        height,
                        gui_scale,
                        &self.gui_style.hud,
                        &mut ui_quads,
                    );

                    if let Some(ref cont) = self.container_state {
                        let ui_carried = UiSlotItem::new_enchanted(
                            self.inventory_sim.carried.item,
                            self.inventory_sim.carried.count,
                            self.inventory_sim.carried.enchantments.0,
                        );
                        let item_lookup = |id: u32| {
                            if let Some(def) = self.registries.item_registry().get_by_id(id) {
                                def.name.as_str()
                            } else if let Some(name) = self.custom_item_names.get(&id) {
                                name.as_str()
                            } else {
                                telos_sim::item_name(id)
                            }
                        };
                        if cont.container_kind == 3 {
                            let mut ui_slots = [UiSlotItem::EMPTY; DUAL_ANVIL_SLOT_COUNT];
                            ui_slots[..3].copy_from_slice(&cont.slots[..3]);
                            for (i, slot) in self.inventory_sim.slots[9..36].iter().enumerate() {
                                ui_slots[3 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            for (i, slot) in self.inventory_sim.slots[0..9].iter().enumerate() {
                                ui_slots[30 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            render_anvil_container_styled(
                                &ui_slots,
                                ui_carried,
                                self.container_hovered_slot,
                                cont.properties[0].max(0) as u32,
                                self.hud_state.xp_level,
                                width,
                                height,
                                gui_scale,
                                font,
                                &self.ui_layers,
                                &cont.title,
                                item_lookup,
                                self.mouse_cursor_pos,
                                &self.gui_style.anvil,
                                &mut ui_quads,
                            );
                        } else if cont.container_kind == 2 {
                            let mut ui_slots = [UiSlotItem::EMPTY; DUAL_CRAFTING_TABLE_SLOT_COUNT];
                            ui_slots[..10].copy_from_slice(&cont.slots[..10]);
                            for (i, slot) in self.inventory_sim.slots[9..36].iter().enumerate() {
                                ui_slots[10 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            for (i, slot) in self.inventory_sim.slots[0..9].iter().enumerate() {
                                ui_slots[37 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            render_crafting_table_container_styled(
                                &ui_slots,
                                ui_carried,
                                self.container_hovered_slot,
                                width,
                                height,
                                gui_scale,
                                font,
                                &self.ui_layers,
                                &cont.title,
                                item_lookup,
                                self.mouse_cursor_pos,
                                &self.gui_style.crafting_table,
                                &mut ui_quads,
                            );
                        } else if cont.container_kind == 1 {
                            let mut ui_slots = [UiSlotItem::EMPTY; DUAL_FURNACE_SLOT_COUNT];
                            ui_slots[..3].copy_from_slice(&cont.slots[..3]);
                            for (i, slot) in self.inventory_sim.slots[9..36].iter().enumerate() {
                                ui_slots[3 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            for (i, slot) in self.inventory_sim.slots[0..9].iter().enumerate() {
                                ui_slots[30 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            render_furnace_container_styled(
                                &ui_slots,
                                ui_carried,
                                self.container_hovered_slot,
                                cont.properties[0],
                                cont.properties[1],
                                cont.properties[2],
                                cont.properties[3],
                                width,
                                height,
                                gui_scale,
                                font,
                                &self.ui_layers,
                                &cont.title,
                                item_lookup,
                                self.mouse_cursor_pos,
                                &self.gui_style.furnace,
                                &mut ui_quads,
                            );
                        } else {
                            let mut ui_slots = [UiSlotItem::EMPTY; DUAL_CONTAINER_SLOT_COUNT];
                            ui_slots[..27].copy_from_slice(&cont.slots);
                            for (i, slot) in self.inventory_sim.slots[9..36].iter().enumerate() {
                                ui_slots[27 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            for (i, slot) in self.inventory_sim.slots[0..9].iter().enumerate() {
                                ui_slots[54 + i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                            render_chest_container_styled(
                                &ui_slots,
                                ui_carried,
                                self.container_hovered_slot,
                                width,
                                height,
                                gui_scale,
                                font,
                                &self.ui_layers,
                                &cont.title,
                                item_lookup,
                                self.mouse_cursor_pos,
                                &self.gui_style.chest,
                                &mut ui_quads,
                            );
                        }
                    } else if self.inventory_open {
                        let mut ui_slots = [UiSlotItem::EMPTY; telos_ui::INVENTORY_SLOT_COUNT];
                        for (i, slot) in self
                            .inventory_sim
                            .slots
                            .iter()
                            .enumerate()
                            .take(telos_ui::INVENTORY_SLOT_COUNT)
                        {
                            ui_slots[i] = UiSlotItem::new_enchanted(
                                slot.item,
                                slot.count,
                                slot.enchantments.0,
                            );
                        }
                        let ui_carried = UiSlotItem::new_enchanted(
                            self.inventory_sim.carried.item,
                            self.inventory_sim.carried.count,
                            self.inventory_sim.carried.enchantments.0,
                        );
                        let item_lookup = |id: u32| {
                            if let Some(def) = self.registries.item_registry().get_by_id(id) {
                                def.name.as_str()
                            } else if let Some(name) = self.custom_item_names.get(&id) {
                                name.as_str()
                            } else {
                                telos_sim::item_name(id)
                            }
                        };
                        render_inventory_screen_styled(
                            &ui_slots,
                            ui_carried,
                            self.inventory_hovered_slot,
                            width,
                            height,
                            gui_scale,
                            font,
                            &self.ui_layers,
                            &self.gui_style.inventory,
                            item_lookup,
                            self.mouse_cursor_pos,
                            &mut ui_quads,
                        );
                    }

                    if self.advancements_open {
                        self.advancements_screen.render(
                            &self.ui_layers,
                            font,
                            width_gui,
                            height_gui,
                            gui_scale,
                            &mut ui_quads,
                        );
                    }

                    let now_secs = self.start_time.elapsed().as_secs_f64();
                    render_chat_hud(
                        &self.chat_state,
                        font,
                        width,
                        height,
                        gui_scale,
                        now_secs,
                        &mut ui_quads,
                    );
                }
            }
        }

        if let Some(form) = &mut self.active_modal_form {
            form.update_layout(width_gui, height_gui);
            let full_w = snap_to_physical(width_gui, gui_scale) as u16;
            let full_h = snap_to_physical(height_gui, gui_scale) as u16;
            ui_quads.push(UiQuad::solid(
                [0, 0],
                [full_w, full_h],
                UiQuad::rgba(0, 0, 0, 160),
            ));
            form.render(
                font,
                gui_scale,
                u64::from(self.frame_counter),
                &mut ui_quads,
            );
        }

        ui_quads
    }

    #[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
    fn render_opengl(&mut self) -> bool {
        let fps_limit = self.game_settings.video.fps_limit;
        if fps_limit > 0 {
            let target_frame_dur = Duration::from_secs_f64(1.0 / f64::from(fps_limit));
            let elapsed = self.last_frame_start.elapsed();
            if elapsed < target_frame_dur {
                std::thread::sleep(target_frame_dur.checked_sub(elapsed).unwrap());
            }
        }
        self.last_frame_start = Instant::now();

        if self.current_screen == AppScreen::InGame {
            self.poll_network();
            self.rebuild_dirty_meshes_gl(16);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        self.update_player_hurt_tilt(dt);
        self.update_toast(dt);

        if self.current_screen == AppScreen::InGame && !self.is_paused {
            let input_state = PlayerInputState {
                forward: self.controller.forward && self.controller.mouse_captured,
                backward: self.controller.backward && self.controller.mouse_captured,
                left: self.controller.left && self.controller.mouse_captured,
                right: self.controller.right && self.controller.mouse_captured,
                jump: self.controller.up && self.controller.mouse_captured,
                sneak: self.controller.down && self.controller.mouse_captured,
                sprint: self.controller.sprint && self.controller.mouse_captured,
            };
            let reg = &self.block_registry;
            let chunks = &self.chunks;
            self.physics.update(
                dt,
                self.camera.yaw.to_degrees(),
                &input_state,
                |bx, by, bz, out| {
                    let (cpos, lidx) =
                        telos_voxel::coords::split_block_pos(BlockPos::new(bx, by, bz));
                    let block = if let Some(snap) = chunks.get(&cpos) {
                        snap.blocks().get(lidx)
                    } else {
                        BlockStateId::AIR
                    };
                    if block == BlockStateId::AIR {
                        return;
                    }
                    match reg.shape(block) {
                        BlockShape::Cube => {
                            out.push(Aabb::from_block(bx, by, bz));
                        }
                        BlockShape::Boxes(boxes) => {
                            for b in boxes {
                                out.push(Aabb::from_sub_box(bx, by, bz, b.min, b.max));
                            }
                        }
                        _ => {}
                    }
                },
            );

            self.camera.position = self.physics.eye_pos();

            let base_fov = self.game_settings.video.fov.to_radians();
            if self.bow_charging && self.bow_charge_ticks >= telos_sim::BOW_MIN_CHARGE_TICKS {
                #[allow(clippy::cast_precision_loss)]
                let charge_ratio = ((f32::from(self.bow_charge_ticks)
                    - f32::from(telos_sim::BOW_MIN_CHARGE_TICKS))
                    / (f32::from(telos_sim::BOW_FULL_CHARGE_TICKS)
                        - f32::from(telos_sim::BOW_MIN_CHARGE_TICKS)))
                .clamp(0.0, 1.0);
                let zoom_mult = 1.0 - 0.15 * charge_ratio;
                self.camera.fov_y = base_fov * zoom_mult;
            } else {
                self.camera.fov_y = base_fov;
            }

            // Raycast targeted block
            let ray_origin = self.camera.position;
            let ray_dir = self.camera.forward();
            let is_solid = |pos: BlockPos| -> bool {
                let (chunk_pos, local_idx) = telos_voxel::coords::split_block_pos(pos);
                if let Some(snap) = chunks.get(&chunk_pos) {
                    let state = snap.blocks().get(local_idx);
                    !reg.flags(state)
                        .contains(telos_voxel::state::StateFlags::AIR)
                } else {
                    false
                }
            };
            self.targeted_block = raycast_voxels(ray_origin, ray_dir, 5.0, is_solid);

            self.audio.set_listener(
                self.camera.position,
                self.camera.forward(),
                self.camera.up(),
            );
        }

        self.frame_counter += 1;
        self.total_frames += 1;
        if self.last_fps_time.elapsed() >= Duration::from_secs(1) {
            let fps = self.frame_counter;
            info!(
                fps,
                backend = "OpenGL 4.5",
                chunks = self.chunks.len(),
                pos = %format!("({:.1}, {:.1}, {:.1})", self.camera.position.x, self.camera.position.y, self.camera.position.z),
                "Client frame stats"
            );
            self.frame_counter = 0;
            self.last_fps_time = Instant::now();
        }

        let (width, height) = if let Some(gl_ctx) = &self.gl_context {
            (gl_ctx.width(), gl_ctx.height())
        } else {
            (1280, 720)
        };
        let aspect = width as f32 / height.max(1) as f32;
        let view_proj_mat = self.camera.view_proj_matrix(aspect);
        let inv_view_proj = view_proj_mat.inverse();

        let day_number = self.client_world_age / telos_core::time::DAY_TICKS;
        let moon_phase = telos_core::time::moon_phase(day_number);
        let sun_angle = telos_core::time::sun_angle(
            self.client_time_of_day as u64,
            self.client_time_of_day.fract(),
        );
        let sun_dir = telos_core::time::sun_direction(sun_angle);
        let sun_elevation = sun_dir.y;
        let daylight = telos_core::time::daylight_factor(sun_elevation);
        let sunset = telos_core::time::sunset_factor(sun_elevation);

        self.lightmap_lut.update(
            sun_elevation,
            self.client_time_of_day,
            self.weather_rain_level,
            self.weather_thunder_level,
            self.weather_lightning_flash,
        );
        if let Some(renderer) = &mut self.gl_renderer {
            renderer.update_lightmap(&self.lightmap_lut.data);
        }

        let highlight_box = self.targeted_block.as_ref().map(|hit| {
            (
                Vec3::new(hit.pos.x() as f32, hit.pos.y() as f32, hit.pos.z() as f32),
                Vec3::new(
                    hit.pos.x() as f32 + 1.0,
                    hit.pos.y() as f32 + 1.0,
                    hit.pos.z() as f32 + 1.0,
                ),
            )
        });

        self.update_hud_state(dt, day_number, moon_phase);
        let ui_quads = self.build_ui_quads(width, height);
        let mut gl_ui_verts = Vec::new();
        opengl_renderer::ui_quads_to_gl_vertices(&ui_quads, &mut gl_ui_verts);

        if let Some(renderer) = &mut self.gl_renderer {
            renderer.render_frame(
                &view_proj_mat,
                &inv_view_proj,
                self.camera.position,
                sun_dir,
                daylight,
                sunset,
                highlight_box,
                &gl_ui_verts,
                width,
                height,
            );
        }

        if let Some(gl_ctx) = &self.gl_context {
            let _ = gl_ctx.swap_buffers();
        }

        // Handle screenshot capture
        let take_cli_screenshot =
            self.args.screenshot.is_some() && self.total_frames >= self.args.frames;
        let take_manual = self.manual_screenshot_requested;
        if take_cli_screenshot || take_manual {
            self.manual_screenshot_requested = false;
            let target_path = if let Some(ref p) = self.args.screenshot {
                p.clone()
            } else {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                PathBuf::from(format!(
                    "dev-assets/screenshots/screenshot_gl_{timestamp}.png"
                ))
            };

            if let Some(parent) = target_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            if let Some(gl_ctx) = &self.gl_context {
                let w = gl_ctx.width();
                let h = gl_ctx.height();
                let mut pixels = vec![0u8; (w * h * 4) as usize];
                unsafe {
                    use glow::HasContext;
                    gl_ctx.gl().read_pixels(
                        0,
                        0,
                        w as i32,
                        h as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut pixels)),
                    );
                }
                let stride = (w * 4) as usize;
                let mut flipped = vec![0u8; pixels.len()];
                for y in 0..h as usize {
                    let src_y = h as usize - 1 - y;
                    flipped[y * stride..(y + 1) * stride]
                        .copy_from_slice(&pixels[src_y * stride..(src_y + 1) * stride]);
                }
                if let Some(img) = image::RgbaImage::from_raw(w, h, flipped) {
                    if let Err(e) = img.save(&target_path) {
                        tracing::error!("Failed to save OpenGL screenshot: {e}");
                    } else {
                        info!(
                            path = %target_path.display(),
                            width = w,
                            height = h,
                            frame = self.total_frames,
                            "Saved OpenGL screenshot successfully"
                        );
                    }
                }
            }

            if take_cli_screenshot {
                return true;
            }
        }

        false
    }

    fn init_opengl(
        &mut self,
        attributes: winit::window::WindowAttributes,
        event_loop: &ActiveEventLoop,
    ) {
        let (gl_context, window) =
            match telos_gpu::opengl::GlContext::create_desktop(attributes, event_loop) {
                Ok((ctx, win)) => (ctx, win),
                Err(err) => {
                    tracing::error!("Failed to create OpenGL desktop context: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let mut gl_renderer = match opengl_renderer::OpenGlRenderer::new(&gl_context) {
            Ok(r) => r,
            Err(err) => {
                tracing::error!("Failed to initialize OpenGlRenderer: {err}");
                event_loop.exit();
                return;
            }
        };

        // Lightmap texture
        match telos_gpu::opengl::GlTexture2d::new(gl_context.gl().clone(), 16, 16) {
            Ok(tex) => gl_renderer.set_lightmap(tex),
            Err(err) => tracing::error!("Failed to create GL lightmap texture: {err}"),
        }

        // Bake block textures and upload
        match load_and_upload_textures_gl(gl_context.gl()) {
            Ok((gl_array, anim)) => {
                gl_renderer.set_terrain_textures(gl_array);
                self.anim_textures = anim;
            }
            Err(err) => tracing::error!("Failed to load GL block textures: {err}"),
        }

        // Bake UI textures and upload
        match load_and_upload_ui_textures_gl(gl_context.gl()) {
            Ok((ui_array, font)) => {
                gl_renderer.set_ui_textures(ui_array);
                self.ui_font = Some(font);
            }
            Err(err) => tracing::error!("Failed to load GL UI textures: {err}"),
        }

        info!(
            backend = gl_context.caps().backend.name(),
            renderer = %gl_context.caps().device_name,
            version = %gl_context.caps().driver_info,
            "OpenGL fallback renderer initialized successfully"
        );

        self.window = Some(window);
        self.gl_context = Some(gl_context);
        self.gl_renderer = Some(gl_renderer);
        self.active_backend = telos_gpu::rhi::RenderBackendType::OpenGl;
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::too_many_lines)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let requested_backend = self.args.backend.trim().to_lowercase();
        let use_opengl = requested_backend == "opengl" || requested_backend == "gl";
        let auto = requested_backend == "auto";

        let attributes = Window::default_attributes()
            .with_title("Telos")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        if use_opengl {
            info!("Initializing OpenGL 4.5 fallback renderer (explicit CLI flag)");
            self.init_opengl(attributes, event_loop);
            return;
        }

        let window = match event_loop.create_window(attributes.clone()) {
            Ok(w) => w,
            Err(err) => {
                if auto {
                    tracing::warn!(
                        "Failed to create Vulkan window ({err}), falling back to OpenGL 4.5"
                    );
                    self.init_opengl(attributes, event_loop);
                    return;
                }
                tracing::error!("Failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        let gpu_context = match GpuContext::new(&window, size.width, size.height, self.validation) {
            Ok(ctx) => ctx,
            Err(err) => {
                if auto {
                    tracing::warn!(
                        "Failed to initialize Vulkan GpuContext ({err}), falling back to OpenGL 4.5"
                    );
                    drop(window);
                    self.init_opengl(attributes, event_loop);
                    return;
                }
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

        let (texture_array, anim_textures) = match load_and_upload_textures(&gpu_context) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!("Failed to load and upload texture array: {err}");
                event_loop.exit();
                return;
            }
        };
        self.anim_textures = anim_textures;

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

        // Load T2 SPIR-V bytecode
        let t2_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_t2.vert.spv"));
        let t2_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_t2.frag.spv"));

        let t2_vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), t2_vert_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create T2 vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let t2_frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), t2_frag_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create T2 fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let t2_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<T2PushConstants>() as u32);

        let t2_cutout_pipeline = match GraphicsPipeline::create_dynamic_cutout(
            gpu_context.device().raw(),
            t2_vert_module.raw(),
            t2_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[descriptor_set_layout],
            &[t2_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic T2 cutout graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        let t2_trans_pipeline = match GraphicsPipeline::create_dynamic_translucent(
            gpu_context.device().raw(),
            t2_vert_module.raw(),
            t2_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[descriptor_set_layout],
            &[t2_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic T2 translucent graphics pipeline: {err}");
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
        // Initialize 3D Rotating Panorama Skybox Resources (Phase 53)
        // ---------------------------------------------------------------------
        let panorama_texture = match load_and_upload_panorama_textures(&gpu_context) {
            Ok(tex) => Some(tex),
            Err(err) => {
                tracing::warn!("Failed to load 3D panorama textures: {err}");
                None
            }
        };

        let pano_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let pano_bindings = [pano_binding];
        let pano_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&pano_bindings);
        let panorama_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&pano_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create panorama descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let pano_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)];
        let pano_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pano_pool_sizes)
            .max_sets(1);
        let panorama_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&pano_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create panorama descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let pano_alloc_layouts = [panorama_descriptor_set_layout];
        let pano_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(panorama_descriptor_pool)
            .set_layouts(&pano_alloc_layouts);
        let panorama_descriptor_set = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&pano_alloc_info)
            {
                Ok(mut sets) => sets.pop().unwrap(),
                Err(err) => {
                    tracing::error!("Failed to allocate panorama descriptor set: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        if let Some(pano_tex) = &panorama_texture {
            let pano_image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(pano_tex.view())
                .sampler(pano_tex.sampler())];
            let pano_descriptor_write = [vk::WriteDescriptorSet::default()
                .dst_set(panorama_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&pano_image_info)];
            unsafe {
                gpu_context
                    .device()
                    .raw()
                    .update_descriptor_sets(&pano_descriptor_write, &[]);
            }
        }

        let pano_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/panorama.frag.spv"));
        let panorama_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), pano_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create panorama fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        #[allow(clippy::cast_possible_truncation)]
        let pano_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<PanoramaPushConstants>() as u32);

        let panorama_pipeline = match GraphicsPipeline::create_dynamic_sky(
            gpu_context.device().raw(),
            sky_vert_module.raw(),
            panorama_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            &[panorama_descriptor_set_layout],
            &[pano_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create panorama graphics pipeline: {err}");
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
        // Initialize Visual Particle Billboard Pipeline (Phase 35)
        // ---------------------------------------------------------------------
        let particle_texture = match load_and_upload_particle_textures(&gpu_context) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!("Failed to load particle textures: {err}");
                event_loop.exit();
                return;
            }
        };

        let particle_binding_part = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let particle_binding_terr = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let particle_bindings = [particle_binding_part, particle_binding_terr];
        let particle_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&particle_bindings);
        let particle_descriptor_set_layout = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_set_layout(&particle_layout_info, None)
            {
                Ok(l) => l,
                Err(err) => {
                    tracing::error!("Failed to create particle descriptor set layout: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let particle_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2);
        let particle_pool_sizes = [particle_pool_size];
        let particle_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&particle_pool_sizes);
        let particle_descriptor_pool = unsafe {
            match gpu_context
                .device()
                .raw()
                .create_descriptor_pool(&particle_pool_info, None)
            {
                Ok(p) => p,
                Err(err) => {
                    tracing::error!("Failed to create particle descriptor pool: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let particle_set_layouts = [particle_descriptor_set_layout];
        let particle_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(particle_descriptor_pool)
            .set_layouts(&particle_set_layouts);
        let particle_descriptor_set = unsafe {
            match gpu_context
                .device()
                .raw()
                .allocate_descriptor_sets(&particle_alloc_info)
            {
                Ok(sets) => sets[0],
                Err(err) => {
                    tracing::error!("Failed to allocate particle descriptor set: {err}");
                    event_loop.exit();
                    return;
                }
            }
        };

        let particle_image_info = [vk::DescriptorImageInfo::default()
            .sampler(particle_texture.sampler())
            .image_view(particle_texture.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let terrain_image_info = [vk::DescriptorImageInfo::default()
            .sampler(texture_array.sampler())
            .image_view(texture_array.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let particle_descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(particle_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&particle_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(particle_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&terrain_image_info),
        ];
        unsafe {
            gpu_context
                .device()
                .raw()
                .update_descriptor_sets(&particle_descriptor_writes, &[]);
        }

        let particle_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/particle.vert.spv"));
        let particle_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/particle.frag.spv"));
        let particle_vert_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), particle_vert_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create particle vertex shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };
        let particle_frag_module =
            match ShaderModule::from_spv(gpu_context.device().raw(), particle_frag_spv) {
                Ok(m) => m,
                Err(err) => {
                    tracing::error!("Failed to create particle fragment shader module: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let particle_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<ParticlePushConstants>() as u32);
        let particle_pipeline = match GraphicsPipeline::create_dynamic_translucent(
            gpu_context.device().raw(),
            particle_vert_module.raw(),
            particle_frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::NONE,
            &[particle_descriptor_set_layout],
            &[particle_push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create particle graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        let particle_buffer = match GpuBuffer::new(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            "visual_particle_buffer",
            (4096 * size_of::<ParticleGpu>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        ) {
            Ok(buf) => buf,
            Err(err) => {
                tracing::error!("Failed to allocate visual particle buffer: {err}");
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

        self.panorama_texture = panorama_texture;
        self.panorama_descriptor_set_layout = Some(panorama_descriptor_set_layout);
        self.panorama_descriptor_pool = Some(panorama_descriptor_pool);
        self.panorama_descriptor_set = Some(panorama_descriptor_set);
        self.panorama_frag_shader = Some(panorama_frag_module);
        self.panorama_pipeline = Some(panorama_pipeline);

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

        self.particle_texture = Some(particle_texture);
        self.particle_descriptor_set_layout = Some(particle_descriptor_set_layout);
        self.particle_descriptor_pool = Some(particle_descriptor_pool);
        self.particle_descriptor_set = Some(particle_descriptor_set);
        self.particle_vert_shader = Some(particle_vert_module);
        self.particle_frag_shader = Some(particle_frag_module);
        self.particle_pipeline = Some(particle_pipeline);
        self.particle_buffer = Some(particle_buffer);

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

        self.t2_cutout_pipeline = Some(t2_cutout_pipeline);
        self.t2_translucent_pipeline = Some(t2_trans_pipeline);
        self.t2_vert_shader = Some(t2_vert_module);
        self.t2_frag_shader = Some(t2_frag_module);
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

        let post_fullscreen_spv =
            include_bytes!(concat!(env!("OUT_DIR"), "/post_fullscreen.vert.spv"));
        let post_ssao_spv = include_bytes!(concat!(env!("OUT_DIR"), "/post_ssao.frag.spv"));
        let post_composite_spv =
            include_bytes!(concat!(env!("OUT_DIR"), "/post_composite.frag.spv"));

        let postprocess = match PostProcessFrameGraph::new(
            gpu_context.device().raw(),
            gpu_context.allocator(),
            gpu_context.extent(),
            gpu_context.swapchain().format(),
            self.depth_buffer.as_ref().unwrap().view(),
            post_fullscreen_spv,
            post_ssao_spv,
            post_composite_spv,
        ) {
            Ok(pp) => {
                info!("PostProcessFrameGraph successfully initialized");
                Some(pp)
            }
            Err(err) => {
                tracing::warn!(
                    "Failed to initialize PostProcessFrameGraph: {err}; continuing without post-processing"
                );
                None
            }
        };
        self.postprocess = postprocess;

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
        if self.current_screen != AppScreen::InGame
            || self.is_paused
            || self.advancements_open
            || self.inventory_open
            || self.container_state.is_some()
            || self.chat_state.is_open
            || !self.controller.mouse_captured
        {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            let sens = f64::from(self.game_settings.controls.mouse_sensitivity);
            let invert_y = if self.game_settings.controls.invert_mouse_y {
                -1.0
            } else {
                1.0
            };
            self.controller
                .on_mouse_move(&mut self.camera, dx * sens, dy * sens * invert_y);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window.as_ref().is_none_or(|w| w.id() != window_id) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested, exiting application");
                event_loop.exit();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_cursor_pos = [position.x as f32, position.y as f32];
                let win_size = self.window_size();
                let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                let (mx, my) = (
                    position.x as f32 / gui_scale as f32,
                    position.y as f32 / gui_scale as f32,
                );

                if let Some(form) = &mut self.active_modal_form {
                    form.handle_mouse_move(mx, my);
                    if let Some(window) = &self.window {
                        let icon = if form.is_hovered() {
                            winit::window::CursorIcon::Pointer
                        } else {
                            winit::window::CursorIcon::Default
                        };
                        window.set_cursor(icon);
                    }
                    return;
                }

                match &mut self.current_screen {
                    AppScreen::MainMenu => {
                        self.main_menu.handle_mouse_move(mx, my);
                    }
                    AppScreen::WorldSelect => {
                        self.world_select.handle_mouse_move(mx, my);
                    }
                    AppScreen::WorldCreate => {
                        self.world_create.handle_mouse_move(mx, my);
                    }
                    AppScreen::Settings { .. } => {
                        self.settings_screen.handle_mouse_move(mx, my);
                    }
                    AppScreen::Multiplayer => {
                        self.server_list.handle_mouse_move(mx, my);
                    }
                    AppScreen::InGame => {
                        if self.is_paused {
                            self.pause_menu.handle_mouse_move(mx, my);
                        } else if self.advancements_open {
                            let width_gui = win_size.width as f32 / gui_scale as f32;
                            let height_gui = win_size.height as f32 / gui_scale as f32;
                            self.advancements_screen
                                .handle_mouse_move(mx, my, width_gui, height_gui);
                        } else if let Some(ref cont) = self.container_state {
                            self.container_hovered_slot = if cont.container_kind == 3 {
                                anvil_slot_at_pos_styled(
                                    self.mouse_cursor_pos,
                                    win_size.width,
                                    win_size.height,
                                    gui_scale,
                                    &self.gui_style.anvil,
                                )
                            } else if cont.container_kind == 2 {
                                crafting_table_slot_at_pos_styled(
                                    self.mouse_cursor_pos,
                                    win_size.width,
                                    win_size.height,
                                    gui_scale,
                                    &self.gui_style.crafting_table,
                                )
                            } else if cont.container_kind == 1 {
                                furnace_slot_at_pos_styled(
                                    self.mouse_cursor_pos,
                                    win_size.width,
                                    win_size.height,
                                    gui_scale,
                                    &self.gui_style.furnace,
                                )
                            } else {
                                chest_slot_at_pos_styled(
                                    self.mouse_cursor_pos,
                                    win_size.width,
                                    win_size.height,
                                    gui_scale,
                                    &self.gui_style.chest,
                                )
                            };
                        } else if self.inventory_open {
                            self.inventory_hovered_slot = inventory_slot_at_pos_styled(
                                self.mouse_cursor_pos,
                                win_size.width,
                                win_size.height,
                                gui_scale,
                                &self.gui_style.inventory,
                            );
                        }
                    }
                }

                let is_hovered = match &self.current_screen {
                    AppScreen::MainMenu => self
                        .main_menu
                        .buttons
                        .iter()
                        .any(|b| b.hovered && b.enabled),
                    AppScreen::WorldSelect => self.world_select.is_hovered(),
                    AppScreen::WorldCreate => self
                        .world_create
                        .buttons
                        .iter()
                        .any(|b| b.hovered && b.enabled),
                    AppScreen::Settings { .. } => {
                        self.settings_screen.tab_buttons.iter().any(|b| b.hovered)
                            || self.settings_screen.sliders.iter().any(|s| s.hovered)
                            || self
                                .settings_screen
                                .toggle_buttons
                                .iter()
                                .any(|b| b.hovered)
                            || self.settings_screen.done_button.hovered
                    }
                    AppScreen::Multiplayer => {
                        if self.server_list.direct_connect_mode {
                            self.server_list
                                .direct_buttons
                                .iter()
                                .any(|b| b.hovered && b.enabled)
                        } else {
                            self.server_list
                                .buttons
                                .iter()
                                .any(|b| b.hovered && b.enabled)
                        }
                    }
                    AppScreen::InGame => {
                        if self.is_paused {
                            self.pause_menu
                                .buttons
                                .iter()
                                .any(|b| b.hovered && b.enabled)
                        } else if self.advancements_open {
                            self.advancements_screen.hovered_node.is_some()
                        } else if self.container_state.is_some() {
                            self.container_hovered_slot.is_some()
                        } else if self.inventory_open {
                            self.inventory_hovered_slot.is_some()
                        } else {
                            false
                        }
                    }
                };

                if let Some(window) = &self.window {
                    let icon = if is_hovered {
                        winit::window::CursorIcon::Pointer
                    } else {
                        winit::window::CursorIcon::Default
                    };
                    window.set_cursor(icon);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let win_size = self.window_size();
                let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                let width_gui = win_size.width as f32 / gui_scale as f32;
                let height_gui = win_size.height as f32 / gui_scale as f32;
                let (mx, my) = (
                    self.mouse_cursor_pos[0] / gui_scale as f32,
                    self.mouse_cursor_pos[1] / gui_scale as f32,
                );

                if let Some(form) = &mut self.active_modal_form {
                    if state == ElementState::Released {
                        form.handle_mouse_up();
                        return;
                    }
                    if button == MouseButton::Left
                        && let Some(action) = form.handle_mouse_click(mx, my)
                    {
                        let form_id = form.form_id;
                        match action {
                            ModalFormAction::Submit(resp) => {
                                if let Ok(resp_payload) =
                                    C2sModalFormResponse::success(form_id, &resp)
                                {
                                    let resp_msg = C2sMessage::ModalFormResponse(resp_payload);
                                    let _ = self
                                        .client_conn
                                        .send(Lane::Control, Payload::Msg(resp_msg));
                                }
                            }
                            ModalFormAction::Cancel(reason) => {
                                let cancel_msg = C2sMessage::ModalFormResponse(
                                    C2sModalFormResponse::cancelled(form_id, reason),
                                );
                                let _ = self
                                    .client_conn
                                    .send(Lane::Control, Payload::Msg(cancel_msg));
                            }
                        }
                        self.active_modal_form = None;
                        if self.current_screen == AppScreen::InGame
                            && !self.is_paused
                            && !self.advancements_open
                            && !self.chat_state.is_open
                            && self.container_state.is_none()
                            && !self.inventory_open
                        {
                            self.controller.mouse_captured = true;
                            self.set_cursor_captured(true);
                        }
                    }
                    return;
                }

                if state == ElementState::Released {
                    if let AppScreen::Settings { .. } = self.current_screen {
                        self.settings_screen.handle_mouse_up();
                    }
                    if self.advancements_open {
                        self.advancements_screen.handle_mouse_up();
                        return;
                    }
                    if self.current_screen == AppScreen::InGame
                        && button == MouseButton::Right
                        && self.bow_charging
                    {
                        let charge_ticks = self.bow_charge_ticks;
                        self.bow_charging = false;
                        self.bow_charge_ticks = 0;
                        if charge_ticks >= telos_sim::BOW_MIN_CHARGE_TICKS {
                            let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                command: PlayerCommandKind::ShootBow { charge_ticks },
                            });
                            let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                            self.audio.play_bow_shoot(self.camera.position);
                        }
                    }
                    return;
                }

                match &mut self.current_screen {
                    AppScreen::MainMenu => {
                        if button == MouseButton::Left
                            && let Some(action) = self.main_menu.handle_mouse_click(mx, my)
                        {
                            match action {
                                MainMenuAction::Singleplayer => {
                                    self.world_select.scan_worlds(&self.worlds_dir);
                                    self.current_screen = AppScreen::WorldSelect;
                                }
                                MainMenuAction::Multiplayer => {
                                    self.server_list.update_layout(width_gui, height_gui);
                                    self.current_screen = AppScreen::Multiplayer;
                                }
                                MainMenuAction::Options => {
                                    self.settings_screen =
                                        SettingsScreen::new(self.game_settings.clone());
                                    self.settings_screen.update_layout_i18n(
                                        width_gui,
                                        height_gui,
                                        &self.catalog,
                                    );
                                    self.current_screen = AppScreen::Settings {
                                        return_to_pause: false,
                                    };
                                }
                                MainMenuAction::Quit => {
                                    info!("Quit button clicked on Main Menu");
                                    event_loop.exit();
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::WorldSelect => {
                        if button == MouseButton::Left
                            && let Some(action) =
                                self.world_select.handle_mouse_click(mx, my, width_gui)
                        {
                            match action {
                                WorldSelectAction::PlayWorld(entry) => {
                                    self.start_singleplayer_server(
                                        entry.path,
                                        entry.seed,
                                        &entry.generator,
                                    );
                                    self.current_screen = AppScreen::InGame;
                                    self.is_paused = false;
                                    self.controller.mouse_captured = true;
                                    self.set_cursor_captured(true);
                                }
                                WorldSelectAction::CreateNewWorld => {
                                    self.world_create = WorldCreateWizard::new();
                                    self.current_screen = AppScreen::WorldCreate;
                                }
                                WorldSelectAction::DeleteWorld(entry) => {
                                    info!(world = %entry.name, path = ?entry.path, "Moving world to trash");
                                    match telos_core::trash::move_to_trash(&entry.path) {
                                        Ok(dest) => {
                                            info!(world = %entry.name, destination = ?dest, "Successfully moved world to trash");
                                        }
                                        Err(err) => {
                                            warn!(world = %entry.name, error = %err, "Failed to move world to trash");
                                        }
                                    }
                                    self.world_select.scan_worlds(&self.worlds_dir);
                                }
                                WorldSelectAction::BackToTitle => {
                                    self.current_screen = AppScreen::MainMenu;
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::WorldCreate => {
                        if button == MouseButton::Left
                            && let Some(action) =
                                self.world_create
                                    .handle_mouse_click(mx, my, &self.worlds_dir)
                        {
                            match action {
                                WorldCreateAction::CreateWorld {
                                    path,
                                    seed,
                                    generator,
                                    ..
                                } => {
                                    self.start_singleplayer_server(path, seed, &generator);
                                    self.current_screen = AppScreen::InGame;
                                    self.is_paused = false;
                                    self.controller.mouse_captured = true;
                                    self.set_cursor_captured(true);
                                }
                                WorldCreateAction::Cancel => {
                                    self.world_select.scan_worlds(&self.worlds_dir);
                                    self.current_screen = AppScreen::WorldSelect;
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::Settings { return_to_pause } => {
                        let to_pause = *return_to_pause;
                        if button == MouseButton::Right && self.settings_screen.is_rebinding() {
                            self.settings_screen.cancel_rebinding();
                            self.settings_screen.update_layout_i18n(
                                width_gui,
                                height_gui,
                                &self.catalog,
                            );
                            return;
                        }
                        if button == MouseButton::Left {
                            let old_lang = self.settings_screen.settings.gameplay.language.clone();
                            let done = self.settings_screen.handle_mouse_click_i18n(
                                mx,
                                my,
                                width_gui,
                                height_gui,
                                &self.catalog,
                            );
                            if self.settings_screen.settings.gameplay.language != old_lang {
                                self.catalog.set_active_locale(
                                    &self.settings_screen.settings.gameplay.language,
                                );
                                self.game_settings.gameplay.language =
                                    self.settings_screen.settings.gameplay.language.clone();
                                let _ = self.game_settings.save(&self.settings_path);
                            }
                            if done {
                                self.game_settings = self.settings_screen.settings.clone();
                                self.catalog
                                    .set_active_locale(&self.game_settings.gameplay.language);
                                let _ = self.game_settings.save(&self.settings_path);
                                if to_pause {
                                    self.current_screen = AppScreen::InGame;
                                    self.is_paused = true;
                                    #[allow(clippy::cast_possible_truncation)]
                                    let settings_msg =
                                        C2sMessage::ClientSettings(C2sClientSettings {
                                            view_distance: self.game_settings.video.view_distance
                                                as u16,
                                            simulation_distance: self
                                                .game_settings
                                                .video
                                                .simulation_distance
                                                as u16,
                                            locale: BoundedString::new(
                                                &self.game_settings.gameplay.language,
                                            )
                                            .unwrap_or_else(|_| {
                                                BoundedString::new("en_US").unwrap()
                                            }),
                                        });
                                    let _ = self
                                        .client_conn
                                        .send(Lane::Control, Payload::Msg(settings_msg));
                                } else {
                                    self.current_screen = AppScreen::MainMenu;
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::Multiplayer => {
                        if button == MouseButton::Left
                            && let Some(action) = self
                                .server_list
                                .handle_mouse_click(mx, my, width_gui, height_gui)
                        {
                            match action {
                                ServerListAction::Connect(target) => {
                                    self.connect_to_remote_server(&target);
                                }
                                ServerListAction::Cancel => {
                                    self.current_screen = AppScreen::MainMenu;
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::InGame => {
                        if self.is_paused {
                            if button == MouseButton::Left
                                && let Some(action) = self.pause_menu.handle_mouse_click(mx, my)
                            {
                                match action {
                                    PauseMenuAction::Resume => {
                                        self.is_paused = false;
                                        self.controller.mouse_captured = true;
                                        self.set_cursor_captured(true);
                                    }
                                    PauseMenuAction::Advancements => {
                                        self.is_paused = false;
                                        self.advancements_open = true;
                                        self.controller.mouse_captured = false;
                                        self.set_cursor_captured(false);
                                    }
                                    PauseMenuAction::Options => {
                                        self.settings_screen =
                                            SettingsScreen::new(self.game_settings.clone());
                                        self.settings_screen.update_layout_i18n(
                                            width_gui,
                                            height_gui,
                                            &self.catalog,
                                        );
                                        self.current_screen = AppScreen::Settings {
                                            return_to_pause: true,
                                        };
                                    }
                                    PauseMenuAction::OpenToLan => {
                                        self.open_singleplayer_to_lan_and_invite();
                                    }
                                    PauseMenuAction::SaveAndQuit => {
                                        self.stop_singleplayer_server();
                                        self.is_paused = false;
                                        self.advancements_open = false;
                                        self.current_screen = AppScreen::MainMenu;
                                        self.set_cursor_captured(false);
                                    }
                                }
                            }
                            return;
                        }
                        if self.advancements_open {
                            if button == MouseButton::Left {
                                self.advancements_screen
                                    .handle_mouse_down(mx, my, width_gui, height_gui);
                            }
                            return;
                        }
                    }
                }

                if self.chat_state.is_open {
                    return;
                }
                if let Some(ref mut cont) = self.container_state {
                    let click_btn = match button {
                        MouseButton::Left => 0u8,
                        MouseButton::Right => 1u8,
                        _ => 255u8,
                    };
                    if click_btn <= 1
                        && let Some(hovered) = self.container_hovered_slot
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
                            telos_sim::ClickButton::Left
                        } else {
                            telos_sim::ClickButton::Right
                        };
                        let mode_sim = if mode == 1 {
                            telos_sim::ClickMode::QuickMove
                        } else {
                            telos_sim::ClickMode::Pickup
                        };

                        if cont.container_kind == 3 {
                            let mut anvil_inv = telos_sim::AnvilInventory::new();
                            anvil_inv.left = telos_sim::ItemStack::new_enchanted(
                                cont.slots[0].item,
                                cont.slots[0].count,
                                telos_sim::CompactEnchantments(cont.slots[0].enchantments),
                            );
                            anvil_inv.right = telos_sim::ItemStack::new_enchanted(
                                cont.slots[1].item,
                                cont.slots[1].count,
                                telos_sim::CompactEnchantments(cont.slots[1].enchantments),
                            );
                            anvil_inv.result = telos_sim::ItemStack::new_enchanted(
                                cont.slots[2].item,
                                cont.slots[2].count,
                                telos_sim::CompactEnchantments(cont.slots[2].enchantments),
                            );
                            anvil_inv.level_cost = cont.properties[0].max(0) as u32;
                            let _ = telos_sim::anvil_container_click(
                                &mut anvil_inv,
                                &mut self.inventory_sim,
                                self.hud_state.xp_level,
                                hovered,
                                btn_sim,
                                mode_sim,
                            );
                            cont.slots[0] = UiSlotItem::new_enchanted(
                                anvil_inv.left.item,
                                anvil_inv.left.count,
                                anvil_inv.left.enchantments.0,
                            );
                            cont.slots[1] = UiSlotItem::new_enchanted(
                                anvil_inv.right.item,
                                anvil_inv.right.count,
                                anvil_inv.right.enchantments.0,
                            );
                            cont.slots[2] = UiSlotItem::new_enchanted(
                                anvil_inv.result.item,
                                anvil_inv.result.count,
                                anvil_inv.result.enchantments.0,
                            );
                            cont.properties[0] = anvil_inv.level_cost as i16;
                        } else if cont.container_kind == 2 {
                            let mut ct_inv = telos_sim::CraftingTableInventory::new();
                            ct_inv.result =
                                telos_sim::ItemStack::new(cont.slots[0].item, cont.slots[0].count);
                            for (i, slot) in cont.slots[1..10].iter().enumerate() {
                                ct_inv.grid[i] = telos_sim::ItemStack::new(slot.item, slot.count);
                            }
                            let _ = telos_sim::crafting_table_container_click(
                                &mut ct_inv,
                                &mut self.inventory_sim,
                                hovered,
                                btn_sim,
                                mode_sim,
                                &self.recipe_registry,
                            );
                            cont.slots[0] =
                                UiSlotItem::new(ct_inv.result.item, ct_inv.result.count);
                            for (i, slot) in ct_inv.grid.iter().enumerate() {
                                cont.slots[1 + i] = UiSlotItem::new(slot.item, slot.count);
                            }
                        } else if cont.container_kind == 1 {
                            let mut furnace_inv = telos_sim::FurnaceInventory::default();
                            for (i, slot) in cont.slots.iter().take(3).enumerate() {
                                furnace_inv.slots[i] =
                                    telos_sim::ItemStack::new(slot.item, slot.count);
                            }
                            let _ = telos_sim::furnace_container_click(
                                &mut furnace_inv,
                                &mut self.inventory_sim,
                                hovered,
                                btn_sim,
                                mode_sim,
                                &self.fuel_registry,
                                &self.smelting_registry,
                            );
                            for (i, slot) in furnace_inv.slots.iter().enumerate() {
                                cont.slots[i] = UiSlotItem::new(slot.item, slot.count);
                            }
                        } else {
                            let mut chest_inv = telos_sim::ChestInventory::default();
                            for (i, slot) in cont.slots.iter().enumerate() {
                                chest_inv.slots[i] = telos_sim::ItemStack::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    telos_sim::CompactEnchantments(slot.enchantments),
                                );
                            }
                            let _ = telos_sim::container_click(
                                &mut chest_inv,
                                &mut self.inventory_sim,
                                hovered,
                                btn_sim,
                                mode_sim,
                            );
                            for (i, slot) in chest_inv.slots.iter().enumerate() {
                                cont.slots[i] = UiSlotItem::new_enchanted(
                                    slot.item,
                                    slot.count,
                                    slot.enchantments.0,
                                );
                            }
                        }
                        self.selected_block_state =
                            BlockStateId::new(self.inventory_sim.selected_item().item);
                    }
                    return;
                }
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
                            telos_sim::ClickButton::Left
                        } else {
                            telos_sim::ClickButton::Right
                        };
                        let mode_sim = if mode == 1 {
                            telos_sim::ClickMode::QuickMove
                        } else {
                            telos_sim::ClickMode::Pickup
                        };
                        let _ = telos_sim::inventory_click_with_registry(
                            &mut self.inventory_sim,
                            hovered,
                            btn_sim,
                            mode_sim,
                            &self.recipe_registry,
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

                                let ent_pos = self
                                    .entity_store
                                    .get(target_net_id)
                                    .map_or(origin + forward * 2.0, |e| e.pos.as_vec3());
                                self.audio.play_procedural_hurt(ent_pos, 1.0);
                                self.particle_system
                                    .spawn_crit(ent_pos + Vec3::new(0.0, 1.0, 0.0), 16);
                                return;
                            }

                            if !self.physics.capabilities.can_build {
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
                                let broken_state = self.get_block_at(hit.pos);
                                self.apply_block_update(hit.pos, BlockStateId::AIR);

                                let hit_pos_f = Vec3::new(
                                    hit.pos.x() as f32 + 0.5,
                                    hit.pos.y() as f32 + 0.5,
                                    hit.pos.z() as f32 + 0.5,
                                );
                                let (layer, tint) = block_state_to_particle_layer(broken_state);
                                if let Some(hook) = &self.js_particle_hook {
                                    let p = hook.on_particle_spawn(
                                        "block_break",
                                        telos_mod::JsParticleParams {
                                            pos: [hit_pos_f.x, hit_pos_f.y, hit_pos_f.z],
                                            velocity: [0.0, 0.0, 0.0],
                                            color_tint: u32::from_le_bytes(tint),
                                            scale: 1.0,
                                            lifetime: 32.0,
                                        },
                                    );
                                    let mod_tint = p.color_tint.to_le_bytes();
                                    let mod_count = (p.lifetime as usize).clamp(1, 128);
                                    let mod_pos = Vec3::new(p.pos[0], p.pos[1], p.pos[2]);
                                    self.particle_system
                                        .spawn_block_break(mod_pos, layer, mod_tint, mod_count);
                                } else {
                                    self.particle_system
                                        .spawn_block_break(hit_pos_f, layer, tint, 32);
                                }

                                let mut pitch = 0.92
                                    + (hit.pos.x().abs() as f32 * 0.1
                                        + hit.pos.z().abs() as f32 * 0.2)
                                        .fract()
                                        * 0.16;
                                if let Some(hook) = &self.js_particle_hook {
                                    let (_, mod_pitch) =
                                        hook.on_sound_play("block_break", 1.0, pitch);
                                    pitch = mod_pitch;
                                }
                                self.audio.play_procedural_break(hit_pos_f, pitch);
                            }
                        }
                        MouseButton::Right => {
                            if self.inventory_sim.selected_item().item == telos_sim::ITEM_BOW {
                                self.bow_charging = true;
                                self.bow_charge_ticks = 0;
                                return;
                            }

                            if telos_sim::is_potion(self.inventory_sim.selected_item().item) {
                                let msg = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::DrinkPotion { potion_type: 4 },
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
                                return;
                            }

                            if let Some(hit) = self.targeted_block {
                                let target_state = self.get_block_at(hit.pos);
                                if self.block_registry.is_chest(target_state) && !self.shift_held {
                                    self.action_sequence += 1;
                                    let msg = C2sMessage::BlockAction(C2sBlockAction {
                                        sequence: self.action_sequence,
                                        action: BlockActionKind::Interact,
                                        x: hit.pos.x(),
                                        y: hit.pos.y(),
                                        z: hit.pos.z(),
                                        input_tick: self.frame_counter,
                                    });
                                    let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
                                    return;
                                }

                                let is_lever = self
                                    .block_registry
                                    .identifier(target_state)
                                    .is_some_and(|id| {
                                        id.path() == "logic_lever" || id.path() == "logic_lever_on"
                                    });
                                if is_lever {
                                    self.action_sequence += 1;
                                    let msg = C2sMessage::BlockAction(C2sBlockAction {
                                        sequence: self.action_sequence,
                                        action: BlockActionKind::Interact,
                                        x: hit.pos.x(),
                                        y: hit.pos.y(),
                                        z: hit.pos.z(),
                                        input_tick: self.frame_counter,
                                    });
                                    let _ = self.client_conn.send(Lane::Control, Payload::Msg(msg));
                                    let toggled_name = if self
                                        .block_registry
                                        .identifier(target_state)
                                        .is_some_and(|id| id.path() == "logic_lever")
                                    {
                                        "logic_lever_on"
                                    } else {
                                        "logic_lever"
                                    };
                                    if let Ok(id) =
                                        telos_core::ident::Identifier::new("telos", toggled_name)
                                        && let Some(b) = self.block_registry.get(&id)
                                    {
                                        self.apply_block_update(hit.pos, b.default_state());
                                    }
                                    let hit_pos_f = Vec3::new(
                                        hit.pos.x() as f32 + 0.5,
                                        hit.pos.y() as f32 + 0.5,
                                        hit.pos.z() as f32 + 0.5,
                                    );
                                    self.audio.play_procedural_place(hit_pos_f, 1.25);
                                    return;
                                }

                                if !self.physics.capabilities.can_build {
                                    return;
                                }

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

                                let place_pos_f = Vec3::new(
                                    place_pos.x() as f32 + 0.5,
                                    place_pos.y() as f32 + 0.5,
                                    place_pos.z() as f32 + 0.5,
                                );
                                let (layer, tint) =
                                    block_state_to_particle_layer(self.selected_block_state);
                                if let Some(hook) = &self.js_particle_hook {
                                    let p = hook.on_particle_spawn(
                                        "block_place",
                                        telos_mod::JsParticleParams {
                                            pos: [place_pos_f.x, place_pos_f.y, place_pos_f.z],
                                            velocity: [0.0, 0.0, 0.0],
                                            color_tint: u32::from_le_bytes(tint),
                                            scale: 1.0,
                                            lifetime: 12.0,
                                        },
                                    );
                                    let mod_tint = p.color_tint.to_le_bytes();
                                    let mod_count = (p.lifetime as usize).clamp(1, 128);
                                    let mod_pos = Vec3::new(p.pos[0], p.pos[1], p.pos[2]);
                                    self.particle_system
                                        .spawn_block_place(mod_pos, layer, mod_tint, mod_count);
                                } else {
                                    self.particle_system.spawn_block_place(
                                        place_pos_f,
                                        layer,
                                        tint,
                                        12,
                                    );
                                }
                                let mut pitch = 0.95
                                    + (place_pos.x().abs() as f32 * 0.13
                                        + place_pos.z().abs() as f32 * 0.17)
                                        .fract()
                                        * 0.12;
                                if let Some(hook) = &self.js_particle_hook {
                                    let (_, mod_pitch) =
                                        hook.on_sound_play("block_place", 1.0, pitch);
                                    pitch = mod_pitch;
                                }
                                self.audio.play_procedural_place(place_pos_f, pitch);
                            }
                        }
                        _ => {}
                    }
                } else if button == MouseButton::Left {
                    self.controller.mouse_captured = true;
                    self.set_cursor_captured(true);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        text,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();

                if let Some(form) = &mut self.active_modal_form {
                    if pressed {
                        match code {
                            KeyCode::Escape => {
                                let cancel_msg =
                                    C2sMessage::ModalFormResponse(C2sModalFormResponse::cancelled(
                                        form.form_id,
                                        telos_core::form::FormCancelReason::UserClosed,
                                    ));
                                let _ = self
                                    .client_conn
                                    .send(Lane::Control, Payload::Msg(cancel_msg));
                                self.active_modal_form = None;
                                if self.current_screen == AppScreen::InGame
                                    && !self.is_paused
                                    && !self.advancements_open
                                    && !self.chat_state.is_open
                                    && self.container_state.is_none()
                                    && !self.inventory_open
                                {
                                    self.controller.mouse_captured = true;
                                    self.set_cursor_captured(true);
                                }
                            }
                            KeyCode::Backspace => {
                                form.handle_backspace();
                            }
                            _ => {
                                if let Some(txt) = text {
                                    for ch in txt.chars() {
                                        if !ch.is_control() {
                                            form.handle_char(ch);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                match &mut self.current_screen {
                    AppScreen::Multiplayer => {
                        if pressed {
                            match code {
                                KeyCode::Escape => {
                                    if self.server_list.direct_connect_mode {
                                        self.server_list.direct_connect_mode = false;
                                        self.server_list.direct_connect_focused = false;
                                    } else {
                                        self.current_screen = AppScreen::MainMenu;
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some(action) = self.server_list.handle_enter() {
                                        match action {
                                            ServerListAction::Connect(target) => {
                                                self.connect_to_remote_server(&target);
                                            }
                                            ServerListAction::Cancel => {
                                                self.current_screen = AppScreen::MainMenu;
                                            }
                                        }
                                    }
                                }
                                KeyCode::Backspace => {
                                    self.server_list.handle_backspace();
                                }
                                _ => {
                                    if let Some(txt) = text {
                                        for ch in txt.chars() {
                                            if !ch.is_control() {
                                                self.server_list.handle_char(ch);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::WorldCreate => {
                        if pressed {
                            match code {
                                KeyCode::Escape => {
                                    self.world_select.scan_worlds(&self.worlds_dir);
                                    self.current_screen = AppScreen::WorldSelect;
                                }
                                KeyCode::Backspace => {
                                    self.world_create.handle_backspace();
                                }
                                _ => {
                                    if let Some(txt) = text {
                                        for ch in txt.chars() {
                                            if !ch.is_control() {
                                                self.world_create.handle_char(ch);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::Settings { return_to_pause } => {
                        let to_pause = *return_to_pause;
                        if pressed {
                            if self.settings_screen.is_rebinding() {
                                let key = key_code_to_input_key(code);
                                if code == KeyCode::Escape || key == InputKey::Escape {
                                    self.settings_screen.cancel_rebinding();
                                } else if key != InputKey::None {
                                    self.settings_screen.handle_key_input(key);
                                    self.game_settings = self.settings_screen.settings.clone();
                                    let _ = self.game_settings.save(&self.settings_path);
                                }
                                let win_size = self.window_size();
                                let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                                let width_gui = win_size.width as f32 / gui_scale as f32;
                                let height_gui = win_size.height as f32 / gui_scale as f32;
                                self.settings_screen.update_layout_i18n(
                                    width_gui,
                                    height_gui,
                                    &self.catalog,
                                );
                                return;
                            }

                            if code == KeyCode::Escape {
                                self.game_settings = self.settings_screen.settings.clone();
                                self.catalog
                                    .set_active_locale(&self.game_settings.gameplay.language);
                                let _ = self.game_settings.save(&self.settings_path);
                                if to_pause {
                                    self.current_screen = AppScreen::InGame;
                                    self.is_paused = true;
                                    #[allow(clippy::cast_possible_truncation)]
                                    let settings_msg =
                                        C2sMessage::ClientSettings(C2sClientSettings {
                                            view_distance: self.game_settings.video.view_distance
                                                as u16,
                                            simulation_distance: self
                                                .game_settings
                                                .video
                                                .simulation_distance
                                                as u16,
                                            locale: BoundedString::new(
                                                &self.game_settings.gameplay.language,
                                            )
                                            .unwrap_or_else(|_| {
                                                BoundedString::new("en_US").unwrap()
                                            }),
                                        });
                                    let _ = self
                                        .client_conn
                                        .send(Lane::Control, Payload::Msg(settings_msg));
                                } else {
                                    self.current_screen = AppScreen::MainMenu;
                                }
                            }
                        }
                        return;
                    }
                    AppScreen::WorldSelect => {
                        if pressed {
                            if code == KeyCode::Escape {
                                if self.world_select.has_prompt() {
                                    self.world_select.cancel_prompt();
                                } else {
                                    self.current_screen = AppScreen::MainMenu;
                                }
                            } else if code == KeyCode::Enter || code == KeyCode::NumpadEnter {
                                if self.world_select.has_prompt() {
                                    if let Some(WorldSelectAction::DeleteWorld(entry)) =
                                        self.world_select.confirm_prompt()
                                    {
                                        info!(world = %entry.name, path = ?entry.path, "Moving world to trash");
                                        let _ = telos_core::trash::move_to_trash(&entry.path);
                                        self.world_select.scan_worlds(&self.worlds_dir);
                                    }
                                } else if let Some(idx) = self.world_select.selected_index
                                    && let Some(entry) = self.world_select.worlds.get(idx).cloned()
                                {
                                    self.start_singleplayer_server(
                                        entry.path,
                                        entry.seed,
                                        &entry.generator,
                                    );
                                    self.current_screen = AppScreen::InGame;
                                    self.is_paused = false;
                                    self.controller.mouse_captured = true;
                                    self.set_cursor_captured(true);
                                }
                            } else if (code == KeyCode::Delete || code == KeyCode::Backspace)
                                && !self.world_select.has_prompt()
                            {
                                self.world_select.open_delete_prompt_i18n(&self.catalog);
                            } else if (code == KeyCode::ArrowUp || code == KeyCode::KeyW)
                                && !self.world_select.has_prompt()
                            {
                                self.world_select.select_previous();
                            } else if (code == KeyCode::ArrowDown || code == KeyCode::KeyS)
                                && !self.world_select.has_prompt()
                            {
                                self.world_select.select_next();
                            }
                        }
                        return;
                    }
                    AppScreen::MainMenu => {
                        return;
                    }
                    AppScreen::InGame => {}
                }

                if self.is_paused {
                    if pressed && code == KeyCode::Escape {
                        self.is_paused = false;
                        self.controller.mouse_captured = true;
                        self.set_cursor_captured(true);
                    }
                    return;
                }

                if code == KeyCode::ShiftLeft || code == KeyCode::ShiftRight {
                    self.shift_held = pressed;
                }

                if self.chat_state.is_open {
                    if pressed {
                        match code {
                            KeyCode::Escape => {
                                self.chat_state.close();
                                self.controller.mouse_captured = true;
                                self.set_cursor_captured(true);
                            }
                            KeyCode::Enter => {
                                if let Some(msg) = self.chat_state.take_submitted_message()
                                    && let Ok(bounded) = BoundedString::new(msg)
                                {
                                    let _ = self.client_conn.send(
                                        Lane::Control,
                                        Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                                            message: bounded,
                                        })),
                                    );
                                }
                                self.controller.mouse_captured = true;
                                self.set_cursor_captured(true);
                            }
                            KeyCode::Backspace => {
                                self.chat_state.handle_backspace();
                                self.request_command_suggestions();
                            }
                            KeyCode::Delete => {
                                self.chat_state.handle_delete();
                                self.request_command_suggestions();
                            }
                            KeyCode::ArrowLeft => {
                                self.chat_state.move_cursor_left();
                            }
                            KeyCode::ArrowRight => {
                                self.chat_state.move_cursor_right();
                            }
                            KeyCode::Home => {
                                self.chat_state.move_cursor_start();
                            }
                            KeyCode::End => {
                                self.chat_state.move_cursor_end();
                            }
                            KeyCode::ArrowUp => {
                                if self.chat_state.suggestions.is_empty() {
                                    self.chat_state.history_up();
                                } else {
                                    self.chat_state.select_prev_suggestion();
                                }
                            }
                            KeyCode::ArrowDown => {
                                if self.chat_state.suggestions.is_empty() {
                                    self.chat_state.history_down();
                                } else {
                                    self.chat_state.select_next_suggestion();
                                }
                            }
                            KeyCode::Tab => {
                                if self.chat_state.suggestions.is_empty() {
                                    self.request_command_suggestions();
                                } else {
                                    self.chat_state.apply_selected_suggestion();
                                }
                            }
                            _ => {
                                if let Some(txt) = text {
                                    for c in txt.chars() {
                                        if !c.is_control() {
                                            self.chat_state.handle_char(c);
                                        }
                                    }
                                    self.request_command_suggestions();
                                }
                            }
                        }
                    }
                    return;
                }

                if self.container_state.is_some() {
                    let key = key_code_to_input_key(code);
                    let kb = &self.game_settings.controls.keybinds;
                    if pressed && (code == KeyCode::Escape || kb.matches(KeyAction::Inventory, key))
                    {
                        self.close_active_container();
                        return;
                    }
                    if pressed {
                        if kb.matches(KeyAction::Hotbar1, key) || code == KeyCode::Digit1 {
                            self.handle_container_swap_hotbar(0);
                        } else if kb.matches(KeyAction::Hotbar2, key) || code == KeyCode::Digit2 {
                            self.handle_container_swap_hotbar(1);
                        } else if kb.matches(KeyAction::Hotbar3, key) || code == KeyCode::Digit3 {
                            self.handle_container_swap_hotbar(2);
                        } else if kb.matches(KeyAction::Hotbar4, key) || code == KeyCode::Digit4 {
                            self.handle_container_swap_hotbar(3);
                        } else if kb.matches(KeyAction::Hotbar5, key) || code == KeyCode::Digit5 {
                            self.handle_container_swap_hotbar(4);
                        } else if kb.matches(KeyAction::Hotbar6, key) || code == KeyCode::Digit6 {
                            self.handle_container_swap_hotbar(5);
                        } else if kb.matches(KeyAction::Hotbar7, key) || code == KeyCode::Digit7 {
                            self.handle_container_swap_hotbar(6);
                        } else if kb.matches(KeyAction::Hotbar8, key) || code == KeyCode::Digit8 {
                            self.handle_container_swap_hotbar(7);
                        } else if kb.matches(KeyAction::Hotbar9, key) || code == KeyCode::Digit9 {
                            self.handle_container_swap_hotbar(8);
                        }
                    }
                    return;
                }

                if self.inventory_open {
                    let key = key_code_to_input_key(code);
                    let kb = &self.game_settings.controls.keybinds;
                    if pressed && (code == KeyCode::Escape || kb.matches(KeyAction::Inventory, key))
                    {
                        self.inventory_open = false;
                        self.controller.mouse_captured = true;
                        self.set_cursor_captured(true);
                        self.inventory_hovered_slot = None;
                        return;
                    }
                    if pressed && kb.matches(KeyAction::Drop, key) {
                        if let Some(hovered) = self.inventory_hovered_slot {
                            if hovered < 9 {
                                self.selected_hotbar_slot = hovered;
                            }
                            let entire_stack = self.controller.sprint;
                            let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                command: PlayerCommandKind::DropItem { entire_stack },
                            });
                            let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        }
                        return;
                    }
                    if pressed {
                        if kb.matches(KeyAction::Hotbar1, key) || code == KeyCode::Digit1 {
                            self.handle_inventory_swap_hotbar(0);
                        } else if kb.matches(KeyAction::Hotbar2, key) || code == KeyCode::Digit2 {
                            self.handle_inventory_swap_hotbar(1);
                        } else if kb.matches(KeyAction::Hotbar3, key) || code == KeyCode::Digit3 {
                            self.handle_inventory_swap_hotbar(2);
                        } else if kb.matches(KeyAction::Hotbar4, key) || code == KeyCode::Digit4 {
                            self.handle_inventory_swap_hotbar(3);
                        } else if kb.matches(KeyAction::Hotbar5, key) || code == KeyCode::Digit5 {
                            self.handle_inventory_swap_hotbar(4);
                        } else if kb.matches(KeyAction::Hotbar6, key) || code == KeyCode::Digit6 {
                            self.handle_inventory_swap_hotbar(5);
                        } else if kb.matches(KeyAction::Hotbar7, key) || code == KeyCode::Digit7 {
                            self.handle_inventory_swap_hotbar(6);
                        } else if kb.matches(KeyAction::Hotbar8, key) || code == KeyCode::Digit8 {
                            self.handle_inventory_swap_hotbar(7);
                        } else if kb.matches(KeyAction::Hotbar9, key) || code == KeyCode::Digit9 {
                            self.handle_inventory_swap_hotbar(8);
                        }
                    }
                    return;
                }

                if self.advancements_open {
                    let key = key_code_to_input_key(code);
                    let kb = &self.game_settings.controls.keybinds;
                    if pressed
                        && (code == KeyCode::Escape || kb.matches(KeyAction::Advancements, key))
                    {
                        self.advancements_open = false;
                        self.controller.mouse_captured = true;
                        self.set_cursor_captured(true);
                    }
                    return;
                }

                let key = key_code_to_input_key(code);
                let kb = &self.game_settings.controls.keybinds;

                if kb.matches(KeyAction::Forward, key) {
                    self.controller.forward = pressed;
                }
                if kb.matches(KeyAction::Backward, key) {
                    self.controller.backward = pressed;
                }
                if kb.matches(KeyAction::Left, key) {
                    self.controller.left = pressed;
                }
                if kb.matches(KeyAction::Right, key) {
                    self.controller.right = pressed;
                }
                if kb.matches(KeyAction::Jump, key) {
                    self.controller.up = pressed;
                }
                if kb.matches(KeyAction::Sneak, key) {
                    self.controller.down = pressed;
                }
                if kb.matches(KeyAction::Sprint, key) {
                    self.controller.sprint = pressed;
                }

                if pressed {
                    if kb.matches(KeyAction::Drop, key) {
                        let entire_stack = self.controller.sprint;
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::DropItem { entire_stack },
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                    } else if kb.matches(KeyAction::Inventory, key) {
                        self.inventory_open = true;
                        self.controller.mouse_captured = false;
                        self.set_cursor_captured(false);
                        self.controller.forward = false;
                        self.controller.backward = false;
                        self.controller.left = false;
                        self.controller.right = false;
                        self.controller.up = false;
                        self.controller.down = false;
                        let win_size = self.window_size();
                        let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                        self.inventory_hovered_slot = inventory_slot_at_pos_styled(
                            self.mouse_cursor_pos,
                            win_size.width,
                            win_size.height,
                            gui_scale,
                            &self.gui_style.inventory,
                        );
                    } else if kb.matches(KeyAction::Advancements, key) {
                        self.advancements_open = true;
                        self.controller.mouse_captured = false;
                        self.set_cursor_captured(false);
                        self.controller.forward = false;
                        self.controller.backward = false;
                        self.controller.left = false;
                        self.controller.right = false;
                        self.controller.up = false;
                        self.controller.down = false;
                        self.controller.sprint = false;
                    } else if kb.matches(KeyAction::Chat, key) {
                        self.chat_state.open(None);
                        self.controller.mouse_captured = false;
                        self.set_cursor_captured(false);
                        self.controller.forward = false;
                        self.controller.backward = false;
                        self.controller.left = false;
                        self.controller.right = false;
                        self.controller.up = false;
                        self.controller.down = false;
                    } else if kb.matches(KeyAction::Command, key) {
                        self.chat_state.open(Some("/"));
                        self.controller.mouse_captured = false;
                        self.set_cursor_captured(false);
                        self.controller.forward = false;
                        self.controller.backward = false;
                        self.controller.left = false;
                        self.controller.right = false;
                        self.controller.up = false;
                        self.controller.down = false;
                        self.request_command_suggestions();
                    } else if kb.matches(KeyAction::Hotbar1, key) || code == KeyCode::Digit1 {
                        self.select_hotbar_slot(0);
                    } else if kb.matches(KeyAction::Hotbar2, key) || code == KeyCode::Digit2 {
                        self.select_hotbar_slot(1);
                    } else if kb.matches(KeyAction::Hotbar3, key) || code == KeyCode::Digit3 {
                        self.select_hotbar_slot(2);
                    } else if kb.matches(KeyAction::Hotbar4, key) || code == KeyCode::Digit4 {
                        self.select_hotbar_slot(3);
                    } else if kb.matches(KeyAction::Hotbar5, key) || code == KeyCode::Digit5 {
                        self.select_hotbar_slot(4);
                    } else if kb.matches(KeyAction::Hotbar6, key) || code == KeyCode::Digit6 {
                        self.select_hotbar_slot(5);
                    } else if kb.matches(KeyAction::Hotbar7, key) || code == KeyCode::Digit7 {
                        self.select_hotbar_slot(6);
                    } else if kb.matches(KeyAction::Hotbar8, key) || code == KeyCode::Digit8 {
                        self.select_hotbar_slot(7);
                    } else if kb.matches(KeyAction::Hotbar9, key) || code == KeyCode::Digit9 {
                        self.select_hotbar_slot(8);
                    } else if kb.matches(KeyAction::Screenshot, key) {
                        self.manual_screenshot_requested = true;
                        info!("Screenshot requested (F2)");
                    } else if kb.matches(KeyAction::ToggleF3, key) {
                        self.hud_state.f3_open = !self.hud_state.f3_open;
                        info!(f3_open = self.hud_state.f3_open, "Toggled F3 debug overlay");
                    } else if kb.matches(KeyAction::ToggleGameMode, key) {
                        let next_mode = self.physics.toggle_game_mode();
                        let mode_u8 = next_mode.id();
                        let mode_key = next_mode.translation_key();
                        self.hud_state.game_mode = self.catalog.translate(mode_key).to_string();
                        self.hud_state.is_flying = self.physics.flying;
                        let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                            command: PlayerCommandKind::SetGameMode(mode_u8),
                        });
                        let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                        info!(?next_mode, "Toggled game mode (F4)");
                    } else if kb.matches(KeyAction::ReloadShaders, key) {
                        if let (Some(gpu_context), Some(pp)) =
                            (&self.gpu_context, &mut self.postprocess)
                        {
                            let shaders_dir = if std::path::Path::new("shaders").exists() {
                                PathBuf::from("shaders")
                            } else if std::path::Path::new("../../shaders").exists() {
                                PathBuf::from("../../shaders")
                            } else {
                                PathBuf::from("shaders")
                            };
                            if shaders_dir.exists() {
                                match pp.reload_shaders(gpu_context.device().raw(), &shaders_dir) {
                                    Ok(()) => {
                                        info!(
                                            "Hot-reloaded post-processing shaders successfully (F5)"
                                        );
                                    }
                                    Err(err) => {
                                        tracing::error!("Failed to reload shaders: {err}");
                                    }
                                }
                            } else {
                                tracing::warn!("Shaders directory not found for hot-reload");
                            }
                        }
                    } else {
                        match code {
                            KeyCode::F7 => {
                                let next_kind = if self.weather_rain_level < 0.1
                                    && self.weather_thunder_level < 0.1
                                {
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
                            KeyCode::F8 => {
                                let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::TriggerLightning,
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                                info!("Sent trigger lightning command");
                            }
                            KeyCode::F9 => {
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
                            KeyCode::F10 => {
                                let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::ClearMobs,
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                                info!("Sent clear mobs command (F10)");
                            }
                            KeyCode::KeyK => {
                                let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::Damage(2.0),
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                                info!("Sent test damage command (2.0 HP)");
                            }
                            KeyCode::KeyJ => {
                                let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::AddXp(50),
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                                info!("Sent test add XP command (50 XP)");
                            }
                            KeyCode::KeyH => {
                                let cmd = C2sMessage::PlayerCommand(C2sPlayerCommand {
                                    command: PlayerCommandKind::SetFood(6),
                                });
                                let _ = self.client_conn.send(Lane::Control, Payload::Msg(cmd));
                                info!("Sent test set food command (6 food)");
                            }
                            KeyCode::Escape => {
                                self.is_paused = true;
                                self.controller.mouse_captured = false;
                                self.set_cursor_captured(false);
                                self.controller.forward = false;
                                self.controller.backward = false;
                                self.controller.left = false;
                                self.controller.right = false;
                                self.controller.up = false;
                                self.controller.down = false;
                                self.controller.sprint = false;
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    #[allow(clippy::cast_possible_truncation)]
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                if let Some(form) = &mut self.active_modal_form {
                    form.handle_scroll(scroll);
                    return;
                }
                if matches!(self.current_screen, AppScreen::Settings { .. }) {
                    let win_size = self.window_size();
                    let gui_scale = compute_gui_scale(win_size.width, win_size.height);
                    let width_gui = win_size.width as f32 / gui_scale as f32;
                    let height_gui = win_size.height as f32 / gui_scale as f32;
                    self.settings_screen.handle_mouse_wheel(scroll);
                    self.settings_screen
                        .update_layout_i18n(width_gui, height_gui, &self.catalog);
                    return;
                }
                if self.current_screen == AppScreen::Multiplayer {
                    self.server_list.scroll_offset =
                        (self.server_list.scroll_offset - scroll * 16.0).max(0.0);
                    return;
                }
                if self.current_screen == AppScreen::WorldSelect {
                    self.world_select.handle_mouse_wheel(scroll);
                } else if self.current_screen == AppScreen::InGame
                    && !self.is_paused
                    && !self.inventory_open
                    && !self.advancements_open
                    && self.container_state.is_none()
                {
                    if scroll > 0.0 {
                        let new_slot = (self.selected_hotbar_slot + 8) % 9;
                        self.select_hotbar_slot(new_slot);
                    } else if scroll < 0.0 {
                        let new_slot = (self.selected_hotbar_slot + 1) % 9;
                        self.select_hotbar_slot(new_slot);
                    }
                }
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    if let Some(gpu_context) = &mut self.gpu_context {
                        if let Err(err) =
                            gpu_context.resize(physical_size.width, physical_size.height)
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
                        if let (Some(pp), Some(depth)) = (&mut self.postprocess, &self.depth_buffer)
                            && let Err(err) = pp.resize(
                                gpu_context.device().raw(),
                                gpu_context.allocator(),
                                extent,
                                depth.view(),
                            )
                        {
                            tracing::error!("Failed to resize post-process frame graph: {err}");
                        }
                        if let Err(err) = self.recreate_hiz_resources(extent) {
                            tracing::error!("Failed to recreate Hi-Z resources on resize: {err}");
                        }
                    } else if let Some(gl_context) = &mut self.gl_context {
                        gl_context.resize(physical_size.width, physical_size.height);
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if !self.controller.mouse_captured {
                    self.set_cursor_captured(false);
                }
            }
            WindowEvent::Focused(focused) => {
                if !focused {
                    self.set_cursor_captured(false);
                    self.controller.mouse_captured = false;
                    self.controller.forward = false;
                    self.controller.backward = false;
                    self.controller.left = false;
                    self.controller.right = false;
                    self.controller.up = false;
                    self.controller.down = false;
                    self.controller.sprint = false;
                    if self.current_screen == AppScreen::InGame
                        && !self.inventory_open
                        && !self.advancements_open
                        && self.container_state.is_none()
                        && !self.chat_state.is_open
                        && self.args.screenshot.is_none()
                    {
                        self.is_paused = true;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if self.render() {
                    event_loop.exit();
                } else if let Some(window) = &self.window {
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

            if let Some(mut pipeline) = self.t2_cutout_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut pipeline) = self.t2_translucent_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.t2_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.t2_frag_shader.take() {
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

            if let Some(mut pp) = self.postprocess.take() {
                pp.destroy(device, allocator);
            }

            if let Some(mut pipeline) = self.sky_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut pipeline) = self.panorama_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut frag) = self.panorama_frag_shader.take() {
                frag.destroy(device);
            }
            if let Some(pool) = self.panorama_descriptor_pool.take() {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            if let Some(layout) = self.panorama_descriptor_set_layout.take() {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
            if let Some(mut tex) = self.panorama_texture.take() {
                tex.destroy(device, allocator);
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

            if let Some(mut pipeline) = self.particle_pipeline.take() {
                pipeline.destroy(device);
            }
            if let Some(mut vert) = self.particle_vert_shader.take() {
                vert.destroy(device);
            }
            if let Some(mut frag) = self.particle_frag_shader.take() {
                frag.destroy(device);
            }
            if let Some(pool) = self.particle_descriptor_pool.take() {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            if let Some(layout) = self.particle_descriptor_set_layout.take() {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
            if let Some(mut tex) = self.particle_texture.take() {
                tex.destroy(device, allocator);
            }
            if let Some(mut buf) = self.particle_buffer.take() {
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

fn mount_asset_roots(stack: &mut ResourcePackStack) {
    if let Ok(dir) = std::env::var("TELOS_ASSETS_DIR") {
        stack.add_root(dir);
    }
    if let Ok(dir) = std::env::var("VOXEL_ASSETS_DIR") {
        stack.add_root(dir);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        stack.add_root(exe_dir.join("assets"));
        stack.add_root(exe_dir.join("assets/telos"));
        stack.add_root(exe_dir.join("assets/voxel"));
        stack.add_root(exe_dir.join("../share/telos/assets"));
        stack.add_root(exe_dir.join("../share/voxel/assets"));
        stack.add_root(exe_dir.join("../dev-assets/faithful-32x"));
        stack.add_root(exe_dir.join("../dev-assets/classic-pack"));
    }
    stack.add_root("/usr/share/telos/assets");
    stack.add_root("/usr/share/voxel/assets");
    stack.add_root("/usr/local/share/telos/assets");
    stack.add_root("/usr/local/share/voxel/assets");
    if let Ok(home) = std::env::var("HOME") {
        stack.add_root(format!("{home}/.local/share/telos/assets"));
        stack.add_root(format!("{home}/.local/share/voxel/assets"));
        stack.add_root(format!(
            "{home}/Projects/larvance/telos/dev-assets/faithful-32x"
        ));
        stack.add_root(format!(
            "{home}/Projects/larvance/telos/dev-assets/classic-pack"
        ));
        stack.add_root(format!("{home}/Projects/larvance/telos/assets/telos"));
        stack.add_root(format!("{home}/Projects/larvance/telos/assets/voxel"));
    }
    stack.add_root("dev-assets/faithful-32x");
    stack.add_root("dev-assets/classic-pack");
    stack.add_root("assets/telos");
    stack.add_root("assets/voxel");
}

#[allow(clippy::too_many_lines)]
fn bake_block_textures() -> (telos_assets::BakedTextureArray, AnimatedTextures) {
    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

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

    let load_texture_with_fallback = |name: &str| -> telos_assets::RgbaImage {
        stack.load_block_texture(name).unwrap_or_else(|_| {
            let mut img = telos_assets::RgbaImage::new(target_res, target_res);
            let color = match name {
                "stone" | "bedrock" => [128, 128, 128, 255],
                "dirt" | "grass_block_side" => [134, 96, 67, 255],
                "grass_block_top" => [124, 189, 81, 255],
                "sand" => [219, 211, 160, 255],
                "oak_planks" => [162, 130, 78, 255],
                "oak_leaves" => [60, 140, 40, 255],
                "birch_leaves" => [100, 160, 60, 255],
                "spruce_leaves" => [65, 110, 65, 255],
                "oak_log" => [103, 82, 49, 255],
                "oak_log_top" => [140, 115, 75, 255],
                "birch_log" => [215, 215, 210, 255],
                "birch_log_top" => [180, 160, 125, 255],
                "spruce_log" => [45, 30, 15, 255],
                "spruce_log_top" => [105, 80, 50, 255],
                "glass" => [200, 220, 255, 128],
                "poppy" | "red_mushroom" => [220, 40, 40, 255],
                "dandelion" => [255, 230, 40, 255],
                "torch" => [255, 200, 50, 255],
                "short_grass" | "tall_grass" | "fern" => [100, 180, 60, 255],
                "cornflower" => [70, 120, 240, 255],
                "oxeye_daisy" => [240, 240, 230, 255],
                "brown_mushroom" => [150, 110, 80, 255],
                "dead_bush" | "lever" => [140, 110, 70, 255],
                "redstone_torch" => [255, 60, 60, 255],
                "redstone_torch_off" => [100, 30, 30, 255],
                "redstone_block" => [220, 30, 30, 255],
                "redstone_lamp" => [90, 55, 30, 255],
                "redstone_lamp_on" => [255, 210, 100, 255],
                "redstone_dust_line0" => [255, 255, 255, 255],
                "repeater" | "repeater_on" => [180, 180, 180, 255],
                _ => [255, 0, 255, 255],
            };
            for pixel in img.data.as_chunks_mut::<4>().0 {
                *pixel = color;
            }
            img
        })
    };

    let mut builder = TextureArrayBuilder::new(target_res);
    builder.insert("stone", load_texture_with_fallback("stone"));
    builder.insert("dirt", load_texture_with_fallback("dirt"));
    builder.insert(
        "grass_block_top",
        load_texture_with_fallback("grass_block_top"),
    );
    builder.insert(
        "grass_block_side",
        load_texture_with_fallback("grass_block_side"),
    );
    builder.insert("bedrock", load_texture_with_fallback("bedrock"));
    builder.insert("sand", load_texture_with_fallback("sand"));
    builder.insert("oak_planks", load_texture_with_fallback("oak_planks"));
    builder.insert("oak_leaves", load_texture_with_fallback("oak_leaves"));
    builder.insert("glass", load_texture_with_fallback("glass"));
    builder.insert("poppy", load_texture_with_fallback("poppy"));
    builder.insert("dandelion", load_texture_with_fallback("dandelion"));
    builder.insert("torch", load_texture_with_fallback("torch"));
    builder.insert(
        "short_grass",
        stack
            .load_block_texture("short_grass")
            .or_else(|_| stack.load_block_texture("tall_grass"))
            .unwrap_or_else(|_| load_texture_with_fallback("short_grass")),
    );
    builder.insert("fern", load_texture_with_fallback("fern"));
    builder.insert("dead_bush", load_texture_with_fallback("dead_bush"));
    builder.insert(
        "redstone_torch",
        load_texture_with_fallback("redstone_torch"),
    );
    builder.insert(
        "redstone_torch_off",
        load_texture_with_fallback("redstone_torch_off"),
    );
    builder.insert(
        "redstone_block",
        load_texture_with_fallback("redstone_block"),
    );
    builder.insert("redstone_lamp", load_texture_with_fallback("redstone_lamp"));
    builder.insert(
        "redstone_lamp_on",
        load_texture_with_fallback("redstone_lamp_on"),
    );
    builder.insert(
        "redstone_dust_line0",
        load_texture_with_fallback("redstone_dust_line0"),
    );
    builder.insert("repeater", load_texture_with_fallback("repeater"));
    builder.insert("repeater_on", load_texture_with_fallback("repeater_on"));
    builder.insert("lever", load_texture_with_fallback("lever"));
    builder.insert("oak_log", load_texture_with_fallback("oak_log"));
    builder.insert("oak_log_top", load_texture_with_fallback("oak_log_top"));
    builder.insert("birch_log", load_texture_with_fallback("birch_log"));
    builder.insert("birch_log_top", load_texture_with_fallback("birch_log_top"));
    builder.insert("spruce_log", load_texture_with_fallback("spruce_log"));
    builder.insert(
        "spruce_log_top",
        load_texture_with_fallback("spruce_log_top"),
    );
    builder.insert("birch_leaves", load_texture_with_fallback("birch_leaves"));
    builder.insert("spruce_leaves", load_texture_with_fallback("spruce_leaves"));
    builder.insert("tall_grass", load_texture_with_fallback("tall_grass"));
    builder.insert("cornflower", load_texture_with_fallback("cornflower"));
    builder.insert("oxeye_daisy", load_texture_with_fallback("oxeye_daisy"));
    builder.insert(
        "brown_mushroom",
        load_texture_with_fallback("brown_mushroom"),
    );
    builder.insert("red_mushroom", load_texture_with_fallback("red_mushroom"));

    let mut load_animated = |name: &str,
                             alt_name: Option<&str>,
                             fallback_color: [u8; 4],
                             default_frametime: u32|
     -> AnimatedTextureInfo {
        let res = stack.load_block_texture_with_animation(name).or_else(|_| {
            if let Some(alt) = alt_name {
                stack.load_block_texture_with_animation(alt)
            } else {
                Err(telos_assets::AssetError::MissingTexture("missing".into()))
            }
        });

        let (frames, mcmeta) = res.unwrap_or_else(|_| {
            let mut img = telos_assets::RgbaImage::new(target_res, target_res);
            for pixel in img.data.as_chunks_mut::<4>().0 {
                *pixel = fallback_color;
            }
            (vec![img], None)
        });
        let mut anim_def = mcmeta;
        if anim_def.is_none() && default_frametime > 1 {
            anim_def = Some(telos_assets::AnimationDef {
                frametime: default_frametime,
                interpolate: false,
                frames: None,
            });
        }
        builder.insert_animated_with_meta(name, &frames, anim_def.as_ref())
    };

    let water_still_anim = load_animated("water_still", None, [40, 80, 200, 200], 2);
    let water_flow_anim = load_animated("water_flow", None, [40, 80, 200, 200], 2);
    let lava_still_anim = load_animated("lava_still", None, [230, 90, 10, 255], 2);
    let lava_flow_anim = load_animated("lava_flow", None, [230, 90, 10, 255], 2);
    let fire_anim = load_animated("fire_0", Some("fire"), [240, 140, 20, 255], 1);
    let nether_portal_anim = load_animated("nether_portal", None, [150, 40, 200, 200], 1);

    let anim_textures = AnimatedTextures {
        water_still: water_still_anim,
        water_flow: water_flow_anim,
        lava_still: lava_still_anim,
        lava_flow: lava_flow_anim,
        fire: fire_anim,
        nether_portal: nether_portal_anim,
    };

    let baked = builder.bake();
    info!(
        resolution = baked.resolution,
        layers = baked.layer_count,
        mips = baked.mip_levels,
        "Texture array baked with full mip chains"
    );

    (baked, anim_textures)
}

fn load_and_upload_textures(
    gpu_context: &GpuContext,
) -> Result<(GpuTextureArray, AnimatedTextures)> {
    let (baked, anim_textures) = bake_block_textures();
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

    Ok((texture_array, anim_textures))
}

fn load_and_upload_textures_gl(
    gl: &Arc<glow::Context>,
) -> Result<(telos_gpu::opengl::GlTextureArray, AnimatedTextures)> {
    let (baked, anim_textures) = bake_block_textures();
    let gl_array = telos_gpu::opengl::GlTextureArray::new(
        gl.clone(),
        baked.resolution,
        baked.resolution,
        baked.layer_count,
        baked.mip_levels,
    )?;

    for r in &baked.copy_regions {
        let start = r.buffer_offset as usize;
        let len = (r.width * r.height * 4) as usize;
        if start + len <= baked.pixel_data.len() {
            gl_array.upload_mip_region(
                r.mip_level,
                r.layer,
                r.width,
                r.height,
                &baked.pixel_data[start..start + len],
            );
        }
    }

    Ok((gl_array, anim_textures))
}

#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
fn bake_ui_textures() -> (Vec<u8>, Vec<TextureMipRegion>, BitmapFont) {
    const UI_RES: u32 = 256;

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    let mut pixel_data = vec![0u8; (UI_RES * UI_RES * 4 * 11) as usize];

    // Helper to copy a sub-image into a 256x256 layer at specified offset
    let copy_to_layer_at = |dest: &mut [u8],
                            layer: usize,
                            x_offset: usize,
                            y_offset: usize,
                            img: &telos_assets::RgbaImage| {
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

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
        copy_to_layer_at(dest, layer, 0, 0, img);
    };

    let ensure_size =
        |img: telos_assets::RgbaImage, target_w: u32, target_h: u32| -> telos_assets::RgbaImage {
            if img.width != target_w || img.height != target_h {
                img.rescale(target_w, target_h)
            } else {
                img
            }
        };

    // Layer 0: Hotbar (182x22)
    let hotbar_img = stack.load_gui_sprite("hud/hotbar").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(182, 22);
        for i in (0..img.data.len()).step_by(4) {
            img.data[i] = 40;
            img.data[i + 1] = 40;
            img.data[i + 2] = 40;
            img.data[i + 3] = 200;
        }
        img
    });
    let hotbar_img = ensure_size(hotbar_img, 182, 22);
    copy_to_layer(&mut pixel_data, 0, &hotbar_img);

    // Layer 1: Hotbar Selection (24x23)
    let selection_img = stack
        .load_gui_sprite("hud/hotbar_selection")
        .unwrap_or_else(|_| {
            let mut img = telos_assets::RgbaImage::new(24, 23);
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
    let selection_img = ensure_size(selection_img, 24, 23);
    copy_to_layer(&mut pixel_data, 1, &selection_img);

    // Layer 2: Crosshair (15x15)
    let crosshair_img = stack.load_gui_sprite("hud/crosshair").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(15, 15);
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
    let crosshair_img = ensure_size(crosshair_img, 15, 15);
    copy_to_layer(&mut pixel_data, 2, &crosshair_img);

    // Layer 3: Ascii Font (128x128 -> scale 2x to 256x256)
    let ascii_img = stack.load_font_texture("ascii").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(128, 128);
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
            let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let heart_container_img = ensure_size(heart_container_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 0, 0, &heart_container_img);

    let heart_full_img = stack.load_gui_sprite("hud/heart/full").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let heart_full_img = ensure_size(heart_full_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 16, 0, &heart_full_img);

    let heart_half_img = stack.load_gui_sprite("hud/heart/half").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let heart_half_img = ensure_size(heart_half_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 32, 0, &heart_half_img);

    let food_empty_img = stack.load_gui_sprite("hud/food_empty").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let food_empty_img = ensure_size(food_empty_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 48, 0, &food_empty_img);

    let food_full_img = stack.load_gui_sprite("hud/food_full").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let food_full_img = ensure_size(food_full_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 64, 0, &food_full_img);

    let food_half_img = stack.load_gui_sprite("hud/food_half").unwrap_or_else(|_| {
        let mut img = telos_assets::RgbaImage::new(9, 9);
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
    let food_half_img = ensure_size(food_half_img, 9, 9);
    copy_to_layer_at(&mut pixel_data, 4, 80, 0, &food_half_img);

    let xp_bar_bg_img = stack
        .load_gui_sprite("hud/experience_bar_background")
        .unwrap_or_else(|_| {
            let mut img = telos_assets::RgbaImage::new(182, 5);
            for i in (0..img.data.len()).step_by(4) {
                img.data[i] = 30;
                img.data[i + 1] = 30;
                img.data[i + 2] = 30;
                img.data[i + 3] = 200;
            }
            img
        });
    let xp_bar_bg_img = ensure_size(xp_bar_bg_img, 182, 5);
    copy_to_layer_at(&mut pixel_data, 4, 0, 16, &xp_bar_bg_img);

    let xp_bar_progress_img = stack
        .load_gui_sprite("hud/experience_bar_progress")
        .unwrap_or_else(|_| {
            let mut img = telos_assets::RgbaImage::new(182, 5);
            for i in (0..img.data.len()).step_by(4) {
                img.data[i] = 120;
                img.data[i + 1] = 230;
                img.data[i + 2] = 30;
                img.data[i + 3] = 255;
            }
            img
        });
    let xp_bar_progress_img = ensure_size(xp_bar_progress_img, 182, 5);
    copy_to_layer_at(&mut pixel_data, 4, 0, 24, &xp_bar_progress_img);

    // Layer 5: Inventory Container Background (176x166 window on 256x256 canvas)
    let inv_bg_img = if let Some(img) = stack
        .find_texture("textures/gui/container/inventory.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 256, 256)
    } else {
        let mut img = telos_assets::RgbaImage::new(176, 166);
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
    };
    copy_to_layer_at(&mut pixel_data, 5, 0, 0, &inv_bg_img);

    // Layer 6: Item Icons Atlas (256x256 holding 16x16 icons for registered items)
    let item_textures: [(u32, &str, [u8; 4]); 54] = [
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
        (49, "textures/item/porkchop.png", [230, 150, 140, 255]),
        (50, "textures/item/beef.png", [180, 60, 50, 255]),
        (52, "textures/item/iron_ingot.png", [210, 210, 210, 255]),
        (53, "textures/item/gold_ingot.png", [250, 220, 60, 255]),
        (54, "textures/item/coal.png", [40, 40, 40, 255]),
        (61, "textures/item/bow.png", [140, 100, 50, 255]),
        (62, "textures/item/arrow.png", [200, 200, 200, 255]),
        (63, "textures/block/chest_front.png", [150, 100, 40, 255]),
        (64, "textures/block/furnace_front.png", [100, 100, 100, 255]),
        (65, "textures/item/cooked_porkchop.png", [190, 110, 80, 255]),
        (66, "textures/item/cooked_beef.png", [130, 70, 50, 255]),
        (67, "textures/item/charcoal.png", [45, 45, 45, 255]),
        (68, "textures/item/wooden_pickaxe.png", [160, 130, 80, 255]),
        (69, "textures/item/stone_pickaxe.png", [130, 130, 130, 255]),
        (70, "textures/item/iron_pickaxe.png", [220, 220, 220, 255]),
        (71, "textures/item/wooden_axe.png", [160, 130, 80, 255]),
        (72, "textures/item/stone_axe.png", [130, 130, 130, 255]),
        (73, "textures/item/iron_axe.png", [220, 220, 220, 255]),
        (74, "textures/item/wooden_shovel.png", [160, 130, 80, 255]),
        (75, "textures/item/stone_shovel.png", [130, 130, 130, 255]),
        (76, "textures/item/iron_shovel.png", [220, 220, 220, 255]),
        (77, "textures/item/wooden_sword.png", [160, 130, 80, 255]),
        (78, "textures/item/stone_sword.png", [130, 130, 130, 255]),
        (79, "textures/item/iron_sword.png", [220, 220, 220, 255]),
        (80, "textures/item/wooden_hoe.png", [160, 130, 80, 255]),
        (81, "textures/item/stone_hoe.png", [130, 130, 130, 255]),
        (82, "textures/item/iron_hoe.png", [220, 220, 220, 255]),
        (83, "textures/item/leather_helmet.png", [150, 90, 50, 255]),
        (
            84,
            "textures/item/leather_chestplate.png",
            [150, 90, 50, 255],
        ),
        (85, "textures/item/leather_leggings.png", [150, 90, 50, 255]),
        (86, "textures/item/leather_boots.png", [150, 90, 50, 255]),
        (87, "textures/item/golden_helmet.png", [240, 210, 60, 255]),
        (
            88,
            "textures/item/golden_chestplate.png",
            [240, 210, 60, 255],
        ),
        (89, "textures/item/golden_leggings.png", [240, 210, 60, 255]),
        (90, "textures/item/golden_boots.png", [240, 210, 60, 255]),
    ];

    let copy_icon =
        |dest: &mut [u8], x_offset: usize, y_offset: usize, img: &telos_assets::RgbaImage| {
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

        let icon_img = if let Some(img) = stack
            .find_texture(rel_path)
            .and_then(|p| telos_assets::RgbaImage::from_file(&p).ok())
        {
            ensure_size(img, 16, 16)
        } else {
            let mut img = telos_assets::RgbaImage::new(16, 16);
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
        };
        copy_icon(&mut pixel_data, x_offset, y_offset, &icon_img);
    }

    // Layer 7: Furnace Container Background and Sprites
    let furnace_bg_img = if let Some(img) = stack
        .find_texture("textures/gui/container/furnace.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 256, 256)
    } else {
        let mut img = telos_assets::RgbaImage::new(176, 166);
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
    };
    copy_to_layer_at(&mut pixel_data, 7, 0, 0, &furnace_bg_img);

    // Lit flame sprite (14x14 at (176, 0))
    let flame_img = if let Some(img) = stack
        .find_texture("textures/gui/sprites/container/furnace/lit_progress.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 14, 14)
    } else {
        let mut img = telos_assets::RgbaImage::new(14, 14);
        for y in 0..14 {
            for x in 0..14 {
                let idx = ((y * 14 + x) * 4) as usize;
                img.data[idx] = 255;
                img.data[idx + 1] = 160;
                img.data[idx + 2] = 20;
                img.data[idx + 3] = 255;
            }
        }
        img
    };
    copy_to_layer_at(&mut pixel_data, 7, 176, 0, &flame_img);

    // Cook progress arrow sprite (24x16 at (176, 16))
    let arrow_img = if let Some(img) = stack
        .find_texture("textures/gui/sprites/container/furnace/burn_progress.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 24, 16)
    } else {
        let mut img = telos_assets::RgbaImage::new(24, 16);
        for y in 0..16 {
            for x in 0..24 {
                let idx = ((y * 24 + x) * 4) as usize;
                img.data[idx] = 255;
                img.data[idx + 1] = 255;
                img.data[idx + 2] = 255;
                img.data[idx + 3] = 255;
            }
        }
        img
    };
    copy_to_layer_at(&mut pixel_data, 7, 176, 16, &arrow_img);

    // Layer 8: Crafting Table Container Background (176x166 window on 256x256 canvas)
    let crafting_table_bg_img = if let Some(img) = stack
        .find_texture("textures/gui/container/crafting_table.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 256, 256)
    } else {
        let mut img = telos_assets::RgbaImage::new(176, 166);
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
    };
    copy_to_layer_at(&mut pixel_data, 8, 0, 0, &crafting_table_bg_img);

    // Layer 9: Advancements Window Background (252x140 window on 256x256 canvas)
    let adv_window_img = if let Some(img) = stack
        .find_texture("textures/gui/advancements/window.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 252, 140)
    } else {
        let mut img = telos_assets::RgbaImage::new(252, 140);
        for y in 0..140 {
            for x in 0..252 {
                let idx = ((y * 252 + x) * 4) as usize;
                let is_border = x == 0 || x == 251 || y == 0 || y == 139;
                let is_inner_border = x == 9 || x == 242 || y == 18 || y == 130;
                let color = if is_border || is_inner_border {
                    40
                } else {
                    198
                };
                img.data[idx] = color;
                img.data[idx + 1] = color;
                img.data[idx + 2] = color;
                img.data[idx + 3] = 255;
            }
        }
        img
    };
    copy_to_layer_at(&mut pixel_data, 9, 0, 0, &adv_window_img);

    // Layer 10: Advancement Toast Notification Banner (160x32 on 256x256 canvas)
    let toast_bg_img = if let Some(img) = stack
        .find_texture("textures/gui/sprites/toast/advancement.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        ensure_size(img, 160, 32)
    } else {
        let mut img = telos_assets::RgbaImage::new(160, 32);
        for y in 0..32 {
            for x in 0..160 {
                let idx = ((y * 160 + x) * 4) as usize;
                let is_border = x == 0 || x == 159 || y == 0 || y == 31;
                let color = if is_border { 50 } else { 210 };
                img.data[idx] = color;
                img.data[idx + 1] = color;
                img.data[idx + 2] = color;
                img.data[idx + 3] = 245;
            }
        }
        img
    };
    copy_to_layer_at(&mut pixel_data, 10, 0, 0, &toast_bg_img);

    let regions: Vec<TextureMipRegion> = (0..11)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * UI_RES * UI_RES * 4),
            layer,
            mip_level: 0,
            width: UI_RES,
            height: UI_RES,
        })
        .collect();

    (pixel_data, regions, font)
}

fn load_and_upload_ui_textures(gpu_context: &GpuContext) -> Result<(GpuTextureArray, BitmapFont)> {
    const UI_RES: u32 = 256;
    let (pixel_data, regions, font) = bake_ui_textures();
    let texture_array = gpu_context.create_texture_array(UI_RES, 11, 1, &pixel_data, &regions)?;

    info!(
        "UI texture array loaded (11 layers, 256x256, font baked, survival icons, inventory background, item icons, furnace GUI, crafting table GUI, advancements window, toast banner)"
    );

    Ok((texture_array, font))
}

fn load_and_upload_ui_textures_gl(
    gl: &Arc<glow::Context>,
) -> Result<(telos_gpu::opengl::GlTextureArray, BitmapFont)> {
    const UI_RES: u32 = 256;
    let (pixel_data, regions, font) = bake_ui_textures();
    let gl_array = telos_gpu::opengl::GlTextureArray::new(gl.clone(), UI_RES, UI_RES, 11, 1)?;

    for r in &regions {
        let start = r.buffer_offset as usize;
        let len = (r.width * r.height * 4) as usize;
        if start + len <= pixel_data.len() {
            gl_array.upload_mip_region(
                0,
                r.layer,
                r.width,
                r.height,
                &pixel_data[start..start + len],
            );
        }
    }

    Ok((gl_array, font))
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

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
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

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    // Layer 0: Sun
    let sun_img = stack
        .find_texture("textures/environment/celestial/sun.png")
        .and_then(|p| telos_assets::RgbaImage::from_file(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(CELESTIAL_RES, CELESTIAL_RES);
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
        let moon_img = stack
            .find_texture(&format!("textures/environment/celestial/moon/{name}.png"))
            .and_then(|p| telos_assets::RgbaImage::from_file(&p).ok())
            .unwrap_or_else(|| {
                let mut img = telos_assets::RgbaImage::new(CELESTIAL_RES, CELESTIAL_RES);
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
fn load_and_upload_particle_textures(gpu_context: &GpuContext) -> Result<GpuTextureArray> {
    const PARTICLE_RES: u32 = 16;
    const LAYER_COUNT: u32 = 11;

    let mut pixel_data = vec![0u8; (PARTICLE_RES * PARTICLE_RES * 4 * LAYER_COUNT) as usize];

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
        let layer_offset = layer * (PARTICLE_RES * PARTICLE_RES * 4) as usize;
        for y in 0..PARTICLE_RES {
            let src_y = if img.height > 0 {
                (y * img.height) / PARTICLE_RES
            } else {
                0
            };
            for x in 0..PARTICLE_RES {
                let src_x = if img.width > 0 {
                    (x * img.width) / PARTICLE_RES
                } else {
                    0
                };
                let src_idx = ((src_y * img.width + src_x) * 4) as usize;
                let dst_idx = layer_offset + ((y * PARTICLE_RES + x) * 4) as usize;
                if src_idx + 4 <= img.data.len() && dst_idx + 4 <= dest.len() {
                    dest[dst_idx..dst_idx + 4].copy_from_slice(&img.data[src_idx..src_idx + 4]);
                }
            }
        }
    };

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    // Layer 0..=7: generic puffs
    for i in 0..8 {
        let path = format!("textures/particle/generic_{i}.png");
        let puff_img = stack
            .find_texture(&path)
            .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
            .unwrap_or_else(|| {
                let mut img = telos_assets::RgbaImage::new(PARTICLE_RES, PARTICLE_RES);
                let radius = 2.5 + (i as f32) * 0.7;
                let center = 7.5f32;
                for y in 0..PARTICLE_RES {
                    for x in 0..PARTICLE_RES {
                        let dx = x as f32 - center;
                        let dy = y as f32 - center;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist <= radius {
                            let idx = ((y * PARTICLE_RES + x) * 4) as usize;
                            let alpha =
                                (230.0 * (1.0 - dist / (radius + 1.0)).clamp(0.0, 1.0)) as u8;
                            img.data[idx] = 230;
                            img.data[idx + 1] = 230;
                            img.data[idx + 2] = 230;
                            img.data[idx + 3] = alpha;
                        }
                    }
                }
                img
            });
        copy_to_layer(&mut pixel_data, i, &puff_img);
    }

    // Layer 8: Flame
    let flame_img = stack
        .find_texture("textures/particle/flame.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(PARTICLE_RES, PARTICLE_RES);
            for y in 0..PARTICLE_RES {
                for x in 0..PARTICLE_RES {
                    let dx = (x as f32 - 7.5).abs();
                    let dy = y as f32;
                    let idx = ((y * PARTICLE_RES + x) * 4) as usize;
                    if dy > 3.0 && dx <= (dy - 3.0) * 0.7 && dy < 14.0 {
                        img.data[idx] = 255;
                        img.data[idx + 1] = if dy < 8.0 { 240 } else { 120 };
                        img.data[idx + 2] = if dy < 6.0 { 180 } else { 30 };
                        img.data[idx + 3] = 255;
                    }
                }
            }
            img
        });
    copy_to_layer(&mut pixel_data, 8, &flame_img);

    // Layer 9: Spark / Crit
    let spark_img = stack
        .find_texture("textures/particle/spark_0.png")
        .or_else(|| stack.find_texture("textures/particle/critical_hit.png"))
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(PARTICLE_RES, PARTICLE_RES);
            for y in 0..PARTICLE_RES {
                for x in 0..PARTICLE_RES {
                    let idx = ((y * PARTICLE_RES + x) * 4) as usize;
                    let is_cross = (x == 7 || x == 8) || (y == 7 || y == 8);
                    let is_center = (6..=9).contains(&x) && (6..=9).contains(&y);
                    if is_cross || is_center {
                        img.data[idx] = 255;
                        img.data[idx + 1] = 245;
                        img.data[idx + 2] = 160;
                        img.data[idx + 3] = 255;
                    }
                }
            }
            img
        });
    copy_to_layer(&mut pixel_data, 9, &spark_img);

    // Layer 10: Heart
    let heart_img = stack
        .find_texture("textures/particle/heart.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(PARTICLE_RES, PARTICLE_RES);
            for y in 0..PARTICLE_RES {
                for x in 0..PARTICLE_RES {
                    let idx = ((y * PARTICLE_RES + x) * 4) as usize;
                    let is_heart = ((4..=11).contains(&y) && (3..=12).contains(&x))
                        && !(y == 4 && (x == 7 || x == 8))
                        && !(y > 8 && (x < (y - 8) + 3 || x > 12 - (y - 8)));
                    if is_heart {
                        img.data[idx] = 235;
                        img.data[idx + 1] = 35;
                        img.data[idx + 2] = 55;
                        img.data[idx + 3] = 255;
                    }
                }
            }
            img
        });
    copy_to_layer(&mut pixel_data, 10, &heart_img);

    let regions: Vec<TextureMipRegion> = (0..LAYER_COUNT)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * PARTICLE_RES * PARTICLE_RES * 4),
            layer,
            mip_level: 0,
            width: PARTICLE_RES,
            height: PARTICLE_RES,
        })
        .collect();

    let texture_array =
        gpu_context.create_texture_array(PARTICLE_RES, LAYER_COUNT, 1, &pixel_data, &regions)?;

    info!(
        "Particle billboard texture array loaded (11 layers: 0..7 puff, 8 flame, 9 spark, 10 heart, 16x16)"
    );
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

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
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

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    // Layer 0: Rain
    let rain_img = stack
        .find_texture("textures/environment/rain.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(WEATHER_W, WEATHER_H);
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
    let snow_img = stack
        .find_texture("textures/environment/snow.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
        .unwrap_or_else(|| {
            let mut img = telos_assets::RgbaImage::new(WEATHER_W, WEATHER_H);
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

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
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

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    // Layer 0: Zombie
    let zombie_img = if let Some(img) = stack
        .find_texture("textures/entity/zombie/zombie.png")
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        if img.width != ENTITY_RES || img.height != ENTITY_RES {
            img.rescale(ENTITY_RES, ENTITY_RES)
        } else {
            img
        }
    } else {
        let mut img = telos_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
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
    };
    copy_to_layer(&mut pixel_data, 0, &zombie_img);

    // Layer 1: Pig
    let pig_img = if let Some(img) = stack
        .find_texture("textures/entity/pig/pig_temperate.png")
        .or_else(|| stack.find_texture("textures/entity/pig/pig.png"))
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        if img.width != ENTITY_RES || img.height != ENTITY_RES {
            img.rescale(ENTITY_RES, ENTITY_RES)
        } else {
            img
        }
    } else {
        let mut img = telos_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
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
    };
    copy_to_layer(&mut pixel_data, 1, &pig_img);

    // Layer 2: Cow
    let cow_img = if let Some(img) = stack
        .find_texture("textures/entity/cow/cow_temperate.png")
        .or_else(|| stack.find_texture("textures/entity/cow/cow.png"))
        .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
    {
        if img.width != ENTITY_RES || img.height != ENTITY_RES {
            img.rescale(ENTITY_RES, ENTITY_RES)
        } else {
            img
        }
    } else {
        let mut img = telos_assets::RgbaImage::new(ENTITY_RES, ENTITY_RES);
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
    };
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

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn load_and_upload_panorama_textures(gpu_context: &GpuContext) -> Result<GpuTextureArray> {
    const PANO_RES: u32 = 512;
    const LAYER_COUNT: u32 = 6;

    let mut pixel_data = vec![0u8; (PANO_RES * PANO_RES * 4 * LAYER_COUNT) as usize];

    let copy_to_layer = |dest: &mut [u8], layer: usize, img: &telos_assets::RgbaImage| {
        let layer_offset = layer * (PANO_RES * PANO_RES * 4) as usize;
        let w = img.width.min(PANO_RES);
        let h = img.height.min(PANO_RES);
        for y in 0..h {
            let src_start = ((y * img.width) * 4) as usize;
            let src_end = src_start + (w * 4) as usize;
            let dst_start = layer_offset + ((y * PANO_RES) * 4) as usize;
            dest[dst_start..dst_start + (w * 4) as usize]
                .copy_from_slice(&img.data[src_start..src_end]);
        }
    };

    let mut stack = ResourcePackStack::new();
    mount_asset_roots(&mut stack);

    let pano_names = [
        "textures/gui/title/background/panorama_0.png",
        "textures/gui/title/background/panorama_1.png",
        "textures/gui/title/background/panorama_2.png",
        "textures/gui/title/background/panorama_3.png",
        "textures/gui/title/background/panorama_4.png",
        "textures/gui/title/background/panorama_5.png",
    ];

    for (layer, name) in pano_names.iter().enumerate() {
        let pano_img = if let Some(img) = stack
            .find_texture(name)
            .and_then(|p| telos_assets::RgbaImage::from_file_exact(&p).ok())
            .filter(|img| img.width >= 16 && img.height >= 16)
        {
            if img.width != PANO_RES || img.height != PANO_RES {
                img.rescale(PANO_RES, PANO_RES)
            } else {
                img
            }
        } else {
            let mut img = telos_assets::RgbaImage::new(PANO_RES, PANO_RES);
            for y in 0..PANO_RES {
                let v = y as f32 / PANO_RES as f32;
                for x in 0..PANO_RES {
                    let idx = ((y * PANO_RES + x) * 4) as usize;
                    let col = match layer {
                        4 => [70, 140, 240, 255],
                        5 => [50, 110, 45, 255],
                        _ => {
                            let r = (50.0 * (1.0 - v) + 120.0 * v) as u8;
                            let g = (100.0 * (1.0 - v) + 180.0 * v) as u8;
                            let b = (200.0 * (1.0 - v) + 100.0 * v) as u8;
                            [r, g, b, 255]
                        }
                    };
                    img.data[idx..idx + 4].copy_from_slice(&col);
                }
            }
            img
        };

        copy_to_layer(&mut pixel_data, layer, &pano_img);
    }

    let regions: Vec<TextureMipRegion> = (0..LAYER_COUNT)
        .map(|layer| TextureMipRegion {
            buffer_offset: u64::from(layer * PANO_RES * PANO_RES * 4),
            layer,
            mip_level: 0,
            width: PANO_RES,
            height: PANO_RES,
        })
        .collect();

    let texture_array =
        gpu_context.create_texture_array(PANO_RES, LAYER_COUNT, 1, &pixel_data, &regions)?;

    info!("3D Panorama cubemap texture array loaded (6 layers, {PANO_RES}x{PANO_RES})");

    Ok(texture_array)
}

fn main() -> Result<()> {
    let args = Args::parse();

    init_telemetry(&TelemetryConfig {
        app_name: "telos-client",
        default_filter: args.log.clone().into(),
        ansi_colors: true,
    });

    info!(
        version = env!("CARGO_PKG_VERSION"),
        seed = args.seed,
        view_distance = args.view_distance,
        portable = args.portable,
        world_dir = ?args.world_dir,
        "Starting Telos client"
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(args);
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
