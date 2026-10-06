//! In-game HUD layout: crosshair, 9-slot hotbar with selection indicator, and F3 debug overlay.

use crate::font::BitmapFont;
use crate::quad::UiQuad;
use crate::scale::to_physical_pixels;

/// Textures layer indices within the UI texture array.
/// Textures layer indices and UV rectangles within the UI texture array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiLayers {
    /// Hotbar frame texture layer.
    pub hotbar: u32,
    /// Hotbar frame UV bounds `[u0, v0, u1, v1]`.
    pub hotbar_uv: [f32; 4],
    /// Hotbar selection slot indicator texture layer.
    pub selection: u32,
    /// Hotbar selection UV bounds `[u0, v0, u1, v1]`.
    pub selection_uv: [f32; 4],
    /// Crosshair texture layer.
    pub crosshair: u32,
    /// Crosshair UV bounds `[u0, v0, u1, v1]`.
    pub crosshair_uv: [f32; 4],
    /// Bitmap font sheet texture layer.
    pub font: u32,
}

impl Default for UiLayers {
    fn default() -> Self {
        Self {
            hotbar: 0,
            hotbar_uv: [0.0, 0.0, 182.0 / 256.0, 22.0 / 256.0],
            selection: 1,
            selection_uv: [0.0, 0.0, 24.0 / 256.0, 23.0 / 256.0],
            crosshair: 2,
            crosshair_uv: [0.0, 0.0, 15.0 / 256.0, 15.0 / 256.0],
            font: 3,
        }
    }
}

/// Dynamic game state feeding the HUD elements.
#[derive(Debug, Clone)]
pub struct HudState {
    /// Active hotbar slot index (0..=8).
    pub selected_slot: usize,
    /// Player world position `[x, y, z]`.
    pub player_pos: [f64; 3],
    /// Player chunk coordinate `[cx, cy, cz]`.
    pub chunk_pos: [i32; 3],
    /// Camera look yaw in degrees.
    pub yaw: f32,
    /// Camera look pitch in degrees.
    pub pitch: f32,
    /// Cardinal facing direction string (e.g. `"North (-Z)"`).
    pub facing: String,
    /// Current rendered frames per second.
    pub fps: u32,
    /// Current frame time in milliseconds.
    pub frame_time_ms: f32,
    /// Number of chunk mesh layers drawn this frame.
    pub chunks_rendered: u32,
    /// Number of chunk draws culled by Hi-Z GPU compute.
    pub chunks_culled_hiz: u32,
    /// Number of LOD nodes drawn this frame.
    pub lod_nodes_rendered: u32,
    /// Physical GPU device description string.
    pub gpu_name: String,
    /// Whether the F3 debug performance overlay is toggled on.
    pub f3_open: bool,
    /// Current world time of day in ticks (0..24000).
    pub time_of_day: u64,
    /// Current world day count.
    pub day_number: u64,
    /// Current lunar phase display name.
    pub moon_phase_name: String,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            selected_slot: 0,
            player_pos: [0.0, 64.0, 0.0],
            chunk_pos: [0, 2, 0],
            yaw: 0.0,
            pitch: 0.0,
            facing: "North (-Z)".to_string(),
            fps: 60,
            frame_time_ms: 16.6,
            chunks_rendered: 0,
            chunks_culled_hiz: 0,
            lod_nodes_rendered: 0,
            gpu_name: "Vulkan Device".to_string(),
            f3_open: false,
            time_of_day: 6000,
            day_number: 0,
            moon_phase_name: "Full Moon".to_string(),
        }
    }
}

/// Generates all HUD quads for the active frame.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn render_hud(
    state: &HudState,
    font: &BitmapFont,
    layers: &UiLayers,
    screen_width: u32,
    screen_height: u32,
    gui_scale: u32,
    out: &mut Vec<UiQuad>,
) {
    let sw = screen_width as i32;
    let sh = screen_height as i32;

    // 1. Crosshair (15×15 source px) centered on screen
    let crosshair_sz = to_physical_pixels(15, gui_scale) as u16;
    let crosshair_x = (sw - i32::from(crosshair_sz)) / 2;
    let crosshair_y = (sh - i32::from(crosshair_sz)) / 2;
    out.push(UiQuad::crosshair(
        [crosshair_x, crosshair_y],
        [crosshair_sz, crosshair_sz],
        [layers.crosshair_uv[0], layers.crosshair_uv[1]],
        [layers.crosshair_uv[2], layers.crosshair_uv[3]],
        layers.crosshair,
    ));

    // 2. Hotbar (182×22 source px) anchored to bottom-center
    let hotbar_w = to_physical_pixels(182, gui_scale) as u16;
    let hotbar_h = to_physical_pixels(22, gui_scale) as u16;
    let hotbar_x = (sw - i32::from(hotbar_w)) / 2;
    let hotbar_y = sh - i32::from(hotbar_h);

    out.push(UiQuad::sprite(
        [hotbar_x, hotbar_y],
        [hotbar_w, hotbar_h],
        [layers.hotbar_uv[0], layers.hotbar_uv[1]],
        [layers.hotbar_uv[2], layers.hotbar_uv[3]],
        layers.hotbar,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    // 3. Hotbar selection indicator (24×23 source px)
    let slot = state.selected_slot.min(8) as i32;
    let sel_w = to_physical_pixels(24, gui_scale) as u16;
    let sel_h = to_physical_pixels(23, gui_scale) as u16;
    let sel_x =
        hotbar_x - to_physical_pixels(1, gui_scale) + slot * to_physical_pixels(20, gui_scale);
    let sel_y = sh - to_physical_pixels(23, gui_scale);

    out.push(UiQuad::sprite(
        [sel_x, sel_y],
        [sel_w, sel_h],
        [layers.selection_uv[0], layers.selection_uv[1]],
        [layers.selection_uv[2], layers.selection_uv[3]],
        layers.selection,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    // 4. F3 Debug Overlay (if toggled on)
    if state.f3_open {
        render_f3_overlay(state, font, screen_width, gui_scale, out);
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn render_f3_overlay(
    state: &HudState,
    font: &BitmapFont,
    screen_width: u32,
    gui_scale: u32,
    out: &mut Vec<UiQuad>,
) {
    // Left column lines
    let block_x = state.player_pos[0].floor() as i32;
    let block_y = state.player_pos[1].floor() as i32;
    let block_z = state.player_pos[2].floor() as i32;

    let total_minutes = ((state.time_of_day + 6000) % 24000) * 1440 / 24000;
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;

    let left_lines = [
        "§fVoxel Engine 0.1.0-dev".to_string(),
        format!("§fFPS: §a{} §f({:.2} ms)", state.fps, state.frame_time_ms),
        format!(
            "§fXYZ: §e{:.3} §f/ §e{:.3} §f/ §e{:.3}",
            state.player_pos[0], state.player_pos[1], state.player_pos[2]
        ),
        format!(
            "§fBlock: §b{} {} {} §fin Chunk [{}, {}, {}]",
            block_x, block_y, block_z, state.chunk_pos[0], state.chunk_pos[1], state.chunk_pos[2]
        ),
        format!(
            "§fFacing: §6{} §f(Yaw: {:.1}, Pitch: {:.1})",
            state.facing, state.yaw, state.pitch
        ),
        format!(
            "§fTime: §e{:02}:{:02} §f(Day {}, §b{}§f)",
            hours, minutes, state.day_number, state.moon_phase_name
        ),
    ];

    let mut cur_y = 2.0f32;
    for line in &left_lines {
        let (line_w, line_h) = font.measure_text(line);
        let pad_w = line_w + 4.0;
        let pad_h = line_h;

        let px_x = 2 * (gui_scale as i32);
        let px_y = (cur_y * gui_scale as f32).round() as i32;
        let px_w = (pad_w * gui_scale as f32).round() as u16;
        let px_h = (pad_h * gui_scale as f32).round() as u16;

        // Gray translucent backing bar (#90505050)
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(80, 80, 80, 144),
        ));

        // Text
        font.layout_text(
            line,
            4.0,
            cur_y + 1.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            gui_scale,
            out,
        );

        cur_y += 10.0;
    }

    // Right column lines
    let right_lines = [
        format!("§fGPU: §b{}", state.gpu_name),
        format!(
            "§fChunks: §a{} §frendered / §c{} §fculled (Hi-Z)",
            state.chunks_rendered, state.chunks_culled_hiz
        ),
        format!("§fLOD Nodes: §e{} §factive", state.lod_nodes_rendered),
        format!("§fGUI Scale: §a{gui_scale}x"),
    ];

    let mut cur_y_r = 2.0f32;
    for line in &right_lines {
        let (line_w, line_h) = font.measure_text(line);
        let pad_w = line_w + 4.0;
        let pad_h = line_h;

        let start_x_gui = (screen_width as f32 / gui_scale as f32) - pad_w - 2.0;
        let px_x = (start_x_gui * gui_scale as f32).round() as i32;
        let px_y = (cur_y_r * gui_scale as f32).round() as i32;
        let px_w = (pad_w * gui_scale as f32).round() as u16;
        let px_h = (pad_h * gui_scale as f32).round() as u16;

        // Gray translucent backing bar
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(80, 80, 80, 144),
        ));

        // Text
        font.layout_text(
            line,
            start_x_gui + 2.0,
            cur_y_r + 1.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            gui_scale,
            out,
        );

        cur_y_r += 10.0;
    }
}
