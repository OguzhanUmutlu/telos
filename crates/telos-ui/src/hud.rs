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
    /// Survival icons and bars texture layer.
    pub icons: u32,
    /// Heart container (empty heart) UV bounds `[u0, v0, u1, v1]`.
    pub heart_container_uv: [f32; 4],
    /// Full heart UV bounds `[u0, v0, u1, v1]`.
    pub heart_full_uv: [f32; 4],
    /// Half heart UV bounds `[u0, v0, u1, v1]`.
    pub heart_half_uv: [f32; 4],
    /// Food empty drumstick UV bounds `[u0, v0, u1, v1]`.
    pub food_empty_uv: [f32; 4],
    /// Food full drumstick UV bounds `[u0, v0, u1, v1]`.
    pub food_full_uv: [f32; 4],
    /// Food half drumstick UV bounds `[u0, v0, u1, v1]`.
    pub food_half_uv: [f32; 4],
    /// Experience bar background UV bounds `[u0, v0, u1, v1]`.
    pub xp_bar_bg_uv: [f32; 4],
    /// Experience bar progress fill UV bounds `[u0, v0, u1, v1]`.
    pub xp_bar_progress_uv: [f32; 4],
    /// Inventory container background texture layer.
    pub inventory_bg: u32,
    /// Inventory container background UV bounds `[u0, v0, u1, v1]`.
    pub inventory_bg_uv: [f32; 4],
    /// Item icons atlas texture layer.
    pub item_icons: u32,
    /// Furnace container background and sprite texture layer.
    pub furnace_bg: u32,
    /// Furnace container background UV bounds `[u0, v0, u1, v1]`.
    pub furnace_bg_uv: [f32; 4],
    /// Furnace lit flame sprite UV bounds `[u0, v0, u1, v1]`.
    pub furnace_flame_uv: [f32; 4],
    /// Furnace cook progress arrow sprite UV bounds `[u0, v0, u1, v1]`.
    pub furnace_arrow_uv: [f32; 4],
    /// Crafting table container background texture layer.
    pub crafting_table_bg: u32,
    /// Crafting table container background UV bounds `[u0, v0, u1, v1]`.
    pub crafting_table_bg_uv: [f32; 4],
    /// Advancement tree window texture layer.
    pub advancement_window: u32,
    /// Advancement tree window UV bounds `[u0, v0, u1, v1]`.
    pub advancement_window_uv: [f32; 4],
    /// Toast banner notification texture layer.
    pub toast_bg: u32,
    /// Toast banner notification UV bounds `[u0, v0, u1, v1]`.
    pub toast_bg_uv: [f32; 4],
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
            icons: 4,
            heart_container_uv: [0.0 / 256.0, 0.0 / 256.0, 9.0 / 256.0, 9.0 / 256.0],
            heart_full_uv: [16.0 / 256.0, 0.0 / 256.0, 25.0 / 256.0, 9.0 / 256.0],
            heart_half_uv: [32.0 / 256.0, 0.0 / 256.0, 41.0 / 256.0, 9.0 / 256.0],
            food_empty_uv: [48.0 / 256.0, 0.0 / 256.0, 57.0 / 256.0, 9.0 / 256.0],
            food_full_uv: [64.0 / 256.0, 0.0 / 256.0, 73.0 / 256.0, 9.0 / 256.0],
            food_half_uv: [80.0 / 256.0, 0.0 / 256.0, 89.0 / 256.0, 9.0 / 256.0],
            xp_bar_bg_uv: [0.0 / 256.0, 16.0 / 256.0, 182.0 / 256.0, 21.0 / 256.0],
            xp_bar_progress_uv: [0.0 / 256.0, 24.0 / 256.0, 182.0 / 256.0, 29.0 / 256.0],
            inventory_bg: 5,
            inventory_bg_uv: [0.0, 0.0, 176.0 / 256.0, 166.0 / 256.0],
            item_icons: 6,
            furnace_bg: 7,
            furnace_bg_uv: [0.0, 0.0, 176.0 / 256.0, 166.0 / 256.0],
            furnace_flame_uv: [176.0 / 256.0, 0.0 / 256.0, 190.0 / 256.0, 14.0 / 256.0],
            furnace_arrow_uv: [176.0 / 256.0, 16.0 / 256.0, 200.0 / 256.0, 32.0 / 256.0],
            crafting_table_bg: 8,
            crafting_table_bg_uv: [0.0, 0.0, 176.0 / 256.0, 166.0 / 256.0],
            advancement_window: 9,
            advancement_window_uv: [0.0, 0.0, 252.0 / 256.0, 140.0 / 256.0],
            toast_bg: 10,
            toast_bg_uv: [0.0, 0.0, 160.0 / 256.0, 32.0 / 256.0],
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
    /// Current player health points (0.0..=20.0).
    pub health: f32,
    /// Maximum player health points.
    pub max_health: f32,
    /// Food / hunger points (0..=20).
    pub food: u32,
    /// Saturation points (0.0..=food).
    pub saturation: f32,
    /// Current experience level.
    pub xp_level: u32,
    /// Progress fraction to the next level (0.0..1.0).
    pub xp_progress: f32,
    /// Active weather condition name (e.g. "Clear", "Rain", "Thunder").
    pub weather_name: String,
    /// Active rain level in [0.0, 1.0].
    pub weather_rain_level: f32,
    /// Active thunder level in [0.0, 1.0].
    pub weather_thunder_level: f32,
    /// Local precipitation type (e.g. "None", "Rain", "Snow").
    pub local_precipitation: String,
    /// Number of active entities tracked on client.
    pub entities_rendered: u32,
    /// Number of active audio channels playing.
    pub audio_channels: u32,
    /// Current player game mode ("Survival", "Creative").
    pub game_mode: String,
    /// Whether the player is currently in flying mode.
    pub is_flying: bool,
    /// Active status effects displayed in top-right HUD corner.
    pub active_effects: Vec<HudEffectDisplay>,
    /// Active advancement toast notification sliding banner.
    pub active_toast: Option<ToastState>,
}

/// Active advancement toast notification state displayed on the HUD.
#[derive(Debug, Clone, PartialEq)]
pub struct ToastState {
    /// Identifier of the advancement.
    pub id: String,
    /// Title displayed on the toast banner.
    pub title: String,
    /// Icon item ID (rendered on the left of the banner).
    pub icon_item: u32,
    /// Frame tier (0 = Task, 1 = Goal, 2 = Challenge).
    pub frame: u8,
    /// Elapsed display time in seconds.
    pub elapsed_secs: f32,
    /// Total display duration in seconds (typically 5.0s).
    pub duration_secs: f32,
}

/// Active status effect badge presentation on client HUD.
#[derive(Debug, Clone, PartialEq)]
pub struct HudEffectDisplay {
    /// Effect numeric identifier.
    pub effect_id: u8,
    /// Human-readable effect name (e.g. "Speed II").
    pub name: String,
    /// Effect amplifier (0 = I, 1 = II, etc.).
    pub amplifier: u8,
    /// Remaining duration in ticks.
    pub duration_ticks: i32,
    /// Particle and badge accent RGB color.
    pub color: [u8; 3],
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
            health: 20.0,
            max_health: 20.0,
            food: 20,
            saturation: 5.0,
            xp_level: 0,
            xp_progress: 0.0,
            weather_name: "Clear".to_string(),
            weather_rain_level: 0.0,
            weather_thunder_level: 0.0,
            local_precipitation: "None".to_string(),
            entities_rendered: 0,
            audio_channels: 0,
            game_mode: "Creative".to_string(),
            is_flying: true,
            active_effects: Vec::new(),
            active_toast: None,
        }
    }
}

/// Generates all HUD quads for the active frame.
#[allow(
    clippy::too_many_lines,
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

    // 4. Experience Bar (182x5) sitting directly above hotbar
    let xp_w = hotbar_w;
    let xp_h = to_physical_pixels(5, gui_scale) as u16;
    let xp_x = hotbar_x;
    let xp_y = hotbar_y - to_physical_pixels(6, gui_scale);

    out.push(UiQuad::sprite(
        [xp_x, xp_y],
        [xp_w, xp_h],
        [layers.xp_bar_bg_uv[0], layers.xp_bar_bg_uv[1]],
        [layers.xp_bar_bg_uv[2], layers.xp_bar_bg_uv[3]],
        layers.icons,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    let progress_clamped = state.xp_progress.clamp(0.0, 1.0);
    if progress_clamped > 0.0 {
        let prog_logical = (182.0 * progress_clamped).round() as i32;
        if prog_logical > 0 {
            let prog_w = to_physical_pixels(prog_logical, gui_scale) as u16;
            let u0 = layers.xp_bar_progress_uv[0];
            let u1 = u0 + (layers.xp_bar_progress_uv[2] - u0) * (prog_logical as f32 / 182.0);
            out.push(UiQuad::sprite(
                [xp_x, xp_y],
                [prog_w, xp_h],
                [u0, layers.xp_bar_progress_uv[1]],
                [u1, layers.xp_bar_progress_uv[3]],
                layers.icons,
                UiQuad::rgba(255, 255, 255, 255),
            ));
        }
    }

    // Experience Level number centered above the bar
    if state.xp_level > 0 {
        let lvl_str = format!("{}", state.xp_level);
        let (txt_w, _) = font.measure_text(&lvl_str);
        let lvl_x = (sw as f32 / gui_scale as f32 - txt_w) / 2.0;
        let lvl_y = (xp_y as f32 / gui_scale as f32) - 6.0;
        font.layout_text(
            &lvl_str,
            lvl_x,
            lvl_y,
            UiQuad::rgba(128, 255, 32, 255),
            true,
            gui_scale,
            out,
        );
    }

    // 5. Health Hearts (10 containers) anchored left above XP bar
    let icon_sz = to_physical_pixels(9, gui_scale) as u16;
    let icons_y = xp_y - to_physical_pixels(10, gui_scale);
    let half_hearts = state.health.round() as i32;

    for i in 0..10 {
        let heart_x = hotbar_x + to_physical_pixels(i * 8, gui_scale);
        // Container
        out.push(UiQuad::sprite(
            [heart_x, icons_y],
            [icon_sz, icon_sz],
            [layers.heart_container_uv[0], layers.heart_container_uv[1]],
            [layers.heart_container_uv[2], layers.heart_container_uv[3]],
            layers.icons,
            UiQuad::rgba(255, 255, 255, 255),
        ));

        // Fill
        if half_hearts >= (i + 1) * 2 {
            out.push(UiQuad::sprite(
                [heart_x, icons_y],
                [icon_sz, icon_sz],
                [layers.heart_full_uv[0], layers.heart_full_uv[1]],
                [layers.heart_full_uv[2], layers.heart_full_uv[3]],
                layers.icons,
                UiQuad::rgba(255, 255, 255, 255),
            ));
        } else if half_hearts == i * 2 + 1 {
            out.push(UiQuad::sprite(
                [heart_x, icons_y],
                [icon_sz, icon_sz],
                [layers.heart_half_uv[0], layers.heart_half_uv[1]],
                [layers.heart_half_uv[2], layers.heart_half_uv[3]],
                layers.icons,
                UiQuad::rgba(255, 255, 255, 255),
            ));
        }
    }

    // 6. Food Drumsticks (10 icons) anchored right above XP bar (drawn right-to-left)
    let food_points = state.food as i32;
    for i in 0..10 {
        let food_x =
            hotbar_x + i32::from(hotbar_w) - to_physical_pixels((i + 1) * 8 + 1, gui_scale);
        // Container / background
        out.push(UiQuad::sprite(
            [food_x, icons_y],
            [icon_sz, icon_sz],
            [layers.food_empty_uv[0], layers.food_empty_uv[1]],
            [layers.food_empty_uv[2], layers.food_empty_uv[3]],
            layers.icons,
            UiQuad::rgba(255, 255, 255, 255),
        ));

        // Fill
        if food_points >= (i + 1) * 2 {
            out.push(UiQuad::sprite(
                [food_x, icons_y],
                [icon_sz, icon_sz],
                [layers.food_full_uv[0], layers.food_full_uv[1]],
                [layers.food_full_uv[2], layers.food_full_uv[3]],
                layers.icons,
                UiQuad::rgba(255, 255, 255, 255),
            ));
        } else if food_points == i * 2 + 1 {
            out.push(UiQuad::sprite(
                [food_x, icons_y],
                [icon_sz, icon_sz],
                [layers.food_half_uv[0], layers.food_half_uv[1]],
                [layers.food_half_uv[2], layers.food_half_uv[3]],
                layers.icons,
                UiQuad::rgba(255, 255, 255, 255),
            ));
        }
    }

    // 7. Active Status Effect Badges (top-right corner)
    if !state.active_effects.is_empty() {
        render_active_effects(state, font, screen_width, gui_scale, out);
    }

    // 8. F3 Debug Overlay (if toggled on)
    if state.f3_open {
        render_f3_overlay(state, font, screen_width, gui_scale, out);
    }

    // 9. Active Advancement Toast Notification (sliding top-right banner)
    if let Some(toast) = &state.active_toast {
        render_toast(toast, layers, font, screen_width, gui_scale, out);
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn render_active_effects(
    state: &HudState,
    font: &BitmapFont,
    screen_width: u32,
    gui_scale: u32,
    out: &mut Vec<UiQuad>,
) {
    let mut cur_y = if state.f3_open { 74.0f32 } else { 4.0f32 };
    let card_w = 72.0f32;
    let card_h = 18.0f32;
    let right_pad = 4.0f32;
    let sw_logical = screen_width as f32 / gui_scale as f32;
    let card_x = sw_logical - card_w - right_pad;

    for eff in &state.active_effects {
        // Low duration blink (< 100 ticks = 5s): blink when ticks/10 is odd
        let is_blinking = eff.duration_ticks > 0 && eff.duration_ticks < 100;
        let bg_alpha = if is_blinking && ((eff.duration_ticks / 10) % 2 == 1) {
            70
        } else {
            160
        };

        let px_x = (card_x * gui_scale as f32).round() as i32;
        let px_y = (cur_y * gui_scale as f32).round() as i32;
        let px_w = (card_w * gui_scale as f32).round() as u16;
        let px_h = (card_h * gui_scale as f32).round() as u16;

        // Dark translucent background card
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(25, 25, 30, bg_alpha),
        ));

        // Colored accent bar on left edge (3 logical px wide)
        let bar_w = to_physical_pixels(3, gui_scale) as u16;
        out.push(UiQuad::solid(
            [px_x, px_y],
            [bar_w, px_h],
            UiQuad::rgba(eff.color[0], eff.color[1], eff.color[2], 255),
        ));

        // Effect title (e.g. "Speed II")
        let title_x = card_x + 6.0;
        let title_y = cur_y + 1.0;
        font.layout_text(
            &eff.name,
            title_x,
            title_y,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            gui_scale,
            out,
        );

        // Formatted duration (e.g. "0:45" or "**:**")
        let duration_str = if eff.duration_ticks < 0 {
            "**:**".to_string()
        } else {
            let total_sec = eff.duration_ticks / 20;
            let mins = total_sec / 60;
            let secs = total_sec % 60;
            format!("{mins}:{secs:02}")
        };
        let duration_y = cur_y + 9.5;
        let dur_color = if is_blinking {
            UiQuad::rgba(255, 120, 120, 255)
        } else {
            UiQuad::rgba(180, 180, 180, 255)
        };
        font.layout_text(
            &duration_str,
            title_x,
            duration_y,
            dur_color,
            true,
            gui_scale,
            out,
        );

        cur_y += card_h + 3.0;
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn render_toast(
    toast: &ToastState,
    layers: &UiLayers,
    font: &BitmapFont,
    screen_width: u32,
    gui_scale: u32,
    out: &mut Vec<UiQuad>,
) {
    let scale_f = gui_scale as f32;
    let toast_w = 160.0f32;
    let toast_h = 32.0f32;
    let sw_logical = screen_width as f32 / scale_f;
    let toast_x = sw_logical - toast_w - 8.0f32;

    // Slide-in / slide-out animation:
    // First 0.5s slides down, last 0.5s slides up
    let slide_in = (toast.elapsed_secs / 0.5).clamp(0.0, 1.0);
    let slide_out = ((toast.duration_secs - toast.elapsed_secs) / 0.5).clamp(0.0, 1.0);
    let slide = slide_in.min(slide_out);
    let y_offset = (1.0 - slide) * -(toast_h + 12.0);
    let toast_y = 6.0f32 + y_offset;

    if toast_y + toast_h <= 0.0 {
        return;
    }

    let px_x = (toast_x * scale_f).round() as i32;
    let px_y = (toast_y * scale_f).round() as i32;
    let px_w = (toast_w * scale_f).round() as u16;
    let px_h = (toast_h * scale_f).round() as u16;

    // 1. Toast banner background sprite (160x32 in toast_bg layer)
    out.push(UiQuad::sprite(
        [px_x, px_y],
        [px_w, px_h],
        [layers.toast_bg_uv[0], layers.toast_bg_uv[1]],
        [layers.toast_bg_uv[2], layers.toast_bg_uv[3]],
        layers.toast_bg,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    // 2. Icon item (16x16 icon at toast_x + 8, toast_y + 8)
    if toast.icon_item > 0 {
        let icon_x = ((toast_x + 8.0) * scale_f).round() as i32;
        let icon_y = ((toast_y + 8.0) * scale_f).round() as i32;
        let icon_sz = (16.0 * scale_f).round() as u16;
        let uv = crate::inventory::item_icon_uv(toast.icon_item);
        out.push(UiQuad::sprite(
            [icon_x, icon_y],
            [icon_sz, icon_sz],
            [uv[0], uv[1]],
            [uv[2], uv[3]],
            layers.item_icons,
            UiQuad::rgba(255, 255, 255, 255),
        ));
    }

    // 3. Header text ("Advancement Made!", "Goal Reached!", "Challenge Complete!")
    let (header_text, header_color) = match toast.frame {
        1 => ("Goal Reached!", UiQuad::rgba(85, 255, 255, 255)), // Cyan
        2 => ("Challenge Complete!", UiQuad::rgba(255, 85, 255, 255)), // Magenta
        _ => ("Advancement Made!", UiQuad::rgba(255, 255, 85, 255)), // Yellow
    };
    font.layout_text(
        header_text,
        toast_x + 30.0,
        toast_y + 7.0,
        header_color,
        true,
        gui_scale,
        out,
    );

    // 4. Advancement title text
    font.layout_text(
        &toast.title,
        toast_x + 30.0,
        toast_y + 18.0,
        UiQuad::rgba(255, 255, 255, 255),
        true,
        gui_scale,
        out,
    );
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
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
        "§fTelos 0.1.0-dev".to_string(),
        format!(
            "§fMode: §d{} §f(F4 to toggle) | Flying: §b{}",
            state.game_mode,
            if state.is_flying { "Yes" } else { "No" }
        ),
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
        format!(
            "§fHealth: §c{:.1}/{} §f| Food: §6{}/20 §f(Sat: §e{:.1}§f) | XP: §aLvl {} §f({:.0}%)",
            state.health,
            state.max_health as u32,
            state.food,
            state.saturation,
            state.xp_level,
            state.xp_progress * 100.0,
        ),
        format!(
            "§fWeather: §b{} §f(Rain: §a{:.2}§f, Thunder: §9{:.2}§f) | Precip: §e{}",
            state.weather_name,
            state.weather_rain_level,
            state.weather_thunder_level,
            state.local_precipitation,
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
        format!("§fEntities: §d{} §ftracked", state.entities_rendered),
        format!("§fAudio: §6{} §fchannels", state.audio_channels),
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
