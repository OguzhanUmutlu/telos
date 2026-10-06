//! Settings and options screen allowing live adjustment of video, audio, and controls.

use crate::font::BitmapFont;
use crate::menu::widgets::{MenuButton, MenuSlider};
use crate::quad::UiQuad;
use crate::settings::GameSettings;

/// Active tab within the options screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsTab {
    /// Video and graphics options.
    #[default]
    Video,
    /// Audio volume sliders.
    Audio,
    /// Mouse and keyboard controls.
    Controls,
}

/// Settings and options screen state.
#[derive(Debug, Clone)]
pub struct SettingsScreen {
    /// Current editable settings.
    pub settings: GameSettings,
    /// Active tab.
    pub active_tab: SettingsTab,
    /// Category tab buttons.
    pub tab_buttons: Vec<MenuButton>,
    /// Sliders for current active tab.
    pub sliders: Vec<MenuSlider>,
    /// Toggle buttons for current active tab.
    pub toggle_buttons: Vec<MenuButton>,
    /// Done confirmation button.
    pub done_button: MenuButton,
    /// Dragging slider index.
    pub active_drag_slider: Option<usize>,
}

impl SettingsScreen {
    /// Creates a new `SettingsScreen` loaded with the given `GameSettings`.
    #[must_use]
    pub fn new(settings: GameSettings) -> Self {
        Self {
            settings,
            active_tab: SettingsTab::Video,
            tab_buttons: Vec::new(),
            sliders: Vec::new(),
            toggle_buttons: Vec::new(),
            done_button: MenuButton::new(99, 0.0, 0.0, 140.0, 24.0, "Done"),
            active_drag_slider: None,
        }
    }

    /// Rebuilds tab and widget layouts for the given viewport.
    #[allow(clippy::too_many_lines)]
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let center_x = width_gui * 0.5;
        let tab_w = 90.0;
        let tab_h = 22.0;
        let tab_y = 42.0;

        self.tab_buttons = vec![
            MenuButton::new(101, center_x - 140.0, tab_y, tab_w, tab_h, "Video"),
            MenuButton::new(102, center_x - 45.0, tab_y, tab_w, tab_h, "Audio"),
            MenuButton::new(103, center_x + 50.0, tab_y, tab_w, tab_h, "Controls"),
        ];

        let col_w = 145.0;
        let col1_x = center_x - col_w - 6.0;
        let col2_x = center_x + 6.0;
        let start_y = 80.0;
        let row_h = 30.0;

        self.sliders.clear();
        self.toggle_buttons.clear();

        match self.active_tab {
            SettingsTab::Video => {
                self.sliders.push(MenuSlider::new(
                    1,
                    col1_x,
                    start_y,
                    col_w,
                    22.0,
                    "Render Dist",
                    2.0,
                    32.0,
                    self.settings.video.view_distance as f32,
                    " Chunks",
                    true,
                ));
                self.sliders.push(MenuSlider::new(
                    2,
                    col2_x,
                    start_y,
                    col_w,
                    22.0,
                    "Vertical Dist",
                    2.0,
                    24.0,
                    self.settings.video.vertical_view_distance as f32,
                    " Chunks",
                    true,
                ));
                self.sliders.push(MenuSlider::new(
                    3,
                    col1_x,
                    start_y + row_h,
                    col_w,
                    22.0,
                    "FOV",
                    30.0,
                    110.0,
                    self.settings.video.fov,
                    "°",
                    true,
                ));
                self.sliders.push(MenuSlider::new(
                    4,
                    col2_x,
                    start_y + row_h,
                    col_w,
                    22.0,
                    "FPS Limit",
                    30.0,
                    240.0,
                    self.settings.video.fps_limit as f32,
                    " FPS",
                    true,
                ));

                let vsync_str = if self.settings.video.vsync {
                    "VSync: ON"
                } else {
                    "VSync: OFF"
                };
                self.toggle_buttons.push(MenuButton::new(
                    201,
                    col1_x,
                    start_y + row_h * 2.0,
                    col_w,
                    22.0,
                    vsync_str,
                ));

                let cull_str = if self.settings.video.no_cull {
                    "Hi-Z Culling: OFF"
                } else {
                    "Hi-Z Culling: ON"
                };
                self.toggle_buttons.push(MenuButton::new(
                    202,
                    col2_x,
                    start_y + row_h * 2.0,
                    col_w,
                    22.0,
                    cull_str,
                ));

                let gui_str = if self.settings.video.gui_scale == 0 {
                    "GUI Scale: Auto".to_string()
                } else {
                    format!("GUI Scale: {}", self.settings.video.gui_scale)
                };
                self.toggle_buttons.push(MenuButton::new(
                    203,
                    col1_x,
                    start_y + row_h * 3.0,
                    col_w,
                    22.0,
                    gui_str,
                ));
            }
            SettingsTab::Audio => {
                self.sliders.push(MenuSlider::new(
                    11,
                    col1_x,
                    start_y,
                    col_w,
                    22.0,
                    "Master",
                    0.0,
                    1.0,
                    self.settings.audio.master_volume,
                    "%",
                    false,
                ));
                self.sliders.push(MenuSlider::new(
                    12,
                    col2_x,
                    start_y,
                    col_w,
                    22.0,
                    "Music",
                    0.0,
                    1.0,
                    self.settings.audio.music_volume,
                    "%",
                    false,
                ));
                self.sliders.push(MenuSlider::new(
                    13,
                    col1_x,
                    start_y + row_h,
                    col_w,
                    22.0,
                    "Weather",
                    0.0,
                    1.0,
                    self.settings.audio.weather_volume,
                    "%",
                    false,
                ));
                self.sliders.push(MenuSlider::new(
                    14,
                    col2_x,
                    start_y + row_h,
                    col_w,
                    22.0,
                    "Blocks",
                    0.0,
                    1.0,
                    self.settings.audio.blocks_volume,
                    "%",
                    false,
                ));
                self.sliders.push(MenuSlider::new(
                    15,
                    col1_x,
                    start_y + row_h * 2.0,
                    col_w,
                    22.0,
                    "Entities",
                    0.0,
                    1.0,
                    self.settings.audio.entities_volume,
                    "%",
                    false,
                ));
            }
            SettingsTab::Controls => {
                self.sliders.push(MenuSlider::new(
                    21,
                    col1_x,
                    start_y,
                    col_w,
                    22.0,
                    "Sensitivity",
                    0.1,
                    3.0,
                    self.settings.controls.mouse_sensitivity,
                    "x",
                    false,
                ));

                let inv_str = if self.settings.controls.invert_mouse_y {
                    "Invert Y: ON"
                } else {
                    "Invert Y: OFF"
                };
                self.toggle_buttons
                    .push(MenuButton::new(204, col2_x, start_y, col_w, 22.0, inv_str));
            }
        }

        let done_w = 140.0;
        self.done_button = MenuButton::new(
            99,
            center_x - done_w * 0.5,
            height_gui - 34.0,
            done_w,
            24.0,
            "Done",
        );
    }

    /// Handles mouse motion in GUI pixels.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        if let Some(idx) = self.active_drag_slider {
            if let Some(slider) = self.sliders.get_mut(idx) {
                slider.update_from_mouse_x(mouse_x);
                self.sync_settings_from_sliders();
            }
            return;
        }

        for btn in &mut self.tab_buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
        for s in &mut self.sliders {
            s.hovered = s.contains(mouse_x, mouse_y);
        }
        for b in &mut self.toggle_buttons {
            b.hovered = b.contains(mouse_x, mouse_y);
        }
        self.done_button.hovered = self.done_button.contains(mouse_x, mouse_y);
    }

    /// Handles mouse button press. Returns true if "Done" was clicked.
    pub fn handle_mouse_click(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
    ) -> bool {
        if self.done_button.contains(mouse_x, mouse_y) {
            return true;
        }

        // Tab buttons
        for btn in &self.tab_buttons {
            if btn.contains(mouse_x, mouse_y) {
                match btn.id {
                    101 => self.active_tab = SettingsTab::Video,
                    102 => self.active_tab = SettingsTab::Audio,
                    103 => self.active_tab = SettingsTab::Controls,
                    _ => {}
                }
                self.update_layout(width_gui, height_gui);
                return false;
            }
        }

        // Sliders
        for (i, slider) in self.sliders.iter_mut().enumerate() {
            if slider.contains(mouse_x, mouse_y) {
                slider.dragging = true;
                slider.update_from_mouse_x(mouse_x);
                self.active_drag_slider = Some(i);
                self.sync_settings_from_sliders();
                return false;
            }
        }

        // Toggles
        for btn in &mut self.toggle_buttons {
            if btn.contains(mouse_x, mouse_y) {
                match btn.id {
                    201 => {
                        self.settings.video.vsync = !self.settings.video.vsync;
                    }
                    202 => {
                        self.settings.video.no_cull = !self.settings.video.no_cull;
                    }
                    203 => {
                        self.settings.video.gui_scale = (self.settings.video.gui_scale + 1) % 5;
                    }
                    204 => {
                        self.settings.controls.invert_mouse_y =
                            !self.settings.controls.invert_mouse_y;
                    }
                    _ => {}
                }
                self.update_layout(width_gui, height_gui);
                return false;
            }
        }

        false
    }

    /// Handles mouse release.
    pub fn handle_mouse_up(&mut self) {
        if let Some(idx) = self.active_drag_slider {
            if let Some(slider) = self.sliders.get_mut(idx) {
                slider.dragging = false;
            }
            self.active_drag_slider = None;
        }
    }

    fn sync_settings_from_sliders(&mut self) {
        for s in &self.sliders {
            match s.id {
                1 => self.settings.video.view_distance = s.value as u32,
                2 => self.settings.video.vertical_view_distance = s.value as u32,
                3 => self.settings.video.fov = s.value,
                4 => self.settings.video.fps_limit = s.value as u32,
                11 => self.settings.audio.master_volume = s.value,
                12 => self.settings.audio.music_volume = s.value,
                13 => self.settings.audio.weather_volume = s.value,
                14 => self.settings.audio.blocks_volume = s.value,
                15 => self.settings.audio.entities_volume = s.value,
                21 => self.settings.controls.mouse_sensitivity = s.value,
                _ => {}
            }
        }
    }

    /// Renders the options screen.
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        let px_w = (width_gui * scale as f32).round() as u16;
        let px_h = (height_gui * scale as f32).round() as u16;
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, px_h],
            UiQuad::rgba(25, 28, 35, 255),
        ));

        // Title
        let header = "Settings & Options";
        let (hw, _) = font.measure_text(header);
        font.layout_text(
            header,
            (width_gui - hw) * 0.5,
            16.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Tab buttons
        for btn in &self.tab_buttons {
            btn.render(font, scale, out);
        }

        // Active tab highlight indicator bar under tab
        for (i, btn) in self.tab_buttons.iter().enumerate() {
            let is_cur = matches!(
                (i, self.active_tab),
                (0, SettingsTab::Video) | (1, SettingsTab::Audio) | (2, SettingsTab::Controls)
            );
            if is_cur {
                let px_x = (btn.x * scale as f32).round() as i32;
                let px_y = ((btn.y + btn.height) * scale as f32).round() as i32;
                let px_w = (btn.width * scale as f32).round() as u16;
                out.push(UiQuad::solid(
                    [px_x, px_y],
                    [px_w, scale.max(1) as u16 * 2],
                    UiQuad::rgba(255, 215, 0, 255),
                ));
            }
        }

        // Sliders & toggle buttons
        for s in &self.sliders {
            s.render(font, scale, out);
        }
        for b in &self.toggle_buttons {
            b.render(font, scale, out);
        }

        // Done button
        self.done_button.render(font, scale, out);
    }
}
