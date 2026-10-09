//! Modernized card-based settings and options screen inspired by Sodium and Hytale.
//!
//! Features spacious category panels, refined sliders, localized tab headers,
//! and interactive key rebinding with conflict detection and smooth scrolling.

use crate::font::BitmapFont;
use crate::keybinds::{InputKey, KeyAction, KeyCategory};
use crate::menu::widgets::{MenuButton, MenuSlider};
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;
use crate::settings::GameSettings;
use telos_core::i18n::LanguageCatalog;

/// Active tab within the options screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsTab {
    /// Video and graphics options.
    #[default]
    Video,
    /// Audio volume sliders.
    Audio,
    /// Mouse and keyboard controls with interactive rebinding.
    Controls,
    /// Language selection.
    Language,
}

/// Settings and options screen state with Sodium/Hytale-inspired layout.
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
    /// Interactive keybind buttons for Controls tab: `(action, button)`.
    pub keybind_buttons: Vec<(KeyAction, MenuButton)>,
    /// Action currently awaiting a physical key press for rebinding.
    pub listening_action: Option<KeyAction>,
    /// Button to reset keybinds to defaults.
    pub reset_keybinds_button: MenuButton,
    /// Done confirmation button.
    pub done_button: MenuButton,
    /// Dragging slider index.
    pub active_drag_slider: Option<usize>,
    /// Vertical scroll offset in GUI pixels for scrollable tabs.
    pub scroll_y: f32,
    /// Maximum scrollable distance in GUI pixels.
    pub max_scroll_y: f32,
    /// Screen title text.
    pub title: String,
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
            keybind_buttons: Vec::new(),
            listening_action: None,
            reset_keybinds_button: MenuButton::new(501, 0.0, 0.0, 160.0, 22.0, "Reset Keybinds"),
            done_button: MenuButton::new(99, 0.0, 0.0, 140.0, 24.0, "Done"),
            active_drag_slider: None,
            scroll_y: 0.0,
            max_scroll_y: 0.0,
            title: "Settings & Options".to_string(),
        }
    }

    /// Returns true if an action is currently waiting for a physical key press.
    #[must_use]
    pub const fn is_rebinding(&self) -> bool {
        self.listening_action.is_some()
    }

    /// Cancels any active key rebinding.
    pub fn cancel_rebinding(&mut self) {
        self.listening_action = None;
    }

    /// Processes an incoming physical key for rebinding. Returns true if a key was rebound.
    pub fn handle_key_input(&mut self, key: InputKey) -> bool {
        if let Some(action) = self.listening_action {
            if key == InputKey::Escape {
                self.listening_action = None;
                return false;
            }
            if key != InputKey::None {
                self.settings.controls.keybinds.set(action, key);
                self.listening_action = None;
                return true;
            }
        }
        false
    }

    /// Adjusts the scroll offset from a mouse wheel delta.
    pub fn handle_mouse_wheel(&mut self, delta: f32) {
        if self.max_scroll_y > 0.0 {
            self.scroll_y = (self.scroll_y - delta * 24.0).clamp(0.0, self.max_scroll_y);
        }
    }

    /// Returns true if any interactive button or slider is currently hovered by the mouse.
    #[must_use]
    pub fn has_hovered_element(&self) -> bool {
        self.tab_buttons.iter().any(|b| b.hovered)
            || self.sliders.iter().any(|s| s.hovered)
            || self.toggle_buttons.iter().any(|b| b.hovered)
            || self.keybind_buttons.iter().any(|(_, b)| b.hovered)
            || self.reset_keybinds_button.hovered
            || self.done_button.hovered
    }

    /// Rebuilds tab and widget layouts for the given viewport using the specified language catalog.
    #[allow(clippy::too_many_lines)]
    pub fn update_layout_i18n(
        &mut self,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) {
        self.title = catalog
            .translate("menu.options")
            .trim_end_matches('.')
            .to_string();

        let center_x = width_gui * 0.5;
        let tab_w = 78.0;
        let tab_h = 24.0;
        let tab_gap = 8.0;
        let total_tabs_w = 4.0 * tab_w + 3.0 * tab_gap;
        let start_tab_x = center_x - total_tabs_w * 0.5;
        let tab_y = 38.0;

        self.tab_buttons = vec![
            MenuButton::new(
                101,
                start_tab_x,
                tab_y,
                tab_w,
                tab_h,
                catalog.translate("options.tab.video"),
            ),
            MenuButton::new(
                102,
                start_tab_x + tab_w + tab_gap,
                tab_y,
                tab_w,
                tab_h,
                catalog.translate("options.tab.audio"),
            ),
            MenuButton::new(
                103,
                start_tab_x + (tab_w + tab_gap) * 2.0,
                tab_y,
                tab_w,
                tab_h,
                catalog.translate("options.tab.controls"),
            ),
            MenuButton::new(
                104,
                start_tab_x + (tab_w + tab_gap) * 3.0,
                tab_y,
                tab_w,
                tab_h,
                catalog.translate("options.tab.language"),
            ),
        ];

        let col_w = 155.0;
        let col1_x = center_x - col_w - 8.0;
        let col2_x = center_x + 8.0;
        let start_y = 74.0;
        let row_h = 26.0;

        self.sliders.clear();
        self.toggle_buttons.clear();
        self.keybind_buttons.clear();

        match self.active_tab {
            SettingsTab::Video => {
                let total_unscrolled_h = 328.0;
                let visible_h = height_gui - 36.0;
                self.max_scroll_y = (total_unscrolled_h - visible_h).max(0.0);
                self.scroll_y = self.scroll_y.clamp(0.0, self.max_scroll_y);

                let start_y = 66.0 - self.scroll_y;
                let row_h = 26.0;

                // Card 1: Display & Camera
                self.sliders.push(MenuSlider::new(
                    1,
                    col1_x,
                    start_y + 10.0,
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
                    start_y + 10.0,
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
                    start_y + 10.0 + row_h,
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
                    start_y + 10.0 + row_h,
                    col_w,
                    22.0,
                    "FPS Limit",
                    0.0,
                    360.0,
                    self.settings.video.fps_limit as f32,
                    " FPS",
                    true,
                ));

                // Card 2: Graphics & Performance
                let card2_start_y = start_y + 68.0;
                let on_str = catalog.translate("options.on");
                let off_str = catalog.translate("options.off");

                let vsync_val = if self.settings.video.vsync {
                    on_str
                } else {
                    off_str
                };
                let vsync_str = format!("VSync: {vsync_val}");
                self.toggle_buttons.push(MenuButton::new(
                    201,
                    col1_x,
                    card2_start_y + 10.0,
                    col_w,
                    22.0,
                    vsync_str,
                ));

                let cull_val = if self.settings.video.no_cull {
                    off_str
                } else {
                    on_str
                };
                let cull_str = format!("Hi-Z Culling: {cull_val}");
                self.toggle_buttons.push(MenuButton::new(
                    202,
                    col2_x,
                    card2_start_y + 10.0,
                    col_w,
                    22.0,
                    cull_str,
                ));

                let gui_str = if self.settings.video.gui_scale == 0 {
                    catalog.translate("options.guiScale.auto").to_string()
                } else {
                    format!("GUI Scale: {}", self.settings.video.gui_scale)
                };
                self.toggle_buttons.push(MenuButton::new(
                    203,
                    col1_x,
                    card2_start_y + 10.0 + row_h,
                    col_w,
                    22.0,
                    gui_str,
                ));
                self.sliders.push(MenuSlider::new(
                    5,
                    col2_x,
                    card2_start_y + 10.0 + row_h,
                    col_w,
                    22.0,
                    "Sim Dist",
                    2.0,
                    32.0,
                    self.settings.video.simulation_distance as f32,
                    " Chunks",
                    true,
                ));

                // Card 3: Post-Processing Pipeline
                let card3_row_y = card2_start_y + 68.0 + 10.0;
                let pp_val = if self.settings.video.post_processing {
                    on_str
                } else {
                    off_str
                };
                let pp_str = format!("Post-Process: {pp_val}");
                self.toggle_buttons.push(MenuButton::new(
                    205,
                    col1_x,
                    card3_row_y,
                    col_w,
                    22.0,
                    pp_str,
                ));

                let ssao_val = if self.settings.video.ssao {
                    on_str
                } else {
                    off_str
                };
                let ssao_str = format!("SSAO: {ssao_val}");
                self.toggle_buttons.push(MenuButton::new(
                    206,
                    col2_x,
                    card3_row_y,
                    col_w,
                    22.0,
                    ssao_str,
                ));

                let fog_val = if self.settings.video.volumetric_fog {
                    on_str
                } else {
                    off_str
                };
                let fog_str = format!("Volumetric Fog: {fog_val}");
                self.toggle_buttons.push(MenuButton::new(
                    207,
                    col1_x,
                    card3_row_y + row_h,
                    col_w,
                    22.0,
                    fog_str,
                ));

                let tone_val = if self.settings.video.tonemapping {
                    on_str
                } else {
                    off_str
                };
                let tone_str = format!("Tonemapping: {tone_val}");
                self.toggle_buttons.push(MenuButton::new(
                    208,
                    col2_x,
                    card3_row_y + row_h,
                    col_w,
                    22.0,
                    tone_str,
                ));

                let clouds_val = if self.settings.video.volumetric_clouds {
                    on_str
                } else {
                    off_str
                };
                let clouds_str = format!("Clouds: {clouds_val}");
                self.toggle_buttons.push(MenuButton::new(
                    209,
                    col1_x,
                    card3_row_y + row_h * 2.0,
                    col_w,
                    22.0,
                    clouds_str,
                ));

                let shadows_val = if self.settings.video.shadows {
                    on_str
                } else {
                    off_str
                };
                let shadows_str = format!("Shadows: {shadows_val}");
                self.toggle_buttons.push(MenuButton::new(
                    212,
                    col2_x,
                    card3_row_y + row_h * 2.0,
                    col_w,
                    22.0,
                    shadows_str,
                ));

                let fxaa_val = if self.settings.video.fxaa {
                    on_str
                } else {
                    off_str
                };
                let fxaa_str = format!("FXAA: {fxaa_val}");
                self.toggle_buttons.push(MenuButton::new(
                    213,
                    col1_x,
                    card3_row_y + row_h * 3.0,
                    col_w,
                    22.0,
                    fxaa_str,
                ));

                let underwater_val = if self.settings.video.underwater_effects {
                    on_str
                } else {
                    off_str
                };
                let underwater_str = format!("Underwater: {underwater_val}");
                self.toggle_buttons.push(MenuButton::new(
                    214,
                    col2_x,
                    card3_row_y + row_h * 3.0,
                    col_w,
                    22.0,
                    underwater_str,
                ));
            }
            SettingsTab::Audio => {
                self.scroll_y = 0.0;
                self.max_scroll_y = 0.0;

                self.sliders.push(MenuSlider::new(
                    11,
                    col1_x,
                    start_y + 12.0,
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
                    start_y + 12.0,
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
                    start_y + 12.0 + row_h,
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
                    start_y + 12.0 + row_h,
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
                    start_y + 12.0 + row_h * 2.0,
                    col_w,
                    22.0,
                    "Entities",
                    0.0,
                    1.0,
                    self.settings.audio.entities_volume,
                    "%",
                    false,
                ));

                let on_str = catalog.translate("options.on");
                let off_str = catalog.translate("options.off");

                let vc_val = if self.settings.audio.voice_chat_enabled {
                    on_str
                } else {
                    off_str
                };
                self.toggle_buttons.push(MenuButton::new(
                    210,
                    col2_x,
                    start_y + 12.0 + row_h * 2.0,
                    col_w,
                    22.0,
                    format!("Voice Chat: {vc_val}"),
                ));

                self.sliders.push(MenuSlider::new(
                    16,
                    col1_x,
                    start_y + 12.0 + row_h * 3.0,
                    col_w,
                    22.0,
                    "Voice Volume",
                    0.0,
                    2.0,
                    self.settings.audio.voice_chat_volume,
                    "%",
                    false,
                ));

                let ptt_val = if self.settings.audio.push_to_talk {
                    "PTT (V)"
                } else {
                    "Open Mic"
                };
                self.toggle_buttons.push(MenuButton::new(
                    211,
                    col2_x,
                    start_y + 12.0 + row_h * 3.0,
                    col_w,
                    22.0,
                    format!("Mic Mode: {ptt_val}"),
                ));

                self.sliders.push(MenuSlider::new(
                    17,
                    col1_x,
                    start_y + 12.0 + row_h * 4.0,
                    col_w,
                    22.0,
                    "Mic Sensitivity",
                    0.005,
                    0.20,
                    self.settings.audio.mic_sensitivity,
                    "%",
                    false,
                ));
            }
            SettingsTab::Controls => {
                // Controls tab supports scrollable keybind list
                let mut cur_y = start_y + 12.0 - self.scroll_y;

                // Mouse Input section
                self.sliders.push(MenuSlider::new(
                    21,
                    col1_x,
                    cur_y,
                    col_w,
                    22.0,
                    "Sensitivity",
                    0.1,
                    3.0,
                    self.settings.controls.mouse_sensitivity,
                    "x",
                    false,
                ));

                let on_str = catalog.translate("options.on");
                let off_str = catalog.translate("options.off");
                let inv_val = if self.settings.controls.invert_mouse_y {
                    on_str
                } else {
                    off_str
                };
                let inv_str = format!("Invert Y: {inv_val}");
                self.toggle_buttons
                    .push(MenuButton::new(204, col2_x, cur_y, col_w, 22.0, inv_str));

                cur_y += row_h + 16.0;

                // Keybinds grouped into categories
                let categories = [
                    (KeyCategory::Movement, "Movement Controls"),
                    (KeyCategory::Gameplay, "Gameplay & Interaction"),
                    (KeyCategory::Hotbar, "Hotbar Slots"),
                    (KeyCategory::Multiplayer, "Multiplayer & Voice"),
                    (KeyCategory::System, "System & Debug"),
                ];

                let kb_btn_w = 95.0;
                let kb_btn_h = 20.0;
                let kb_btn_x = center_x + col_w - kb_btn_w;

                for (cat, _title) in categories {
                    cur_y += 18.0; // Header space
                    for &action in KeyAction::all() {
                        if action.category() != cat {
                            continue;
                        }

                        let is_listening = self.listening_action == Some(action);
                        let label = if is_listening {
                            "> PRESS KEY <".to_string()
                        } else {
                            let key = self.settings.controls.keybinds.get(action);
                            format!("[ {} ]", key.display_name())
                        };

                        let btn_id = 1000 + action as u32;
                        let btn =
                            MenuButton::new(btn_id, kb_btn_x, cur_y, kb_btn_w, kb_btn_h, label);
                        self.keybind_buttons.push((action, btn));

                        cur_y += 24.0;
                    }
                }

                // Reset keybinds button
                cur_y += 8.0;
                let rst_w = 160.0;
                self.reset_keybinds_button = MenuButton::new(
                    501,
                    center_x - rst_w * 0.5,
                    cur_y,
                    rst_w,
                    22.0,
                    "Reset Keybinds",
                );
                cur_y += 30.0;

                // Calculate total content height for scroll bounds
                let total_unscrolled_h = cur_y + self.scroll_y - start_y;
                let visible_h = height_gui - start_y - 36.0;
                self.max_scroll_y = (total_unscrolled_h - visible_h).max(0.0);
                self.scroll_y = self.scroll_y.clamp(0.0, self.max_scroll_y);
            }
            SettingsTab::Language => {
                self.scroll_y = 0.0;
                self.max_scroll_y = 0.0;

                let langs: [(&str, &str); 5] = [
                    ("en_us", "English (US)"),
                    ("es_es", "Español (España)"),
                    ("de_de", "Deutsch (Deutschland)"),
                    ("fr_fr", "Français (France)"),
                    ("tr_tr", "Türkçe (Türkiye)"),
                ];
                let lang_btn_w = 240.0;
                let lang_btn_h = 24.0;
                let lang_gap = 6.0;
                for (i, (code, display)) in langs.iter().enumerate() {
                    let is_active = self.settings.gameplay.language == *code;
                    let label = if is_active {
                        format!("> {display} <")
                    } else {
                        (*display).to_string()
                    };
                    let btn_y = start_y + 14.0 + (lang_btn_h + lang_gap) * i as f32;
                    self.toggle_buttons.push(MenuButton::new(
                        301 + i as u32,
                        center_x - lang_btn_w * 0.5,
                        btn_y,
                        lang_btn_w,
                        lang_btn_h,
                        label,
                    ));
                }
            }
        }

        let done_w = 140.0;
        self.done_button = MenuButton::new(
            99,
            center_x - done_w * 0.5,
            height_gui - 28.0,
            done_w,
            22.0,
            catalog.translate("gui.done"),
        );
    }

    /// Rebuilds tab and widget layouts using embedded default catalog.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let catalog = LanguageCatalog::with_default_embedded();
        self.update_layout_i18n(width_gui, height_gui, &catalog);
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
        for slider in &mut self.sliders {
            slider.hovered = slider.contains(mouse_x, mouse_y);
        }
        for btn in &mut self.toggle_buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
        for (_, btn) in &mut self.keybind_buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
        self.reset_keybinds_button.hovered = self.reset_keybinds_button.contains(mouse_x, mouse_y);
        self.done_button.hovered = self.done_button.contains(mouse_x, mouse_y);
    }

    /// Handles mouse click with internationalization support. Returns true if "Done" was clicked.
    #[allow(clippy::too_many_lines)]
    pub fn handle_mouse_click_i18n(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) -> bool {
        if self.done_button.contains(mouse_x, mouse_y) {
            self.listening_action = None;
            return true;
        }

        // Tab selection
        for btn in &self.tab_buttons {
            if btn.contains(mouse_x, mouse_y) {
                match btn.id {
                    101 => self.active_tab = SettingsTab::Video,
                    102 => self.active_tab = SettingsTab::Audio,
                    103 => self.active_tab = SettingsTab::Controls,
                    104 => self.active_tab = SettingsTab::Language,
                    _ => {}
                }
                self.scroll_y = 0.0;
                self.listening_action = None;
                self.update_layout_i18n(width_gui, height_gui, catalog);
                return false;
            }
        }

        // Keybind buttons in Controls tab
        if self.active_tab == SettingsTab::Controls {
            for (action, btn) in &self.keybind_buttons {
                if btn.contains(mouse_x, mouse_y) {
                    if self.listening_action == Some(*action) {
                        self.listening_action = None;
                    } else {
                        self.listening_action = Some(*action);
                    }
                    self.update_layout_i18n(width_gui, height_gui, catalog);
                    return false;
                }
            }

            // Reset keybinds button
            if self.reset_keybinds_button.contains(mouse_x, mouse_y) {
                self.settings.controls.keybinds.reset_defaults();
                self.listening_action = None;
                self.update_layout_i18n(width_gui, height_gui, catalog);
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

        // Toggles & language selectors
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
                    205 => {
                        self.settings.video.post_processing = !self.settings.video.post_processing;
                    }
                    206 => {
                        self.settings.video.ssao = !self.settings.video.ssao;
                    }
                    207 => {
                        self.settings.video.volumetric_fog = !self.settings.video.volumetric_fog;
                    }
                    208 => {
                        self.settings.video.tonemapping = !self.settings.video.tonemapping;
                    }
                    209 => {
                        self.settings.video.volumetric_clouds =
                            !self.settings.video.volumetric_clouds;
                    }
                    212 => {
                        self.settings.video.shadows = !self.settings.video.shadows;
                    }
                    213 => {
                        self.settings.video.fxaa = !self.settings.video.fxaa;
                    }
                    214 => {
                        self.settings.video.underwater_effects =
                            !self.settings.video.underwater_effects;
                    }
                    210 => {
                        self.settings.audio.voice_chat_enabled =
                            !self.settings.audio.voice_chat_enabled;
                    }
                    211 => {
                        self.settings.audio.push_to_talk = !self.settings.audio.push_to_talk;
                    }
                    301 => self.settings.gameplay.language = "en_us".to_string(),
                    302 => self.settings.gameplay.language = "es_es".to_string(),
                    303 => self.settings.gameplay.language = "de_de".to_string(),
                    304 => self.settings.gameplay.language = "fr_fr".to_string(),
                    305 => self.settings.gameplay.language = "tr_tr".to_string(),
                    _ => {}
                }

                // If language was changed, adapt the active locale for this layout update
                let mut local_catalog;
                let active_cat = if self.settings.gameplay.language == catalog.active_locale() {
                    catalog
                } else {
                    local_catalog = catalog.clone();
                    local_catalog.set_active_locale(&self.settings.gameplay.language);
                    &local_catalog
                };

                self.update_layout_i18n(width_gui, height_gui, active_cat);
                return false;
            }
        }

        false
    }

    /// Handles mouse button press. Returns true if "Done" was clicked.
    pub fn handle_mouse_click(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
    ) -> bool {
        let catalog = LanguageCatalog::with_default_embedded();
        self.handle_mouse_click_i18n(mouse_x, mouse_y, width_gui, height_gui, &catalog)
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
                5 => self.settings.video.simulation_distance = s.value as u32,
                11 => self.settings.audio.master_volume = s.value,
                12 => self.settings.audio.music_volume = s.value,
                13 => self.settings.audio.weather_volume = s.value,
                14 => self.settings.audio.blocks_volume = s.value,
                15 => self.settings.audio.entities_volume = s.value,
                16 => self.settings.audio.voice_chat_volume = s.value,
                17 => self.settings.audio.mic_sensitivity = s.value,
                21 => self.settings.controls.mouse_sensitivity = s.value,
                _ => {}
            }
        }
    }

    /// Renders the modernized settings screen.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::similar_names
    )]
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        let px_w = snap_to_physical(width_gui, scale) as u16;
        let px_h = snap_to_physical(height_gui, scale) as u16;

        // Modern slate background
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, px_h],
            UiQuad::rgba(18, 21, 28, 255),
        ));

        let center_x = width_gui * 0.5;

        // Top decorative accent bar
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, scale.max(1) as u16 * 2],
            UiQuad::rgba(56, 189, 248, 255),
        ));

        // Screen title
        let header = self.title.as_str();
        let (hw, _) = font.measure_text(header);
        font.layout_text(
            header,
            (width_gui - hw) * 0.5,
            14.0,
            UiQuad::rgba(248, 250, 252, 255),
            true,
            scale,
            out,
        );

        // Tab pill buttons
        for btn in &self.tab_buttons {
            btn.render(font, scale, out);
        }

        // Active tab cyan indicator bar
        for (i, btn) in self.tab_buttons.iter().enumerate() {
            let is_cur = matches!(
                (i, self.active_tab),
                (0, SettingsTab::Video)
                    | (1, SettingsTab::Audio)
                    | (2, SettingsTab::Controls)
                    | (3, SettingsTab::Language)
            );
            if is_cur {
                let px_x = snap_to_physical(btn.x, scale);
                let px_y = snap_to_physical(btn.y + btn.height, scale);
                let px_bw = snap_to_physical(btn.width, scale) as u16;
                out.push(UiQuad::solid(
                    [px_x, px_y],
                    [px_bw, scale.max(1) as u16 * 3],
                    UiQuad::rgba(56, 189, 248, 255),
                ));
            }
        }

        // Card container backgrounds for active tab
        let card_w = 340.0;
        let card_x = center_x - card_w * 0.5;

        match self.active_tab {
            SettingsTab::Video => {
                let card1_y = 66.0 - self.scroll_y;
                let card2_y = card1_y + 68.0;
                let card3_y = card2_y + 68.0;

                // Card 1: Display & Camera
                render_card(
                    card_x,
                    card1_y,
                    card_w,
                    62.0,
                    "DISPLAY & CAMERA",
                    font,
                    scale,
                    out,
                );
                // Card 2: Graphics & Performance
                render_card(
                    card_x,
                    card2_y,
                    card_w,
                    62.0,
                    "GRAPHICS & PERFORMANCE",
                    font,
                    scale,
                    out,
                );
                // Card 3: Shaders & Post-Processing
                render_card(
                    card_x,
                    card3_y,
                    card_w,
                    116.0,
                    "SHADERS & POST-PROCESSING",
                    font,
                    scale,
                    out,
                );
            }
            SettingsTab::Audio => {
                // Card 1: Volume Channels
                render_card(
                    card_x,
                    70.0,
                    card_w,
                    94.0,
                    "VOLUME CHANNELS",
                    font,
                    scale,
                    out,
                );
            }
            SettingsTab::Controls => {
                // Render category headers and keybind rows
                let conflicts = self.settings.controls.keybinds.find_conflicts();

                // Mouse section card
                let mouse_y = 70.0 - self.scroll_y;
                if mouse_y + 40.0 >= 60.0 && mouse_y <= height_gui - 35.0 {
                    render_card(
                        card_x,
                        mouse_y,
                        card_w,
                        38.0,
                        "MOUSE CONTROLS",
                        font,
                        scale,
                        out,
                    );
                }

                // Action labels for keybinds
                for (action, btn) in &self.keybind_buttons {
                    if btn.y + btn.height < 65.0 || btn.y > height_gui - 35.0 {
                        continue;
                    }

                    // Check conflict
                    let has_conflict = conflicts
                        .iter()
                        .any(|(a1, a2, _)| *a1 == *action || *a2 == *action);

                    let label_x = card_x + 12.0;
                    let label_y = btn.y + (btn.height - 8.0) * 0.5;
                    let label_col = if has_conflict {
                        UiQuad::rgba(251, 146, 60, 255) // Amber conflict alert
                    } else {
                        UiQuad::rgba(226, 232, 240, 255)
                    };

                    font.layout_text(
                        action.display_name(),
                        label_x,
                        label_y,
                        label_col,
                        true,
                        scale,
                        out,
                    );

                    // If listening, render glowing button border
                    if self.listening_action == Some(*action) {
                        let px_bx = snap_to_physical(btn.x, scale);
                        let px_by = snap_to_physical(btn.y, scale);
                        let px_bw = snap_to_physical(btn.width, scale) as u16;
                        let px_bh = snap_to_physical(btn.height, scale) as u16;
                        out.push(UiQuad::solid(
                            [px_bx, px_by],
                            [px_bw, px_bh],
                            UiQuad::rgba(234, 179, 8, 220),
                        ));
                    }

                    btn.render(font, scale, out);
                }

                // Reset button
                if self.reset_keybinds_button.y >= 65.0
                    && self.reset_keybinds_button.y <= height_gui - 35.0
                {
                    self.reset_keybinds_button.render(font, scale, out);
                }
            }
            SettingsTab::Language => {
                // Card 1: Language Selection
                render_card(
                    card_x,
                    70.0,
                    card_w,
                    160.0,
                    "SELECT LANGUAGE",
                    font,
                    scale,
                    out,
                );
            }
        }

        // Scrollbar indicator if scrollable
        if self.max_scroll_y > 0.0 {
            let track_x = center_x + card_w * 0.5 + 8.0;
            let track_y = 74.0;
            let track_h = height_gui - track_y - 36.0;
            let px_tx = snap_to_physical(track_x, scale);
            let px_ty = snap_to_physical(track_y, scale);
            let px_tw = snap_to_physical(4.0, scale) as u16;
            let px_th = snap_to_physical(track_h, scale) as u16;

            // Track
            out.push(UiQuad::solid(
                [px_tx, px_ty],
                [px_tw, px_th],
                UiQuad::rgba(30, 36, 48, 200),
            ));

            // Thumb
            let frac = (self.scroll_y / self.max_scroll_y).clamp(0.0, 1.0);
            let thumb_h = (track_h * (visible_ratio(self.max_scroll_y, track_h))).max(16.0);
            let thumb_y = track_y + frac * (track_h - thumb_h);
            let px_sy = snap_to_physical(thumb_y, scale);
            let px_sh = snap_to_physical(thumb_h, scale) as u16;

            out.push(UiQuad::solid(
                [px_tx, px_sy],
                [px_tw, px_sh],
                UiQuad::rgba(94, 234, 212, 240),
            ));
        }

        // Sliders & toggle buttons (clipped to visible area)
        for s in &self.sliders {
            if s.y + s.height >= 65.0 && s.y <= height_gui - 32.0 {
                s.render(font, scale, out);
            }
        }
        for b in &self.toggle_buttons {
            if b.y + b.height >= 65.0 && b.y <= height_gui - 32.0 {
                b.render(font, scale, out);
            }
        }

        // Bottom backdrop bar behind Done button
        let bar_h = 32.0;
        let bar_y = height_gui - bar_h;
        let px_by = snap_to_physical(bar_y, scale);
        let px_bh = snap_to_physical(bar_h, scale) as u16;
        out.push(UiQuad::solid(
            [0, px_by],
            [px_w, px_bh],
            UiQuad::rgba(14, 17, 24, 240),
        ));

        // Done button
        self.done_button.render(font, scale, out);
    }
}

/// Helper function to compute visible scrollbar thumb ratio.
fn visible_ratio(max_scroll: f32, visible_h: f32) -> f32 {
    let total = max_scroll + visible_h;
    if total > 0.0 {
        (visible_h / total).clamp(0.15, 0.8)
    } else {
        1.0
    }
}

/// Helper function rendering a clean Sodium/Hytale-inspired card panel.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn render_card(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    title: &str,
    font: &BitmapFont,
    scale: u32,
    out: &mut Vec<UiQuad>,
) {
    let px_x = snap_to_physical(x, scale);
    let px_y = snap_to_physical(y, scale);
    let px_w = snap_to_physical(w, scale) as u16;
    let px_h = snap_to_physical(h, scale) as u16;
    let border_th = scale.max(1) as u16;

    // Card outer border
    out.push(UiQuad::solid(
        [px_x, px_y],
        [px_w, px_h],
        UiQuad::rgba(39, 45, 60, 255),
    ));

    // Card interior fill
    let in_x = px_x + i32::from(border_th);
    let in_y = px_y + i32::from(border_th);
    let in_w = px_w.saturating_sub(border_th * 2);
    let in_h = px_h.saturating_sub(border_th * 2);
    out.push(UiQuad::solid(
        [in_x, in_y],
        [in_w, in_h],
        UiQuad::rgba(25, 30, 42, 230),
    ));

    // Section title with colored accent pip
    if !title.is_empty() {
        let pip_x = px_x + 8 * scale as i32;
        let pip_y = px_y + 4 * scale as i32;
        let pip_w = scale.max(1) as u16 * 2;
        let pip_h = scale.max(1) as u16 * 6;
        out.push(UiQuad::solid(
            [pip_x, pip_y],
            [pip_w, pip_h],
            UiQuad::rgba(56, 189, 248, 255),
        ));

        font.layout_text(
            title,
            x + 14.0,
            y + 3.0,
            UiQuad::rgba(148, 163, 184, 255),
            true,
            scale,
            out,
        );
    }
}
