//! Settings and options screen allowing live adjustment of video, audio, and controls.

use crate::font::BitmapFont;
use crate::menu::widgets::{MenuButton, MenuSlider};
use crate::quad::UiQuad;
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
    /// Mouse and keyboard controls.
    Controls,
    /// Language selection.
    Language,
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
            done_button: MenuButton::new(99, 0.0, 0.0, 140.0, 24.0, "Done"),
            active_drag_slider: None,
            title: "Settings & Options".to_string(),
        }
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
        let tab_w = 72.0;
        let tab_h = 22.0;
        let tab_gap = 6.0;
        let total_tabs_w = 4.0 * tab_w + 3.0 * tab_gap;
        let start_tab_x = center_x - total_tabs_w * 0.5;
        let tab_y = 42.0;

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
                    start_y + row_h * 2.0,
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
                    start_y + row_h * 2.0,
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
                    start_y + row_h * 3.0,
                    col_w,
                    22.0,
                    gui_str,
                ));
                self.sliders.push(MenuSlider::new(
                    5,
                    col2_x,
                    start_y + row_h * 3.0,
                    col_w,
                    22.0,
                    "Sim Dist",
                    2.0,
                    32.0,
                    self.settings.video.simulation_distance as f32,
                    " Chunks",
                    true,
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

                let on_str = catalog.translate("options.on");
                let off_str = catalog.translate("options.off");
                let inv_val = if self.settings.controls.invert_mouse_y {
                    on_str
                } else {
                    off_str
                };
                let inv_str = format!("Invert Y: {inv_val}");
                self.toggle_buttons
                    .push(MenuButton::new(204, col2_x, start_y, col_w, 22.0, inv_str));
            }
            SettingsTab::Language => {
                let langs: [(&str, &str); 5] = [
                    ("en_us", "English (US)"),
                    ("es_es", "Español (España)"),
                    ("de_de", "Deutsch (Deutschland)"),
                    ("fr_fr", "Français (France)"),
                    ("tr_tr", "Türkçe (Türkiye)"),
                ];
                let lang_btn_w = 230.0;
                let lang_btn_h = 24.0;
                let lang_gap = 6.0;
                for (i, (code, display)) in langs.iter().enumerate() {
                    let is_active = self.settings.gameplay.language == *code;
                    let label = if is_active {
                        format!("> {display} <")
                    } else {
                        (*display).to_string()
                    };
                    let btn_y = start_y + (lang_btn_h + lang_gap) * i as f32;
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
            height_gui - 34.0,
            done_w,
            24.0,
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
        for s in &mut self.sliders {
            s.hovered = s.contains(mouse_x, mouse_y);
        }
        for b in &mut self.toggle_buttons {
            b.hovered = b.contains(mouse_x, mouse_y);
        }
        self.done_button.hovered = self.done_button.contains(mouse_x, mouse_y);
    }

    /// Handles mouse button press with localized catalog. Returns true if "Done" was clicked.
    pub fn handle_mouse_click_i18n(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
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
                    104 => self.active_tab = SettingsTab::Language,
                    _ => {}
                }
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
        let header = self.title.as_str();
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
                (0, SettingsTab::Video)
                    | (1, SettingsTab::Audio)
                    | (2, SettingsTab::Controls)
                    | (3, SettingsTab::Language)
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
