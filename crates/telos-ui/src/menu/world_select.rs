//! Singleplayer world selection and world manager screen.

use std::fs;
use std::path::{Path, PathBuf};

use crate::font::BitmapFont;
use crate::menu::widgets::MenuButton;
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;
use telos_core::i18n::LanguageCatalog;

/// Information for a single discovered world save directory.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldEntry {
    /// World display name.
    pub name: String,
    /// Directory folder name under `worlds/`.
    pub dir_name: String,
    /// Absolute path to the world directory.
    pub path: PathBuf,
    /// Seed used to generate terrain.
    pub seed: u64,
    /// World generator mode name (e.g. "Standard", "Flat", "Void").
    pub generator: String,
}

/// Action triggered by user interaction on the world select screen.
#[derive(Debug, Clone, PartialEq)]
pub enum WorldSelectAction {
    /// Play the selected world.
    PlayWorld(WorldEntry),
    /// Open the world creator wizard.
    CreateNewWorld,
    /// Delete the selected world.
    DeleteWorld(WorldEntry),
    /// Return to main menu title screen.
    BackToTitle,
}

/// Singleplayer world selection screen.
#[derive(Debug, Clone)]
pub struct WorldSelectScreen {
    /// Discovered worlds.
    pub worlds: Vec<WorldEntry>,
    /// Selected index in `worlds`.
    pub selected_index: Option<usize>,
    /// Action buttons.
    pub buttons: Vec<MenuButton>,
    /// Vertical scroll offset.
    pub scroll_offset: f32,
    /// Header title text.
    pub title: String,
    /// Last card clicked index and timestamp for double-click detection.
    pub last_card_click: Option<(usize, std::time::Instant)>,
}

impl Default for WorldSelectScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldSelectScreen {
    /// Creates a new empty `WorldSelectScreen`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            worlds: Vec::new(),
            selected_index: None,
            buttons: Vec::new(),
            scroll_offset: 0.0,
            title: "Select World".to_string(),
            last_card_click: None,
        }
    }

    /// Scans a directory for world folders and populates the world list.
    pub fn scan_worlds(&mut self, worlds_dir: &Path) {
        self.worlds.clear();
        if !worlds_dir.exists() {
            let _ = fs::create_dir_all(worlds_dir);
            return;
        }

        if let Ok(entries) = fs::read_dir(worlds_dir) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                    let dir_name = entry.file_name().to_string_lossy().to_string();
                    let path = entry.path();
                    let mut name = dir_name.clone();
                    let mut seed = 1337u64;
                    let mut generator = "Standard".to_string();

                    // Read world.toml if present
                    let meta_file = path.join("world.toml");
                    if meta_file.exists()
                        && let Ok(content) = fs::read_to_string(&meta_file)
                    {
                        for line in content.lines() {
                            let line = line.trim();
                            if let Some(val) = line.strip_prefix("name = ") {
                                name = val.trim_matches('"').to_string();
                            } else if let Some(val) = line.strip_prefix("seed = ") {
                                if let Ok(s) = val.parse::<u64>() {
                                    seed = s;
                                }
                            } else if let Some(val) = line.strip_prefix("generator = ") {
                                generator = val.trim_matches('"').to_string();
                            }
                        }
                    }

                    self.worlds.push(WorldEntry {
                        name,
                        dir_name,
                        path,
                        seed,
                        generator,
                    });
                }
            }
        }

        if !self.worlds.is_empty() && self.selected_index.is_none() {
            self.selected_index = Some(0);
        }
    }

    /// Updates widget layout and localized text for the given GUI dimensions and catalog.
    pub fn update_layout_i18n(
        &mut self,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) {
        let center_x = width_gui * 0.5;
        let btn_h = 24.0;
        let bottom_y = height_gui - 36.0;

        self.title = catalog.translate("selectWorld.title").to_string();
        let has_sel = self.selected_index.is_some() && !self.worlds.is_empty();

        let w_play = 130.0;
        let w_create = 130.0;
        let w_del = 90.0;
        let w_cancel = 90.0;
        let gap = 10.0;
        let total_w = w_play + gap + w_create + gap + w_del + gap + w_cancel;
        let start_x = center_x - total_w * 0.5;

        let mut b1 = MenuButton::new(
            1,
            start_x,
            bottom_y,
            w_play,
            btn_h,
            catalog.translate("selectWorld.select"),
        );
        b1.enabled = has_sel;

        let b2 = MenuButton::new(
            2,
            start_x + w_play + gap,
            bottom_y,
            w_create,
            btn_h,
            catalog.translate("selectWorld.create"),
        );

        let mut b3 = MenuButton::new(
            3,
            start_x + w_play + gap + w_create + gap,
            bottom_y,
            w_del,
            btn_h,
            catalog.translate("selectWorld.delete"),
        );
        b3.enabled = has_sel;

        let b4 = MenuButton::new(
            4,
            start_x + w_play + gap + w_create + gap + w_del + gap,
            bottom_y,
            w_cancel,
            btn_h,
            catalog.translate("gui.cancel"),
        );

        self.buttons = vec![b1, b2, b3, b4];
    }

    /// Updates widget layout for the given GUI dimensions using default embedded translations.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let catalog = LanguageCatalog::with_default_embedded();
        self.update_layout_i18n(width_gui, height_gui, &catalog);
    }

    /// Handles mouse motion in GUI units.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        for btn in &mut self.buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
    }

    /// Handles mouse clicks in GUI units.
    pub fn handle_mouse_click(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
    ) -> Option<WorldSelectAction> {
        for btn in &self.buttons {
            if btn.enabled && btn.contains(mouse_x, mouse_y) {
                return match btn.id {
                    1 => self
                        .selected_index
                        .and_then(|idx| self.worlds.get(idx).cloned())
                        .map(WorldSelectAction::PlayWorld),
                    2 => Some(WorldSelectAction::CreateNewWorld),
                    3 => self
                        .selected_index
                        .and_then(|idx| self.worlds.get(idx).cloned())
                        .map(WorldSelectAction::DeleteWorld),
                    4 => Some(WorldSelectAction::BackToTitle),
                    _ => None,
                };
            }
        }

        // Check cards click
        let card_w = 320.0f32;
        let card_h = 36.0f32;
        let card_x = (width_gui - card_w) * 0.5;
        let start_y = 50.0 - self.scroll_offset;

        for (i, world) in self.worlds.iter().enumerate() {
            let card_y = start_y + (i as f32) * (card_h + 6.0);
            if mouse_x >= card_x
                && mouse_x <= card_x + card_w
                && mouse_y >= card_y
                && mouse_y <= card_y + card_h
            {
                let now = std::time::Instant::now();
                if let Some((prev_i, prev_t)) = self.last_card_click
                    && prev_i == i
                    && now.duration_since(prev_t).as_millis() < 400
                {
                    self.selected_index = Some(i);
                    self.last_card_click = None;
                    return Some(WorldSelectAction::PlayWorld(world.clone()));
                }
                self.selected_index = Some(i);
                self.last_card_click = Some((i, now));
                break;
            }
        }

        None
    }

    /// Renders the world list screen.
    #[allow(clippy::similar_names)]
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

        // Header Title
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

        // World Cards List
        let card_w = 320.0f32;
        let card_h = 36.0f32;
        let card_x = (width_gui - card_w) * 0.5;
        let start_y = 50.0 - self.scroll_offset;

        if self.worlds.is_empty() {
            let empty_text = "§7No worlds found. Click 'Create New' to generate a world!";
            let (ew, _) = font.measure_text(empty_text);
            font.layout_text(
                empty_text,
                (width_gui - ew) * 0.5,
                height_gui * 0.45,
                UiQuad::rgba(180, 180, 180, 255),
                true,
                scale,
                out,
            );
        } else {
            for (i, world) in self.worlds.iter().enumerate() {
                let card_y = start_y + (i as f32) * (card_h + 6.0);
                if card_y + card_h < 40.0 || card_y > height_gui - 45.0 {
                    continue;
                }

                let is_selected = self.selected_index == Some(i);
                let px_cx = snap_to_physical(card_x, scale);
                let px_cy = snap_to_physical(card_y, scale);
                let px_cw = snap_to_physical(card_w, scale) as u16;
                let px_ch = snap_to_physical(card_h, scale) as u16;

                // Card background
                let bg_col = if is_selected {
                    UiQuad::rgba(45, 55, 80, 240)
                } else {
                    UiQuad::rgba(35, 38, 45, 220)
                };
                out.push(UiQuad::solid([px_cx, px_cy], [px_cw, px_ch], bg_col));

                // Border
                let b_col = if is_selected {
                    UiQuad::rgba(255, 255, 255, 255)
                } else {
                    UiQuad::rgba(50, 55, 65, 255)
                };
                let b_th = scale.max(1) as u16;
                out.push(UiQuad::solid([px_cx, px_cy], [px_cw, b_th], b_col));
                out.push(UiQuad::solid(
                    [px_cx, px_cy + i32::from(px_ch) - i32::from(b_th)],
                    [px_cw, b_th],
                    b_col,
                ));
                out.push(UiQuad::solid([px_cx, px_cy], [b_th, px_ch], b_col));
                out.push(UiQuad::solid(
                    [px_cx + i32::from(px_cw) - i32::from(b_th), px_cy],
                    [b_th, px_ch],
                    b_col,
                ));

                // World Name
                font.layout_text(
                    &world.name,
                    card_x + 8.0,
                    card_y + 6.0,
                    UiQuad::rgba(255, 255, 255, 255),
                    true,
                    scale,
                    out,
                );

                // World meta
                let meta = format!("§7{}, Seed: {}", world.generator, world.seed);
                font.layout_text(
                    &meta,
                    card_x + 8.0,
                    card_y + 19.0,
                    UiQuad::rgba(180, 180, 180, 255),
                    true,
                    scale,
                    out,
                );
            }
        }

        // Render action buttons
        for btn in &self.buttons {
            btn.render(font, scale, out);
        }
    }
}
