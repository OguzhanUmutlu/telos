//! Modernized singleplayer world selection and world manager screen.
//!
//! Features Sodium/Hytale-inspired responsive card layouts, rich world metadata
//! (generator mode, last modified date, disk footprint), smooth scrolling, keyboard
//! navigation, and a modal confirmation prompt that safely moves worlds to the system trash.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::font::BitmapFont;
use crate::menu::widgets::{ButtonStyle, MenuButton};
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;
use telos_core::i18n::LanguageCatalog;
use telos_core::trash::format_iso8601_utc;

/// Formats raw byte count into human-readable representation.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Recursively computes approximate size of directory in bytes.
fn calculate_dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += calculate_dir_size(&entry.path());
                }
            }
        }
    }
    total
}

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
    /// Last modified timestamp of the world folder or metadata.
    pub last_modified: Option<SystemTime>,
    /// Formatted last modified date string.
    pub formatted_date: String,
    /// Total disk footprint in bytes.
    pub size_bytes: u64,
    /// Formatted disk footprint string.
    pub formatted_size: String,
}

impl WorldEntry {
    /// Creates a new `WorldEntry` with computed date and size metadata.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dir_name: impl Into<String>,
        path: PathBuf,
        seed: u64,
        generator: impl Into<String>,
        last_modified: Option<SystemTime>,
        size_bytes: u64,
    ) -> Self {
        let formatted_date = last_modified.map_or_else(
            || "Unknown".to_string(),
            |t| {
                let iso = format_iso8601_utc(t);
                iso.replace('T', " ")
            },
        );
        let formatted_size = format_bytes(size_bytes);

        Self {
            name: name.into(),
            dir_name: dir_name.into(),
            path,
            seed,
            generator: generator.into(),
            last_modified,
            formatted_date,
            size_bytes,
            formatted_size,
        }
    }
}

/// Modal confirmation dialog for safely moving a world save to trash.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldDeletePrompt {
    /// World entry targeted for deletion.
    pub target: WorldEntry,
    /// Dialog box bounds `[x, y, width, height]` in GUI units.
    pub bounds: [f32; 4],
    /// Danger button to confirm and move world to trash.
    pub confirm_button: MenuButton,
    /// Neutral button to cancel and keep the world.
    pub cancel_button: MenuButton,
}

impl WorldDeletePrompt {
    /// Creates a new deletion confirmation modal centered in the GUI.
    #[must_use]
    pub fn new(
        target: WorldEntry,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) -> Self {
        let dialog_w = (width_gui - 60.0).clamp(320.0, 420.0);
        let dialog_h = 145.0;
        let dialog_x = (width_gui - dialog_w) * 0.5;
        let dialog_y = (height_gui - dialog_h) * 0.5;

        let btn_w = (dialog_w - 36.0) * 0.5;
        let btn_h = 24.0;
        let btn_y = dialog_y + dialog_h - 36.0;

        let mut confirm_label = catalog.translate("selectWorld.deletePrompt.confirm");
        if confirm_label == "selectWorld.deletePrompt.confirm" {
            confirm_label = "Move to Trash";
        }

        let cancel_label = catalog.translate("gui.cancel");

        let confirm_button =
            MenuButton::new(101, dialog_x + 12.0, btn_y, btn_w, btn_h, confirm_label)
                .with_style(ButtonStyle::Danger);

        let cancel_button = MenuButton::new(
            102,
            dialog_x + dialog_w - 12.0 - btn_w,
            btn_y,
            btn_w,
            btn_h,
            cancel_label,
        );

        Self {
            target,
            bounds: [dialog_x, dialog_y, dialog_w, dialog_h],
            confirm_button,
            cancel_button,
        }
    }

    /// Hit-tests a coordinate in GUI pixels against the dialog window bounds.
    #[must_use]
    pub fn contains(&self, mouse_x: f32, mouse_y: f32) -> bool {
        let [dx, dy, dw, dh] = self.bounds;
        mouse_x >= dx && mouse_x <= dx + dw && mouse_y >= dy && mouse_y <= dy + dh
    }

    /// Renders the modal confirmation dialog with dimmed backdrop and warning styling.
    #[allow(
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        let px_screen_w = (width_gui * scale as f32).round() as u16;
        let px_screen_h = (height_gui * scale as f32).round() as u16;

        // Dimmed backdrop scrim
        out.push(UiQuad::solid(
            [0, 0],
            [px_screen_w, px_screen_h],
            UiQuad::rgba(0, 0, 0, 190),
        ));

        let [dx, dy, dw, dh] = self.bounds;
        let px_dx = snap_to_physical(dx, scale);
        let px_dy = snap_to_physical(dy, scale);
        let px_dw = snap_to_physical(dw, scale) as u16;
        let px_dh = snap_to_physical(dh, scale) as u16;
        let border_th = scale.max(1) as u16;

        // Modal outer border (crimson warning)
        out.push(UiQuad::solid(
            [px_dx, px_dy],
            [px_dw, px_dh],
            UiQuad::rgba(239, 68, 68, 220),
        ));

        // Modal inner dark slate fill
        let in_x = px_dx + i32::from(border_th);
        let in_y = px_dy + i32::from(border_th);
        let in_w = px_dw.saturating_sub(border_th * 2);
        let in_h = px_dh.saturating_sub(border_th * 2);
        out.push(UiQuad::solid(
            [in_x, in_y],
            [in_w, in_h],
            UiQuad::rgba(20, 24, 34, 252),
        ));

        // Top crimson warning stripe
        let stripe_h = (scale.max(1) * 3) as u16;
        out.push(UiQuad::solid(
            [in_x, in_y],
            [in_w, stripe_h],
            UiQuad::rgba(239, 68, 68, 255),
        ));

        // Header Title: "Delete World?"
        let title_text = "§c§lDelete World?";
        let (tw, _) = font.measure_text(title_text);
        font.layout_text(
            title_text,
            dx + (dw - tw) * 0.5,
            dy + 14.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Confirmation question
        let q_text = "§7Are you sure you want to delete this world?";
        let (qw, _) = font.measure_text(q_text);
        font.layout_text(
            q_text,
            dx + (dw - qw) * 0.5,
            dy + 38.0,
            UiQuad::rgba(200, 200, 200, 255),
            true,
            scale,
            out,
        );

        // Highlighted target world name
        let name_text = format!("§f§l\"{}\"", self.target.name);
        let (nw, _) = font.measure_text(&name_text);
        font.layout_text(
            &name_text,
            dx + (dw - nw) * 0.5,
            dy + 56.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Explanatory note
        let note_text = "§8World will be moved to system Trash and can be restored.";
        let (notew, _) = font.measure_text(note_text);
        font.layout_text(
            note_text,
            dx + (dw - notew) * 0.5,
            dy + 76.0,
            UiQuad::rgba(140, 140, 150, 255),
            true,
            scale,
            out,
        );

        // Buttons
        self.confirm_button.render(font, scale, out);
        self.cancel_button.render(font, scale, out);
    }
}

/// Action triggered by user interaction on the world select screen.
#[derive(Debug, Clone, PartialEq)]
pub enum WorldSelectAction {
    /// Play the selected world.
    PlayWorld(WorldEntry),
    /// Open the world creator wizard.
    CreateNewWorld,
    /// Delete the selected world (safely moving to trash).
    DeleteWorld(WorldEntry),
    /// Return to main menu title screen.
    BackToTitle,
}

/// Singleplayer world selection and world manager screen.
#[derive(Debug, Clone)]
pub struct WorldSelectScreen {
    /// Discovered worlds.
    pub worlds: Vec<WorldEntry>,
    /// Selected index in `worlds`.
    pub selected_index: Option<usize>,
    /// Hovered card index in `worlds`.
    pub hovered_card_index: Option<usize>,
    /// Action buttons docked in bottom toolbar.
    pub buttons: Vec<MenuButton>,
    /// Vertical scroll offset in GUI pixels.
    pub scroll_offset: f32,
    /// Maximum scrollable distance in GUI pixels.
    pub max_scroll: f32,
    /// Header title text.
    pub title: String,
    /// Header subtitle text.
    pub subtitle: String,
    /// Last card clicked index and timestamp for double-click detection.
    pub last_card_click: Option<(usize, std::time::Instant)>,
    /// Active modal confirmation dialog for deleting a world.
    pub delete_prompt: Option<WorldDeletePrompt>,
    /// Cached GUI width.
    pub width_gui: f32,
    /// Cached GUI height.
    pub height_gui: f32,
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
            hovered_card_index: None,
            buttons: Vec::new(),
            scroll_offset: 0.0,
            max_scroll: 0.0,
            title: "Select World".to_string(),
            subtitle: "Choose a world to enter or create a new adventure".to_string(),
            last_card_click: None,
            delete_prompt: None,
            width_gui: 480.0,
            height_gui: 320.0,
        }
    }

    /// Returns true if a modal deletion prompt is currently active.
    #[must_use]
    pub const fn has_prompt(&self) -> bool {
        self.delete_prompt.is_some()
    }

    /// Dismisses the deletion confirmation prompt.
    pub fn cancel_prompt(&mut self) -> bool {
        if self.delete_prompt.is_some() {
            self.delete_prompt = None;
            true
        } else {
            false
        }
    }

    /// Confirms deletion prompt if active, returning the deletion action.
    pub fn confirm_prompt(&mut self) -> Option<WorldSelectAction> {
        self.delete_prompt
            .take()
            .map(|prompt| WorldSelectAction::DeleteWorld(prompt.target))
    }

    /// Opens the deletion confirmation dialog for the currently selected world.
    pub fn open_delete_prompt(&mut self) -> bool {
        let catalog = LanguageCatalog::with_default_embedded();
        self.open_delete_prompt_i18n(&catalog)
    }

    /// Opens the deletion confirmation dialog using the specified language catalog.
    pub fn open_delete_prompt_i18n(&mut self, catalog: &LanguageCatalog) -> bool {
        if let Some(idx) = self.selected_index
            && let Some(world) = self.worlds.get(idx).cloned()
        {
            self.delete_prompt = Some(WorldDeletePrompt::new(
                world,
                self.width_gui,
                self.height_gui,
                catalog,
            ));
            return true;
        }
        false
    }

    /// Moves selection up by one world.
    pub fn select_previous(&mut self) {
        if self.worlds.is_empty() {
            return;
        }
        let cur = self.selected_index.unwrap_or(0);
        let next = cur.saturating_sub(1);
        self.selected_index = Some(next);
        self.scroll_into_view(next);
    }

    /// Moves selection down by one world.
    pub fn select_next(&mut self) {
        if self.worlds.is_empty() {
            return;
        }
        let cur = self.selected_index.unwrap_or(0);
        let next = (cur + 1).min(self.worlds.len().saturating_sub(1));
        self.selected_index = Some(next);
        self.scroll_into_view(next);
    }

    /// Selects a specific world index and ensures it is scrolled into view.
    pub fn select_index(&mut self, index: usize) {
        if index < self.worlds.len() {
            self.selected_index = Some(index);
            self.scroll_into_view(index);
        }
    }

    /// Adjusts scroll offset so the selected index is visible.
    fn scroll_into_view(&mut self, index: usize) {
        let card_h = 52.0;
        let card_gap = 8.0;
        let card_y = index as f32 * (card_h + card_gap);
        let visible_h = (self.height_gui - 96.0).max(100.0);

        if card_y < self.scroll_offset {
            self.scroll_offset = card_y;
        } else if card_y + card_h > self.scroll_offset + visible_h {
            self.scroll_offset = (card_y + card_h - visible_h).min(self.max_scroll);
        }
    }

    /// Adjusts the vertical scroll offset from mouse wheel input.
    pub fn handle_mouse_wheel(&mut self, delta: f32) {
        if self.delete_prompt.is_some() {
            return;
        }
        if self.max_scroll > 0.0 {
            self.scroll_offset = (self.scroll_offset - delta * 24.0).clamp(0.0, self.max_scroll);
        }
    }

    /// Returns true if any interactive GUI button or card is hovered.
    #[must_use]
    pub fn is_hovered(&self) -> bool {
        if let Some(prompt) = &self.delete_prompt {
            prompt.confirm_button.hovered || prompt.cancel_button.hovered
        } else {
            self.buttons.iter().any(|b| b.hovered && b.enabled) || self.hovered_card_index.is_some()
        }
    }

    /// Scans a directory for world folders and populates the world list with full metadata.
    pub fn scan_worlds(&mut self, worlds_dir: &Path) {
        self.worlds.clear();
        self.hovered_card_index = None;
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

                    let last_modified = path
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .or_else(|| {
                            if meta_file.exists() {
                                meta_file.metadata().ok().and_then(|m| m.modified().ok())
                            } else {
                                None
                            }
                        });

                    let size_bytes = calculate_dir_size(&path);

                    self.worlds.push(WorldEntry::new(
                        name,
                        dir_name,
                        path,
                        seed,
                        generator,
                        last_modified,
                        size_bytes,
                    ));
                }
            }
        }

        // Sort worlds by most recently modified first
        self.worlds.sort_by(|a, b| {
            b.last_modified
                .cmp(&a.last_modified)
                .then_with(|| a.name.cmp(&b.name))
        });

        if self.worlds.is_empty() {
            self.selected_index = None;
        } else if let Some(sel) = self.selected_index {
            if sel >= self.worlds.len() {
                self.selected_index = Some(0);
            }
        } else {
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
        self.width_gui = width_gui;
        self.height_gui = height_gui;
        self.title = catalog.translate("selectWorld.title").to_string();
        self.subtitle = "Choose a world to enter or create a new adventure".to_string();

        let center_x = width_gui * 0.5;
        let btn_h = 24.0;
        let dock_h = 44.0;
        let bottom_y = height_gui - dock_h + 10.0;

        let has_sel = self.selected_index.is_some() && !self.worlds.is_empty();

        let w_play = 140.0;
        let w_create = 130.0;
        let w_del = 100.0;
        let w_cancel = 90.0;
        let gap = 10.0;
        let total_w = w_play + gap + w_create + gap + w_del + gap + w_cancel;
        let start_x = center_x - total_w * 0.5;

        // Button 1: Play Selected World (Primary Cyan Style)
        let mut b1 = MenuButton::new(
            1,
            start_x,
            bottom_y,
            w_play,
            btn_h,
            catalog.translate("selectWorld.select"),
        )
        .with_style(ButtonStyle::Primary);
        b1.enabled = has_sel;

        // Button 2: Create New World
        let b2 = MenuButton::new(
            2,
            start_x + w_play + gap,
            bottom_y,
            w_create,
            btn_h,
            catalog.translate("selectWorld.create"),
        );

        // Button 3: Delete World (Alert Crimson Danger Style)
        let mut del_label = catalog.translate("selectWorld.delete");
        if del_label == "selectWorld.delete" {
            del_label = "Delete World";
        }
        let mut b3 = MenuButton::new(
            3,
            start_x + w_play + gap + w_create + gap,
            bottom_y,
            w_del,
            btn_h,
            del_label,
        )
        .with_style(ButtonStyle::Danger);
        b3.enabled = has_sel;

        // Button 4: Cancel / Back
        let b4 = MenuButton::new(
            4,
            start_x + w_play + gap + w_create + gap + w_del + gap,
            bottom_y,
            w_cancel,
            btn_h,
            catalog.translate("gui.cancel"),
        );

        self.buttons = vec![b1, b2, b3, b4];

        // Recalculate max scroll
        let card_h = 52.0;
        let card_gap = 8.0;
        let top_y = 48.0;
        let visible_h = height_gui - dock_h - top_y;
        let total_h = self.worlds.len() as f32 * (card_h + card_gap);
        self.max_scroll = (total_h - visible_h + 16.0).max(0.0);
        self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll);

        // Refresh delete prompt bounds if currently active
        if let Some(prompt) = &mut self.delete_prompt {
            *prompt = WorldDeletePrompt::new(prompt.target.clone(), width_gui, height_gui, catalog);
        }
    }

    /// Updates widget layout for the given GUI dimensions using default embedded translations.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let catalog = LanguageCatalog::with_default_embedded();
        self.update_layout_i18n(width_gui, height_gui, &catalog);
    }

    /// Handles mouse motion in GUI pixels.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        if let Some(prompt) = &mut self.delete_prompt {
            prompt.confirm_button.hovered = prompt.confirm_button.contains(mouse_x, mouse_y);
            prompt.cancel_button.hovered = prompt.cancel_button.contains(mouse_x, mouse_y);
            self.hovered_card_index = None;
            for btn in &mut self.buttons {
                btn.hovered = false;
            }
            return;
        }

        for btn in &mut self.buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }

        let card_w = (self.width_gui - 80.0).clamp(340.0, 520.0);
        let card_h = 52.0;
        let card_gap = 8.0;
        let card_x = (self.width_gui - card_w) * 0.5;
        let start_y = 50.0 - self.scroll_offset;
        let top_limit = 44.0;
        let bottom_limit = self.height_gui - 44.0;

        self.hovered_card_index = None;
        for (i, _) in self.worlds.iter().enumerate() {
            let card_y = start_y + (i as f32) * (card_h + card_gap);
            if card_y + card_h < top_limit || card_y > bottom_limit {
                continue;
            }
            if mouse_x >= card_x
                && mouse_x <= card_x + card_w
                && mouse_y >= card_y
                && mouse_y <= card_y + card_h
            {
                self.hovered_card_index = Some(i);
                break;
            }
        }
    }

    /// Handles mouse click interaction using default embedded translations.
    pub fn handle_mouse_click(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
    ) -> Option<WorldSelectAction> {
        let catalog = LanguageCatalog::with_default_embedded();
        self.handle_mouse_click_i18n(mouse_x, mouse_y, width_gui, self.height_gui, &catalog)
    }

    /// Handles mouse clicks with localized dialog support.
    pub fn handle_mouse_click_i18n(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) -> Option<WorldSelectAction> {
        // Modal Delete Prompt takes exclusive focus
        if let Some(prompt) = &self.delete_prompt {
            if prompt.confirm_button.contains(mouse_x, mouse_y) {
                let target = prompt.target.clone();
                self.delete_prompt = None;
                return Some(WorldSelectAction::DeleteWorld(target));
            }
            if prompt.cancel_button.contains(mouse_x, mouse_y) {
                self.delete_prompt = None;
                return None;
            }
            // Click outside dialog dismisses prompt
            if !prompt.contains(mouse_x, mouse_y) {
                self.delete_prompt = None;
                return None;
            }
            return None;
        }

        // Toolbar Action Buttons
        for btn in &self.buttons {
            if btn.enabled && btn.contains(mouse_x, mouse_y) {
                return match btn.id {
                    1 => self
                        .selected_index
                        .and_then(|idx| self.worlds.get(idx).cloned())
                        .map(WorldSelectAction::PlayWorld),
                    2 => Some(WorldSelectAction::CreateNewWorld),
                    3 => {
                        // Open confirmation prompt instead of immediately deleting
                        if let Some(idx) = self.selected_index
                            && let Some(world) = self.worlds.get(idx).cloned()
                        {
                            self.delete_prompt = Some(WorldDeletePrompt::new(
                                world, width_gui, height_gui, catalog,
                            ));
                        }
                        None
                    }
                    4 => Some(WorldSelectAction::BackToTitle),
                    _ => None,
                };
            }
        }

        // World Cards List
        let card_w = (width_gui - 80.0).clamp(340.0, 520.0);
        let card_h = 52.0;
        let card_gap = 8.0;
        let card_x = (width_gui - card_w) * 0.5;
        let start_y = 50.0 - self.scroll_offset;
        let top_limit = 44.0;
        let bottom_limit = height_gui - 44.0;

        for (i, world) in self.worlds.iter().enumerate() {
            let card_y = start_y + (i as f32) * (card_h + card_gap);
            if card_y + card_h < top_limit || card_y > bottom_limit {
                continue;
            }

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

    /// Renders the world list screen, header, cards, docked bottom dock, and modal dialog.
    #[allow(
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::too_many_lines
    )]
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

        // Dark slate screen background
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, px_h],
            UiQuad::rgba(18, 21, 28, 255),
        ));

        // Top cyan accent line
        let top_stripe_h = (scale.max(1) * 2) as u16;
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, top_stripe_h],
            UiQuad::rgba(56, 189, 248, 255),
        ));

        // Header Title
        let header = self.title.as_str();
        let (hw, _) = font.measure_text(header);
        font.layout_text(
            header,
            (width_gui - hw) * 0.5,
            14.0,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Header Subtitle
        let sub = self.subtitle.as_str();
        let (subw, _) = font.measure_text(sub);
        font.layout_text(
            sub,
            (width_gui - subw) * 0.5,
            27.0,
            UiQuad::rgba(150, 160, 175, 255),
            true,
            scale,
            out,
        );

        // Header divider line
        let div_y = snap_to_physical(42.0, scale);
        out.push(UiQuad::solid(
            [0, div_y],
            [px_w, scale.max(1) as u16],
            UiQuad::rgba(39, 45, 60, 200),
        ));

        // World Cards List parameters
        let card_w = (width_gui - 80.0).clamp(340.0, 520.0);
        let card_h = 52.0;
        let card_gap = 8.0;
        let card_x = (width_gui - card_w) * 0.5;
        let start_y = 50.0 - self.scroll_offset;
        let dock_h = 44.0;
        let list_bottom = height_gui - dock_h;
        let border_th = scale.max(1) as u16;

        if self.worlds.is_empty() {
            // Modern empty state card
            let empty_card_w = (width_gui - 100.0).clamp(280.0, 400.0);
            let empty_card_h = 70.0;
            let empty_x = (width_gui - empty_card_w) * 0.5;
            let empty_y = (height_gui - empty_card_h) * 0.42;

            let px_ex = snap_to_physical(empty_x, scale);
            let px_ey = snap_to_physical(empty_y, scale);
            let px_ew = snap_to_physical(empty_card_w, scale) as u16;
            let px_eh = snap_to_physical(empty_card_h, scale) as u16;

            out.push(UiQuad::solid(
                [px_ex, px_ey],
                [px_ew, px_eh],
                UiQuad::rgba(45, 55, 75, 255),
            ));
            let in_ex = px_ex + i32::from(border_th);
            let in_ey = px_ey + i32::from(border_th);
            let in_ew = px_ew.saturating_sub(border_th * 2);
            let in_eh = px_eh.saturating_sub(border_th * 2);
            out.push(UiQuad::solid(
                [in_ex, in_ey],
                [in_ew, in_eh],
                UiQuad::rgba(22, 26, 36, 230),
            ));

            let empty_title = "§eNo worlds found yet";
            let (etw, _) = font.measure_text(empty_title);
            font.layout_text(
                empty_title,
                empty_x + (empty_card_w - etw) * 0.5,
                empty_y + 16.0,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                scale,
                out,
            );

            let empty_sub = "§7Click 'Create New World' to begin your adventure!";
            let (esw, _) = font.measure_text(empty_sub);
            font.layout_text(
                empty_sub,
                empty_x + (empty_card_w - esw) * 0.5,
                empty_y + 36.0,
                UiQuad::rgba(180, 180, 190, 255),
                true,
                scale,
                out,
            );
        } else {
            for (i, world) in self.worlds.iter().enumerate() {
                let card_y = start_y + (i as f32) * (card_h + card_gap);
                if card_y + card_h < 42.0 || card_y > list_bottom {
                    continue;
                }

                let is_selected = self.selected_index == Some(i);
                let is_hovered = self.hovered_card_index == Some(i);

                let px_cx = snap_to_physical(card_x, scale);
                let px_cy = snap_to_physical(card_y, scale);
                let px_cw = snap_to_physical(card_w, scale) as u16;
                let px_ch = snap_to_physical(card_h, scale) as u16;

                // Card border color
                let border_col = if is_selected {
                    UiQuad::rgba(56, 189, 248, 255)
                } else if is_hovered {
                    UiQuad::rgba(90, 105, 130, 240)
                } else {
                    UiQuad::rgba(39, 45, 60, 255)
                };

                // Card background fill
                let bg_col = if is_selected {
                    UiQuad::rgba(32, 48, 70, 245)
                } else if is_hovered {
                    UiQuad::rgba(28, 35, 48, 235)
                } else {
                    UiQuad::rgba(22, 26, 36, 220)
                };

                // Outer border
                out.push(UiQuad::solid(
                    [px_cx, px_cy],
                    [px_cw, border_th],
                    border_col,
                ));
                out.push(UiQuad::solid(
                    [px_cx, px_cy + i32::from(px_ch) - i32::from(border_th)],
                    [px_cw, border_th],
                    border_col,
                ));
                out.push(UiQuad::solid(
                    [px_cx, px_cy],
                    [border_th, px_ch],
                    border_col,
                ));
                out.push(UiQuad::solid(
                    [px_cx + i32::from(px_cw) - i32::from(border_th), px_cy],
                    [border_th, px_ch],
                    border_col,
                ));

                // Interior fill
                let in_cx = px_cx + i32::from(border_th);
                let in_cy = px_cy + i32::from(border_th);
                let in_cw = px_cw.saturating_sub(border_th * 2);
                let in_ch = px_ch.saturating_sub(border_th * 2);
                out.push(UiQuad::solid([in_cx, in_cy], [in_cw, in_ch], bg_col));

                // Selected left edge vertical accent stripe
                if is_selected {
                    let accent_w = (scale.max(1) * 3) as u16;
                    out.push(UiQuad::solid(
                        [in_cx, in_cy],
                        [accent_w, in_ch],
                        UiQuad::rgba(56, 189, 248, 255),
                    ));
                }

                // World Icon Badge box
                let icon_box_x = card_x + 10.0;
                let icon_box_y = card_y + 8.0;
                let icon_box_sz = 36.0;
                let px_ibx = snap_to_physical(icon_box_x, scale);
                let px_iby = snap_to_physical(icon_box_y, scale);
                let px_ibsz = snap_to_physical(icon_box_sz, scale) as u16;

                out.push(UiQuad::solid(
                    [px_ibx, px_iby],
                    [px_ibsz, px_ibsz],
                    UiQuad::rgba(45, 55, 75, 255),
                ));
                let in_ibx = px_ibx + i32::from(border_th);
                let in_iby = px_iby + i32::from(border_th);
                let in_ibsz = px_ibsz.saturating_sub(border_th * 2);
                out.push(UiQuad::solid(
                    [in_ibx, in_iby],
                    [in_ibsz, in_ibsz],
                    UiQuad::rgba(14, 18, 26, 255),
                ));

                // World Icon Glyph (World Initial)
                let initial = world.name.chars().next().unwrap_or('W').to_uppercase();
                let glyph_str = format!("§b{initial}");
                let (gw, gh) = font.measure_text(&glyph_str);
                font.layout_text(
                    &glyph_str,
                    icon_box_x + (icon_box_sz - gw) * 0.5,
                    icon_box_y + (icon_box_sz - gh) * 0.5,
                    UiQuad::rgba(255, 255, 255, 255),
                    true,
                    scale,
                    out,
                );

                // World Name
                let name_x = card_x + 54.0;
                let name_formatted = format!("§f§l{}", world.name);
                font.layout_text(
                    &name_formatted,
                    name_x,
                    card_y + 7.0,
                    UiQuad::rgba(255, 255, 255, 255),
                    true,
                    scale,
                    out,
                );

                // Generator Badge
                let badge = format!("§3[{}]", world.generator);
                let (badgew, _) = font.measure_text(&badge);
                font.layout_text(
                    &badge,
                    card_x + card_w - badgew - 12.0,
                    card_y + 7.0,
                    UiQuad::rgba(125, 211, 252, 255),
                    true,
                    scale,
                    out,
                );

                // Metadata Row 1: Folder & Seed
                let row1 = format!("§7Folder: §8{} §7| Seed: §8{}", world.dir_name, world.seed);
                font.layout_text(
                    &row1,
                    name_x,
                    card_y + 22.0,
                    UiQuad::rgba(180, 180, 190, 255),
                    true,
                    scale,
                    out,
                );

                // Metadata Row 2: Last Played & Size
                let row2 = format!(
                    "§8Last Played: §7{} §7| Size: §8{}",
                    world.formatted_date, world.formatted_size
                );
                font.layout_text(
                    &row2,
                    name_x,
                    card_y + 35.0,
                    UiQuad::rgba(160, 160, 170, 255),
                    true,
                    scale,
                    out,
                );
            }

            // Scrollbar track & thumb when list overflows
            if self.max_scroll > 0.0 {
                let track_x = card_x + card_w + 6.0;
                let track_y = 48.0;
                let track_h = list_bottom - track_y - 6.0;
                let track_w = 4.0;

                let px_tx = snap_to_physical(track_x, scale);
                let px_ty = snap_to_physical(track_y, scale);
                let px_tw = snap_to_physical(track_w, scale) as u16;
                let px_th = snap_to_physical(track_h, scale) as u16;

                out.push(UiQuad::solid(
                    [px_tx, px_ty],
                    [px_tw, px_th],
                    UiQuad::rgba(14, 17, 24, 180),
                ));

                let ratio = (track_h / (self.max_scroll + track_h)).clamp(0.15, 0.85);
                let thumb_h = (track_h * ratio).max(14.0);
                let frac = (self.scroll_offset / self.max_scroll).clamp(0.0, 1.0);
                let thumb_y = track_y + frac * (track_h - thumb_h);

                let px_sy = snap_to_physical(thumb_y, scale);
                let px_sh = snap_to_physical(thumb_h, scale) as u16;

                out.push(UiQuad::solid(
                    [px_tx, px_sy],
                    [px_tw, px_sh],
                    UiQuad::rgba(56, 189, 248, 220),
                ));
            }
        }

        // Docked bottom toolbar backdrop
        let bar_y = height_gui - dock_h;
        let px_by = snap_to_physical(bar_y, scale);
        let px_bh = snap_to_physical(dock_h, scale) as u16;
        out.push(UiQuad::solid(
            [0, px_by],
            [px_w, px_bh],
            UiQuad::rgba(14, 17, 24, 245),
        ));

        // Dock top divider line
        out.push(UiQuad::solid(
            [0, px_by],
            [px_w, scale.max(1) as u16],
            UiQuad::rgba(39, 45, 60, 220),
        ));

        // Action buttons
        for btn in &self.buttons {
            btn.render(font, scale, out);
        }

        // Modal Confirmation Dialog if active
        if let Some(prompt) = &self.delete_prompt {
            prompt.render(font, width_gui, height_gui, scale, out);
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::duration_suboptimal_units)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_scales() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_world_entry_construction_and_formatting() {
        let entry = WorldEntry::new(
            "My World",
            "world_01",
            PathBuf::from("/tmp/world_01"),
            42,
            "Standard",
            Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_791_500_700)),
            2048,
        );
        assert_eq!(entry.name, "My World");
        assert_eq!(entry.seed, 42);
        assert_eq!(entry.generator, "Standard");
        assert_eq!(entry.formatted_size, "2.0 KB");
        assert!(entry.formatted_date.starts_with("2026-10-08"));
    }

    #[test]
    fn test_scan_worlds_empty_and_populated() {
        let temp = tempfile::tempdir().unwrap();
        let worlds_dir = temp.path().join("worlds");
        let mut screen = WorldSelectScreen::new();

        screen.scan_worlds(&worlds_dir);
        assert!(screen.worlds.is_empty());
        assert_eq!(screen.selected_index, None);

        // Create sample world
        let w1 = worlds_dir.join("w1");
        fs::create_dir_all(&w1).unwrap();
        fs::write(
            w1.join("world.toml"),
            "name = \"Alpha\"\nseed = 999\ngenerator = \"Flat\"\n",
        )
        .unwrap();

        screen.scan_worlds(&worlds_dir);
        assert_eq!(screen.worlds.len(), 1);
        assert_eq!(screen.worlds[0].name, "Alpha");
        assert_eq!(screen.worlds[0].seed, 999);
        assert_eq!(screen.worlds[0].generator, "Flat");
        assert_eq!(screen.selected_index, Some(0));
    }

    #[test]
    fn test_layout_and_buttons_state() {
        let catalog = LanguageCatalog::with_default_embedded();
        let mut screen = WorldSelectScreen::new();
        screen.update_layout_i18n(600.0, 400.0, &catalog);

        assert_eq!(screen.buttons.len(), 4);
        // Initially no worlds, so Play and Delete are disabled
        assert!(!screen.buttons[0].enabled);
        assert!(screen.buttons[1].enabled);
        assert!(!screen.buttons[2].enabled);
        assert!(screen.buttons[3].enabled);

        // Add a world
        screen.worlds.push(WorldEntry::new(
            "World1",
            "w1",
            PathBuf::from("/tmp/w1"),
            1,
            "Standard",
            None,
            100,
        ));
        screen.selected_index = Some(0);
        screen.update_layout_i18n(600.0, 400.0, &catalog);

        assert!(screen.buttons[0].enabled);
        assert!(screen.buttons[2].enabled);
        assert_eq!(screen.buttons[0].style, ButtonStyle::Primary);
        assert_eq!(screen.buttons[2].style, ButtonStyle::Danger);
    }

    #[test]
    fn test_delete_button_opens_modal_prompt_not_instant_delete() {
        let catalog = LanguageCatalog::with_default_embedded();
        let mut screen = WorldSelectScreen::new();
        screen.worlds.push(WorldEntry::new(
            "Target World",
            "target_dir",
            PathBuf::from("/tmp/target_dir"),
            123,
            "Standard",
            None,
            500,
        ));
        screen.selected_index = Some(0);
        screen.update_layout_i18n(640.0, 480.0, &catalog);

        // Click delete button
        let del_btn = &screen.buttons[2];
        let action = screen.handle_mouse_click_i18n(
            del_btn.x + 2.0,
            del_btn.y + 2.0,
            640.0,
            480.0,
            &catalog,
        );

        // Action must be NONE! Not an immediate delete!
        assert_eq!(action, None);
        assert!(screen.has_prompt());
        assert_eq!(
            screen.delete_prompt.as_ref().unwrap().target.name,
            "Target World"
        );
    }

    #[test]
    fn test_prompt_cancel_dismisses_without_action() {
        let catalog = LanguageCatalog::with_default_embedded();
        let mut screen = WorldSelectScreen::new();
        screen.worlds.push(WorldEntry::new(
            "World",
            "dir",
            PathBuf::from("/tmp/dir"),
            1,
            "Standard",
            None,
            100,
        ));
        screen.selected_index = Some(0);
        screen.open_delete_prompt_i18n(&catalog);
        assert!(screen.has_prompt());

        // Click cancel button on dialog
        let cancel_btn = screen.delete_prompt.as_ref().unwrap().cancel_button.clone();
        let action = screen.handle_mouse_click_i18n(
            cancel_btn.x + 2.0,
            cancel_btn.y + 2.0,
            640.0,
            480.0,
            &catalog,
        );

        assert_eq!(action, None);
        assert!(!screen.has_prompt());
    }

    #[test]
    fn test_prompt_click_outside_dismisses_modal() {
        let catalog = LanguageCatalog::with_default_embedded();
        let mut screen = WorldSelectScreen::new();
        screen.worlds.push(WorldEntry::new(
            "World",
            "dir",
            PathBuf::from("/tmp/dir"),
            1,
            "Standard",
            None,
            100,
        ));
        screen.selected_index = Some(0);
        screen.update_layout_i18n(640.0, 480.0, &catalog);
        screen.open_delete_prompt_i18n(&catalog);
        assert!(screen.has_prompt());

        // Click at top-left outside modal dialog bounds
        let action = screen.handle_mouse_click_i18n(10.0, 10.0, 640.0, 480.0, &catalog);
        assert_eq!(action, None);
        assert!(!screen.has_prompt());
    }

    #[test]
    fn test_prompt_confirm_emits_delete_world_action() {
        let catalog = LanguageCatalog::with_default_embedded();
        let mut screen = WorldSelectScreen::new();
        screen.worlds.push(WorldEntry::new(
            "World",
            "dir",
            PathBuf::from("/tmp/dir"),
            1,
            "Standard",
            None,
            100,
        ));
        screen.selected_index = Some(0);
        screen.update_layout_i18n(640.0, 480.0, &catalog);
        screen.open_delete_prompt_i18n(&catalog);

        // Click confirm button on dialog
        let confirm_btn = screen
            .delete_prompt
            .as_ref()
            .unwrap()
            .confirm_button
            .clone();
        let action = screen.handle_mouse_click_i18n(
            confirm_btn.x + 2.0,
            confirm_btn.y + 2.0,
            640.0,
            480.0,
            &catalog,
        );

        assert_eq!(
            action,
            Some(WorldSelectAction::DeleteWorld(screen.worlds[0].clone()))
        );
        assert!(!screen.has_prompt());
    }

    #[test]
    fn test_keyboard_navigation_and_clamping() {
        let mut screen = WorldSelectScreen::new();
        for i in 0..5 {
            screen.worlds.push(WorldEntry::new(
                format!("World {i}"),
                format!("dir_{i}"),
                PathBuf::from(format!("/tmp/dir_{i}")),
                i as u64,
                "Standard",
                None,
                100,
            ));
        }
        screen.selected_index = Some(0);

        screen.select_next();
        assert_eq!(screen.selected_index, Some(1));

        screen.select_next();
        assert_eq!(screen.selected_index, Some(2));

        screen.select_previous();
        assert_eq!(screen.selected_index, Some(1));

        screen.select_previous();
        assert_eq!(screen.selected_index, Some(0));

        // Clamps at 0
        screen.select_previous();
        assert_eq!(screen.selected_index, Some(0));

        screen.select_index(4);
        assert_eq!(screen.selected_index, Some(4));
        // Clamps at max
        screen.select_next();
        assert_eq!(screen.selected_index, Some(4));
    }

    #[test]
    fn test_mouse_wheel_scroll_clamping() {
        let mut screen = WorldSelectScreen::new();
        screen.max_scroll = 100.0;
        screen.scroll_offset = 0.0;

        screen.handle_mouse_wheel(-1.0); // scroll down
        assert_eq!(screen.scroll_offset, 24.0);

        screen.handle_mouse_wheel(1.0); // scroll up
        assert_eq!(screen.scroll_offset, 0.0);

        screen.handle_mouse_wheel(1.0); // clamps at 0
        assert_eq!(screen.scroll_offset, 0.0);

        screen.handle_mouse_wheel(-10.0); // clamps at max_scroll
        assert_eq!(screen.scroll_offset, 100.0);
    }
}
