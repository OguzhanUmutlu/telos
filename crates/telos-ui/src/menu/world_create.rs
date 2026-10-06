//! Interactive world creation wizard screen (Seed, Name, Generator kind).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::font::BitmapFont;
use crate::menu::widgets::{MenuButton, MenuTextInput};
use crate::quad::UiQuad;

/// Action triggered by interaction in the world creator wizard.
#[derive(Debug, Clone, PartialEq)]
pub enum WorldCreateAction {
    /// World creation confirmed.
    CreateWorld {
        /// World display name.
        name: String,
        /// Folder directory name under `worlds/`.
        dir_name: String,
        /// Full path to created directory.
        path: PathBuf,
        /// Seed.
        seed: u64,
        /// Generator kind: "Standard", "Flat", or "Void".
        generator: String,
    },
    /// Canceled, return to world select.
    Cancel,
}

/// Interactive world creation wizard.
#[derive(Debug, Clone)]
pub struct WorldCreateWizard {
    /// World name text input.
    pub name_input: MenuTextInput,
    /// World seed text input.
    pub seed_input: MenuTextInput,
    /// Active generator kind index (0 = Standard, 1 = Flat, 2 = Void).
    pub generator_index: usize,
    /// Action buttons.
    pub buttons: Vec<MenuButton>,
}

impl Default for WorldCreateWizard {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldCreateWizard {
    /// Available terrain generators.
    pub const GENERATORS: [&'static str; 3] = ["Standard", "Flat", "Void"];

    /// Creates a new `WorldCreateWizard`.
    #[must_use]
    pub fn new() -> Self {
        let name_input = MenuTextInput::new(10, 0.0, 0.0, 240.0, 22.0, "New World");
        let seed_input =
            MenuTextInput::new(11, 0.0, 0.0, 240.0, 22.0, "Leave blank for random seed");

        Self {
            name_input,
            seed_input,
            generator_index: 0,
            buttons: Vec::new(),
        }
    }

    /// Updates widget layout for GUI viewport dimensions.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let center_x = width_gui * 0.5;
        let box_w = 240.0;
        let box_x = center_x - box_w * 0.5;

        self.name_input.x = box_x;
        self.name_input.y = height_gui * 0.28;
        self.name_input.width = box_w;

        self.seed_input.x = box_x;
        self.seed_input.y = height_gui * 0.44;
        self.seed_input.width = box_w;

        let btn_w = 240.0;
        let gen_y = height_gui * 0.56;
        let gen_btn = MenuButton::new(
            1,
            box_x,
            gen_y,
            btn_w,
            24.0,
            format!("Generator: {}", Self::GENERATORS[self.generator_index]),
        );

        let bottom_y = height_gui - 36.0;
        let create_btn =
            MenuButton::new(2, center_x - 125.0, bottom_y, 120.0, 24.0, "Create World");
        let cancel_btn = MenuButton::new(3, center_x + 5.0, bottom_y, 120.0, 24.0, "Cancel");

        self.buttons = vec![gen_btn, create_btn, cancel_btn];
    }

    /// Handles mouse motion in GUI units.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        self.name_input.hovered = self.name_input.contains(mouse_x, mouse_y);
        self.seed_input.hovered = self.seed_input.contains(mouse_x, mouse_y);
        for btn in &mut self.buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
    }

    /// Handles mouse click in GUI units.
    pub fn handle_mouse_click(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        worlds_base_dir: &Path,
    ) -> Option<WorldCreateAction> {
        self.name_input.focused = self.name_input.contains(mouse_x, mouse_y);
        self.seed_input.focused = self.seed_input.contains(mouse_x, mouse_y);

        for btn in &self.buttons {
            if btn.enabled && btn.contains(mouse_x, mouse_y) {
                match btn.id {
                    1 => {
                        // Cycle generator
                        self.generator_index = (self.generator_index + 1) % Self::GENERATORS.len();
                        return None;
                    }
                    2 => {
                        // Confirm create
                        let name = if self.name_input.text.trim().is_empty() {
                            "New World".to_string()
                        } else {
                            self.name_input.text.trim().to_string()
                        };

                        let seed = if self.seed_input.text.trim().is_empty() {
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map_or(1337, |d| d.as_nanos() as u64)
                        } else if let Ok(s) = self.seed_input.text.trim().parse::<u64>() {
                            s
                        } else {
                            // Hash arbitrary string input
                            let mut hash = 0xcbf2_9ce4_8422_2325_u64;
                            for byte in self.seed_input.text.trim().bytes() {
                                hash ^= u64::from(byte);
                                hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
                            }
                            hash
                        };

                        let dir_sanitized = name
                            .chars()
                            .map(|c| {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    c
                                } else {
                                    '_'
                                }
                            })
                            .collect::<String>();
                        let dir_name = if dir_sanitized.is_empty() {
                            "world".to_string()
                        } else {
                            dir_sanitized
                        };

                        let path = worlds_base_dir.join(&dir_name);
                        let generator = Self::GENERATORS[self.generator_index].to_string();

                        // Write metadata world.toml
                        let _ = fs::create_dir_all(&path);
                        let meta = format!(
                            "name = \"{name}\"\nseed = {seed}\ngenerator = \"{generator}\"\n"
                        );
                        let _ = fs::write(path.join("world.toml"), meta);

                        return Some(WorldCreateAction::CreateWorld {
                            name,
                            dir_name,
                            path,
                            seed,
                            generator,
                        });
                    }
                    3 => return Some(WorldCreateAction::Cancel),
                    _ => {}
                }
            }
        }
        None
    }

    /// Handles keyboard input character typing.
    pub fn handle_char(&mut self, ch: char) {
        if self.name_input.focused {
            self.name_input.insert_char(ch);
        } else if self.seed_input.focused {
            self.seed_input.insert_char(ch);
        }
    }

    /// Handles backspace key.
    pub fn handle_backspace(&mut self) {
        if self.name_input.focused {
            self.name_input.backspace();
        } else if self.seed_input.focused {
            self.seed_input.backspace();
        }
    }

    /// Renders the world create wizard.
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        tick: u64,
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
        let header = "Create New World";
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

        // Name label
        font.layout_text(
            "World Name:",
            self.name_input.x,
            self.name_input.y - 12.0,
            UiQuad::rgba(180, 180, 180, 255),
            true,
            scale,
            out,
        );
        self.name_input.render(font, scale, tick, out);

        // Seed label
        font.layout_text(
            "Seed for World Generator:",
            self.seed_input.x,
            self.seed_input.y - 12.0,
            UiQuad::rgba(180, 180, 180, 255),
            true,
            scale,
            out,
        );
        self.seed_input.render(font, scale, tick, out);

        // Render buttons
        for btn in &self.buttons {
            btn.render(font, scale, out);
        }
    }
}
