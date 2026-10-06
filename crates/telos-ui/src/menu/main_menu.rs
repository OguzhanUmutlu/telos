//! Title main menu screen with logo banner, animated splash text, and navigation buttons.

use crate::font::BitmapFont;
use crate::menu::widgets::MenuButton;
use crate::quad::UiQuad;

/// Action triggered by user clicking a button on the main menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainMenuAction {
    /// Open singleplayer world select screen.
    Singleplayer,
    /// Open multiplayer server list screen.
    Multiplayer,
    /// Open options / settings screen.
    Options,
    /// Quit application.
    Quit,
}

/// Title main menu screen state and button layout.
#[derive(Debug, Clone)]
pub struct MainMenuScreen {
    /// Action buttons.
    pub buttons: Vec<MenuButton>,
    /// Splash text string.
    pub splash_text: String,
}

impl Default for MainMenuScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl MainMenuScreen {
    /// Creates a new `MainMenuScreen` with standard navigation buttons.
    #[must_use]
    pub fn new() -> Self {
        let splashes = [
            "Over-engineered in Rust!",
            "144 FPS pure Vulkan 1.3!",
            "Extreme render distance!",
            "Deterministic cubic chunks!",
            "Infinite possibilities!",
            "GPU-driven Hi-Z occlusion!",
        ];
        let splash_text = splashes[0].to_string();

        Self {
            buttons: Vec::new(),
            splash_text,
        }
    }

    /// Updates button layouts based on current viewport size.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let center_x = width_gui * 0.5;
        let btn_w = 200.0;
        let btn_h = 24.0;
        let btn_x = center_x - btn_w * 0.5;
        let start_y = height_gui * 0.42;

        self.buttons = vec![
            MenuButton::new(1, btn_x, start_y, btn_w, btn_h, "Singleplayer"),
            MenuButton::new(2, btn_x, start_y + 30.0, btn_w, btn_h, "Multiplayer"),
            MenuButton::new(3, btn_x, start_y + 60.0, btn_w, btn_h, "Options..."),
            MenuButton::new(4, btn_x, start_y + 90.0, btn_w, btn_h, "Quit Game"),
        ];
    }

    /// Handles mouse motion in GUI units and updates hover states.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        for btn in &mut self.buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
    }

    /// Handles mouse click and returns the triggered action, if any.
    #[must_use]
    pub fn handle_mouse_click(&mut self, mouse_x: f32, mouse_y: f32) -> Option<MainMenuAction> {
        for btn in &self.buttons {
            if btn.enabled && btn.contains(mouse_x, mouse_y) {
                return match btn.id {
                    1 => Some(MainMenuAction::Singleplayer),
                    2 => Some(MainMenuAction::Multiplayer),
                    3 => Some(MainMenuAction::Options),
                    4 => Some(MainMenuAction::Quit),
                    _ => None,
                };
            }
        }
        None
    }

    /// Renders the title banner, animated splash, buttons, and version footer to `out`.
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        tick: u64,
        out: &mut Vec<UiQuad>,
    ) {
        // Dark background backdrop (translucent overlay or solid background)
        let px_w = (width_gui * scale as f32).round() as u16;
        let px_h = (height_gui * scale as f32).round() as u16;
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, px_h],
            UiQuad::rgba(20, 22, 28, 255),
        ));

        // Large stylized title "TELOS"
        let title = "§6§lT E L O S";
        let (tw, th) = font.measure_text(title);
        let title_x = (width_gui - tw) * 0.5;
        let title_y = height_gui * 0.22;
        font.layout_text(
            title,
            title_x,
            title_y,
            UiQuad::rgba(255, 215, 0, 255),
            true,
            scale,
            out,
        );

        // Subtitle
        let subtitle = "§7Rust & Vulkan 1.3 Voxel Engine";
        let (sw, _) = font.measure_text(subtitle);
        font.layout_text(
            subtitle,
            (width_gui - sw) * 0.5,
            title_y + th + 4.0,
            UiQuad::rgba(180, 180, 190, 255),
            true,
            scale,
            out,
        );

        // Animated pulsing splash text
        let pulse = ((tick as f32 * 0.1).sin() * 2.0).round();
        let splash = format!("§e* {} *", self.splash_text);
        let (_sp_w, _) = font.measure_text(&splash);
        font.layout_text(
            &splash,
            title_x + tw - 20.0,
            title_y + th + 18.0 + pulse,
            UiQuad::rgba(255, 255, 80, 255),
            true,
            scale,
            out,
        );

        // Render buttons
        for btn in &self.buttons {
            btn.render(font, scale, out);
        }

        // Version & platform footer
        let footer = "§8Telos v0.1.0-dev | Linux x86_64";
        font.layout_text(
            footer,
            6.0,
            height_gui - 14.0,
            UiQuad::rgba(120, 120, 130, 255),
            false,
            scale,
            out,
        );
    }
}
