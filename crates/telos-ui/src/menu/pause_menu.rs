//! In-game pause menu modal screen.

use crate::font::BitmapFont;
use crate::menu::widgets::MenuButton;
use crate::quad::UiQuad;

/// Action triggered by the pause menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseMenuAction {
    /// Resume game.
    Resume,
    /// Open options screen.
    Options,
    /// Save and cleanly quit to title.
    SaveAndQuit,
}

/// In-game pause menu modal overlay.
#[derive(Debug, Clone)]
pub struct PauseMenuScreen {
    /// Action buttons.
    pub buttons: Vec<MenuButton>,
}

impl Default for PauseMenuScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl PauseMenuScreen {
    /// Creates a new `PauseMenuScreen`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buttons: Vec::new(),
        }
    }

    /// Updates button layout.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let center_x = width_gui * 0.5;
        let btn_w = 180.0;
        let btn_h = 24.0;
        let btn_x = center_x - btn_w * 0.5;
        let start_y = height_gui * 0.35;

        self.buttons = vec![
            MenuButton::new(1, btn_x, start_y, btn_w, btn_h, "Back to Game"),
            MenuButton::new(2, btn_x, start_y + 32.0, btn_w, btn_h, "Options..."),
            MenuButton::new(
                3,
                btn_x,
                start_y + 64.0,
                btn_w,
                btn_h,
                "Save & Quit to Title",
            ),
        ];
    }

    /// Handles mouse motion in GUI pixels.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        for btn in &mut self.buttons {
            btn.hovered = btn.contains(mouse_x, mouse_y);
        }
    }

    /// Handles mouse clicks in GUI pixels.
    #[must_use]
    pub fn handle_mouse_click(&self, mouse_x: f32, mouse_y: f32) -> Option<PauseMenuAction> {
        for btn in &self.buttons {
            if btn.enabled && btn.contains(mouse_x, mouse_y) {
                return match btn.id {
                    1 => Some(PauseMenuAction::Resume),
                    2 => Some(PauseMenuAction::Options),
                    3 => Some(PauseMenuAction::SaveAndQuit),
                    _ => None,
                };
            }
        }
        None
    }

    /// Renders modal background and buttons to `out`.
    pub fn render(
        &self,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        // Translucent dark backdrop over in-game world
        let px_w = (width_gui * scale as f32).round() as u16;
        let px_h = (height_gui * scale as f32).round() as u16;
        out.push(UiQuad::solid(
            [0, 0],
            [px_w, px_h],
            UiQuad::rgba(0, 0, 0, 160),
        ));

        // Title
        let title = "Game Paused";
        let (tw, _) = font.measure_text(title);
        font.layout_text(
            title,
            (width_gui - tw) * 0.5,
            height_gui * 0.25,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Buttons
        for btn in &self.buttons {
            btn.render(font, scale, out);
        }
    }
}
