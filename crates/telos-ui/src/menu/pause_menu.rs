//! In-game pause menu modal screen.

use crate::font::BitmapFont;
use crate::menu::widgets::MenuButton;
use crate::quad::UiQuad;
use telos_core::i18n::LanguageCatalog;

/// Action triggered by the pause menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseMenuAction {
    /// Resume game.
    Resume,
    /// Open advancements screen.
    Advancements,
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
    /// Screen title text.
    pub title: String,
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
            title: "Game Paused".to_string(),
        }
    }

    /// Updates button layout and localized labels based on current viewport size and language catalog.
    pub fn update_layout_i18n(
        &mut self,
        width_gui: f32,
        height_gui: f32,
        catalog: &LanguageCatalog,
    ) {
        let center_x = width_gui * 0.5;
        let btn_w = 180.0;
        let btn_h = 24.0;
        let btn_x = center_x - btn_w * 0.5;
        let start_y = height_gui * 0.30;

        self.title = catalog.translate("menu.game").to_string();

        self.buttons = vec![
            MenuButton::new(
                1,
                btn_x,
                start_y,
                btn_w,
                btn_h,
                catalog.translate("menu.returnToGame"),
            ),
            MenuButton::new(
                4,
                btn_x,
                start_y + 32.0,
                btn_w,
                btn_h,
                catalog.translate("menu.advancements"),
            ),
            MenuButton::new(
                2,
                btn_x,
                start_y + 64.0,
                btn_w,
                btn_h,
                catalog.translate("menu.options"),
            ),
            MenuButton::new(
                3,
                btn_x,
                start_y + 96.0,
                btn_w,
                btn_h,
                catalog.translate("menu.returnToMenu"),
            ),
        ];
    }

    /// Updates button layout using default embedded translations.
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let catalog = LanguageCatalog::with_default_embedded();
        self.update_layout_i18n(width_gui, height_gui, &catalog);
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
                    4 => Some(PauseMenuAction::Advancements),
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
        let title = self.title.as_str();
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
