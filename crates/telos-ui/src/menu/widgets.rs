//! Reusable pure-CPU menu widgets for title, settings, and modal screens.

use crate::font::BitmapFont;
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;

/// Standard interactive button widget rendered to `UiQuad` primitives.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuButton {
    /// Unique identifier for event routing.
    pub id: u32,
    /// Top-left X in GUI pixels.
    pub x: f32,
    /// Top-left Y in GUI pixels.
    pub y: f32,
    /// Width in GUI pixels.
    pub width: f32,
    /// Height in GUI pixels.
    pub height: f32,
    /// Button display text label.
    pub label: String,
    /// Whether this button is enabled and clickable.
    pub enabled: bool,
    /// Whether the mouse cursor is currently hovering over this button.
    pub hovered: bool,
}

impl MenuButton {
    /// Creates a new `MenuButton`.
    #[must_use]
    pub fn new(id: u32, x: f32, y: f32, width: f32, height: f32, label: impl Into<String>) -> Self {
        Self {
            id,
            x,
            y,
            width,
            height,
            label: label.into(),
            enabled: true,
            hovered: false,
        }
    }

    /// Hit-tests a mouse position in GUI pixels against this button.
    #[must_use]
    pub fn contains(&self, mouse_x: f32, mouse_y: f32) -> bool {
        mouse_x >= self.x
            && mouse_x <= self.x + self.width
            && mouse_y >= self.y
            && mouse_y <= self.y + self.height
    }

    /// Renders the button to the `UiQuad` output draw list.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn render(&self, font: &BitmapFont, scale: u32, out: &mut Vec<UiQuad>) {
        let px_x = snap_to_physical(self.x, scale);
        let px_y = snap_to_physical(self.y, scale);
        let px_w = snap_to_physical(self.width, scale) as u16;
        let px_h = snap_to_physical(self.height, scale) as u16;
        let border_th = scale.max(1) as u16;

        // Background color
        let bg_col = if !self.enabled {
            UiQuad::rgba(30, 30, 30, 200)
        } else if self.hovered {
            UiQuad::rgba(70, 70, 75, 240)
        } else {
            UiQuad::rgba(45, 45, 50, 220)
        };

        // Outer border
        let border_col = if !self.enabled {
            UiQuad::rgba(20, 20, 20, 255)
        } else if self.hovered {
            UiQuad::rgba(255, 255, 255, 255)
        } else {
            UiQuad::rgba(15, 15, 15, 255)
        };

        // Border quads (top, bottom, left, right)
        out.push(UiQuad::solid([px_x, px_y], [px_w, border_th], border_col));
        out.push(UiQuad::solid(
            [px_x, px_y + i32::from(px_h) - i32::from(border_th)],
            [px_w, border_th],
            border_col,
        ));
        out.push(UiQuad::solid([px_x, px_y], [border_th, px_h], border_col));
        out.push(UiQuad::solid(
            [px_x + i32::from(px_w) - i32::from(border_th), px_y],
            [border_th, px_h],
            border_col,
        ));

        // Center fill panel
        let in_x = px_x + i32::from(border_th);
        let in_y = px_y + i32::from(border_th);
        let in_w = px_w.saturating_sub(border_th * 2);
        let in_h = px_h.saturating_sub(border_th * 2);
        out.push(UiQuad::solid([in_x, in_y], [in_w, in_h], bg_col));

        // Centered label text
        let (text_w, text_h) = font.measure_text(&self.label);
        let text_x = self.x + (self.width - text_w) * 0.5;
        let text_y = self.y + (self.height - text_h) * 0.5;

        let text_col = if !self.enabled {
            UiQuad::rgba(140, 140, 140, 255)
        } else if self.hovered {
            UiQuad::rgba(255, 255, 160, 255)
        } else {
            UiQuad::rgba(240, 240, 240, 255)
        };

        font.layout_text(&self.label, text_x, text_y, text_col, true, scale, out);
    }
}

/// Draggable value slider widget with numeric/custom formatting.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuSlider {
    /// Unique identifier.
    pub id: u32,
    /// Top-left X in GUI pixels.
    pub x: f32,
    /// Top-left Y in GUI pixels.
    pub y: f32,
    /// Width in GUI pixels.
    pub width: f32,
    /// Height in GUI pixels.
    pub height: f32,
    /// Label prefix (e.g. "FOV", "Master Volume").
    pub prefix: String,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Current value.
    pub value: f32,
    /// Display format mode: suffix or percentage.
    pub suffix: String,
    /// Whether value is an integer step.
    pub is_int: bool,
    /// Whether hovered.
    pub hovered: bool,
    /// Whether actively dragged.
    pub dragging: bool,
}

impl MenuSlider {
    /// Creates a new `MenuSlider`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        id: u32,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        prefix: impl Into<String>,
        min: f32,
        max: f32,
        value: f32,
        suffix: impl Into<String>,
        is_int: bool,
    ) -> Self {
        Self {
            id,
            x,
            y,
            width,
            height,
            prefix: prefix.into(),
            min,
            max,
            value: value.clamp(min, max),
            suffix: suffix.into(),
            is_int,
            hovered: false,
            dragging: false,
        }
    }

    /// Hit tests mouse coordinate against slider bounding box.
    #[must_use]
    pub fn contains(&self, mouse_x: f32, mouse_y: f32) -> bool {
        mouse_x >= self.x
            && mouse_x <= self.x + self.width
            && mouse_y >= self.y
            && mouse_y <= self.y + self.height
    }

    /// Updates current value from mouse X coordinate.
    pub fn update_from_mouse_x(&mut self, mouse_x: f32) {
        let handle_w = 8.0f32;
        let scrub_w = (self.width - handle_w).max(1.0);
        let rel_x = (mouse_x - (self.x + handle_w * 0.5)).clamp(0.0, scrub_w);
        let frac = rel_x / scrub_w;
        let mut val = self.min + frac * (self.max - self.min);
        if self.is_int {
            val = val.round();
        }
        self.value = val.clamp(self.min, self.max);
    }

    /// Formats the current display string.
    #[must_use]
    pub fn display_text(&self) -> String {
        if self.prefix == "FPS Limit" {
            let val = self.value.round() as i32;
            if val <= 0 {
                return "FPS Limit: VSync".to_string();
            }
            if val >= self.max.round() as i32 {
                return "FPS Limit: Unlimited".to_string();
            }
            return format!("FPS Limit: {val} FPS");
        }
        if self.is_int {
            format!("{}: {}{}", self.prefix, self.value as i32, self.suffix)
        } else if self.suffix == "%" {
            format!("{}: {}%", self.prefix, (self.value * 100.0).round() as i32)
        } else {
            format!("{}: {:.1}{}", self.prefix, self.value, self.suffix)
        }
    }

    /// Renders track, fill, handle, and text to `out`.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn render(&self, font: &BitmapFont, scale: u32, out: &mut Vec<UiQuad>) {
        let px_x = snap_to_physical(self.x, scale);
        let px_y = snap_to_physical(self.y, scale);
        let px_w = snap_to_physical(self.width, scale) as u16;
        let px_h = snap_to_physical(self.height, scale) as u16;
        let border_th = scale.max(1) as u16;

        // Background track
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(30, 30, 35, 230),
        ));
        // Border
        let border_col = if self.hovered || self.dragging {
            UiQuad::rgba(255, 255, 255, 255)
        } else {
            UiQuad::rgba(15, 15, 15, 255)
        };
        out.push(UiQuad::solid([px_x, px_y], [px_w, border_th], border_col));
        out.push(UiQuad::solid(
            [px_x, px_y + i32::from(px_h) - i32::from(border_th)],
            [px_w, border_th],
            border_col,
        ));
        out.push(UiQuad::solid([px_x, px_y], [border_th, px_h], border_col));
        out.push(UiQuad::solid(
            [px_x + i32::from(px_w) - i32::from(border_th), px_y],
            [border_th, px_h],
            border_col,
        ));

        // Active fill bar
        let frac = if (self.max - self.min).abs() > 1e-4 {
            ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let fill_w = (f32::from(px_w) * frac).round() as u16;
        if fill_w > border_th * 2 {
            out.push(UiQuad::solid(
                [px_x + i32::from(border_th), px_y + i32::from(border_th)],
                [
                    fill_w.saturating_sub(border_th * 2),
                    px_h.saturating_sub(border_th * 2),
                ],
                UiQuad::rgba(50, 70, 120, 180),
            ));
        }

        // Draggable handle indicator
        let handle_w = 8.0f32;
        let scrub_w = (self.width - handle_w).max(1.0);
        let handle_x_gui = self.x + frac * scrub_w;
        let h_px_x = snap_to_physical(handle_x_gui, scale);
        let h_px_w = snap_to_physical(handle_w, scale) as u16;
        let handle_col = if self.dragging {
            UiQuad::rgba(255, 255, 200, 255)
        } else if self.hovered {
            UiQuad::rgba(220, 220, 220, 255)
        } else {
            UiQuad::rgba(180, 180, 180, 255)
        };
        out.push(UiQuad::solid([h_px_x, px_y], [h_px_w, px_h], handle_col));

        // Label
        let label = self.display_text();
        let (tw, th) = font.measure_text(&label);
        let tx = self.x + (self.width - tw) * 0.5;
        let ty = self.y + (self.height - th) * 0.5;
        let t_col = if self.hovered || self.dragging {
            UiQuad::rgba(255, 255, 180, 255)
        } else {
            UiQuad::rgba(240, 240, 240, 255)
        };
        font.layout_text(&label, tx, ty, t_col, true, scale, out);
    }
}

/// Interactive single-line text input widget with cursor and backspace support.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuTextInput {
    /// Unique identifier.
    pub id: u32,
    /// Top-left X in GUI pixels.
    pub x: f32,
    /// Top-left Y in GUI pixels.
    pub y: f32,
    /// Width in GUI pixels.
    pub width: f32,
    /// Height in GUI pixels.
    pub height: f32,
    /// Current text content.
    pub text: String,
    /// Placeholder text shown when text is empty.
    pub placeholder: String,
    /// Cursor character index.
    pub cursor_pos: usize,
    /// Whether currently focused and receiving keystrokes.
    pub focused: bool,
    /// Whether hovered.
    pub hovered: bool,
}

impl MenuTextInput {
    /// Creates a new `MenuTextInput`.
    #[must_use]
    pub fn new(
        id: u32,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        placeholder: impl Into<String>,
    ) -> Self {
        Self {
            id,
            x,
            y,
            width,
            height,
            text: String::new(),
            placeholder: placeholder.into(),
            cursor_pos: 0,
            focused: false,
            hovered: false,
        }
    }

    /// Hit tests mouse coordinate against input box.
    #[must_use]
    pub fn contains(&self, mouse_x: f32, mouse_y: f32) -> bool {
        mouse_x >= self.x
            && mouse_x <= self.x + self.width
            && mouse_y >= self.y
            && mouse_y <= self.y + self.height
    }

    /// Inserts a typed character at the active cursor position.
    pub fn insert_char(&mut self, ch: char) {
        if !ch.is_control() {
            self.text.insert(self.cursor_pos, ch);
            self.cursor_pos += 1;
        }
    }

    /// Deletes the character before the active cursor.
    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 && !self.text.is_empty() {
            self.cursor_pos -= 1;
            self.text.remove(self.cursor_pos);
        }
    }

    /// Renders the text box, text, and active cursor.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn render(&self, font: &BitmapFont, scale: u32, tick: u64, out: &mut Vec<UiQuad>) {
        let px_x = snap_to_physical(self.x, scale);
        let px_y = snap_to_physical(self.y, scale);
        let px_w = snap_to_physical(self.width, scale) as u16;
        let px_h = snap_to_physical(self.height, scale) as u16;
        let border_th = scale.max(1) as u16;

        // Background
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(20, 20, 25, 240),
        ));

        // Border: gold/white if focused
        let border_col = if self.focused {
            UiQuad::rgba(255, 255, 255, 255)
        } else if self.hovered {
            UiQuad::rgba(160, 160, 160, 255)
        } else {
            UiQuad::rgba(40, 40, 45, 255)
        };
        out.push(UiQuad::solid([px_x, px_y], [px_w, border_th], border_col));
        out.push(UiQuad::solid(
            [px_x, px_y + i32::from(px_h) - i32::from(border_th)],
            [px_w, border_th],
            border_col,
        ));
        out.push(UiQuad::solid([px_x, px_y], [border_th, px_h], border_col));
        out.push(UiQuad::solid(
            [px_x + i32::from(px_w) - i32::from(border_th), px_y],
            [border_th, px_h],
            border_col,
        ));

        // Text & placeholder
        let text_y = self.y + (self.height - 8.0) * 0.5;
        let text_x = self.x + 6.0;

        if self.text.is_empty() && !self.focused {
            font.layout_text(
                &self.placeholder,
                text_x,
                text_y,
                UiQuad::rgba(120, 120, 120, 255),
                false,
                scale,
                out,
            );
        } else {
            font.layout_text(
                &self.text,
                text_x,
                text_y,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                scale,
                out,
            );
        }

        // Blinking cursor
        if self.focused && (tick / 30).is_multiple_of(2) {
            let prefix = &self.text[..self.cursor_pos.min(self.text.len())];
            let (cursor_offset_x, _) = font.measure_text(prefix);
            let cur_x = text_x + cursor_offset_x;
            font.layout_text(
                "_",
                cur_x,
                text_y,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                scale,
                out,
            );
        }
    }
}
