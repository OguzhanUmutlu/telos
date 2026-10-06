//! Bitmap font parser, text metrics, and quad layout generator.

use crate::quad::UiQuad;
use crate::scale::snap_to_physical;

/// Glyph metrics extracted from a font sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphMetrics {
    /// Column offset within cell (0..8) where glyph starts.
    pub min_x: u8,
    /// Column offset within cell (0..8) where glyph ends.
    pub max_x: u8,
    /// Proportional horizontal advance in GUI pixels.
    pub advance: f32,
    /// Normalized UV bounds: `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
}

impl Default for GlyphMetrics {
    fn default() -> Self {
        Self {
            min_x: 0,
            max_x: 7,
            advance: 8.0,
            uv: [0.0, 0.0, 0.0625, 0.0625],
        }
    }
}

/// Bitmap font engine supporting proportional glyph advances and drop shadows.
#[derive(Debug, Clone)]
pub struct BitmapFont {
    /// 256 ASCII glyph metrics.
    glyphs: [GlyphMetrics; 256],
    /// Cell dimensions in source pixels (e.g. 8x8).
    cell_size: [u32; 2],
    /// Layer index of the font sheet in the UI texture array.
    pub texture_layer: u32,
}

impl BitmapFont {
    /// Builds a `BitmapFont` from raw RGBA8 image pixels (e.g. `ascii.png`).
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn from_rgba(rgba_pixels: &[u8], width: u32, height: u32, texture_layer: u32) -> Self {
        let cell_w = (width / 16).max(1);
        let cell_h = (height / 16).max(1);
        let mut glyphs = [GlyphMetrics::default(); 256];

        for char_code in 0..256 {
            let row = char_code / 16;
            let col = char_code % 16;

            let cell_x0 = col * cell_w;
            let cell_y0 = row * cell_h;

            let mut min_x = cell_w;
            let mut max_x = 0;
            let mut has_pixel = false;

            for cx in 0..cell_w {
                for cy in 0..cell_h {
                    let px = cell_x0 + cx;
                    let py = cell_y0 + cy;
                    let idx = ((py * width + px) * 4) as usize;
                    if idx + 3 < rgba_pixels.len() && rgba_pixels[idx + 3] > 16 {
                        has_pixel = true;
                        if cx < min_x {
                            min_x = cx;
                        }
                        if cx > max_x {
                            max_x = cx;
                        }
                    }
                }
            }

            let (glyph_min_x, glyph_max_x, advance) = if has_pixel {
                let char_width = (max_x - min_x + 1) as f32;
                // Proportional advance with 1 GUI pixel letter spacing
                let adv = char_width * (8.0 / cell_w as f32) + 1.0;
                (min_x as u8, max_x as u8, adv)
            } else if char_code == 32 {
                // Space character: 4 GUI pixels advance
                (0u8, 0u8, 4.0)
            } else {
                (0u8, (cell_w - 1) as u8, cell_w as f32)
            };

            let u0 = (cell_x0 as f32) / (width as f32);
            let v0 = (cell_y0 as f32) / (height as f32);
            let u1 = ((cell_x0 + cell_w) as f32) / (width as f32);
            let v1 = ((cell_y0 + cell_h) as f32) / (height as f32);

            glyphs[char_code as usize] = GlyphMetrics {
                min_x: glyph_min_x,
                max_x: glyph_max_x,
                advance,
                uv: [u0, v0, u1, v1],
            };
        }

        Self {
            glyphs,
            cell_size: [cell_w, cell_h],
            texture_layer,
        }
    }

    /// Returns the cell dimensions in source pixels.
    #[must_use]
    pub const fn cell_size(&self) -> [u32; 2] {
        self.cell_size
    }

    /// Returns the metrics for a given ASCII character code.
    #[must_use]
    pub const fn glyph(&self, ch: u8) -> &GlyphMetrics {
        &self.glyphs[ch as usize]
    }

    /// Creates a fallback procedural font for testing when no assets are loaded.
    #[must_use]
    pub fn new_fallback(texture_layer: u32) -> Self {
        let mut glyphs = [GlyphMetrics::default(); 256];
        for (i, glyph) in glyphs.iter_mut().enumerate() {
            let row = (i / 16) as f32;
            let col = (i % 16) as f32;
            let u0 = col / 16.0;
            let v0 = row / 16.0;
            let u1 = (col + 1.0) / 16.0;
            let v1 = (row + 1.0) / 16.0;
            let advance = if i == 32 {
                4.0 // space
            } else if i == b'i' as usize || i == b'l' as usize || i == b'.' as usize {
                3.0 // narrow letters
            } else if i == b'W' as usize || i == b'M' as usize {
                7.0 // wide letters
            } else {
                6.0 // standard letters
            };
            *glyph = GlyphMetrics {
                min_x: 0,
                max_x: 7,
                advance,
                uv: [u0, v0, u1, v1],
            };
        }

        Self {
            glyphs,
            cell_size: [8, 8],
            texture_layer,
        }
    }

    /// Measures the total width and height of `text` in GUI pixels.
    #[must_use]
    pub fn measure_text(&self, text: &str) -> (f32, f32) {
        let mut max_width = 0.0f32;
        let mut current_width = 0.0f32;
        let mut lines = 1;

        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '§' {
                // Skip color code character
                let _ = chars.next();
                continue;
            }
            if ch == '\n' {
                lines += 1;
                if current_width > max_width {
                    max_width = current_width;
                }
                current_width = 0.0;
                continue;
            }

            let idx = (ch as usize).min(255);
            current_width += self.glyphs[idx].advance;
        }

        if current_width > max_width {
            max_width = current_width;
        }

        (max_width, (lines * 9) as f32)
    }

    /// Lays out `text` into `out` quad buffer with optional drop shadows and color codes.
    pub fn layout_text(
        &self,
        text: &str,
        start_x: f32,
        start_y: f32,
        default_color: u32,
        shadow: bool,
        gui_scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        if shadow {
            // First pass: render darkened drop shadow offset by 1 GUI pixel
            let shadow_color = Self::darken_color(default_color);
            self.layout_text_pass(
                text,
                start_x + 1.0,
                start_y + 1.0,
                shadow_color,
                true,
                gui_scale,
                out,
            );
        }

        // Second pass: render foreground text
        self.layout_text_pass(text, start_x, start_y, default_color, false, gui_scale, out);
    }

    fn layout_text_pass(
        &self,
        text: &str,
        start_x: f32,
        start_y: f32,
        initial_color: u32,
        is_shadow: bool,
        gui_scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        let mut cur_x = start_x;
        let mut cur_y = start_y;
        let mut cur_color = initial_color;

        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '§' {
                if let Some(code) = chars.next() {
                    let parsed_color = Self::parse_color_code(code, initial_color);
                    cur_color = if is_shadow {
                        Self::darken_color(parsed_color)
                    } else {
                        parsed_color
                    };
                }
                continue;
            }

            if ch == '\n' {
                cur_x = start_x;
                cur_y += 9.0;
                continue;
            }

            let idx = (ch as usize).min(255);
            let glyph = &self.glyphs[idx];

            if ch != ' ' {
                let px_x = snap_to_physical(cur_x, gui_scale);
                let px_y = snap_to_physical(cur_y, gui_scale);
                let glyph_w = (8 * gui_scale) as u16;
                let glyph_h = (8 * gui_scale) as u16;

                let quad = UiQuad::glyph(
                    [px_x, px_y],
                    [glyph_w, glyph_h],
                    [glyph.uv[0], glyph.uv[1]],
                    [glyph.uv[2], glyph.uv[3]],
                    self.texture_layer,
                    cur_color,
                );
                out.push(quad);
            }

            cur_x += glyph.advance;
        }
    }

    fn darken_color(color: u32) -> u32 {
        let r = ((color & 0xFF) as f32 * 0.25).round() as u32;
        let g = (((color >> 8) & 0xFF) as f32 * 0.25).round() as u32;
        let b = (((color >> 16) & 0xFF) as f32 * 0.25).round() as u32;
        let a = color & 0xFF00_0000;
        r | (g << 8) | (b << 16) | a
    }

    fn parse_color_code(code: char, default_color: u32) -> u32 {
        match code {
            '0' => UiQuad::rgba(0, 0, 0, 255),
            '1' => UiQuad::rgba(0, 0, 170, 255),
            '2' => UiQuad::rgba(0, 170, 0, 255),
            '3' => UiQuad::rgba(0, 170, 170, 255),
            '4' => UiQuad::rgba(170, 0, 0, 255),
            '5' => UiQuad::rgba(170, 0, 170, 255),
            '6' => UiQuad::rgba(255, 170, 0, 255),
            '7' => UiQuad::rgba(170, 170, 170, 255),
            '8' => UiQuad::rgba(85, 85, 85, 255),
            '9' => UiQuad::rgba(85, 85, 255, 255),
            'a' => UiQuad::rgba(85, 255, 85, 255),
            'b' => UiQuad::rgba(85, 255, 255, 255),
            'c' => UiQuad::rgba(255, 85, 85, 255),
            'd' => UiQuad::rgba(255, 85, 255, 255),
            'e' => UiQuad::rgba(255, 255, 85, 255),
            'f' => UiQuad::rgba(255, 255, 255, 255),
            _ => default_color,
        }
    }
}
