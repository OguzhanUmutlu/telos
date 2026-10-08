//! UI quad instance layout and builders for GPU-batched instanced rendering.

use bytemuck::{Pod, Zeroable};

/// Rendering kind of a `UiQuad`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum QuadKind {
    /// Solid untextured quad with RGBA8 color tint.
    Solid = 0,
    /// Textured sprite quad sampled with nearest filtering.
    Sprite = 1,
    /// Bitmap font glyph sampled with nearest filtering and drop shadow logic.
    GlyphBitmap = 2,
    /// Nine-slice scalable sprite with fixed corner borders.
    NineSlice = 3,
    /// Centered crosshair with inverted color blending or alpha mask.
    Crosshair = 4,
}

/// Fixed 48-byte instanced quad layout matching `ui-framework/SKILL.md §4.1` (std430).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct UiQuad {
    /// Physical screen pixel coordinates `[x, y]` of the top-left corner.
    pub pos: [i32; 2],
    /// Physical pixel dimensions `[w, h]` packed as low/high `u16` in `size_kind[0]`,
    /// and `QuadKind` (u16) + flags (u16) in `size_kind[1]`.
    pub size_kind: [u32; 2],
    /// UV rectangle: `[u0, v0, u1, v1]` normalized unorm16 in `uv[0..=1]`,
    /// texture layer index in `uv[2]`, and texture sheet index in `uv[3]`.
    pub uv: [u32; 4],
    /// RGBA8 tint color packed into a `u32` (`0xAABBGGRR` in little-endian).
    pub color: u32,
    /// Clip rectangle index in SSBO (0 = no clipping).
    pub clip: u32,
    /// Parameter 0: 9-slice borders `[left, top, right, bottom]` as 4× `u8`, or corner radius.
    pub param0: u32,
    /// Parameter 1: 9-slice source image size `[w, h]` as 2× `u16`, or MSDF pixel range.
    pub param1: u32,
}

impl UiQuad {
    /// Pack dimensions `[w, h]` into a single `u32`.
    #[must_use]
    pub const fn pack_size(w: u16, h: u16) -> u32 {
        (w as u32) | ((h as u32) << 16)
    }

    /// Pack `QuadKind` and flags into a single `u32`.
    #[must_use]
    pub const fn pack_kind(kind: QuadKind, flags: u16) -> u32 {
        (kind as u32) | ((flags as u32) << 16)
    }

    /// Pack normalized UV rect `[u0, v0, u1, v1]` into two `u32` words using `unorm16`.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn pack_uv_rect(u0: f32, v0: f32, u1: f32, v1: f32) -> [u32; 2] {
        let u0_16 = (u0.clamp(0.0, 1.0) * 65535.0).round() as u32;
        let v0_16 = (v0.clamp(0.0, 1.0) * 65535.0).round() as u32;
        let u1_16 = (u1.clamp(0.0, 1.0) * 65535.0).round() as u32;
        let v1_16 = (v1.clamp(0.0, 1.0) * 65535.0).round() as u32;
        [u0_16 | (v0_16 << 16), u1_16 | (v1_16 << 16)]
    }

    /// Pack four 8-bit borders `[left, top, right, bottom]` into a single `u32`.
    #[must_use]
    pub const fn pack_borders(left: u8, top: u8, right: u8, bottom: u8) -> u32 {
        (left as u32) | ((top as u32) << 8) | ((right as u32) << 16) | ((bottom as u32) << 24)
    }

    /// Pack RGBA8 bytes into `u32` little-endian.
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
    }

    /// Creates an untextured solid color quad.
    #[must_use]
    pub fn solid(pos: [i32; 2], size: [u16; 2], color: u32) -> Self {
        Self {
            pos,
            size_kind: [
                Self::pack_size(size[0], size[1]),
                Self::pack_kind(QuadKind::Solid, 0),
            ],
            uv: [0, 0, 0, 0],
            color,
            clip: 0,
            param0: 0,
            param1: 0,
        }
    }

    /// Creates a textured sprite quad sampled from the sprite texture array layer.
    #[must_use]
    pub fn sprite(
        pos: [i32; 2],
        size: [u16; 2],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        layer: u32,
        color: u32,
    ) -> Self {
        let uv_packed = Self::pack_uv_rect(uv_min[0], uv_min[1], uv_max[0], uv_max[1]);
        Self {
            pos,
            size_kind: [
                Self::pack_size(size[0], size[1]),
                Self::pack_kind(QuadKind::Sprite, 0),
            ],
            uv: [uv_packed[0], uv_packed[1], layer, 0],
            color,
            clip: 0,
            param0: 0,
            param1: 0,
        }
    }

    /// Creates a bitmap font glyph quad sampled from the font texture sheet.
    #[must_use]
    pub fn glyph(
        pos: [i32; 2],
        size: [u16; 2],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        layer: u32,
        color: u32,
    ) -> Self {
        let uv_packed = Self::pack_uv_rect(uv_min[0], uv_min[1], uv_max[0], uv_max[1]);
        Self {
            pos,
            size_kind: [
                Self::pack_size(size[0], size[1]),
                Self::pack_kind(QuadKind::GlyphBitmap, 0),
            ],
            uv: [uv_packed[0], uv_packed[1], layer, 0],
            color,
            clip: 0,
            param0: 0,
            param1: 0,
        }
    }

    /// Creates a centered crosshair quad.
    #[must_use]
    pub fn crosshair(
        pos: [i32; 2],
        size: [u16; 2],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        layer: u32,
    ) -> Self {
        let uv_packed = Self::pack_uv_rect(uv_min[0], uv_min[1], uv_max[0], uv_max[1]);
        Self {
            pos,
            size_kind: [
                Self::pack_size(size[0], size[1]),
                Self::pack_kind(QuadKind::Crosshair, 0),
            ],
            uv: [uv_packed[0], uv_packed[1], layer, 0],
            color: Self::rgba(255, 255, 255, 255),
            clip: 0,
            param0: 0,
            param1: 0,
        }
    }

    /// Creates a scalable nine-slice quad with fixed edge borders.
    #[must_use]
    pub fn nine_slice(
        pos: [i32; 2],
        size: [u16; 2],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        layer: u32,
        borders: [u8; 4],
        src_size: [u16; 2],
        color: u32,
    ) -> Self {
        let uv_packed = Self::pack_uv_rect(uv_min[0], uv_min[1], uv_max[0], uv_max[1]);
        Self {
            pos,
            size_kind: [
                Self::pack_size(size[0], size[1]),
                Self::pack_kind(QuadKind::NineSlice, 0),
            ],
            uv: [uv_packed[0], uv_packed[1], layer, 0],
            color,
            clip: 0,
            param0: Self::pack_borders(borders[0], borders[1], borders[2], borders[3]),
            param1: Self::pack_size(src_size[0], src_size[1]),
        }
    }

    /// Returns the `QuadKind` of this quad.
    #[must_use]
    pub const fn kind(&self) -> QuadKind {
        match (self.size_kind[1] & 0xFFFF) as u16 {
            1 => QuadKind::Sprite,
            2 => QuadKind::GlyphBitmap,
            3 => QuadKind::NineSlice,
            4 => QuadKind::Crosshair,
            _ => QuadKind::Solid,
        }
    }

    /// Unpacks the 9-slice borders `[left, top, right, bottom]`.
    #[must_use]
    pub const fn nine_slice_borders(&self) -> [u8; 4] {
        [
            (self.param0 & 0xFF) as u8,
            ((self.param0 >> 8) & 0xFF) as u8,
            ((self.param0 >> 16) & 0xFF) as u8,
            ((self.param0 >> 24) & 0xFF) as u8,
        ]
    }

    /// Unpacks the 9-slice source image size `[w, h]`.
    #[must_use]
    pub const fn nine_slice_src_size(&self) -> [u16; 2] {
        [
            (self.param1 & 0xFFFF) as u16,
            ((self.param1 >> 16) & 0xFFFF) as u16,
        ]
    }
}
