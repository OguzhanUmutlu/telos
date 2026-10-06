//! 8-byte color-only LOD quad layout matching GPU vertex pulling specifications.

use telos_core::coords::Face;

/// Compact 8-byte quad representation for far-field terrain rendering.
///
/// Encoded as two 32-bit unsigned words (`uvec2` on GPU):
///
/// ```text
/// Word 0:
///   [0..5)   x (0..=31)
///   [5..10)  y (0..=31)
///   [10..15) z (0..=31)
///   [15..20) width - 1 (0..=31)
///   [20..25) height - 1 (0..=31)
///   [25..28) face direction (0..=5: -Y, +Y, -Z, +Z, -X, +X)
///   [28..32) sky light level (0..=15)
///
/// Word 1:
///   [0..12)  color palette index (0..=4095)
///   [12..20) 4-corner ambient occlusion (2 bits per corner: c0..c3)
///   [20..24) block light level (0..=15)
///   [24..28) surface flags (bit 24: water, 25: emissive, 26: foliage, 27: reserved)
///   [28..32) reserved (0)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct LodQuad {
    /// Packed word 0 (coordinates, extents, direction, sky light).
    pub data0: u32,
    /// Packed word 1 (color index, AO, block light, surface flags).
    pub data1: u32,
}

impl std::fmt::Debug for LodQuad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LodQuad")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .field("width", &self.width())
            .field("height", &self.height())
            .field("face", &self.face())
            .field("sky_light", &self.sky_light())
            .field("color_index", &self.color_index())
            .field("ao", &self.ao())
            .field("block_light", &self.block_light())
            .field("flags", &self.flags())
            .finish()
    }
}

impl LodQuad {
    /// Water surface flag bit (bit 24 of data1).
    pub const FLAG_WATER: u8 = 1 << 0;
    /// Emissive surface flag bit (bit 25 of data1).
    pub const FLAG_EMISSIVE: u8 = 1 << 1;
    /// Foliage surface flag bit (bit 26 of data1).
    pub const FLAG_FOLIAGE: u8 = 1 << 2;

    /// Constructs a packed `LodQuad`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[must_use]
    pub const fn new(
        x: u32,
        y: u32,
        z: u32,
        width: u32,
        height: u32,
        face: Face,
        sky_light: u8,
        color_index: u16,
        ao: [u8; 4],
        block_light: u8,
        flags: u8,
    ) -> Self {
        let dir = face as u32;
        let w_minus_1 = width.saturating_sub(1) & 0x1F;
        let h_minus_1 = height.saturating_sub(1) & 0x1F;
        let sky = (sky_light as u32) & 0x0F;

        let data0 = (x & 0x1F)
            | ((y & 0x1F) << 5)
            | ((z & 0x1F) << 10)
            | (w_minus_1 << 15)
            | (h_minus_1 << 20)
            | ((dir & 0x07) << 25)
            | (sky << 28);

        let color = (color_index as u32) & 0x0FFF;
        let c0 = (ao[0] as u32) & 0x03;
        let c1 = (ao[1] as u32) & 0x03;
        let c2 = (ao[2] as u32) & 0x03;
        let c3 = (ao[3] as u32) & 0x03;
        let ao_packed = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        let block = (block_light as u32) & 0x0F;
        let fl = (flags as u32) & 0x0F;

        let data1 = color | (ao_packed << 12) | (block << 20) | (fl << 24);

        Self { data0, data1 }
    }

    /// Local X position (0..=31).
    #[inline]
    #[must_use]
    pub const fn x(&self) -> u32 {
        self.data0 & 0x1F
    }

    /// Local Y position (0..=31).
    #[inline]
    #[must_use]
    pub const fn y(&self) -> u32 {
        (self.data0 >> 5) & 0x1F
    }

    /// Local Z position (0..=31).
    #[inline]
    #[must_use]
    pub const fn z(&self) -> u32 {
        (self.data0 >> 10) & 0x1F
    }

    /// Quad width in voxels along the tangent axis (1..=32).
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        ((self.data0 >> 15) & 0x1F) + 1
    }

    /// Quad height in voxels along the bitangent axis (1..=32).
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        ((self.data0 >> 20) & 0x1F) + 1
    }

    /// Face normal direction.
    #[inline]
    #[must_use]
    pub const fn face(&self) -> Face {
        match (self.data0 >> 25) & 0x07 {
            0 => Face::Down,
            1 => Face::Up,
            2 => Face::North,
            3 => Face::South,
            4 => Face::West,
            _ => Face::East,
        }
    }

    /// Sky light level (0..=15).
    #[inline]
    #[must_use]
    pub const fn sky_light(&self) -> u8 {
        ((self.data0 >> 28) & 0x0F) as u8
    }

    /// Palette color index (0..=4095).
    #[inline]
    #[must_use]
    pub const fn color_index(&self) -> u16 {
        (self.data1 & 0x0FFF) as u16
    }

    /// 4-corner ambient occlusion values `[c0, c1, c2, c3]` (each 0..=3).
    #[inline]
    #[must_use]
    pub const fn ao(&self) -> [u8; 4] {
        let ao_packed = (self.data1 >> 12) & 0xFF;
        [
            (ao_packed & 0x03) as u8,
            ((ao_packed >> 2) & 0x03) as u8,
            ((ao_packed >> 4) & 0x03) as u8,
            ((ao_packed >> 6) & 0x03) as u8,
        ]
    }

    /// Block light level (0..=15).
    #[inline]
    #[must_use]
    pub const fn block_light(&self) -> u8 {
        ((self.data1 >> 20) & 0x0F) as u8
    }

    /// Surface attribute flags.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> u8 {
        ((self.data1 >> 24) & 0x0F) as u8
    }

    /// Serializes this quad as two consecutive `u32` values.
    #[inline]
    pub fn write_to_slice(&self, out: &mut [u32]) {
        out[0] = self.data0;
        out[1] = self.data1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_quad_packing_roundtrip() {
        let quad = LodQuad::new(
            17,
            29,
            3,
            8,
            16,
            Face::South,
            14,
            1023,
            [0, 1, 2, 3],
            7,
            LodQuad::FLAG_WATER | LodQuad::FLAG_FOLIAGE,
        );

        assert_eq!(quad.x(), 17);
        assert_eq!(quad.y(), 29);
        assert_eq!(quad.z(), 3);
        assert_eq!(quad.width(), 8);
        assert_eq!(quad.height(), 16);
        assert_eq!(quad.face(), Face::South);
        assert_eq!(quad.sky_light(), 14);
        assert_eq!(quad.color_index(), 1023);
        assert_eq!(quad.ao(), [0, 1, 2, 3]);
        assert_eq!(quad.block_light(), 7);
        assert_eq!(quad.flags(), LodQuad::FLAG_WATER | LodQuad::FLAG_FOLIAGE);
    }
}
