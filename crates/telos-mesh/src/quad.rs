//! Standard 8-byte packed quad (`T0Quad`) and cardinal face direction primitives.

use glam::IVec3;
use telos_voxel::state::BlockStateId;

/// The 6 cardinal face directions for chunk meshing and GPU draw bucketing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum FaceDir {
    /// Positive X (+1, 0, 0)
    PosX = 0,
    /// Negative X (-1, 0, 0)
    NegX = 1,
    /// Positive Y (0, +1, 0)
    PosY = 2,
    /// Negative Y (0, -1, 0)
    NegY = 3,
    /// Positive Z (0, 0, +1)
    PosZ = 4,
    /// Negative Z (0, 0, -1)
    NegZ = 5,
}

impl FaceDir {
    /// Array of all 6 cardinal directions in canonical bucket order.
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    /// Converts a raw discriminant (0..=5) to `FaceDir`.
    #[inline]
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::PosX),
            1 => Some(Self::NegX),
            2 => Some(Self::PosY),
            3 => Some(Self::NegY),
            4 => Some(Self::PosZ),
            5 => Some(Self::NegZ),
            _ => None,
        }
    }

    /// Normal unit vector of this face.
    #[inline]
    #[must_use]
    pub const fn normal(self) -> IVec3 {
        match self {
            Self::PosX => IVec3::new(1, 0, 0),
            Self::NegX => IVec3::new(-1, 0, 0),
            Self::PosY => IVec3::new(0, 1, 0),
            Self::NegY => IVec3::new(0, -1, 0),
            Self::PosZ => IVec3::new(0, 0, 1),
            Self::NegZ => IVec3::new(0, 0, -1),
        }
    }

    /// Tangent frame $(u, v)$ for this face direction.
    /// Right-handed coordinate frame: $u \times v = n$.
    #[inline]
    #[must_use]
    pub const fn tangent_frame(self) -> (IVec3, IVec3) {
        match self {
            Self::PosX => (IVec3::new(0, 1, 0), IVec3::new(0, 0, 1)),
            Self::NegX => (IVec3::new(0, 0, 1), IVec3::new(0, 1, 0)),
            Self::PosY => (IVec3::new(0, 0, 1), IVec3::new(1, 0, 0)),
            Self::NegY => (IVec3::new(1, 0, 0), IVec3::new(0, 0, 1)),
            Self::PosZ => (IVec3::new(1, 0, 0), IVec3::new(0, 1, 0)),
            Self::NegZ => (IVec3::new(0, 1, 0), IVec3::new(1, 0, 0)),
        }
    }

    /// Plane offset along the normal: 1 for positive faces, 0 for negative faces.
    #[inline]
    #[must_use]
    pub const fn plane_offset(self) -> i32 {
        match self {
            Self::PosX | Self::PosY | Self::PosZ => 1,
            Self::NegX | Self::NegY | Self::NegZ => 0,
        }
    }

    /// Opposite cardinal direction.
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::PosX => Self::NegX,
            Self::NegX => Self::PosX,
            Self::PosY => Self::NegY,
            Self::NegY => Self::PosY,
            Self::PosZ => Self::NegZ,
            Self::NegZ => Self::PosZ,
        }
    }
}

/// Standard 8-byte packed quad representation for GPU vertex pulling (`T0Quad`).
///
/// Bit layout matches `vulkan-rendering/SKILL.md` §6.1:
/// - Word 0 (`u32`):
///   - `x`: bits 0..4 (chunk-local min cell, 0..31)
///   - `y`: bits 5..9 (0..31)
///   - `z`: bits 10..14 (0..31)
///   - `w - 1`: bits 15..19 (width along tangent $u$, 1..32)
///   - `h - 1`: bits 20..24 (height along tangent $v$, 1..32)
///   - `dir`: bits 25..27 (0..5)
///   - `reserved`: bits 28..31 (0)
/// - Word 1 (`u32`):
///   - `material`: bits 0..15 (`BlockStateId.0 as u16`)
///   - `light_pattern`: bits 16..29 (default 0)
///   - `reserved`: bits 30..31 (0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(4))]
pub struct T0Quad {
    data: [u32; 2],
}

impl T0Quad {
    /// Creates a packed `T0Quad`.
    ///
    /// # Panics
    /// Panics in debug mode if coordinates or dimensions exceed bounds:
    /// - `x, y, z < 32`
    /// - `w, h ∈ 1..=32`
    #[inline]
    #[must_use]
    pub const fn new(
        x: u32,
        y: u32,
        z: u32,
        w: u32,
        h: u32,
        dir: FaceDir,
        material: u16,
        light_pattern: u16,
    ) -> Self {
        debug_assert!(x < 32 && y < 32 && z < 32, "Coordinates must be < 32");
        debug_assert!(w >= 1 && w <= 32, "Width must be 1..=32");
        debug_assert!(h >= 1 && h <= 32, "Height must be 1..=32");
        debug_assert!(light_pattern < 16384, "Light pattern index must be < 16384");

        let w0 = (x & 0x1F)
            | ((y & 0x1F) << 5)
            | ((z & 0x1F) << 10)
            | (((w - 1) & 0x1F) << 15)
            | (((h - 1) & 0x1F) << 20)
            | (((dir as u8 as u32) & 0x7) << 25);

        let w1 = (material as u32) | (((light_pattern as u32) & 0x3FFF) << 16);

        Self { data: [w0, w1] }
    }

    /// Raw 64-bit pair.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> [u32; 2] {
        self.data
    }

    /// Chunk-local min cell X (0..31).
    #[inline]
    #[must_use]
    pub const fn x(&self) -> u32 {
        self.data[0] & 0x1F
    }

    /// Chunk-local min cell Y (0..31).
    #[inline]
    #[must_use]
    pub const fn y(&self) -> u32 {
        (self.data[0] >> 5) & 0x1F
    }

    /// Chunk-local min cell Z (0..31).
    #[inline]
    #[must_use]
    pub const fn z(&self) -> u32 {
        (self.data[0] >> 10) & 0x1F
    }

    /// Quad width along tangent axis $u$ (1..=32).
    #[inline]
    #[must_use]
    pub const fn w(&self) -> u32 {
        ((self.data[0] >> 15) & 0x1F) + 1
    }

    /// Quad height along tangent axis $v$ (1..=32).
    #[inline]
    #[must_use]
    pub const fn h(&self) -> u32 {
        ((self.data[0] >> 20) & 0x1F) + 1
    }

    /// Cardinal face direction.
    #[inline]
    #[must_use]
    pub const fn dir(&self) -> FaceDir {
        match (self.data[0] >> 25) & 0x7 {
            0 => FaceDir::PosX,
            1 => FaceDir::NegX,
            2 => FaceDir::PosY,
            3 => FaceDir::NegY,
            4 => FaceDir::PosZ,
            _ => FaceDir::NegZ,
        }
    }

    /// Block material ID (`u16`).
    #[inline]
    #[must_use]
    pub const fn material(&self) -> u16 {
        self.data[1] as u16
    }

    /// Block state ID as typed `BlockStateId`.
    #[inline]
    #[must_use]
    pub const fn block_state_id(&self) -> BlockStateId {
        BlockStateId::new(self.material() as u32)
    }

    /// Light pattern table index.
    #[inline]
    #[must_use]
    pub const fn light_pattern(&self) -> u16 {
        ((self.data[1] >> 16) & 0x3FFF) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_round_trip() {
        let quad = T0Quad::new(5, 12, 31, 16, 32, FaceDir::PosY, 142, 3);
        assert_eq!(quad.x(), 5);
        assert_eq!(quad.y(), 12);
        assert_eq!(quad.z(), 31);
        assert_eq!(quad.w(), 16);
        assert_eq!(quad.h(), 32);
        assert_eq!(quad.dir(), FaceDir::PosY);
        assert_eq!(quad.material(), 142);
        assert_eq!(quad.light_pattern(), 3);
    }

    #[test]
    fn test_face_dir_tangent_frames() {
        for dir in FaceDir::ALL {
            let n = dir.normal();
            let (u, v) = dir.tangent_frame();
            // In right-handed coords: u × v = n
            let cross = u.cross(v);
            assert_eq!(cross, n, "Tangent frame cross product mismatch for {dir:?}");
            assert_eq!(dir.opposite().opposite(), dir);
        }
    }
}
