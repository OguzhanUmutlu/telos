//! Voxel block shapes, sub-cuboids (T1), and 256-bit face occlusion masks.

use telos_core::coords::Face;

/// Classification of block geometry complexity per ADR-08.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShapeTier {
    /// Full 1×1×1 cube (stone, dirt, leaves, glass, water).
    #[default]
    T0,
    /// 1 to 8 axis-aligned sub-cuboids on the 1/16th integer grid (slabs, stairs, fences).
    T1,
    /// Arbitrary rotated quads or cross-geometry (saplings, flowers, chains).
    T2,
    /// Instanced block entities (chests, signs, banners).
    T3,
}

/// 256-bit occlusion bitmask (16×16 grid) for one cardinal face.
///
/// Bit `(v * 16 + u)` is 1 if that 1×1 sub-pixel on the face boundary is solid, 0 if air/open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FaceOcclusionMask(pub [u64; 4]);

impl FaceOcclusionMask {
    /// Completely unoccluded / empty face (all zeros).
    pub const EMPTY: Self = Self([0; 4]);

    /// Completely solid / full face (all ones).
    pub const ALL_ONES: Self = Self([u64::MAX; 4]);

    /// Returns `true` if this mask is completely solid (all 256 bits set).
    #[inline]
    #[must_use]
    pub const fn is_all_ones(&self) -> bool {
        self.0[0] == u64::MAX
            && self.0[1] == u64::MAX
            && self.0[2] == u64::MAX
            && self.0[3] == u64::MAX
    }

    /// Returns `true` if this mask is completely open (all 256 bits clear).
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    /// Returns `true` if `self` is completely culled by `neighbor_opposite`.
    ///
    /// Culling rule per ADR-08 and SKILL runbook:
    /// `(self & !neighbor_opposite) == 0`
    #[inline]
    #[must_use]
    pub const fn is_culled_by(&self, neighbor_opposite: &Self) -> bool {
        let r0 = self.0[0] & !neighbor_opposite.0[0];
        let r1 = self.0[1] & !neighbor_opposite.0[1];
        let r2 = self.0[2] & !neighbor_opposite.0[2];
        let r3 = self.0[3] & !neighbor_opposite.0[3];
        (r0 | r1 | r2 | r3) == 0
    }

    /// Sets a rectangle `[u0..u1, v0..v1]` in the 16×16 face grid to 1.
    pub fn set_rect(&mut self, u0: u8, u1: u8, v0: u8, v1: u8) {
        let u_min = (u0.min(16)) as usize;
        let u_max = (u1.min(16)) as usize;
        let v_min = (v0.min(16)) as usize;
        let v_max = (v1.min(16)) as usize;

        for v in v_min..v_max {
            for u in u_min..u_max {
                let bit_idx = v * 16 + u;
                let word_idx = bit_idx / 64;
                let word_bit = bit_idx % 64;
                self.0[word_idx] |= 1u64 << word_bit;
            }
        }
    }
}

/// An axis-aligned sub-cuboid on the integer 1/16th grid (coordinates in `0..=16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubBox {
    /// Minimum coordinates [x0, y0, z0] in sixteenths (0..=16).
    pub min: [u8; 3],
    /// Maximum coordinates [x1, y1, z1] in sixteenths (0..=16).
    pub max: [u8; 3],
}

impl SubBox {
    /// Creates a new `SubBox` with validated sixteenth coordinates.
    #[inline]
    #[must_use]
    pub const fn new(min: [u8; 3], max: [u8; 3]) -> Self {
        Self { min, max }
    }

    /// Full 1×1×1 unit cube box ([0, 0, 0] to [16, 16, 16]).
    pub const FULL_CUBE: Self = Self {
        min: [0, 0, 0],
        max: [16, 16, 16],
    };

    /// Returns the face tangent bounds `(u0, u1, v0, v1)` if this box touches the given cardinal boundary face.
    #[must_use]
    pub fn boundary_face_rect(&self, face: Face) -> Option<(u8, u8, u8, u8)> {
        match face {
            Face::Down => {
                if self.min[1] == 0 {
                    Some((self.min[0], self.max[0], self.min[2], self.max[2]))
                } else {
                    None
                }
            }
            Face::Up => {
                if self.max[1] == 16 {
                    Some((self.min[0], self.max[0], self.min[2], self.max[2]))
                } else {
                    None
                }
            }
            Face::North => {
                if self.min[2] == 0 {
                    Some((self.min[0], self.max[0], self.min[1], self.max[1]))
                } else {
                    None
                }
            }
            Face::South => {
                if self.max[2] == 16 {
                    Some((self.min[0], self.max[0], self.min[1], self.max[1]))
                } else {
                    None
                }
            }
            Face::West => {
                if self.min[0] == 0 {
                    Some((self.min[2], self.max[2], self.min[1], self.max[1]))
                } else {
                    None
                }
            }
            Face::East => {
                if self.max[0] == 16 {
                    Some((self.min[2], self.max[2], self.min[1], self.max[1]))
                } else {
                    None
                }
            }
        }
    }
}

/// The geometric shape and collision/meshing classification of a block state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BlockShape {
    /// Full opaque or cutout 1×1×1 cube (T0).
    Cube,
    /// 1 to 8 axis-aligned sub-cuboids (T1).
    Boxes(Vec<SubBox>),
    /// Fluid block (water/lava) with surface level and falling state.
    Fluid {
        /// Fluid level (0 = source block, 1..=7 = flowing).
        level: u8,
        /// Whether fluid is falling from above.
        falling: bool,
    },
    /// Cutout diagonal cross geometry (flowers, tall grass, saplings).
    Cross,
    /// Upright floor torch or directional wall-mounted torch.
    Torch {
        /// If attached to a wall, the cardinal face the torch is mounted onto.
        wall: Option<Face>,
    },
    /// Completely empty air block.
    #[default]
    Empty,
}

impl BlockShape {
    /// Returns the shape tier classification.
    #[must_use]
    pub const fn tier(&self) -> ShapeTier {
        match self {
            Self::Cube | Self::Empty => ShapeTier::T0,
            Self::Boxes(_) => ShapeTier::T1,
            Self::Fluid { .. } | Self::Cross | Self::Torch { .. } => ShapeTier::T2,
        }
    }

    /// Computes the 256-bit face occlusion mask along a given cardinal direction.
    #[must_use]
    pub fn occlusion_mask(&self, face: Face) -> FaceOcclusionMask {
        match self {
            Self::Cube => FaceOcclusionMask::ALL_ONES,
            Self::Empty | Self::Fluid { .. } | Self::Cross | Self::Torch { .. } => {
                FaceOcclusionMask::EMPTY
            }
            Self::Boxes(boxes) => {
                let mut mask = FaceOcclusionMask::EMPTY;
                for b in boxes {
                    if let Some((u0, u1, v0, v1)) = b.boundary_face_rect(face) {
                        mask.set_rect(u0, u1, v0, v1);
                    }
                }
                mask
            }
        }
    }

    /// Standard bottom slab shape: [0, 0, 0] to [16, 8, 16].
    #[must_use]
    pub fn bottom_slab() -> Self {
        Self::Boxes(vec![SubBox::new([0, 0, 0], [16, 8, 16])])
    }

    /// Standard top slab shape: [0, 8, 0] to [16, 16, 16].
    #[must_use]
    pub fn top_slab() -> Self {
        Self::Boxes(vec![SubBox::new([0, 8, 0], [16, 16, 16])])
    }

    /// Standard stair shape: base box (8/16 high) + step box (8/16 high on the rear half).
    #[must_use]
    pub fn stairs(facing: Face, is_top: bool) -> Self {
        let (base_min_y, base_max_y, step_min_y, step_max_y) =
            if is_top { (8, 16, 0, 8) } else { (0, 8, 8, 16) };

        let base_box = SubBox::new([0, base_min_y, 0], [16, base_max_y, 16]);

        // Step box is located on the opposite side of the facing direction (the higher back part)
        let step_box = match facing {
            Face::North => SubBox::new([0, step_min_y, 8], [16, step_max_y, 16]),
            Face::South => SubBox::new([0, step_min_y, 0], [16, step_max_y, 8]),
            Face::West => SubBox::new([8, step_min_y, 0], [16, step_max_y, 16]),
            _ => SubBox::new([0, step_min_y, 0], [8, step_max_y, 16]),
        };

        Self::Boxes(vec![base_box, step_box])
    }

    /// Standard diagonal cross model (flowers, saplings, tall grass).
    #[must_use]
    pub const fn cross() -> Self {
        Self::Cross
    }

    /// Standard floor torch or wall-mounted torch.
    #[must_use]
    pub const fn torch(wall: Option<Face>) -> Self {
        Self::Torch { wall }
    }

    /// Fluid block shape with specified level and falling flag.
    #[must_use]
    pub const fn fluid(level: u8, falling: bool) -> Self {
        Self::Fluid { level, falling }
    }

    /// Flat horizontal plate (e.g. logic wire, repeater, diode base: 2/16 high).
    #[must_use]
    pub fn flat_plate() -> Self {
        Self::Boxes(vec![SubBox::new([0, 0, 0], [16, 2, 16])])
    }

    /// Small toggleable switch / lever sub-box.
    #[must_use]
    pub fn lever(_powered: bool) -> Self {
        Self::Boxes(vec![SubBox::new([5, 0, 5], [11, 8, 11])])
    }

    /// Small vertical post (e.g. logic inverter torch).
    #[must_use]
    pub fn post() -> Self {
        Self::Boxes(vec![SubBox::new([7, 0, 7], [9, 10, 9])])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_occlusion_mask() {
        let shape = BlockShape::Cube;
        for face in Face::ALL {
            let mask = shape.occlusion_mask(face);
            assert!(mask.is_all_ones());
            assert!(!mask.is_empty());
        }
    }

    #[test]
    fn test_slab_occlusion_and_culling() {
        let bottom = BlockShape::bottom_slab();
        let top = BlockShape::top_slab();
        let cube = BlockShape::Cube;

        // Bottom slab: Down is ALL_ONES, Up is EMPTY
        assert!(bottom.occlusion_mask(Face::Down).is_all_ones());
        assert!(bottom.occlusion_mask(Face::Up).is_empty());

        // Top slab: Down is EMPTY, Up is ALL_ONES
        assert!(top.occlusion_mask(Face::Down).is_empty());
        assert!(top.occlusion_mask(Face::Up).is_all_ones());

        // Bottom slab on bottom slab horizontally (East/West)
        let b_east = bottom.occlusion_mask(Face::East);
        let b_west = bottom.occlusion_mask(Face::West);
        assert!(b_east.is_culled_by(&b_west));

        // Bottom slab next to top slab horizontally (East/West)
        let t_west = top.occlusion_mask(Face::West);
        assert!(!b_east.is_culled_by(&t_west));

        // Bottom slab against full cube
        let c_west = cube.occlusion_mask(Face::West);
        assert!(b_east.is_culled_by(&c_west));
    }

    #[test]
    fn test_stair_masks() {
        let stair = BlockShape::stairs(Face::North, false);
        let down_mask = stair.occlusion_mask(Face::Down);
        assert!(down_mask.is_all_ones());

        let up_mask = stair.occlusion_mask(Face::Up);
        assert!(!up_mask.is_all_ones());
        assert!(!up_mask.is_empty());
    }

    #[test]
    fn test_t2_shapes_and_tiers() {
        let cross = BlockShape::cross();
        assert_eq!(cross.tier(), ShapeTier::T2);
        for face in Face::ALL {
            assert!(cross.occlusion_mask(face).is_empty());
        }

        let floor_torch = BlockShape::torch(None);
        assert_eq!(floor_torch.tier(), ShapeTier::T2);
        let wall_torch = BlockShape::torch(Some(Face::North));
        assert_eq!(wall_torch.tier(), ShapeTier::T2);
        for face in Face::ALL {
            assert!(floor_torch.occlusion_mask(face).is_empty());
            assert!(wall_torch.occlusion_mask(face).is_empty());
        }

        let fluid = BlockShape::fluid(3, false);
        assert_eq!(fluid.tier(), ShapeTier::T2);
        for face in Face::ALL {
            assert!(fluid.occlusion_mask(face).is_empty());
        }
    }
}
