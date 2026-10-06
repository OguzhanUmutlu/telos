//! T1 axis-aligned sub-cuboid detail quad format and mesher.

use vx_core::coords::Face;
use vx_voxel::{
    coords::LocalIdx,
    registry::BlockRegistry,
    shape::{BlockShape, ShapeTier, SubBox},
    storage::Blocks,
};

use crate::{
    light::{LightPattern, LightPatternTable, VoxelNeighborhood, compute_face_pattern},
    mesh::QuadRange,
    quad::FaceDir,
};

/// 16-byte packed detail quad for sub-voxel geometry on the integer 1/16th grid.
///
/// Bit layout matches ADR-08 and analysis/vulkan-rendering/SKILL.md §6.2.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct T1Quad {
    /// [x: 10b | y: 10b | z: 10b | `uv_rot`: 2b] (coords in 1/16th units within chunk 0..512)
    pub word0: u32,
    /// [w-1: 9b | h-1: 9b | dir: 3b | u0: 5b | v0: 5b | `positional_uv`: 1b]
    pub word1: u32,
    /// [u1: 5b | v1: 5b | `no_shade`: 1b | reserved: 5b | material: 16b]
    pub word2: u32,
    /// [`light_pattern`: 14b | reserved: 18b]
    pub word3: u32,
}

impl T1Quad {
    /// Creates a new packed `T1Quad`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[must_use]
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        w: u32,
        h: u32,
        dir: FaceDir,
        material: u16,
        light_pattern: u16,
    ) -> Self {
        assert!(x <= 512, "x out of range: {x}");
        assert!(y <= 512, "y out of range: {y}");
        assert!(z <= 512, "z out of range: {z}");
        assert!((1..=512).contains(&w), "w out of range: {w}");
        assert!((1..=512).contains(&h), "h out of range: {h}");
        assert!(light_pattern < 16384, "light pattern index overflow");

        let word0 = (x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20);
        let word1 = ((w - 1) & 0x1FF)
            | (((h - 1) & 0x1FF) << 9)
            | (((dir as u32) & 0x7) << 18)
            | (1u32 << 31); // positional_uv = 1
        let word2 = (u32::from(material)) << 16;
        let word3 = u32::from(light_pattern) & 0x3FFF;

        Self {
            word0,
            word1,
            word2,
            word3,
        }
    }

    /// Origin coordinate X in sixteenths within chunk (0..512).
    #[inline]
    #[must_use]
    pub const fn x(self) -> u32 {
        self.word0 & 0x3FF
    }

    /// Origin coordinate Y in sixteenths within chunk (0..512).
    #[inline]
    #[must_use]
    pub const fn y(self) -> u32 {
        (self.word0 >> 10) & 0x3FF
    }

    /// Origin coordinate Z in sixteenths within chunk (0..512).
    #[inline]
    #[must_use]
    pub const fn z(self) -> u32 {
        (self.word0 >> 20) & 0x3FF
    }

    /// Width in sixteenths along tangent U (1..512).
    #[inline]
    #[must_use]
    pub const fn w(self) -> u32 {
        (self.word1 & 0x1FF) + 1
    }

    /// Height in sixteenths along tangent V (1..512).
    #[inline]
    #[must_use]
    pub const fn h(self) -> u32 {
        ((self.word1 >> 9) & 0x1FF) + 1
    }

    /// Cardinal face direction.
    #[inline]
    #[must_use]
    pub fn dir(self) -> FaceDir {
        match (self.word1 >> 18) & 0x7 {
            0 => FaceDir::PosX,
            1 => FaceDir::NegX,
            2 => FaceDir::PosY,
            3 => FaceDir::NegY,
            4 => FaceDir::PosZ,
            _ => FaceDir::NegZ,
        }
    }

    /// Texture material identifier.
    #[inline]
    #[must_use]
    pub const fn material(self) -> u16 {
        (self.word2 >> 16) as u16
    }

    /// Light pattern table index.
    #[inline]
    #[must_use]
    pub const fn light_pattern(self) -> u16 {
        (self.word3 & 0x3FFF) as u16
    }
}

/// Mesh buffer containing packed 16-byte `T1Quad` entries organized by cardinal direction,
/// followed by deduplicated `LightPattern` entries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct T1Mesh {
    /// Contiguous array of all T1 quads in the chunk mesh.
    pub quads: Vec<T1Quad>,
    /// Deduplicated light pattern table referenced by `quads`.
    pub patterns: Vec<LightPattern>,
    /// Sub-ranges corresponding to each cardinal direction (+X, -X, +Y, -Y, +Z, -Z).
    pub ranges: [QuadRange; 6],
}

impl T1Mesh {
    /// Creates an empty T1 mesh with 0 quads.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            quads: Vec::new(),
            patterns: Vec::new(),
            ranges: [QuadRange {
                offset: 0,
                count: 0,
            }; 6],
        }
    }

    /// Whether this mesh contains 0 quads.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Total number of T1 quads.
    #[inline]
    #[must_use]
    pub fn total_quads(&self) -> usize {
        self.quads.len()
    }

    /// Writes all quads (4 `u32` each) followed by patterns (2 `u32` each) into a flat buffer.
    pub fn write_to_u32_buffer(&self, out: &mut Vec<u32>) {
        out.reserve(self.quads.len() * 4 + self.patterns.len() * 2);
        for q in &self.quads {
            out.push(q.word0);
            out.push(q.word1);
            out.push(q.word2);
            out.push(q.word3);
        }
        for p in &self.patterns {
            out.extend_from_slice(&p.raw);
        }
    }
}

/// Meshes all T1 sub-cube blocks in a chunk given its block container, neighborhood light sampler, and block registry.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
#[must_use]
pub fn mesh_chunk_t1(
    blocks: &Blocks,
    neighbors: &[Option<&vx_voxel::chunk::ChunkSnapshot>; 6],
    neighborhood: &VoxelNeighborhood<'_>,
    reg: &BlockRegistry,
) -> T1Mesh {
    if blocks.is_uniform() {
        let state = blocks.get(LocalIdx::from_coords_unchecked(0, 0, 0));
        if reg.shape(state).tier() != ShapeTier::T1 {
            return T1Mesh::empty();
        }
    }

    let mut buckets: [Vec<T1Quad>; 6] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut pattern_table = LightPatternTable::new();

    // Helper to query block shape at given chunk or neighbor coordinates
    let get_shape_at = |bx: i32, by: i32, bz: i32| -> &BlockShape {
        if (0..32).contains(&bx) && (0..32).contains(&by) && (0..32).contains(&bz) {
            let idx = LocalIdx::from_coords_unchecked(bx as u32, by as u32, bz as u32);
            reg.shape(blocks.get(idx))
        } else {
            // Find neighbor snapshot
            let (n_idx, nx, ny, nz) = if bx < 0 {
                (1, bx + 32, by, bz) // NegX
            } else if bx >= 32 {
                (0, bx - 32, by, bz) // PosX
            } else if by < 0 {
                (3, bx, by + 32, bz) // NegY
            } else if by >= 32 {
                (2, bx, by - 32, bz) // PosY
            } else if bz < 0 {
                (5, bx, by, bz + 32) // NegZ
            } else {
                (4, bx, by, bz - 32) // PosZ
            };

            if (0..32).contains(&nx) && (0..32).contains(&ny) && (0..32).contains(&nz) {
                if let Some(snap) = neighbors[n_idx] {
                    let idx = LocalIdx::from_coords_unchecked(nx as u32, ny as u32, nz as u32);
                    reg.shape(snap.blocks().get(idx))
                } else {
                    &BlockShape::Empty
                }
            } else {
                &BlockShape::Empty
            }
        }
    };

    for bz in 0..32u32 {
        for by in 0..32u32 {
            for bx in 0..32u32 {
                let idx = LocalIdx::from_coords_unchecked(bx, by, bz);
                let state = blocks.get(idx);
                let shape = reg.shape(state);
                let BlockShape::Boxes(boxes) = shape else {
                    continue;
                };

                let mat = state.0 as u16;

                for (box_idx, b) in boxes.iter().enumerate() {
                    mesh_single_sub_box(
                        *b,
                        box_idx,
                        boxes,
                        bx,
                        by,
                        bz,
                        mat,
                        &get_shape_at,
                        neighborhood,
                        &mut pattern_table,
                        &mut buckets,
                    );
                }
            }
        }
    }

    let mut all_quads = Vec::new();
    let mut ranges = [QuadRange::default(); 6];

    for (i, bucket) in buckets.into_iter().enumerate() {
        let offset = all_quads.len() as u32;
        let count = bucket.len() as u32;
        ranges[i] = QuadRange::new(offset, count);
        all_quads.extend(bucket);
    }

    T1Mesh {
        quads: all_quads,
        patterns: pattern_table.into_patterns(),
        ranges,
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
fn mesh_single_sub_box<'a, F>(
    b: SubBox,
    box_idx: usize,
    all_boxes: &[SubBox],
    bx: u32,
    by: u32,
    bz: u32,
    mat: u16,
    get_shape_at: &F,
    neighborhood: &VoxelNeighborhood<'_>,
    pattern_table: &mut LightPatternTable,
    buckets: &mut [Vec<T1Quad>; 6],
) where
    F: Fn(i32, i32, i32) -> &'a BlockShape,
{
    let base_x = bx * 16;
    let base_y = by * 16;
    let base_z = bz * 16;

    // Evaluate each of the 6 faces for this sub-box
    for dir in FaceDir::ALL {
        let (touches_boundary, is_internal_coplanar) = match dir {
            FaceDir::PosX => (
                b.max[0] == 16,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.min[0] == b.max[0] && boxes_overlap_y_z(b, *other)
                }),
            ),
            FaceDir::NegX => (
                b.min[0] == 0,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.max[0] == b.min[0] && boxes_overlap_y_z(b, *other)
                }),
            ),
            FaceDir::PosY => (
                b.max[1] == 16,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.min[1] == b.max[1] && boxes_overlap_x_z(b, *other)
                }),
            ),
            FaceDir::NegY => (
                b.min[1] == 0,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.max[1] == b.min[1] && boxes_overlap_x_z(b, *other)
                }),
            ),
            FaceDir::PosZ => (
                b.max[2] == 16,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.min[2] == b.max[2] && boxes_overlap_x_y(b, *other)
                }),
            ),
            FaceDir::NegZ => (
                b.min[2] == 0,
                all_boxes.iter().enumerate().any(|(i, other)| {
                    i != box_idx && other.max[2] == b.min[2] && boxes_overlap_x_y(b, *other)
                }),
            ),
        };

        if is_internal_coplanar {
            continue;
        }

        // If touching outer boundary, check occlusion mask of neighbor
        if touches_boundary {
            let (nx, ny, nz, neighbor_face) = match dir {
                FaceDir::PosX => (bx as i32 + 1, by as i32, bz as i32, Face::West),
                FaceDir::NegX => (bx as i32 - 1, by as i32, bz as i32, Face::East),
                FaceDir::PosY => (bx as i32, by as i32 + 1, bz as i32, Face::Down),
                FaceDir::NegY => (bx as i32, by as i32 - 1, bz as i32, Face::Up),
                FaceDir::PosZ => (bx as i32, by as i32, bz as i32 + 1, Face::North),
                FaceDir::NegZ => (bx as i32, by as i32, bz as i32 - 1, Face::South),
            };

            let neighbor_shape = get_shape_at(nx, ny, nz);
            let neighbor_mask = neighbor_shape.occlusion_mask(neighbor_face);

            // Compute this box's outer face rect mask
            let my_face = match dir {
                FaceDir::PosX => Face::East,
                FaceDir::NegX => Face::West,
                FaceDir::PosY => Face::Up,
                FaceDir::NegY => Face::Down,
                FaceDir::PosZ => Face::South,
                FaceDir::NegZ => Face::North,
            };

            if let Some((u0, u1, v0, v1)) = b.boundary_face_rect(my_face) {
                let mut my_mask = vx_voxel::shape::FaceOcclusionMask::EMPTY;
                my_mask.set_rect(u0, u1, v0, v1);
                if my_mask.is_culled_by(&neighbor_mask) {
                    continue;
                }
            }
        }

        // Calculate quad parameters in 1/16th chunk coordinates
        let (qx, qy, qz, qw, qh) = match dir {
            FaceDir::PosX => (
                base_x + u32::from(b.max[0]),
                base_y + u32::from(b.min[1]),
                base_z + u32::from(b.min[2]),
                u32::from(b.max[1] - b.min[1]), // u along +Y
                u32::from(b.max[2] - b.min[2]), // v along +Z
            ),
            FaceDir::NegX => (
                base_x + u32::from(b.min[0]),
                base_y + u32::from(b.min[1]),
                base_z + u32::from(b.min[2]),
                u32::from(b.max[2] - b.min[2]), // u along +Z
                u32::from(b.max[1] - b.min[1]), // v along +Y
            ),
            FaceDir::PosY => (
                base_x + u32::from(b.min[0]),
                base_y + u32::from(b.max[1]),
                base_z + u32::from(b.min[2]),
                u32::from(b.max[2] - b.min[2]), // u along +Z
                u32::from(b.max[0] - b.min[0]), // v along +X
            ),
            FaceDir::NegY => (
                base_x + u32::from(b.min[0]),
                base_y + u32::from(b.min[1]),
                base_z + u32::from(b.min[2]),
                u32::from(b.max[0] - b.min[0]), // u along +X
                u32::from(b.max[2] - b.min[2]), // v along +Z
            ),
            FaceDir::PosZ => (
                base_x + u32::from(b.min[0]),
                base_y + u32::from(b.min[1]),
                base_z + u32::from(b.max[2]),
                u32::from(b.max[0] - b.min[0]), // u along +X
                u32::from(b.max[1] - b.min[1]), // v along +Y
            ),
            FaceDir::NegZ => (
                base_x + u32::from(b.min[0]),
                base_y + u32::from(b.min[1]),
                base_z + u32::from(b.min[2]),
                u32::from(b.max[1] - b.min[1]), // u along +Y
                u32::from(b.max[0] - b.min[0]), // v along +X
            ),
        };

        if qw == 0 || qh == 0 {
            continue;
        }

        let pattern = compute_face_pattern(neighborhood, bx, by, bz, dir);
        let pattern_idx = pattern_table.insert(pattern);

        buckets[dir as usize].push(T1Quad::new(qx, qy, qz, qw, qh, dir, mat, pattern_idx));
    }
}

#[inline]
fn boxes_overlap_y_z(a: SubBox, b: SubBox) -> bool {
    a.min[1] < b.max[1] && a.max[1] > b.min[1] && a.min[2] < b.max[2] && a.max[2] > b.min[2]
}

#[inline]
fn boxes_overlap_x_z(a: SubBox, b: SubBox) -> bool {
    a.min[0] < b.max[0] && a.max[0] > b.min[0] && a.min[2] < b.max[2] && a.max[2] > b.min[2]
}

#[inline]
fn boxes_overlap_x_y(a: SubBox, b: SubBox) -> bool {
    a.min[0] < b.max[0] && a.max[0] > b.min[0] && a.min[1] < b.max[1] && a.max[1] > b.min[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t1_quad_bitpacking_roundtrip() {
        let quad = T1Quad::new(144, 256, 512, 16, 8, FaceDir::PosY, 10, 42);
        assert_eq!(quad.x(), 144);
        assert_eq!(quad.y(), 256);
        assert_eq!(quad.z(), 512);
        assert_eq!(quad.w(), 16);
        assert_eq!(quad.h(), 8);
        assert_eq!(quad.dir(), FaceDir::PosY);
        assert_eq!(quad.material(), 10);
        assert_eq!(quad.light_pattern(), 42);
    }
}
