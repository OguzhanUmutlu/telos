//! T2 arbitrary quad generic vertex format, cross plant mesher, torch mesher, and sloped fluid mesher.
//!
//! Implements ADR-08 and `analysis/vulkan-rendering/SKILL.md` §6.2:
//! - 16-byte packed generic vertex `T2Vertex` (1/1024 coordinate resolution over chunk space).
//! - 64-byte `T2Quad` (4 `T2Vertex` entries) using the shared 6-index quad buffer `[0, 1, 2, 2, 3, 0]`.
//! - `T2Mesh` buffer container with serializable GPU word emission.
//! - Diagonal cross plant geometry for flowers, saplings, and tall grass.
//! - Floor and directional wall-mounted torch geometry.
//! - Continuous fluid surface corner height interpolation with level-gradient side faces.
//! - Waterlogging fluid volume meshing.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use glam::Vec3;
use telos_core::coords::Face;
use telos_voxel::shape::BlockShape;

/// 16-byte packed generic vertex representation for arbitrary/angled voxel geometry (`T2Vertex`).
///
/// Bit layout matches `vulkan-rendering/SKILL.md` §6.2:
/// - `word0`: `pos.x (16b) | pos.y (16b)` (coordinates in 1/1024th units within chunk 0..32 blocks)
/// - `word1`: `pos.z (16b) | uv.u (16b)` (z in 1/1024th units, u in normalized 0..65535)
/// - `word2`: `uv.v (16b) | material (16b)` (v in normalized 0..65535, material layer ID)
/// - `word3`: `light (16b) | normal (16b)` (light components + octahedral normal encoding)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct T2Vertex {
    /// Position X (0..15) and Position Y (16..31) in 1/1024 units.
    pub word0: u32,
    /// Position Z (0..15) in 1/1024 units and Texture U (16..31) in 0..65535.
    pub word1: u32,
    /// Texture V (0..15) in 0..65535 and Material ID (16..31).
    pub word2: u32,
    /// Packed light (0..15) and Octahedral normal (16..31).
    pub word3: u32,
}

impl T2Vertex {
    /// Creates a packed `T2Vertex`.
    ///
    /// # Arguments
    /// - `pos`: Chunk-relative coordinates in blocks $[0.0, 32.0]$.
    /// - `uv`: Texture coordinates in $[0.0, 1.0]$.
    /// - `material`: Texture array layer ID.
    /// - `normal`: Normalized 3D normal vector.
    /// - `ao`: Ambient occlusion factor ($0..=3$).
    /// - `sky`: Sky light level ($0..=15$).
    /// - `block`: Block light level ($0..=15$).
    /// - `emissive`: Whether vertex is self-illuminating.
    /// - `shade`: Whether directional normal face shading is applied.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        pos: Vec3,
        uv: [f32; 2],
        material: u16,
        normal: Vec3,
        ao: u8,
        sky: u8,
        block: u8,
        emissive: bool,
        shade: bool,
    ) -> Self {
        // Quantize position into 1/1024 units (0..32 blocks -> 0..32768)
        let qx = (pos.x * 1024.0).clamp(0.0, 65535.0) as u16;
        let qy = (pos.y * 1024.0).clamp(0.0, 65535.0) as u16;
        let qz = (pos.z * 1024.0).clamp(0.0, 65535.0) as u16;

        // Quantize UVs into 0..65535
        let qu = (uv[0].clamp(0.0, 1.0) * 65535.0) as u16;
        let qv = (uv[1].clamp(0.0, 1.0) * 65535.0) as u16;

        // Encode octahedral normal
        let (nx, ny) = encode_octahedral_normal(normal);

        // Pack light: [ao: 2b, sky: 4b, block: 4b, emissive: 1b, shade: 1b, reserved: 4b]
        let mut light_word =
            (u16::from(ao & 0x03)) | (u16::from(sky & 0x0F) << 2) | (u16::from(block & 0x0F) << 6);
        if emissive {
            light_word |= 1 << 10;
        }
        if shade {
            light_word |= 1 << 11;
        }

        let word0 = u32::from(qx) | (u32::from(qy) << 16);
        let word1 = u32::from(qz) | (u32::from(qu) << 16);
        let word2 = u32::from(qv) | (u32::from(material) << 16);
        let word3 =
            u32::from(light_word) | (u32::from(nx as u8) << 16) | (u32::from(ny as u8) << 24);

        Self {
            word0,
            word1,
            word2,
            word3,
        }
    }

    /// Position in chunk-relative blocks.
    #[must_use]
    pub fn pos(self) -> Vec3 {
        let x = (self.word0 & 0xFFFF) as f32 / 1024.0;
        let y = ((self.word0 >> 16) & 0xFFFF) as f32 / 1024.0;
        let z = (self.word1 & 0xFFFF) as f32 / 1024.0;
        Vec3::new(x, y, z)
    }

    /// Texture UV in $[0.0, 1.0]$.
    #[must_use]
    pub fn uv(self) -> [f32; 2] {
        let u = ((self.word1 >> 16) & 0xFFFF) as f32 / 65535.0;
        let v = (self.word2 & 0xFFFF) as f32 / 65535.0;
        [u, v]
    }

    /// Material texture array layer index.
    #[must_use]
    pub const fn material(self) -> u16 {
        (self.word2 >> 16) as u16
    }

    /// Ambient occlusion factor ($0..=3$).
    #[must_use]
    pub const fn ao(self) -> u8 {
        (self.word3 & 0x03) as u8
    }

    /// Sky light level ($0..=15$).
    #[must_use]
    pub const fn sky(self) -> u8 {
        ((self.word3 >> 2) & 0x0F) as u8
    }

    /// Block light level ($0..=15$).
    #[must_use]
    pub const fn block(self) -> u8 {
        ((self.word3 >> 6) & 0x0F) as u8
    }

    /// Whether vertex is self-illuminating.
    #[must_use]
    pub const fn is_emissive(self) -> bool {
        (self.word3 & (1 << 10)) != 0
    }

    /// Whether face normal directional shading factor is applied.
    #[must_use]
    pub const fn has_shading(self) -> bool {
        (self.word3 & (1 << 11)) != 0
    }

    /// Decoded normal vector.
    #[must_use]
    pub fn normal(self) -> Vec3 {
        let nx = (self.word3 >> 16) as i8;
        let ny = (self.word3 >> 24) as i8;
        decode_octahedral_normal(nx, ny)
    }
}

/// Encodes a normalized 3D vector into 2 octahedral signed bytes.
#[must_use]
pub fn encode_octahedral_normal(n: Vec3) -> (i8, i8) {
    let l1_norm = n.x.abs() + n.y.abs() + n.z.abs();
    if l1_norm < 1e-6 {
        return (0, 0);
    }
    let mut p = Vec3::new(n.x / l1_norm, n.y / l1_norm, n.z / l1_norm);
    if p.y < 0.0 {
        let ox = (1.0 - p.z.abs()) * if p.x >= 0.0 { 1.0 } else { -1.0 };
        let oz = (1.0 - p.x.abs()) * if p.z >= 0.0 { 1.0 } else { -1.0 };
        p.x = ox;
        p.z = oz;
    }
    let ix = (p.x * 127.0).clamp(-127.0, 127.0) as i8;
    let iz = (p.z * 127.0).clamp(-127.0, 127.0) as i8;
    (ix, iz)
}

/// Decodes 2 octahedral signed bytes into a normalized 3D vector.
#[must_use]
pub fn decode_octahedral_normal(nx: i8, nz: i8) -> Vec3 {
    let x = (f32::from(nx) / 127.0).clamp(-1.0, 1.0);
    let z = (f32::from(nz) / 127.0).clamp(-1.0, 1.0);
    let y = 1.0 - x.abs() - z.abs();
    let mut n = Vec3::new(x, y, z);
    if n.y < 0.0 {
        let ox = (1.0 - n.z.abs()) * if n.x >= 0.0 { 1.0 } else { -1.0 };
        let oz = (1.0 - n.x.abs()) * if n.z >= 0.0 { 1.0 } else { -1.0 };
        n.x = ox;
        n.z = oz;
    }
    n.normalize_or_zero()
}

/// 64-byte quad formed by 4 `T2Vertex` corners (`v0, v1, v2, v3`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct T2Quad {
    /// Corner 0 (bottom-left)
    pub v0: T2Vertex,
    /// Corner 1 (bottom-right)
    pub v1: T2Vertex,
    /// Corner 2 (top-right)
    pub v2: T2Vertex,
    /// Corner 3 (top-left)
    pub v3: T2Vertex,
}

impl T2Quad {
    /// Creates a new `T2Quad` from 4 corners.
    #[inline]
    #[must_use]
    pub const fn new(v0: T2Vertex, v1: T2Vertex, v2: T2Vertex, v3: T2Vertex) -> Self {
        Self { v0, v1, v2, v3 }
    }
}

/// Mesh buffer container holding packed arbitrary `T2Quad` entries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct T2Mesh {
    /// Contiguous array of quads in this mesh layer.
    pub quads: Vec<T2Quad>,
}

impl T2Mesh {
    /// Creates an empty `T2Mesh`.
    #[must_use]
    pub const fn empty() -> Self {
        Self { quads: Vec::new() }
    }

    /// Whether this mesh contains 0 quads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Number of quads in this mesh.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.quads.len()
    }

    /// Number of vertices represented (4 per quad).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.quads.len() * 4
    }

    /// Writes all quad vertices into a flat `u32` array for GPU upload.
    ///
    /// Each quad emits $4 \times 4 = 16$ `u32` words (64 bytes).
    pub fn write_to_u32_buffer(&self, out: &mut Vec<u32>) {
        out.reserve(self.quads.len() * 16);
        for q in &self.quads {
            out.push(q.v0.word0);
            out.push(q.v0.word1);
            out.push(q.v0.word2);
            out.push(q.v0.word3);

            out.push(q.v1.word0);
            out.push(q.v1.word1);
            out.push(q.v1.word2);
            out.push(q.v1.word3);

            out.push(q.v2.word0);
            out.push(q.v2.word1);
            out.push(q.v2.word2);
            out.push(q.v2.word3);

            out.push(q.v3.word0);
            out.push(q.v3.word1);
            out.push(q.v3.word2);
            out.push(q.v3.word3);
        }
    }
}

/// Tessellates a diagonal cross model (flowers, saplings, tall grass) at chunk cell `(bx, by, bz)`.
///
/// Emits 4 double-sided quads across the 2 diagonals of the unit block cell.
pub fn mesh_cross_model(
    bx: u32,
    by: u32,
    bz: u32,
    material: u16,
    sky: u8,
    block: u8,
    out: &mut Vec<T2Quad>,
) {
    let fx = bx as f32;
    let fy = by as f32;
    let fz = bz as f32;

    let p000 = Vec3::new(fx, fy, fz);
    let p100 = Vec3::new(fx + 1.0, fy, fz);
    let p001 = Vec3::new(fx, fy, fz + 1.0);
    let p101 = Vec3::new(fx + 1.0, fy, fz + 1.0);

    let p010 = Vec3::new(fx, fy + 1.0, fz);
    let p110 = Vec3::new(fx + 1.0, fy + 1.0, fz);
    let p011 = Vec3::new(fx, fy + 1.0, fz + 1.0);
    let p111 = Vec3::new(fx + 1.0, fy + 1.0, fz + 1.0);

    let n1 = Vec3::new(1.0, 0.0, -1.0).normalize();
    let n2 = Vec3::new(1.0, 0.0, 1.0).normalize();

    // Diagonal 1: (0, 0) to (1, 1)
    // Forward quad
    let v0 = T2Vertex::new(p000, [0.0, 1.0], material, n1, 3, sky, block, false, false);
    let v1 = T2Vertex::new(p101, [1.0, 1.0], material, n1, 3, sky, block, false, false);
    let v2 = T2Vertex::new(p111, [1.0, 0.0], material, n1, 3, sky, block, false, false);
    let v3 = T2Vertex::new(p010, [0.0, 0.0], material, n1, 3, sky, block, false, false);
    out.push(T2Quad::new(v0, v1, v2, v3));

    // Reverse quad
    let v0_r = T2Vertex::new(p101, [0.0, 1.0], material, -n1, 3, sky, block, false, false);
    let v1_r = T2Vertex::new(p000, [1.0, 1.0], material, -n1, 3, sky, block, false, false);
    let v2_r = T2Vertex::new(p010, [1.0, 0.0], material, -n1, 3, sky, block, false, false);
    let v3_r = T2Vertex::new(p111, [0.0, 0.0], material, -n1, 3, sky, block, false, false);
    out.push(T2Quad::new(v0_r, v1_r, v2_r, v3_r));

    // Diagonal 2: (1, 0) to (0, 1)
    // Forward quad
    let v0_2 = T2Vertex::new(p100, [0.0, 1.0], material, n2, 3, sky, block, false, false);
    let v1_2 = T2Vertex::new(p001, [1.0, 1.0], material, n2, 3, sky, block, false, false);
    let v2_2 = T2Vertex::new(p011, [1.0, 0.0], material, n2, 3, sky, block, false, false);
    let v3_2 = T2Vertex::new(p110, [0.0, 0.0], material, n2, 3, sky, block, false, false);
    out.push(T2Quad::new(v0_2, v1_2, v2_2, v3_2));

    // Reverse quad
    let v0_2r = T2Vertex::new(p001, [0.0, 1.0], material, -n2, 3, sky, block, false, false);
    let v1_2r = T2Vertex::new(p100, [1.0, 1.0], material, -n2, 3, sky, block, false, false);
    let v2_2r = T2Vertex::new(p110, [1.0, 0.0], material, -n2, 3, sky, block, false, false);
    let v3_2r = T2Vertex::new(p011, [0.0, 0.0], material, -n2, 3, sky, block, false, false);
    out.push(T2Quad::new(v0_2r, v1_2r, v2_2r, v3_2r));
}

/// Tessellates an upright or wall-mounted torch at chunk cell `(bx, by, bz)`.
pub fn mesh_torch_model(
    bx: u32,
    by: u32,
    bz: u32,
    wall: Option<Face>,
    material: u16,
    out: &mut Vec<T2Quad>,
) {
    let fx = bx as f32;
    let fy = by as f32;
    let fz = bz as f32;

    // Floor torch: centered 2x10x2 sixteenths stick
    let min_x = 7.0 / 16.0;
    let max_x = 9.0 / 16.0;
    let min_y = 0.0;
    let max_y = 10.0 / 16.0;
    let min_z = 7.0 / 16.0;
    let max_z = 9.0 / 16.0;

    let mut corners = [
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, min_y, min_z),
        Vec3::new(max_x, min_y, max_z),
        Vec3::new(min_x, min_y, max_z),
        Vec3::new(min_x, max_y, min_z),
        Vec3::new(max_x, max_y, min_z),
        Vec3::new(max_x, max_y, max_z),
        Vec3::new(min_x, max_y, max_z),
    ];

    // If attached to a wall, apply 22.5-degree lean away from the wall
    if let Some(wall_face) = wall {
        let angle_rad = 22.5f32.to_radians();
        let (rot_axis, translation) = match wall_face {
            Face::North => (Vec3::X, Vec3::new(0.0, 3.5 / 16.0, 7.0 / 16.0)),
            Face::South => (-Vec3::X, Vec3::new(0.0, 3.5 / 16.0, -7.0 / 16.0)),
            Face::West => (-Vec3::Z, Vec3::new(7.0 / 16.0, 3.5 / 16.0, 0.0)),
            Face::East => (Vec3::Z, Vec3::new(-7.0 / 16.0, 3.5 / 16.0, 0.0)),
            _ => (Vec3::ZERO, Vec3::ZERO),
        };

        if rot_axis != Vec3::ZERO {
            let rot = glam::Quat::from_axis_angle(rot_axis, angle_rad);
            for c in &mut corners {
                *c = rot * (*c - Vec3::new(0.5, 0.0, 0.5)) + Vec3::new(0.5, 0.0, 0.5) + translation;
            }
        }
    }

    // Offset to block chunk coordinate
    let block_offset = Vec3::new(fx, fy, fz);
    for c in &mut corners {
        *c += block_offset;
    }

    let emissive = true;
    let shade = false;
    let sky = 15;
    let block = 15;
    let ao = 3;

    // Up face (top of torch)
    let n_up = Vec3::Y;
    let v0 = T2Vertex::new(
        corners[7],
        [7.0 / 16.0, 8.0 / 16.0],
        material,
        n_up,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v1 = T2Vertex::new(
        corners[6],
        [9.0 / 16.0, 8.0 / 16.0],
        material,
        n_up,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v2 = T2Vertex::new(
        corners[5],
        [9.0 / 16.0, 6.0 / 16.0],
        material,
        n_up,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v3 = T2Vertex::new(
        corners[4],
        [7.0 / 16.0, 6.0 / 16.0],
        material,
        n_up,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    out.push(T2Quad::new(v0, v1, v2, v3));

    // North face (-Z)
    let n_north = -Vec3::Z;
    let v0 = T2Vertex::new(
        corners[0],
        [7.0 / 16.0, 1.0],
        material,
        n_north,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v1 = T2Vertex::new(
        corners[1],
        [9.0 / 16.0, 1.0],
        material,
        n_north,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v2 = T2Vertex::new(
        corners[5],
        [9.0 / 16.0, 6.0 / 16.0],
        material,
        n_north,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v3 = T2Vertex::new(
        corners[4],
        [7.0 / 16.0, 6.0 / 16.0],
        material,
        n_north,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    out.push(T2Quad::new(v0, v1, v2, v3));

    // South face (+Z)
    let n_south = Vec3::Z;
    let v0 = T2Vertex::new(
        corners[2],
        [7.0 / 16.0, 1.0],
        material,
        n_south,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v1 = T2Vertex::new(
        corners[3],
        [9.0 / 16.0, 1.0],
        material,
        n_south,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v2 = T2Vertex::new(
        corners[7],
        [9.0 / 16.0, 6.0 / 16.0],
        material,
        n_south,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v3 = T2Vertex::new(
        corners[6],
        [7.0 / 16.0, 6.0 / 16.0],
        material,
        n_south,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    out.push(T2Quad::new(v0, v1, v2, v3));

    // West face (-X)
    let n_west = -Vec3::X;
    let v0 = T2Vertex::new(
        corners[3],
        [7.0 / 16.0, 1.0],
        material,
        n_west,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v1 = T2Vertex::new(
        corners[0],
        [9.0 / 16.0, 1.0],
        material,
        n_west,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v2 = T2Vertex::new(
        corners[4],
        [9.0 / 16.0, 6.0 / 16.0],
        material,
        n_west,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v3 = T2Vertex::new(
        corners[7],
        [7.0 / 16.0, 6.0 / 16.0],
        material,
        n_west,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    out.push(T2Quad::new(v0, v1, v2, v3));

    // East face (+X)
    let n_east = Vec3::X;
    let v0 = T2Vertex::new(
        corners[1],
        [7.0 / 16.0, 1.0],
        material,
        n_east,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v1 = T2Vertex::new(
        corners[2],
        [9.0 / 16.0, 1.0],
        material,
        n_east,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v2 = T2Vertex::new(
        corners[6],
        [9.0 / 16.0, 6.0 / 16.0],
        material,
        n_east,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    let v3 = T2Vertex::new(
        corners[5],
        [7.0 / 16.0, 6.0 / 16.0],
        material,
        n_east,
        ao,
        sky,
        block,
        emissive,
        shade,
    );
    out.push(T2Quad::new(v0, v1, v2, v3));
}

/// Computes the fluid surface height in $[0.0, 1.0]$ for a given fluid block shape.
#[must_use]
pub fn fluid_surface_height(shape: &BlockShape) -> Option<f32> {
    match shape {
        BlockShape::Fluid { level, falling } => {
            if *falling {
                Some(1.0)
            } else if *level == 0 {
                Some(14.0 / 16.0)
            } else {
                let clamped_lvl = f32::from((*level).min(7));
                Some((8.0 - clamped_lvl) / 9.0)
            }
        }
        _ => None,
    }
}

/// Interpolates fluid surface height at a specific horizontal corner `(u, v)` of cell `(bx, by, bz)`.
///
/// Corner `(0, 0)` is min-X, min-Z.
/// Corner `(1, 0)` is max-X, min-Z.
/// Corner `(1, 1)` is max-X, max-Z.
/// Corner `(0, 1)` is min-X, max-Z.
pub fn interpolate_fluid_corner_height<F>(
    bx: i32,
    by: i32,
    bz: i32,
    corner_u: u8,
    corner_v: u8,
    get_shape: &F,
) -> f32
where
    F: Fn(i32, i32, i32) -> BlockShape,
{
    // Check if any fluid exists immediately above this corner at y + 1
    // If so, snap to 1.0 to eliminate visual seams beneath falling fluid columns
    let offsets_x = if corner_u == 0 { [0, -1] } else { [0, 1] };
    let offsets_z = if corner_v == 0 { [0, -1] } else { [0, 1] };

    for &dx in &offsets_x {
        for &dz in &offsets_z {
            if fluid_surface_height(&get_shape(bx + dx, by + 1, bz + dz)).is_some() {
                return 1.0;
            }
        }
    }

    // Otherwise, average surface heights of all fluid neighbors touching this corner
    let mut sum_height = 0.0f32;
    let mut count = 0u32;

    for &dx in &offsets_x {
        for &dz in &offsets_z {
            if let Some(h) = fluid_surface_height(&get_shape(bx + dx, by, bz + dz)) {
                sum_height += h;
                count += 1;
            }
        }
    }

    if count > 0 {
        sum_height / count as f32
    } else {
        14.0 / 16.0
    }
}

/// Meshes a fluid voxel cell with corner height interpolation, sloped surface normals, and side face culling.
pub fn mesh_fluid_cell<F>(
    bx: u32,
    by: u32,
    bz: u32,
    material: u16,
    sky: u8,
    block: u8,
    get_shape: &F,
    out: &mut Vec<T2Quad>,
) where
    F: Fn(i32, i32, i32) -> BlockShape,
{
    let ibx = bx as i32;
    let iby = by as i32;
    let ibz = bz as i32;

    let fx = bx as f32;
    let fy = by as f32;
    let fz = bz as f32;

    // Corner heights
    let h00 = interpolate_fluid_corner_height(ibx, iby, ibz, 0, 0, get_shape);
    let h10 = interpolate_fluid_corner_height(ibx, iby, ibz, 1, 0, get_shape);
    let h11 = interpolate_fluid_corner_height(ibx, iby, ibz, 1, 1, get_shape);
    let h01 = interpolate_fluid_corner_height(ibx, iby, ibz, 0, 1, get_shape);

    let shape_above = get_shape(ibx, iby + 1, ibz);
    let is_fluid_above = fluid_surface_height(&shape_above).is_some();

    // 1. Top face (only if no fluid directly above)
    if !is_fluid_above {
        let p0 = Vec3::new(fx, fy + h00, fz);
        let p1 = Vec3::new(fx + 1.0, fy + h10, fz);
        let p2 = Vec3::new(fx + 1.0, fy + h11, fz + 1.0);
        let p3 = Vec3::new(fx, fy + h01, fz + 1.0);

        let normal = (p1 - p0).cross(p3 - p0).normalize_or_zero();
        let n = if normal.y > 0.0 { normal } else { Vec3::Y };

        let v0 = T2Vertex::new(p0, [0.0, 0.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 1.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }

    // 2. Bottom face (if block below is open/empty)
    let shape_below = get_shape(ibx, iby - 1, ibz);
    if fluid_surface_height(&shape_below).is_none() && shape_below != BlockShape::Cube {
        let p0 = Vec3::new(fx, fy, fz + 1.0);
        let p1 = Vec3::new(fx + 1.0, fy, fz + 1.0);
        let p2 = Vec3::new(fx + 1.0, fy, fz);
        let p3 = Vec3::new(fx, fy, fz);

        let n = -Vec3::Y;
        let v0 = T2Vertex::new(p0, [0.0, 1.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 0.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }

    // 3. Side faces (North, South, West, East)
    // North (-Z): along edge from (0, 0) to (1, 0)
    let shape_n = get_shape(ibx, iby, ibz - 1);
    if should_emit_fluid_side(&shape_n, is_fluid_above, h00.max(h10)) {
        let bottom_y = fy;
        let p0 = Vec3::new(fx, bottom_y, fz);
        let p1 = Vec3::new(fx + 1.0, bottom_y, fz);
        let p2 = Vec3::new(fx + 1.0, fy + h10, fz);
        let p3 = Vec3::new(fx, fy + h00, fz);

        let n = -Vec3::Z;
        let v0 = T2Vertex::new(p0, [0.0, 1.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 0.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }

    // South (+Z): along edge from (1, 1) to (0, 1)
    let shape_s = get_shape(ibx, iby, ibz + 1);
    if should_emit_fluid_side(&shape_s, is_fluid_above, h01.max(h11)) {
        let bottom_y = fy;
        let p0 = Vec3::new(fx + 1.0, bottom_y, fz + 1.0);
        let p1 = Vec3::new(fx, bottom_y, fz + 1.0);
        let p2 = Vec3::new(fx, fy + h01, fz + 1.0);
        let p3 = Vec3::new(fx + 1.0, fy + h11, fz + 1.0);

        let n = Vec3::Z;
        let v0 = T2Vertex::new(p0, [0.0, 1.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 0.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }

    // West (-X): along edge from (0, 1) to (0, 0)
    let shape_w = get_shape(ibx - 1, iby, ibz);
    if should_emit_fluid_side(&shape_w, is_fluid_above, h00.max(h01)) {
        let bottom_y = fy;
        let p0 = Vec3::new(fx, bottom_y, fz + 1.0);
        let p1 = Vec3::new(fx, bottom_y, fz);
        let p2 = Vec3::new(fx, fy + h00, fz);
        let p3 = Vec3::new(fx, fy + h01, fz + 1.0);

        let n = -Vec3::X;
        let v0 = T2Vertex::new(p0, [0.0, 1.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 0.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }

    // East (+X): along edge from (1, 0) to (1, 1)
    let shape_e = get_shape(ibx + 1, iby, ibz);
    if should_emit_fluid_side(&shape_e, is_fluid_above, h10.max(h11)) {
        let bottom_y = fy;
        let p0 = Vec3::new(fx + 1.0, bottom_y, fz);
        let p1 = Vec3::new(fx + 1.0, bottom_y, fz + 1.0);
        let p2 = Vec3::new(fx + 1.0, fy + h11, fz + 1.0);
        let p3 = Vec3::new(fx + 1.0, fy + h10, fz);

        let n = Vec3::X;
        let v0 = T2Vertex::new(p0, [0.0, 1.0], material, n, 3, sky, block, false, true);
        let v1 = T2Vertex::new(p1, [1.0, 1.0], material, n, 3, sky, block, false, true);
        let v2 = T2Vertex::new(p2, [1.0, 0.0], material, n, 3, sky, block, false, true);
        let v3 = T2Vertex::new(p3, [0.0, 0.0], material, n, 3, sky, block, false, true);
        out.push(T2Quad::new(v0, v1, v2, v3));
    }
}

/// Determines whether a fluid side face should be emitted towards `neighbor_shape`.
#[must_use]
pub fn should_emit_fluid_side(
    neighbor_shape: &BlockShape,
    is_fluid_above: bool,
    edge_height: f32,
) -> bool {
    // If neighbor is solid full cube, cull completely
    if *neighbor_shape == BlockShape::Cube {
        return false;
    }

    // If neighbor is fluid, check if neighbor is taller or has fluid above
    if let Some(n_height) = fluid_surface_height(neighbor_shape) {
        if is_fluid_above {
            // Under a falling waterfall, don't cull side if neighbor is lower
            return n_height < 1.0;
        }
        // Emit only if this edge is strictly higher than neighbor's surface
        edge_height > n_height + 0.05
    } else {
        // Neighbor is air or non-full block: emit side face
        true
    }
}

/// Meshes fluid volume inside a waterlogged sub-cube block (e.g. bottom slab with water).
pub fn mesh_waterlogged_volume(
    bx: u32,
    by: u32,
    bz: u32,
    shape: &BlockShape,
    material: u16,
    sky: u8,
    block: u8,
    out: &mut Vec<T2Quad>,
) {
    let fx = bx as f32;
    let fy = by as f32;
    let fz = bz as f32;

    // For a bottom slab, the upper half [8/16, 16/16] is filled with fluid
    let fluid_min_y = match shape {
        BlockShape::Boxes(boxes) if !boxes.is_empty() => {
            // Find max height of solid sub-boxes
            let max_box_y = boxes.iter().map(|b| b.max[1]).max().unwrap_or(0);
            fy + (f32::from(max_box_y) / 16.0)
        }
        _ => fy,
    };

    let fluid_max_y = fy + 1.0;
    if fluid_min_y >= fluid_max_y {
        return;
    }

    // Top face of waterlogged fluid volume
    let p0 = Vec3::new(fx, fluid_max_y, fz);
    let p1 = Vec3::new(fx + 1.0, fluid_max_y, fz);
    let p2 = Vec3::new(fx + 1.0, fluid_max_y, fz + 1.0);
    let p3 = Vec3::new(fx, fluid_max_y, fz + 1.0);

    let n = Vec3::Y;
    let v0 = T2Vertex::new(p0, [0.0, 0.0], material, n, 3, sky, block, false, true);
    let v1 = T2Vertex::new(p1, [1.0, 0.0], material, n, 3, sky, block, false, true);
    let v2 = T2Vertex::new(p2, [1.0, 1.0], material, n, 3, sky, block, false, true);
    let v3 = T2Vertex::new(p3, [0.0, 1.0], material, n, 3, sky, block, false, true);
    out.push(T2Quad::new(v0, v1, v2, v3));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t2_vertex_packing_roundtrip() {
        let pos = Vec3::new(12.375, 4.5, 27.875);
        let uv = [0.25, 0.75];
        let material = 42;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let ao = 2;
        let sky = 14;
        let block = 7;
        let emissive = true;
        let shade = false;

        let v = T2Vertex::new(pos, uv, material, normal, ao, sky, block, emissive, shade);

        assert!((v.pos().x - pos.x).abs() < 0.002);
        assert!((v.pos().y - pos.y).abs() < 0.002);
        assert!((v.pos().z - pos.z).abs() < 0.002);

        assert!((v.uv()[0] - uv[0]).abs() < 0.001);
        assert!((v.uv()[1] - uv[1]).abs() < 0.001);

        assert_eq!(v.material(), material);
        assert_eq!(v.ao(), ao);
        assert_eq!(v.sky(), sky);
        assert_eq!(v.block(), block);
        assert_eq!(v.is_emissive(), emissive);
        assert_eq!(v.has_shading(), shade);

        let decoded_normal = v.normal();
        assert!((decoded_normal.y - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_cross_model_quads() {
        let mut quads = Vec::new();
        mesh_cross_model(5, 10, 15, 7, 15, 0, &mut quads);

        // Cross emits 4 quads (2 diagonals, double-sided)
        assert_eq!(quads.len(), 4);

        // Verify bounding coordinates of quad vertices
        for q in &quads {
            for v in [q.v0, q.v1, q.v2, q.v3] {
                let p = v.pos();
                assert!((5.0..=6.001).contains(&p.x));
                assert!((10.0..=11.001).contains(&p.y));
                assert!((15.0..=16.001).contains(&p.z));
            }
        }
    }

    #[test]
    fn test_torch_model_quads() {
        let mut floor_quads = Vec::new();
        mesh_torch_model(0, 0, 0, None, 13, &mut floor_quads);
        assert_eq!(floor_quads.len(), 5); // 1 top + 4 sides

        for q in &floor_quads {
            assert!(q.v0.is_emissive());
        }

        let mut wall_quads = Vec::new();
        mesh_torch_model(0, 0, 0, Some(Face::North), 13, &mut wall_quads);
        assert_eq!(wall_quads.len(), 5);
    }

    #[test]
    fn test_fluid_corner_height_interpolation() {
        // 3x3 fluid grid with source at center and flowing neighbors
        let mock_world = |x: i32, y: i32, z: i32| -> BlockShape {
            if y == 0 && x == 0 && z == 0 {
                BlockShape::Fluid {
                    level: 0,
                    falling: false,
                } // Source: 14/16 = 0.875
            } else if y == 0 && (x.abs() <= 1 && z.abs() <= 1) {
                BlockShape::Fluid {
                    level: 2,
                    falling: false,
                } // Flow level 2: 6/9 = 0.6667
            } else {
                BlockShape::Empty
            }
        };

        let h00 = interpolate_fluid_corner_height(0, 0, 0, 0, 0, &mock_world);
        // Average of source (0.875) + 3 flow level 2 blocks (0.6667 each)
        let expected = (0.875 + 3.0 * (6.0 / 9.0)) / 4.0;
        assert!((h00 - expected).abs() < 0.01);
    }

    #[test]
    fn test_waterlogging_volume() {
        let mut quads = Vec::new();
        let bottom_slab = BlockShape::bottom_slab();
        mesh_waterlogged_volume(0, 0, 0, &bottom_slab, 6, 15, 0, &mut quads);

        assert_eq!(quads.len(), 1);
        let top_quad = quads[0];
        assert!((top_quad.v0.pos().y - 1.0).abs() < 0.001);
    }
}
