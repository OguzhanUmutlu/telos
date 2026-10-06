//! 4-corner ambient occlusion and smooth lighting calculation and deduplication.
//!
//! Implements ADR-08, ADR-11, and `vulkan-rendering/SKILL.md` §6.1:
//! - `LightPattern`: 64-bit packed struct storing 4-corner sky light, block light, and AO.
//! - `LightPatternTable`: Deduplicator assigning compact `u16` indices (< 16384).
//! - `VoxelLightSampler`: Trait for sampling opacity, sky light, and block light in chunk neighborhoods.
//! - `compute_face_pattern`: Computes 4-corner AO and smooth lighting for any cardinal face.

use glam::IVec3;
use hashbrown::HashMap;
use vx_voxel::{chunk::ChunkSnapshot, occupancy::Occupancy};

use crate::{bitwise::NeighborSlices, quad::FaceDir};

/// Packed 64-bit light pattern containing 4-corner sky light, block light, and ambient occlusion.
///
/// Bit layout matches `vulkan-rendering/SKILL.md` §6.1:
/// - Word 0 (`u32`):
///   - bits 0..15: sky light c0..c3 (4 bits each, values 0..15)
///   - bits 16..31: block light c0..c3 (4 bits each, values 0..15)
/// - Word 1 (`u32`):
///   - bits 0..7: AO c0..c3 (2 bits each, values 0..3: 0 = darkest, 3 = unoccluded)
///   - bits 8..31: reserved (0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(4))]
pub struct LightPattern {
    /// Raw 64-bit representation (two 32-bit words).
    pub raw: [u32; 2],
}

impl LightPattern {
    /// Creates a new `LightPattern` from per-corner sky light, block light, and AO arrays.
    #[inline]
    #[must_use]
    pub const fn new(sky: [u8; 4], block: [u8; 4], ao: [u8; 4]) -> Self {
        let w0 = ((sky[0] as u32) & 0x0F)
            | (((sky[1] as u32) & 0x0F) << 4)
            | (((sky[2] as u32) & 0x0F) << 8)
            | (((sky[3] as u32) & 0x0F) << 12)
            | (((block[0] as u32) & 0x0F) << 16)
            | (((block[1] as u32) & 0x0F) << 20)
            | (((block[2] as u32) & 0x0F) << 24)
            | (((block[3] as u32) & 0x0F) << 28);

        let w1 = ((ao[0] as u32) & 0x03)
            | (((ao[1] as u32) & 0x03) << 2)
            | (((ao[2] as u32) & 0x03) << 4)
            | (((ao[3] as u32) & 0x03) << 6);

        Self { raw: [w0, w1] }
    }

    /// Default pattern representing completely unoccluded open sunlight (sky 15, block 0, AO 3).
    #[inline]
    #[must_use]
    pub const fn full_sky() -> Self {
        Self {
            raw: [0x0000_FFFF, 0x0000_00FF],
        }
    }

    /// Completely dark pattern (sky 0, block 0, unoccluded AO 3).
    #[inline]
    #[must_use]
    pub const fn dark() -> Self {
        Self {
            raw: [0x0000_0000, 0x0000_00FF],
        }
    }

    /// Retrieves sky light level for corner `c` (0..=3).
    #[inline]
    #[must_use]
    pub const fn sky_corner(&self, c: usize) -> u8 {
        ((self.raw[0] >> (c * 4)) & 0x0F) as u8
    }

    /// Retrieves block light level for corner `c` (0..=3).
    #[inline]
    #[must_use]
    pub const fn block_corner(&self, c: usize) -> u8 {
        ((self.raw[0] >> (16 + c * 4)) & 0x0F) as u8
    }

    /// Retrieves ambient occlusion level for corner `c` (0..=3: 0 = darkest, 3 = unoccluded).
    #[inline]
    #[must_use]
    pub const fn ao_corner(&self, c: usize) -> u8 {
        ((self.raw[1] >> (c * 2)) & 0x03) as u8
    }

    /// Checks whether this pattern permits greedy merging along tangent axis $u$.
    ///
    /// Per `vulkan-rendering/SKILL.md` §6.1: requires `c0 == c1 && c3 == c2`.
    #[inline]
    #[must_use]
    pub const fn can_merge_u(&self) -> bool {
        self.sky_corner(0) == self.sky_corner(1)
            && self.sky_corner(3) == self.sky_corner(2)
            && self.block_corner(0) == self.block_corner(1)
            && self.block_corner(3) == self.block_corner(2)
            && self.ao_corner(0) == self.ao_corner(1)
            && self.ao_corner(3) == self.ao_corner(2)
    }

    /// Checks whether this pattern permits greedy merging along tangent axis $v$.
    ///
    /// Per `vulkan-rendering/SKILL.md` §6.1: requires `c0 == c3 && c1 == c2`.
    #[inline]
    #[must_use]
    pub const fn can_merge_v(&self) -> bool {
        self.sky_corner(0) == self.sky_corner(3)
            && self.sky_corner(1) == self.sky_corner(2)
            && self.block_corner(0) == self.block_corner(3)
            && self.block_corner(1) == self.block_corner(2)
            && self.ao_corner(0) == self.ao_corner(3)
            && self.ao_corner(1) == self.ao_corner(2)
    }

    /// Checks whether light and AO are completely homogeneous across all 4 corners.
    #[inline]
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        self.can_merge_u()
            && self.sky_corner(0) == self.sky_corner(3)
            && self.block_corner(0) == self.block_corner(3)
            && self.ao_corner(0) == self.ao_corner(3)
    }
}

/// Deduplicator table for `LightPattern` allocations within a mesh section.
#[derive(Debug, Clone)]
pub struct LightPatternTable {
    patterns: Vec<LightPattern>,
    index_map: HashMap<LightPattern, u16>,
}

impl Default for LightPatternTable {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl LightPatternTable {
    /// Creates a new table pre-populated with pattern 0 as `LightPattern::full_sky()`.
    #[must_use]
    pub fn new() -> Self {
        let mut this = Self {
            patterns: Vec::with_capacity(64),
            index_map: HashMap::with_capacity(64),
        };
        let default_pattern = LightPattern::full_sky();
        this.patterns.push(default_pattern);
        this.index_map.insert(default_pattern, 0);
        this
    }

    /// Inserts a pattern and returns its `u16` index (< 16384).
    pub fn insert(&mut self, pattern: LightPattern) -> u16 {
        if let Some(&idx) = self.index_map.get(&pattern) {
            return idx;
        }
        let idx = self.patterns.len() as u16;
        assert!(
            idx < 16384,
            "Exceeded maximum of 16384 light patterns per section"
        );
        self.patterns.push(pattern);
        self.index_map.insert(pattern, idx);
        idx
    }

    /// Returns the array of deduplicated patterns.
    #[must_use]
    pub fn patterns(&self) -> &[LightPattern] {
        &self.patterns
    }

    /// Converts into the inner patterns vector.
    #[must_use]
    pub fn into_patterns(self) -> Vec<LightPattern> {
        self.patterns
    }
}

/// Sampling abstraction for voxel occupancy and lighting in chunk neighborhoods.
pub trait VoxelLightSampler {
    /// Returns `true` if voxel at `(x, y, z)` is opaque (light blocking).
    fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool;

    /// Returns the 4-bit sky light level (0..=15) at `(x, y, z)`.
    fn get_sky_light(&self, x: i32, y: i32, z: i32) -> u8;

    /// Returns the 4-bit block light level (0..=15) at `(x, y, z)`.
    fn get_block_light(&self, x: i32, y: i32, z: i32) -> u8;
}

/// Computes 4-corner ambient occlusion and smooth lighting for a specific quad corner.
#[inline]
pub fn compute_corner_ao_and_light<S: VoxelLightSampler>(
    sampler: &S,
    pos_p: IVec3,
    s1_offset: IVec3,
    s2_offset: IVec3,
    corner_offset: IVec3,
) -> (u8, u8, u8) {
    let pos_s1 = pos_p + s1_offset;
    let pos_s2 = pos_p + s2_offset;
    let pos_corner = pos_p + corner_offset;

    let s1 = sampler.is_opaque(pos_s1.x, pos_s1.y, pos_s1.z);
    let s2 = sampler.is_opaque(pos_s2.x, pos_s2.y, pos_s2.z);
    let sc = sampler.is_opaque(pos_corner.x, pos_corner.y, pos_corner.z);

    let ao = if s1 && s2 {
        0
    } else {
        3 - (u8::from(s1) + u8::from(s2) + u8::from(sc))
    };

    // Smooth lighting calculation: average valid non-solid voxels
    let mut count = 1u32;
    let mut sky_sum = u32::from(sampler.get_sky_light(pos_p.x, pos_p.y, pos_p.z));
    let mut block_sum = u32::from(sampler.get_block_light(pos_p.x, pos_p.y, pos_p.z));

    if !s1 {
        sky_sum += u32::from(sampler.get_sky_light(pos_s1.x, pos_s1.y, pos_s1.z));
        block_sum += u32::from(sampler.get_block_light(pos_s1.x, pos_s1.y, pos_s1.z));
        count += 1;
    }
    if !s2 {
        sky_sum += u32::from(sampler.get_sky_light(pos_s2.x, pos_s2.y, pos_s2.z));
        block_sum += u32::from(sampler.get_block_light(pos_s2.x, pos_s2.y, pos_s2.z));
        count += 1;
    }
    if (!s1 || !s2) && !sc {
        sky_sum += u32::from(sampler.get_sky_light(pos_corner.x, pos_corner.y, pos_corner.z));
        block_sum += u32::from(sampler.get_block_light(pos_corner.x, pos_corner.y, pos_corner.z));
        count += 1;
    }

    let sky_avg = ((sky_sum + (count / 2)) / count) as u8;
    let block_avg = ((block_sum + (count / 2)) / count) as u8;

    (ao, sky_avg, block_avg)
}

/// Computes the packed `LightPattern` for a 1x1 face at local `(x, y, z)` facing `dir`.
#[must_use]
pub fn compute_face_pattern<S: VoxelLightSampler>(
    sampler: &S,
    x: u32,
    y: u32,
    z: u32,
    dir: FaceDir,
) -> LightPattern {
    #[allow(clippy::cast_possible_wrap)]
    let p = IVec3::new(x as i32, y as i32, z as i32) + dir.normal();
    let (u_dir, v_dir) = dir.tangent_frame();

    let mut sky = [0u8; 4];
    let mut block = [0u8; 4];
    let mut ao = [0u8; 4];

    for c in 0..4 {
        let sign_u = if c == 1 || c == 2 { 1 } else { -1 };
        let sign_v = if c == 2 || c == 3 { 1 } else { -1 };
        let s1_offset = u_dir * sign_u;
        let s2_offset = v_dir * sign_v;
        let corner_offset = s1_offset + s2_offset;

        let (corner_ao, corner_sky, corner_block) =
            compute_corner_ao_and_light(sampler, p, s1_offset, s2_offset, corner_offset);
        ao[c] = corner_ao;
        sky[c] = corner_sky;
        block[c] = corner_block;
    }

    LightPattern::new(sky, block, ao)
}

/// Neighborhood sampler wrapping a center snapshot and its 6 cardinal neighbors.
pub struct VoxelNeighborhood<'a> {
    center: &'a ChunkSnapshot,
    cardinals: &'a [Option<&'a ChunkSnapshot>; 6],
}

impl<'a> VoxelNeighborhood<'a> {
    /// Creates a new `VoxelNeighborhood`.
    #[must_use]
    pub const fn new(
        center: &'a ChunkSnapshot,
        cardinals: &'a [Option<&'a ChunkSnapshot>; 6],
    ) -> Self {
        Self { center, cardinals }
    }
}

impl VoxelLightSampler for VoxelNeighborhood<'_> {
    #[allow(clippy::manual_range_contains)]
    fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool {
        if (0..32).contains(&x) && (0..32).contains(&y) && (0..32).contains(&z) {
            return self
                .center
                .occupancy()
                .is_solid(x as u32, y as u32, z as u32);
        }

        // Cardinal neighbor lookups
        if x < 0 && (0..32).contains(&y) && (0..32).contains(&z) {
            if let Some(n) = self.cardinals[1] {
                return n.occupancy().is_solid(31, y as u32, z as u32);
            }
            return false;
        }
        if x >= 32 && (0..32).contains(&y) && (0..32).contains(&z) {
            if let Some(n) = self.cardinals[0] {
                return n.occupancy().is_solid(0, y as u32, z as u32);
            }
            return false;
        }
        if y < 0 && (0..32).contains(&x) && (0..32).contains(&z) {
            if let Some(n) = self.cardinals[3] {
                return n.occupancy().is_solid(x as u32, 31, z as u32);
            }
            return false;
        }
        if y >= 32 && (0..32).contains(&x) && (0..32).contains(&z) {
            if let Some(n) = self.cardinals[2] {
                return n.occupancy().is_solid(x as u32, 0, z as u32);
            }
            return false;
        }
        if z < 0 && (0..32).contains(&x) && (0..32).contains(&y) {
            if let Some(n) = self.cardinals[5] {
                return n.occupancy().is_solid(x as u32, y as u32, 31);
            }
            return false;
        }
        if z >= 32 && (0..32).contains(&x) && (0..32).contains(&y) {
            if let Some(n) = self.cardinals[4] {
                return n.occupancy().is_solid(x as u32, y as u32, 0);
            }
            return false;
        }

        // Diagonal edge clamp
        let cx = x.clamp(0, 31) as u32;
        let cy = y.clamp(0, 31) as u32;
        let cz = z.clamp(0, 31) as u32;

        if y >= 32 {
            if let Some(n) = self.cardinals[2] {
                return n.occupancy().is_solid(cx, 0, cz);
            }
        } else if y < 0 {
            if let Some(n) = self.cardinals[3] {
                return n.occupancy().is_solid(cx, 31, cz);
            }
        } else if x >= 32 {
            if let Some(n) = self.cardinals[0] {
                return n.occupancy().is_solid(0, cy, cz);
            }
        } else if x < 0 {
            if let Some(n) = self.cardinals[1] {
                return n.occupancy().is_solid(31, cy, cz);
            }
        } else if z >= 32 {
            if let Some(n) = self.cardinals[4] {
                return n.occupancy().is_solid(cx, cy, 0);
            }
        } else if z < 0
            && let Some(n) = self.cardinals[5]
        {
            return n.occupancy().is_solid(cx, cy, 31);
        }

        false
    }

    #[allow(clippy::manual_range_contains)]
    fn get_sky_light(&self, x: i32, y: i32, z: i32) -> u8 {
        if (0..32).contains(&x) && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5) | x;
            return self.center.light().map_or(15, |l| l.get_sky(idx as usize));
        }

        // Cardinal neighbors
        if x < 0 && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5) | 31;
            return self.cardinals[1]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }
        if x >= 32 && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5);
            return self.cardinals[0]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }
        if y < 0 && (0..32).contains(&x) && (0..32).contains(&z) {
            let idx = (31 << 10) | (z << 5) | x;
            return self.cardinals[3]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }
        if y >= 32 && (0..32).contains(&x) && (0..32).contains(&z) {
            let idx = (z << 5) | x;
            return self.cardinals[2]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }
        if z < 0 && (0..32).contains(&x) && (0..32).contains(&y) {
            let idx = (y << 10) | (31 << 5) | x;
            return self.cardinals[5]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }
        if z >= 32 && (0..32).contains(&x) && (0..32).contains(&y) {
            let idx = (y << 10) | x;
            return self.cardinals[4]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize));
        }

        // Clamp diagonal
        let cx = x.clamp(0, 31);
        let cy = y.clamp(0, 31);
        let cz = z.clamp(0, 31);

        if y >= 32 {
            let idx = (cz << 5) | cx;
            self.cardinals[2]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else if y < 0 {
            let idx = (31 << 10) | (cz << 5) | cx;
            self.cardinals[3]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else if x >= 32 {
            let idx = (cy << 10) | (cz << 5);
            self.cardinals[0]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else if x < 0 {
            let idx = (cy << 10) | (cz << 5) | 31;
            self.cardinals[1]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else if z >= 32 {
            let idx = (cy << 10) | cx;
            self.cardinals[4]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else if z < 0 {
            let idx = (cy << 10) | (31 << 5) | cx;
            self.cardinals[5]
                .and_then(|n| n.light())
                .map_or(15, |l| l.get_sky(idx as usize))
        } else {
            15
        }
    }

    #[allow(clippy::manual_range_contains)]
    fn get_block_light(&self, x: i32, y: i32, z: i32) -> u8 {
        if (0..32).contains(&x) && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5) | x;
            return self.center.light().map_or(0, |l| l.get_block(idx as usize));
        }

        // Cardinal neighbors
        if x < 0 && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5) | 31;
            return self.cardinals[1]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }
        if x >= 32 && (0..32).contains(&y) && (0..32).contains(&z) {
            let idx = (y << 10) | (z << 5);
            return self.cardinals[0]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }
        if y < 0 && (0..32).contains(&x) && (0..32).contains(&z) {
            let idx = (31 << 10) | (z << 5) | x;
            return self.cardinals[3]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }
        if y >= 32 && (0..32).contains(&x) && (0..32).contains(&z) {
            let idx = (z << 5) | x;
            return self.cardinals[2]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }
        if z < 0 && (0..32).contains(&x) && (0..32).contains(&y) {
            let idx = (y << 10) | (31 << 5) | x;
            return self.cardinals[5]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }
        if z >= 32 && (0..32).contains(&x) && (0..32).contains(&y) {
            let idx = (y << 10) | x;
            return self.cardinals[4]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize));
        }

        // Clamp diagonal
        let cx = x.clamp(0, 31);
        let cy = y.clamp(0, 31);
        let cz = z.clamp(0, 31);

        if y >= 32 {
            let idx = (cz << 5) | cx;
            self.cardinals[2]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else if y < 0 {
            let idx = (31 << 10) | (cz << 5) | cx;
            self.cardinals[3]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else if x >= 32 {
            let idx = (cy << 10) | (cz << 5);
            self.cardinals[0]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else if x < 0 {
            let idx = (cy << 10) | (cz << 5) | 31;
            self.cardinals[1]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else if z >= 32 {
            let idx = (cy << 10) | cx;
            self.cardinals[4]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else if z < 0 {
            let idx = (cy << 10) | (31 << 5) | cx;
            self.cardinals[5]
                .and_then(|n| n.light())
                .map_or(0, |l| l.get_block(idx as usize))
        } else {
            0
        }
    }
}

/// Standalone occupancy-based sampler when explicit lighting is omitted.
pub struct OccupancyLightSampler<'a> {
    occ: &'a Occupancy,
    neighbors: &'a NeighborSlices,
}

impl<'a> OccupancyLightSampler<'a> {
    /// Creates a new `OccupancyLightSampler`.
    #[must_use]
    pub const fn new(occ: &'a Occupancy, neighbors: &'a NeighborSlices) -> Self {
        Self { occ, neighbors }
    }
}

impl VoxelLightSampler for OccupancyLightSampler<'_> {
    #[allow(clippy::manual_range_contains)]
    fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool {
        if (0..32).contains(&x) && (0..32).contains(&y) && (0..32).contains(&z) {
            return self.occ.is_solid(x as u32, y as u32, z as u32);
        }

        if x < 0 && (0..32).contains(&y) && (0..32).contains(&z) {
            return ((self.neighbors.neg_x[y as usize] >> (z as u32)) & 1) != 0;
        }
        if x >= 32 && (0..32).contains(&y) && (0..32).contains(&z) {
            return ((self.neighbors.pos_x[z as usize] >> (y as u32)) & 1) != 0;
        }
        if y < 0 && (0..32).contains(&x) && (0..32).contains(&z) {
            return ((self.neighbors.neg_y[z as usize] >> (x as u32)) & 1) != 0;
        }
        if y >= 32 && (0..32).contains(&x) && (0..32).contains(&z) {
            return ((self.neighbors.pos_y[x as usize] >> (z as u32)) & 1) != 0;
        }
        if z < 0 && (0..32).contains(&x) && (0..32).contains(&y) {
            return ((self.neighbors.neg_z[x as usize] >> (y as u32)) & 1) != 0;
        }
        if z >= 32 && (0..32).contains(&x) && (0..32).contains(&y) {
            return ((self.neighbors.pos_z[y as usize] >> (x as u32)) & 1) != 0;
        }

        false
    }

    fn get_sky_light(&self, _x: i32, _y: i32, _z: i32) -> u8 {
        15
    }

    fn get_block_light(&self, _x: i32, _y: i32, _z: i32) -> u8 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSampler {
        solid_blocks: Vec<IVec3>,
        sky_light: u8,
        block_light: u8,
    }

    impl VoxelLightSampler for MockSampler {
        fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool {
            self.solid_blocks.contains(&IVec3::new(x, y, z))
        }

        fn get_sky_light(&self, _x: i32, _y: i32, _z: i32) -> u8 {
            self.sky_light
        }

        fn get_block_light(&self, _x: i32, _y: i32, _z: i32) -> u8 {
            self.block_light
        }
    }

    #[test]
    fn test_light_pattern_roundtrip() {
        let pattern = LightPattern::new([15, 12, 8, 4], [0, 5, 10, 14], [3, 2, 1, 0]);

        assert_eq!(pattern.sky_corner(0), 15);
        assert_eq!(pattern.sky_corner(1), 12);
        assert_eq!(pattern.sky_corner(2), 8);
        assert_eq!(pattern.sky_corner(3), 4);

        assert_eq!(pattern.block_corner(0), 0);
        assert_eq!(pattern.block_corner(1), 5);
        assert_eq!(pattern.block_corner(2), 10);
        assert_eq!(pattern.block_corner(3), 14);

        assert_eq!(pattern.ao_corner(0), 3);
        assert_eq!(pattern.ao_corner(1), 2);
        assert_eq!(pattern.ao_corner(2), 1);
        assert_eq!(pattern.ao_corner(3), 0);

        assert!(!pattern.can_merge_u());
        assert!(!pattern.can_merge_v());
        assert!(!pattern.is_uniform());
    }

    #[test]
    fn test_light_pattern_uniform_merge() {
        let pattern = LightPattern::full_sky();
        assert!(pattern.can_merge_u());
        assert!(pattern.can_merge_v());
        assert!(pattern.is_uniform());

        // Horizontal gradient: c0 == c1 and c3 == c2
        let h_pattern = LightPattern::new([15, 15, 10, 10], [0, 0, 0, 0], [3, 3, 3, 3]);
        assert!(h_pattern.can_merge_u());
        assert!(!h_pattern.can_merge_v());
        assert!(!h_pattern.is_uniform());
    }

    #[test]
    fn test_ao_crease_calculation() {
        // Test corner AO with 0, 1, 2, and 3 surrounding solid blocks
        let sampler = MockSampler {
            solid_blocks: vec![IVec3::new(1, 1, 0), IVec3::new(0, 1, 1)],
            sky_light: 15,
            block_light: 0,
        };

        // When side1 (1, 1, 0) and side2 (0, 1, 1) are both solid:
        // corner at (0, 1, 0) facing +Y has s1 and s2 both solid -> ao = 0
        let (ao, sky, block) = compute_corner_ao_and_light(
            &sampler,
            IVec3::new(0, 1, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(1, 0, 1),
        );

        assert_eq!(ao, 0, "Both side walls solid must yield darkest AO = 0");
        assert_eq!(sky, 15);
        assert_eq!(block, 0);
    }
}
