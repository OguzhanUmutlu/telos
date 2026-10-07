//! 3D noise-modulated aquifer level sampler and localized fluid barriers.
//!
//! Replaces uniform global sea level in subterranean caverns with 3D Voronoi
//! cells, local water tables, natural stone barriers, and deep magma reservoirs.

use crate::climate::ClimatePoint;
use crate::density::base_terrain_height;
use crate::math::hash3;
use crate::noise::{noise2, noise3};
use telos_core::coords::ChunkPos;

/// Global sea level in blocks (elevation 0).
pub const SEA_LEVEL: i32 = 0;

/// Subterranean elevation threshold below which aquifers generate magma/lava.
pub const LAVA_DEPTH_THRESHOLD: i32 = -64;

/// Aquifer cell dimension in blocks along X.
pub const CELL_SIZE_X: i32 = 16;
/// Aquifer cell dimension in blocks along Y.
pub const CELL_SIZE_Y: i32 = 12;
/// Aquifer cell dimension in blocks along Z.
pub const CELL_SIZE_Z: i32 = 16;

/// Maximum number of aquifer cells cached for a single 32³ chunk (including margins).
///
/// A 32³ chunk spans at most 4 cells along X, 6 cells along Y, and 4 cells along Z (4 * 6 * 4 = 96).
pub const MAX_CACHE_CELLS: usize = 128;

/// Fluid kind produced by an aquifer cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidKind {
    /// No fluid (dry cavity / air).
    None,
    /// Water fluid source.
    Water,
    /// Molten magma / lava fluid source.
    Lava,
}

/// The result of sampling the aquifer at a 3D block position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AquiferSample {
    /// Solid barrier stone placed to dam differing fluid levels.
    Barrier,
    /// Fluid body (Water or Lava).
    Fluid(FluidKind),
    /// Dry air.
    Air,
}

/// Single Voronoi aquifer cell properties.
#[derive(Debug, Clone, Copy)]
pub struct AquiferCell {
    /// Jittered X center position in world block coordinates.
    pub center_x: i32,
    /// Jittered Y center position in world block coordinates.
    pub center_y: i32,
    /// Jittered Z center position in world block coordinates.
    pub center_z: i32,
    /// Fluid surface level in world Y coordinates.
    pub level: i32,
    /// Fluid kind present in this cell.
    pub kind: FluidKind,
}

impl Default for AquiferCell {
    fn default() -> Self {
        Self {
            center_x: 0,
            center_y: 0,
            center_z: 0,
            level: i32::MIN,
            kind: FluidKind::None,
        }
    }
}

/// Precomputed aquifer cells covering a single 32³ chunk with zero heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct ChunkAquiferCache {
    min_cx: i32,
    min_cy: i32,
    min_cz: i32,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    cells: [AquiferCell; MAX_CACHE_CELLS],
}

impl ChunkAquiferCache {
    /// Looks up the precomputed cell at grid index `(cx, cy, cz)`.
    #[inline]
    #[must_use]
    pub fn get_cell(&self, cx: i32, cy: i32, cz: i32) -> Option<&AquiferCell> {
        let ox = cx.checked_sub(self.min_cx)?;
        let oy = cy.checked_sub(self.min_cy)?;
        let oz = cz.checked_sub(self.min_cz)?;

        if ox < 0 || oy < 0 || oz < 0 {
            return None;
        }

        let ux = ox as usize;
        let uy = oy as usize;
        let uz = oz as usize;

        if ux >= self.dim_x || uy >= self.dim_y || uz >= self.dim_z {
            return None;
        }

        let idx = uz * (self.dim_y * self.dim_x) + uy * self.dim_x + ux;
        self.cells.get(idx)
    }
}

/// Deterministic 3D aquifer sampler.
#[derive(Debug, Clone, Copy)]
pub struct AquiferSampler {
    seed: u64,
}

impl AquiferSampler {
    /// Creates a new `AquiferSampler` from the world seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Evaluates the jittered center and fluid level for a single aquifer cell at `(cx, cy, cz)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn evaluate_cell(&self, cx: i32, cy: i32, cz: i32) -> AquiferCell {
        let cell_hash = hash3(self.seed ^ 0xA901_FE45_C123_4567, cx, cy, cz);

        // Jitter within cell boundaries
        let jx = 3 + (cell_hash & 0x0F) as i32 % 11;
        let jy = 2 + ((cell_hash >> 8) & 0x0F) as i32 % 9;
        let jz = 3 + ((cell_hash >> 16) & 0x0F) as i32 % 11;

        let center_x = cx * CELL_SIZE_X + jx;
        let center_y = cy * CELL_SIZE_Y + jy;
        let center_z = cz * CELL_SIZE_Z + jz;

        // Sample preliminary surface height at cell center
        let climate = ClimatePoint::sample(self.seed, center_x as f32, center_z as f32);
        let h_prelim = base_terrain_height(self.seed, center_x as f32, center_z as f32, &climate);

        // Rule 1: Surface or near-surface level (within 12 blocks of preliminary surface)
        if center_y >= (h_prelim - 12.0) as i32 {
            if h_prelim < SEA_LEVEL as f32 {
                // Ocean / depression below sea level
                return AquiferCell {
                    center_x,
                    center_y,
                    center_z,
                    level: SEA_LEVEL,
                    kind: FluidKind::Water,
                };
            }
            // Dry surface / hills above sea level
            return AquiferCell {
                center_x,
                center_y,
                center_z,
                level: i32::MIN,
                kind: FluidKind::None,
            };
        }

        // Rule 2: Subterranean cavern (y < h_prelim - 12)
        // Floodedness noise (λ = 64)
        let n_f = noise3(
            self.seed ^ 0xF100_DCA7_E001_2345,
            center_x as f32 * (1.0 / 64.0),
            center_y as f32 * (1.0 / 64.0),
            center_z as f32 * (1.0 / 64.0),
        );

        if n_f < -0.3 {
            // Dry cavern room
            AquiferCell {
                center_x,
                center_y,
                center_z,
                level: i32::MIN,
                kind: FluidKind::None,
            }
        } else if n_f > 0.6 && center_y < SEA_LEVEL {
            // Sea-connected aquifer
            AquiferCell {
                center_x,
                center_y,
                center_z,
                level: SEA_LEVEL,
                kind: FluidKind::Water,
            }
        } else {
            // Localized water or magma table
            let n_s = noise2(
                self.seed ^ 0x598E_AD01_9988_7766,
                center_x as f32 * (1.0 / 40.0),
                center_z as f32 * (1.0 / 40.0),
            );
            let raw_level = (center_y as f32 + 16.0 * n_s).round() as i32;

            // Quantize to steps of 3 blocks for terraced water tables
            let quantized = raw_level.div_euclid(3) * 3;

            // Clamp below surface rock ceiling
            let max_allowed = (h_prelim - 8.0).floor() as i32;
            let level = quantized.min(max_allowed);

            // Magma table threshold in deep strata
            let kind = if level < LAVA_DEPTH_THRESHOLD {
                FluidKind::Lava
            } else {
                FluidKind::Water
            };

            AquiferCell {
                center_x,
                center_y,
                center_z,
                level,
                kind,
            }
        }
    }

    /// Precomputes all aquifer cells that can influence the given 32³ chunk.
    #[must_use]
    #[allow(
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn prepare_chunk(&self, chunk_pos: ChunkPos) -> ChunkAquiferCache {
        let min_x = chunk_pos.x() * 32;
        let max_x = min_x + 31;
        let min_y = chunk_pos.y() * 32;
        let max_y = min_y + 31;
        let min_z = chunk_pos.z() * 32;
        let max_z = min_z + 31;

        let min_cx = min_x.div_euclid(CELL_SIZE_X) - 1;
        let max_cx = max_x.div_euclid(CELL_SIZE_X) + 1;
        let min_cy = min_y.div_euclid(CELL_SIZE_Y) - 1;
        let max_cy = max_y.div_euclid(CELL_SIZE_Y) + 1;
        let min_cz = min_z.div_euclid(CELL_SIZE_Z) - 1;
        let max_cz = max_z.div_euclid(CELL_SIZE_Z) + 1;

        let dim_x = (max_cx - min_cx + 1) as usize;
        let dim_y = (max_cy - min_cy + 1) as usize;
        let dim_z = (max_cz - min_cz + 1) as usize;

        let mut cells = [AquiferCell::default(); MAX_CACHE_CELLS];

        for cz in min_cz..=max_cz {
            let uz = (cz - min_cz) as usize;
            for cy in min_cy..=max_cy {
                let uy = (cy - min_cy) as usize;
                for cx in min_cx..=max_cx {
                    let ux = (cx - min_cx) as usize;
                    let idx = uz * (dim_y * dim_x) + uy * dim_x + ux;
                    if idx < MAX_CACHE_CELLS {
                        cells[idx] = self.evaluate_cell(cx, cy, cz);
                    }
                }
            }
        }

        ChunkAquiferCache {
            min_cx,
            min_cy,
            min_cz,
            dim_x,
            dim_y,
            dim_z,
            cells,
        }
    }

    /// Samples the fluid or barrier status at world block coordinate `(wx, wy, wz)`.
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::suboptimal_flops,
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn sample(&self, cache: &ChunkAquiferCache, wx: i32, wy: i32, wz: i32) -> AquiferSample {
        let center_cx = wx.div_euclid(CELL_SIZE_X);
        let center_cy = wy.div_euclid(CELL_SIZE_Y);
        let center_cz = wz.div_euclid(CELL_SIZE_Z);

        // Find nearest 2 cell centers among the 27 neighborhood cells
        let mut d1 = i64::MAX;
        let mut d2 = i64::MAX;
        let mut cell1 = AquiferCell::default();
        let mut cell2 = AquiferCell::default();

        for dz in -1..=1 {
            let cz = center_cz + dz;
            for dy in -1..=1 {
                let cy = center_cy + dy;
                for dx in -1..=1 {
                    let cx = center_cx + dx;
                    let c = if let Some(cached) = cache.get_cell(cx, cy, cz) {
                        *cached
                    } else {
                        self.evaluate_cell(cx, cy, cz)
                    };

                    let diff_x = i64::from(wx - c.center_x);
                    let diff_y = i64::from(wy - c.center_y);
                    let diff_z = i64::from(wz - c.center_z);
                    let dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                    if dist_sq < d1 {
                        d2 = d1;
                        cell2 = cell1;
                        d1 = dist_sq;
                        cell1 = c;
                    } else if dist_sq < d2 {
                        d2 = dist_sq;
                        cell2 = c;
                    }
                }
            }
        }

        let c1 = cell1;
        let c2 = cell2;

        // Check barrier condition:
        // Only evaluate barrier if at least one cell has fluid at height wy,
        // the centers are within distance-squared similarity margin 25, and levels differ.
        if (wy < c1.level || wy < c2.level) && c1.level != c2.level {
            let delta_d = d2.saturating_sub(d1);
            if delta_d < 25 {
                let sim = 1.0 - (delta_d as f32) * (1.0 / 25.0);
                let level_diff = (c1.level.abs_diff(c2.level)) as f32;
                let barrier_strength = sim * level_diff * 0.5;

                let n_barrier = noise3(
                    self.seed ^ 0xBA55_1E80_1122_3344,
                    wx as f32 * 0.1,
                    wy as f32 * 0.1,
                    wz as f32 * 0.1,
                );
                let threshold = 0.35 + 0.35 * n_barrier;

                if barrier_strength > threshold {
                    return AquiferSample::Barrier;
                }
            }
        }

        // Fluid check in primary Voronoi cell
        if wy < c1.level && c1.kind != FluidKind::None {
            AquiferSample::Fluid(c1.kind)
        } else {
            AquiferSample::Air
        }
    }

    /// Samples the aquifer status at world block coordinate `(wx, wy, wz)` directly without precomputed cache.
    #[must_use]
    pub fn sample_world(&self, wx: i32, wy: i32, wz: i32) -> AquiferSample {
        let chunk_pos = ChunkPos::new(wx.div_euclid(32), wy.div_euclid(32), wz.div_euclid(32));
        let cache = self.prepare_chunk(chunk_pos);
        self.sample(&cache, wx, wy, wz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aquifer_cell_determinism() {
        let sampler = AquiferSampler::new(0xCAFE_BABE_1337);
        let cell_a = sampler.evaluate_cell(10, -5, 20);
        let cell_b = sampler.evaluate_cell(10, -5, 20);

        assert_eq!(cell_a.center_x, cell_b.center_x);
        assert_eq!(cell_a.center_y, cell_b.center_y);
        assert_eq!(cell_a.center_z, cell_b.center_z);
        assert_eq!(cell_a.level, cell_b.level);
        assert_eq!(cell_a.kind, cell_b.kind);
    }

    #[test]
    fn test_aquifer_cache_consistency() {
        let sampler = AquiferSampler::new(0x1234_5678);
        let pos = ChunkPos::new(0, -2, 0);
        let cache = sampler.prepare_chunk(pos);

        let cell = cache.get_cell(0, -2, 0).expect("Cell must exist in cache");
        let direct = sampler.evaluate_cell(0, -2, 0);

        assert_eq!(cell.center_x, direct.center_x);
        assert_eq!(cell.center_y, direct.center_y);
        assert_eq!(cell.center_z, direct.center_z);
        assert_eq!(cell.level, direct.level);
        assert_eq!(cell.kind, direct.kind);
    }

    #[test]
    fn test_magma_pools_at_depth() {
        let sampler = AquiferSampler::new(0x9876_5432);
        // Deep subterranean cell around y = -100 (cy = -9)
        let cell = sampler.evaluate_cell(0, -9, 0);
        if cell.kind != FluidKind::None {
            assert!(
                cell.level < LAVA_DEPTH_THRESHOLD,
                "Subterranean cell below -64 with fluid must be below threshold: {}",
                cell.level
            );
            assert_eq!(
                cell.kind,
                FluidKind::Lava,
                "Fluid below -64 must be magma/lava"
            );
        }
    }
}
