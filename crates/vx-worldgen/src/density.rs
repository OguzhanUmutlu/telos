//! 3D coarse density evaluation and trilinear interpolation.

use crate::biome::{BiomeId, lookup_biome};
use crate::climate::ClimatePoint;
use crate::math::lerp;
use crate::noise::{fbm3d, ridged2d};
use vx_core::coords::ChunkPos;

/// Number of cells per chunk along X and Z (each cell is 4 blocks wide).
pub const CELLS_XZ: usize = 8;
/// Number of cells per chunk along Y (each cell is 8 blocks high).
pub const CELLS_Y: usize = 4;

/// Number of corners per chunk along X and Z.
pub const CORNERS_XZ: usize = CELLS_XZ + 1; // 9
/// Number of corners per chunk along Y.
pub const CORNERS_Y: usize = CELLS_Y + 1; // 5

/// Total number of corners evaluated per 32³ chunk.
pub const TOTAL_CORNERS: usize = CORNERS_XZ * CORNERS_Y * CORNERS_XZ; // 405

/// Evaluates base height $h(x, z)$ from climate and ridged noise.
#[must_use]
pub fn base_terrain_height(seed: u64, x: f32, z: f32, climate: &ClimatePoint) -> f32 {
    let c = climate.continentalness;
    let e = climate.erosion;
    let pv = climate.peaks_and_valleys;

    // Continental base height (sea level = 0.0)
    let continental_offset = if c < -0.19 {
        // Deep ocean to shallow shelf
        -48.0 + (c + 1.0) * 20.0
    } else if c < 0.05 {
        // Coastline / beaches
        -4.0 + (c + 0.19) * 40.0
    } else {
        // Inland plains & hills
        6.0 + c * 35.0
    };

    // Mountain elevation: low erosion & high peaks & valleys
    let mountain_factor = (1.0 - e).clamp(0.0, 2.0) * pv.max(0.0);
    let mountain_lift = mountain_factor * 80.0;

    // High jagged ridges
    let ridges = if mountain_factor > 0.3 {
        let r_seed = seed.wrapping_add(999);
        ridged2d(r_seed, x, z, 3, 1.0 / 256.0, 2.0, 0.5) * 35.0 * mountain_factor
    } else {
        0.0
    };

    continental_offset + mountain_lift + ridges
}

/// Evaluates 3D density at a world coordinate $(x, y, z)$.
///
/// Positive values indicate solid terrain; non-positive values indicate air or fluid.
#[must_use]
pub fn sample_density(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let climate = ClimatePoint::sample(seed, x, z);
    let h = base_terrain_height(seed, x, z, &climate);

    // Softness gradient
    let softness = if climate.erosion < -0.3 && climate.peaks_and_valleys > 0.2 {
        24.0 // Steeper mountains
    } else {
        12.0 // Plains & hills
    };

    // Macro vertical gradient: (h - y) / S
    let mut density = (h - y) / softness;

    // 3D detail fBm (2 octaves for coarse scale)
    let detail_seed = seed.wrapping_add(777);
    let detail = fbm3d(detail_seed, x, y, z, 2, 1.0 / 64.0, 2.0, 0.5) * 0.45;
    density += detail;

    // 3D cheese caves
    // Avoid carving within 6 blocks of surface to preserve natural ground
    if y < h - 6.0 {
        let cave_seed = seed.wrapping_add(888);
        let cave_noise = fbm3d(cave_seed, x, y, z, 2, 1.0 / 48.0, 2.0, 0.5);
        if cave_noise > 0.38 {
            // Carve out cave cavity
            density = density.min(-1.0);
        }
    }

    density
}

/// A 3D grid of corner density samples and column biomes for one chunk.
pub struct CoarseGrid {
    densities: [f32; TOTAL_CORNERS],
    biomes: [BiomeId; CELLS_XZ * CELLS_XZ],
}

#[derive(Clone, Copy)]
struct ColumnInfo {
    h: f32,
    softness: f32,
}

impl CoarseGrid {
    /// Evaluates the 405 corner samples and 64 cell biomes for the given chunk position.
    #[must_use]
    pub fn evaluate(seed: u64, chunk_pos: ChunkPos) -> Self {
        let origin_x = (chunk_pos.x() * 32) as f32;
        let origin_y = (chunk_pos.y() * 32) as f32;
        let origin_z = (chunk_pos.z() * 32) as f32;

        // Precompute 2D column climate, height, and softness once per (x, z)
        let mut columns = [ColumnInfo {
            h: 0.0,
            softness: 12.0,
        }; CORNERS_XZ * CORNERS_XZ];
        let mut biomes = [BiomeId::Plains; CELLS_XZ * CELLS_XZ];

        for cz in 0..CORNERS_XZ {
            let z = origin_z + (cz * 4) as f32;
            for cx in 0..CORNERS_XZ {
                let x = origin_x + (cx * 4) as f32;
                let climate = ClimatePoint::sample(seed, x, z);
                let h = base_terrain_height(seed, x, z, &climate);
                let softness = if climate.erosion < -0.3 && climate.peaks_and_valleys > 0.2 {
                    24.0
                } else {
                    12.0
                };
                columns[cz * CORNERS_XZ + cx] = ColumnInfo { h, softness };
                if cz < CELLS_XZ && cx < CELLS_XZ {
                    biomes[cz * CELLS_XZ + cx] = lookup_biome(&climate);
                }
            }
        }

        let detail_seed = seed.wrapping_add(777);
        let cave_seed = seed.wrapping_add(888);
        let mut densities = [0.0f32; TOTAL_CORNERS];

        for cz in 0..CORNERS_XZ {
            let z = origin_z + (cz * 4) as f32;
            for cy in 0..CORNERS_Y {
                let y = origin_y + (cy * 8) as f32;
                for cx in 0..CORNERS_XZ {
                    let x = origin_x + (cx * 4) as f32;
                    let col = columns[cz * CORNERS_XZ + cx];
                    let idx = cz * (CORNERS_Y * CORNERS_XZ) + cy * CORNERS_XZ + cx;

                    let mut density = (col.h - y) / col.softness;

                    let detail = fbm3d(detail_seed, x, y, z, 2, 1.0 / 64.0, 2.0, 0.5) * 0.45;
                    density += detail;

                    if y < col.h - 6.0 {
                        let cave_noise = fbm3d(cave_seed, x, y, z, 2, 1.0 / 48.0, 2.0, 0.5);
                        if cave_noise > 0.38 {
                            density = density.min(-1.0);
                        }
                    }

                    densities[idx] = density;
                }
            }
        }

        Self { densities, biomes }
    }

    /// Returns the 8x8 cell biomes resolved during coarse evaluation.
    #[must_use]
    pub const fn biomes(&self) -> &[BiomeId; CELLS_XZ * CELLS_XZ] {
        &self.biomes
    }

    /// Gets the corner density at corner indices `(cx, cy, cz)`.
    #[inline]
    #[must_use]
    pub fn get(&self, cx: usize, cy: usize, cz: usize) -> f32 {
        self.densities[cz * (CORNERS_Y * CORNERS_XZ) + cy * CORNERS_XZ + cx]
    }

    /// Fills a 32³ boolean occupancy array (`true` = solid, `false` = air)
    /// using exact sign skipping and trilinear interpolation for mixed cells.
    #[allow(clippy::too_many_lines)]
    pub fn fill_occupancy(&self, out: &mut [bool; 32 * 32 * 32]) {
        for cz in 0..CELLS_XZ {
            for cy in 0..CELLS_Y {
                for cx in 0..CELLS_XZ {
                    // 8 corners of cell (cx, cy, cz)
                    let d000 = self.get(cx, cy, cz);
                    let d100 = self.get(cx + 1, cy, cz);
                    let d010 = self.get(cx, cy + 1, cz);
                    let d110 = self.get(cx + 1, cy + 1, cz);
                    let d001 = self.get(cx, cy, cz + 1);
                    let d101 = self.get(cx + 1, cy, cz + 1);
                    let d011 = self.get(cx, cy + 1, cz + 1);
                    let d111 = self.get(cx + 1, cy + 1, cz + 1);

                    let min_d = d000
                        .min(d100)
                        .min(d010)
                        .min(d110)
                        .min(d001)
                        .min(d101)
                        .min(d011)
                        .min(d111);
                    let max_d = d000
                        .max(d100)
                        .max(d010)
                        .max(d110)
                        .max(d001)
                        .max(d101)
                        .max(d011)
                        .max(d111);

                    let base_x = cx * 4;
                    let base_y = cy * 8;
                    let base_z = cz * 4;

                    // Exact sign skip: all positive = 100% solid
                    if min_d > 0.0 {
                        for dz in 0..4 {
                            let z = base_z + dz;
                            for dy in 0..8 {
                                let y = base_y + dy;
                                let y_offset = (y << 10) | (z << 5);
                                for dx in 0..4 {
                                    let x = base_x + dx;
                                    out[y_offset | x] = true;
                                }
                            }
                        }
                        continue;
                    }

                    // Exact sign skip: all non-positive = 100% air
                    if max_d <= 0.0 {
                        // Voxels are already default false/air
                        continue;
                    }

                    // Mixed cell: trilinear interpolation
                    for dz in 0..4 {
                        let tz = dz as f32 * 0.25;
                        let z = base_z + dz;

                        let c00 = lerp(d000, d001, tz);
                        let c10 = lerp(d100, d101, tz);
                        let c01 = lerp(d010, d011, tz);
                        let c11 = lerp(d110, d111, tz);

                        for dy in 0..8 {
                            let ty = dy as f32 * 0.125;
                            let y = base_y + dy;
                            let y_offset = (y << 10) | (z << 5);

                            let c0 = lerp(c00, c01, ty);
                            let c1 = lerp(c10, c11, ty);

                            for dx in 0..4 {
                                let tx = dx as f32 * 0.25;
                                let x = base_x + dx;

                                let d = lerp(c0, c1, tx);
                                if d > 0.0 {
                                    out[y_offset | x] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
