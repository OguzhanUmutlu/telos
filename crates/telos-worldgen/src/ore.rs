//! 3D procedural ore distribution and large sinuous ore veins.
//!
//! Evaluates vertical height-distribution probability density functions (Triangle, Uniform),
//! Poisson-distributed random walk sphere-chains for blob ores across cubic chunk boundaries
//! (via 3×3×3 neighborhood pull generation), and 3D ridge/toggle noise for snaking copper and iron
//! veins with filler rock (granite, tuff) and raw metal concentrates.

use crate::biome::{BiomeId, lookup_biome};
use crate::climate::ClimatePoint;
use crate::density::sample_density;
use crate::math::{hash3, lerp, mix64, unit_f32};
use crate::noise::noise3;
use crate::surface::ResolvedBlocks;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Salting constant for vein toggle noise (separating copper and iron bands).
const SALT_VEIN_TOGGLE: u64 = 0x7E10_5066_1E00_0001;
/// Salting constant for vein ridge noise (generating narrow snaking vein cores).
const SALT_VEIN_RIDGE: u64 = 0x7E10_5066_1E00_0002;
/// Salting constant for vein gap noise (modulating ore cluster frequency).
const SALT_VEIN_GAP: u64 = 0x7E10_5066_1E00_0003;
/// Salting constant for vein material rolls (raw metal vs ore vs filler rock).
const SALT_VEIN_BLOCK: u64 = 0x7E10_5066_1E00_0004;
/// Salting constant for per-voxel air exposure discard checks.
const SALT_AIR_DISCARD: u64 = 0xA18_D15C_A8D0_0001;

/// Frequency for vein toggle noise (lambda = 256).
const FREQ_TOGGLE: f32 = 1.0 / 256.0;
/// Frequency for vein ridge noise (lambda = 48).
const FREQ_RIDGE: f32 = 1.0 / 48.0;
/// Frequency for vein gap noise (lambda = 24).
const FREQ_GAP: f32 = 1.0 / 24.0;

/// Vertical height distribution probability model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeightDistribution {
    /// Triangular distribution with zero density at `min_y` and `max_y`, peaking at `(min_y + max_y) / 2`.
    Triangle {
        /// Lower elevation bound.
        min_y: i32,
        /// Upper elevation bound.
        max_y: i32,
    },
    /// Uniform distribution with constant probability density between `min_y` and `max_y`.
    Uniform {
        /// Lower elevation bound.
        min_y: i32,
        /// Upper elevation bound.
        max_y: i32,
    },
}

impl HeightDistribution {
    /// Evaluates the relative density in `[0.0, 1.0]` at world height `y`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sample_density(&self, y: i32) -> f32 {
        match *self {
            Self::Triangle { min_y, max_y } => {
                if y < min_y || y > max_y || min_y >= max_y {
                    return 0.0;
                }
                let mid = (min_y + max_y) as f32 * 0.5;
                let half_span = (max_y - min_y) as f32 * 0.5;
                if half_span <= 0.0 {
                    return 0.0;
                }
                let dist = (y as f32 - mid).abs();
                (1.0 - dist / half_span).max(0.0)
            }
            Self::Uniform { min_y, max_y } => {
                if y >= min_y && y <= max_y {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Returns the minimum and maximum Y values of this distribution.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> (i32, i32) {
        match *self {
            Self::Triangle { min_y, max_y } | Self::Uniform { min_y, max_y } => (min_y, max_y),
        }
    }
}

/// Target specification for a discrete blob ore type.
#[derive(Debug, Clone, Copy)]
pub struct OreBlobConfig {
    /// Unique feature salt for deterministic pseudo-random sequence derivation.
    pub salt: u64,
    /// Vertical height distribution profile.
    pub distribution: HeightDistribution,
    /// Target nominal blob size (number of generation steps and radius scaling).
    pub size: u32,
    /// Peak attempt count per 32³ cell at optimal altitude.
    pub peak_attempts: f32,
    /// Probability `p in [0.0, 1.0]` of discarding a voxel when exposed to adjacent air.
    pub air_discard_chance: f32,
    /// Restricts placement strictly to mountain biomes evaluated at anchor point.
    pub mountains_only: bool,
    /// Index selecting the stone-hosted and deepslate-hosted block state IDs.
    pub ore_type: OreType,
}

/// Identifies the ore material to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OreType {
    /// Coal ore.
    Coal,
    /// Iron ore.
    Iron,
    /// Copper ore.
    Copper,
    /// Gold ore.
    Gold,
    /// Redstone ore.
    Redstone,
    /// Lapis lazuli ore.
    Lapis,
    /// Diamond ore.
    Diamond,
    /// Emerald ore.
    Emerald,
    /// Granite pocket filler.
    Granite,
    /// Diorite pocket filler.
    Diorite,
    /// Andesite pocket filler.
    Andesite,
    /// Tuff pocket filler.
    Tuff,
}

impl OreType {
    /// Resolves the stone-hosted and deepslate-hosted block state IDs for this ore type.
    #[inline]
    #[must_use]
    pub const fn resolve_blocks(self, blocks: &ResolvedBlocks) -> (BlockStateId, BlockStateId) {
        match self {
            Self::Coal => (blocks.coal_ore, blocks.deepslate_coal_ore),
            Self::Iron => (blocks.iron_ore, blocks.deepslate_iron_ore),
            Self::Copper => (blocks.copper_ore, blocks.deepslate_copper_ore),
            Self::Gold => (blocks.gold_ore, blocks.deepslate_gold_ore),
            Self::Redstone => (blocks.redstone_ore, blocks.deepslate_redstone_ore),
            Self::Lapis => (blocks.lapis_ore, blocks.deepslate_lapis_ore),
            Self::Diamond => (blocks.diamond_ore, blocks.deepslate_diamond_ore),
            Self::Emerald => (blocks.emerald_ore, blocks.deepslate_emerald_ore),
            Self::Granite => (blocks.granite, blocks.granite),
            Self::Diorite => (blocks.diorite, blocks.diorite),
            Self::Andesite => (blocks.andesite, blocks.andesite),
            Self::Tuff => (blocks.tuff, blocks.tuff),
        }
    }
}

/// Pre-configured standard ore blob specifications according to terrain generation specifications.
pub const STANDARD_ORE_BLOBS: [OreBlobConfig; 13] = [
    // 1. Coal: Triangle(-128, 256), size 17, peak 12
    OreBlobConfig {
        salt: 0xC0A1_0000_1111_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -128,
            max_y: 256,
        },
        size: 17,
        peak_attempts: 12.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Coal,
    },
    // 2. Iron (deep/main band): Triangle(-256, 64), size 9, peak 6
    OreBlobConfig {
        salt: 0x1E04_0000_2222_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -256,
            max_y: 64,
        },
        size: 9,
        peak_attempts: 6.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Iron,
    },
    // 3. Iron (high mountain band): Triangle(400, 1200), size 9, peak 4
    OreBlobConfig {
        salt: 0x1E04_0000_2222_0002,
        distribution: HeightDistribution::Triangle {
            min_y: 400,
            max_y: 1200,
        },
        size: 9,
        peak_attempts: 4.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Iron,
    },
    // 4. Copper: Triangle(-32, 128), size 10, peak 8
    OreBlobConfig {
        salt: 0xC099_0000_3333_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -32,
            max_y: 128,
        },
        size: 10,
        peak_attempts: 8.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Copper,
    },
    // 5. Gold: Triangle(-320, 32), size 9, peak 3
    OreBlobConfig {
        salt: 0x601D_0000_4444_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -320,
            max_y: 32,
        },
        size: 9,
        peak_attempts: 3.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Gold,
    },
    // 6. Redstone: Triangle(-960, -320), size 8, peak 4
    OreBlobConfig {
        salt: 0x2ED5_0000_5555_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -960,
            max_y: -320,
        },
        size: 8,
        peak_attempts: 4.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Redstone,
    },
    // 7. Lapis Lazuli: Triangle(-448, 64), size 7, peak 2
    OreBlobConfig {
        salt: 0x1A91_0000_6666_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -448,
            max_y: 64,
        },
        size: 7,
        peak_attempts: 2.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Lapis,
    },
    // 8. Diamond: Triangle(-1024, -384), size 8, peak 3, air discard 0.5
    OreBlobConfig {
        salt: 0xD1A0_0000_7777_0001,
        distribution: HeightDistribution::Triangle {
            min_y: -1024,
            max_y: -384,
        },
        size: 8,
        peak_attempts: 3.0,
        air_discard_chance: 0.5,
        mountains_only: false,
        ore_type: OreType::Diamond,
    },
    // 9. Emerald: Uniform(-16, 600), mountain biomes only, size 3, peak 6
    OreBlobConfig {
        salt: 0xE0E2_0000_8888_0001,
        distribution: HeightDistribution::Uniform {
            min_y: -16,
            max_y: 600,
        },
        size: 3,
        peak_attempts: 6.0,
        air_discard_chance: 0.0,
        mountains_only: true,
        ore_type: OreType::Emerald,
    },
    // 10. Granite rock pockets
    OreBlobConfig {
        salt: 0x62A0_0000_9999_0001,
        distribution: HeightDistribution::Uniform {
            min_y: 0,
            max_y: 256,
        },
        size: 20,
        peak_attempts: 3.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Granite,
    },
    // 11. Diorite rock pockets
    OreBlobConfig {
        salt: 0xD102_0000_AAAA_0001,
        distribution: HeightDistribution::Uniform {
            min_y: 0,
            max_y: 256,
        },
        size: 20,
        peak_attempts: 3.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Diorite,
    },
    // 12. Andesite rock pockets
    OreBlobConfig {
        salt: 0xA0DE_0000_BBBB_0001,
        distribution: HeightDistribution::Uniform {
            min_y: 0,
            max_y: 256,
        },
        size: 20,
        peak_attempts: 3.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Andesite,
    },
    // 13. Tuff pockets in deep subterranean strata
    OreBlobConfig {
        salt: 0x70FF_0000_CCCC_0001,
        distribution: HeightDistribution::Uniform {
            min_y: -512,
            max_y: 0,
        },
        size: 24,
        peak_attempts: 4.0,
        air_discard_chance: 0.0,
        mountains_only: false,
        ore_type: OreType::Tuff,
    },
];

/// Returns `true` if the block state represents an ore-replaceable host rock.
#[inline]
fn is_replaceable_rock(id: BlockStateId, blocks: &ResolvedBlocks) -> bool {
    id == blocks.stone
        || id == blocks.deepslate
        || id == blocks.granite
        || id == blocks.diorite
        || id == blocks.andesite
        || id == blocks.tuff
}

/// Returns `true` if any of the 6 cardinal neighbor voxels is air.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::cast_possible_wrap
)]
fn touches_air(
    dense: &[BlockStateId; CHUNK_VOLUME],
    blocks: &ResolvedBlocks,
    seed: u64,
    lx: usize,
    ly: usize,
    lz: usize,
    wx: i32,
    wy: i32,
    wz: i32,
) -> bool {
    const DIRS: [(i32, i32, i32); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    for &(dx, dy, dz) in &DIRS {
        let nlx = lx as i32 + dx;
        let nly = ly as i32 + dy;
        let nlz = lz as i32 + dz;
        if (0..32).contains(&nlx) && (0..32).contains(&nly) && (0..32).contains(&nlz) {
            let nidx = ((nly as usize) << 10) | ((nlz as usize) << 5) | (nlx as usize);
            if dense[nidx] == blocks.air {
                return true;
            }
        } else {
            // Neighbor voxel is across the chunk boundary: query density
            let density =
                sample_density(seed, (wx + dx) as f32, (wy + dy) as f32, (wz + dz) as f32);
            if density <= 0.0 {
                return true;
            }
        }
    }
    false
}

/// Generates blob ores using 3×3×3 chunk neighborhood pull evaluation.
///
/// Guarantees that ore veins crossing 32³ cubic chunk boundaries match up perfectly
/// without truncation or seams, by evaluating neighbor cells whose blobs extend into this chunk.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn apply_blob_ores(
    seed: u64,
    chunk_pos: ChunkPos,
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let cx = chunk_pos.x();
    let cy = chunk_pos.y();
    let cz = chunk_pos.z();

    let chunk_min_x = cx * 32;
    let chunk_max_x = chunk_min_x + 32;
    let chunk_min_y = cy * 32;
    let chunk_max_y = chunk_min_y + 32;
    let chunk_min_z = cz * 32;
    let chunk_max_z = chunk_min_z + 32;

    for config in &STANDARD_ORE_BLOBS {
        let (min_dist_y, max_dist_y) = config.distribution.bounds();
        // Skip if chunk Y bounds are entirely outside the distribution plus margin
        let margin = (config.size as i32) / 4 + 4;
        if chunk_max_y < min_dist_y - margin || chunk_min_y > max_dist_y + margin {
            continue;
        }

        let (stone_id, deepslate_id) = config.ore_type.resolve_blocks(blocks);

        // Pull evaluation: check 3×3×3 chunk cell neighborhood
        for n_dy in -1..=1 {
            let cell_y = cy + n_dy;
            let cell_center_y = cell_y * 32 + 16;
            let density_factor = config.distribution.sample_density(cell_center_y);
            let lambda = config.peak_attempts * density_factor;
            if lambda <= 0.0 {
                continue;
            }

            for n_dz in -1..=1 {
                let cell_z = cz + n_dz;
                for n_dx in -1..=1 {
                    let cell_x = cx + n_dx;

                    let cell_hash = hash3(seed ^ config.salt, cell_x, cell_y, cell_z);
                    let floor_k = lambda.floor() as u32;
                    let frac = lambda - floor_k as f32;
                    let roll = unit_f32(cell_hash);
                    let attempts = if roll < frac { floor_k + 1 } else { floor_k };

                    for attempt_idx in 0..attempts {
                        let attempt_hash = hash3(
                            cell_hash ^ u64::from(attempt_idx).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            cell_x,
                            cell_y,
                            cell_z,
                        );

                        let ax = cell_x * 32 + ((attempt_hash & 31) as i32);
                        let ay = cell_y * 32 + (((attempt_hash >> 5) & 31) as i32);
                        let az = cell_z * 32 + (((attempt_hash >> 10) & 31) as i32);

                        // If mountain-only, evaluate climate biome at anchor
                        if config.mountains_only {
                            let climate = ClimatePoint::sample(seed, ax as f32, az as f32);
                            let biome = lookup_biome(&climate);
                            if biome != BiomeId::Mountains {
                                continue;
                            }
                        }

                        // Determine sphere-chain direction and length
                        let h2 = mix64(attempt_hash);
                        let angle_theta = unit_f32(h2) * std::f32::consts::TAU;
                        let angle_phi = (unit_f32(h2 >> 24) - 0.5) * std::f32::consts::PI;

                        let length = (config.size as f32) / 8.0;
                        let dx = angle_theta.cos() * angle_phi.cos() * length;
                        let dy = angle_phi.sin() * length;
                        let dz = angle_theta.sin() * angle_phi.cos() * length;

                        let p0 = (ax as f32, ay as f32, az as f32);
                        let p1 = (p0.0 + dx, p0.1 + dy, p0.2 + dz);

                        // Coarse bounding box for this blob attempt
                        let max_r = (config.size as f32) / 8.0 + 1.25;
                        let min_bx = (p0.0.min(p1.0) - max_r).floor() as i32;
                        let max_bx = (p0.0.max(p1.0) + max_r).ceil() as i32;
                        let min_by = (p0.1.min(p1.1) - max_r).floor() as i32;
                        let max_by = (p0.1.max(p1.1) + max_r).ceil() as i32;
                        let min_bz = (p0.2.min(p1.2) - max_r).floor() as i32;
                        let max_bz = (p0.2.max(p1.2) + max_r).ceil() as i32;

                        // Check intersection with current chunk bounds
                        if max_bx < chunk_min_x
                            || min_bx >= chunk_max_x
                            || max_by < chunk_min_y
                            || min_by >= chunk_max_y
                            || max_bz < chunk_min_z
                            || min_bz >= chunk_max_z
                        {
                            continue;
                        }

                        // Walk sphere chain along segment
                        let steps = (config.size / 2).max(2);
                        for s in 0..=steps {
                            let t = s as f32 / steps as f32;
                            let cx_f = lerp(p0.0, p1.0, t);
                            let cy_f = lerp(p0.1, p1.1, t);
                            let cz_f = lerp(p0.2, p1.2, t);

                            let r = (std::f32::consts::PI * t).sin()
                                * ((config.size as f32) / 16.0)
                                + 0.85;
                            let r_sq = r * r;

                            let min_sx = ((cx_f - r).floor() as i32).max(chunk_min_x);
                            let max_sx = ((cx_f + r).ceil() as i32).min(chunk_max_x - 1);
                            let min_sy = ((cy_f - r).floor() as i32).max(chunk_min_y);
                            let max_sy = ((cy_f + r).ceil() as i32).min(chunk_max_y - 1);
                            let min_sz = ((cz_f - r).floor() as i32).max(chunk_min_z);
                            let max_sz = ((cz_f + r).ceil() as i32).min(chunk_max_z - 1);

                            if min_sx > max_sx || min_sy > max_sy || min_sz > max_sz {
                                continue;
                            }

                            for vy in min_sy..=max_sy {
                                let ly = (vy - chunk_min_y) as usize;
                                let y_offset = ly << 10;
                                let d_y = vy as f32 + 0.5 - cy_f;
                                let dy_sq = d_y * d_y;

                                for vz in min_sz..=max_sz {
                                    let lz = (vz - chunk_min_z) as usize;
                                    let z_offset = y_offset | (lz << 5);
                                    let d_z = vz as f32 + 0.5 - cz_f;
                                    let dyz_sq = dy_sq + d_z * d_z;
                                    if dyz_sq > r_sq {
                                        continue;
                                    }

                                    for vx in min_sx..=max_sx {
                                        let lx = (vx - chunk_min_x) as usize;
                                        let idx = z_offset | lx;
                                        let current = dense[idx];

                                        if !is_replaceable_rock(current, blocks) {
                                            continue;
                                        }

                                        let d_x = vx as f32 + 0.5 - cx_f;
                                        if dyz_sq + d_x * d_x <= r_sq {
                                            if config.air_discard_chance > 0.0
                                                && touches_air(
                                                    dense, blocks, seed, lx, ly, lz, vx, vy, vz,
                                                )
                                            {
                                                let h_air =
                                                    hash3(seed ^ SALT_AIR_DISCARD, vx, vy, vz);
                                                if unit_f32(h_air) < config.air_discard_chance {
                                                    continue;
                                                }
                                            }

                                            let is_deep = current == blocks.deepslate
                                                || current == blocks.tuff;
                                            dense[idx] =
                                                if is_deep { deepslate_id } else { stone_id };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Evaluates large sinuous ore veins (Copper at y in [0, 128], Iron at y in [-480, -64])
/// with filler rock and raw metal concentrates.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn apply_large_ore_veins(
    seed: u64,
    chunk_pos: ChunkPos,
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let chunk_min_x = chunk_pos.x() * 32;
    let chunk_min_y = chunk_pos.y() * 32;
    let chunk_min_z = chunk_pos.z() * 32;
    let chunk_max_y = chunk_min_y + 32;

    let intersects_copper = chunk_max_y > 0 && chunk_min_y < 128;
    let intersects_iron = chunk_max_y > -480 && chunk_min_y < -64;

    // Coarse chunk rejection: completely skip chunks outside the active vertical bands
    if !intersects_copper && !intersects_iron {
        return;
    }

    // 1. Chunk-level Lipschitz rejection using toggle noise at chunk center
    let center_x = (chunk_min_x + 16) as f32;
    let center_y = (chunk_min_y + 16) as f32;
    let center_z = (chunk_min_z + 16) as f32;
    let center_toggle = noise3(
        seed ^ SALT_VEIN_TOGGLE,
        center_x * FREQ_TOGGLE,
        center_y * FREQ_TOGGLE,
        center_z * FREQ_TOGGLE,
    );

    // With lambda = 256, max variation across a 32³ chunk from its center is <= 0.16.
    // Margin of 0.22 guarantees that no chunk with valid vein voxels is ever rejected.
    let can_have_copper = intersects_copper && center_toggle >= (0.40 - 0.22);
    let can_have_iron = intersects_iron && center_toggle <= (-0.40 + 0.22);

    if !can_have_copper && !can_have_iron {
        return;
    }

    // 2. Iterate 4×4×4 sub-cells of size 8×8×8 (64 sub-cells per chunk)
    for cby in 0usize..4 {
        let cell_min_y = chunk_min_y + (cby as i32 * 8);
        let cell_max_y = cell_min_y + 8;
        let cell_center_y = (cell_min_y + 4) as f32;

        let cell_copper = can_have_copper && cell_max_y > 0 && cell_min_y < 128;
        let cell_iron = can_have_iron && cell_max_y > -480 && cell_min_y < -64;
        if !cell_copper && !cell_iron {
            continue;
        }

        for cbz in 0usize..4 {
            let cell_min_z = chunk_min_z + (cbz as i32 * 8);
            let cell_center_z = (cell_min_z + 4) as f32;

            for cbx in 0usize..4 {
                let cell_min_x = chunk_min_x + (cbx as i32 * 8);
                let cell_center_x = (cell_min_x + 4) as f32;

                // Sub-cell toggle test (max variation within 8³ from center is <= 0.04)
                let cell_toggle = noise3(
                    seed ^ SALT_VEIN_TOGGLE,
                    cell_center_x * FREQ_TOGGLE,
                    cell_center_y * FREQ_TOGGLE,
                    cell_center_z * FREQ_TOGGLE,
                );

                let cell_has_copper = cell_copper && cell_toggle >= (0.40 - 0.06);
                let cell_has_iron = cell_iron && cell_toggle <= (-0.40 + 0.06);
                if !cell_has_copper && !cell_has_iron {
                    continue;
                }

                // Sub-cell ridge test (freq 1/48, max variation within 8³ from center is <= 0.18)
                let cell_ridge = noise3(
                    seed ^ SALT_VEIN_RIDGE,
                    cell_center_x * FREQ_RIDGE,
                    cell_center_y * FREQ_RIDGE,
                    cell_center_z * FREQ_RIDGE,
                )
                .abs();
                if cell_ridge >= (0.08 + 0.20) {
                    continue;
                }

                // Only evaluate voxels in this active 8×8×8 sub-cell
                for ly in (cby * 8)..((cby + 1) * 8) {
                    let wy = chunk_min_y + ly as i32;
                    let is_copper_band = cell_has_copper && (0..=128).contains(&wy);
                    let is_iron_band = cell_has_iron && (-480..=-64).contains(&wy);
                    if !is_copper_band && !is_iron_band {
                        continue;
                    }

                    let taper = if is_copper_band {
                        let edge_dist = wy.min(128 - wy);
                        if edge_dist < 20 {
                            edge_dist as f32 / 20.0
                        } else {
                            1.0
                        }
                    } else {
                        let edge_dist = (wy - (-480)).min(-64 - wy);
                        if edge_dist < 20 {
                            edge_dist as f32 / 20.0
                        } else {
                            1.0
                        }
                    };

                    if taper <= 0.0 {
                        continue;
                    }

                    let y_offset = ly << 10;
                    let fy = wy as f32;

                    for lz in (cbz * 8)..((cbz + 1) * 8) {
                        let wz = chunk_min_z + lz as i32;
                        let z_offset = y_offset | (lz << 5);
                        let fz = wz as f32;

                        for lx in (cbx * 8)..((cbx + 1) * 8) {
                            let idx = z_offset | lx;
                            let current = dense[idx];
                            if !is_replaceable_rock(current, blocks) {
                                continue;
                            }

                            let wx = chunk_min_x + lx as i32;
                            let fx = wx as f32;

                            let toggle = noise3(
                                seed ^ SALT_VEIN_TOGGLE,
                                fx * FREQ_TOGGLE,
                                fy * FREQ_TOGGLE,
                                fz * FREQ_TOGGLE,
                            );

                            if is_copper_band {
                                if toggle <= 0.0 {
                                    continue;
                                }
                                let abs_toggle = toggle * taper;
                                if abs_toggle <= 0.4 {
                                    continue;
                                }
                                let ridge = noise3(
                                    seed ^ SALT_VEIN_RIDGE,
                                    fx * FREQ_RIDGE,
                                    fy * FREQ_RIDGE,
                                    fz * FREQ_RIDGE,
                                )
                                .abs();
                                if ridge >= 0.08 * taper {
                                    continue;
                                }

                                // Inside Copper Vein Core!
                                let gap = noise3(
                                    seed ^ SALT_VEIN_GAP,
                                    fx * FREQ_GAP,
                                    fy * FREQ_GAP,
                                    fz * FREQ_GAP,
                                );
                                let roll = unit_f32(hash3(seed ^ SALT_VEIN_BLOCK, wx, wy, wz));

                                if roll < 0.02 {
                                    dense[idx] = blocks.raw_copper_block;
                                } else {
                                    let ore_prob = lerp(
                                        0.10,
                                        0.30,
                                        ((abs_toggle - 0.4) / 0.6).clamp(0.0, 1.0),
                                    );
                                    if gap > -0.30 && roll < (0.02 + ore_prob) {
                                        let is_deep =
                                            current == blocks.deepslate || current == blocks.tuff;
                                        dense[idx] = if is_deep {
                                            blocks.deepslate_copper_ore
                                        } else {
                                            blocks.copper_ore
                                        };
                                    } else {
                                        dense[idx] = blocks.granite;
                                    }
                                }
                            } else if is_iron_band {
                                if toggle >= 0.0 {
                                    continue;
                                }
                                let abs_toggle = (-toggle) * taper;
                                if abs_toggle <= 0.4 {
                                    continue;
                                }
                                let ridge = noise3(
                                    seed ^ SALT_VEIN_RIDGE,
                                    fx * FREQ_RIDGE,
                                    fy * FREQ_RIDGE,
                                    fz * FREQ_RIDGE,
                                )
                                .abs();
                                if ridge >= 0.08 * taper {
                                    continue;
                                }

                                // Inside Iron Vein Core!
                                let gap = noise3(
                                    seed ^ SALT_VEIN_GAP,
                                    fx * FREQ_GAP,
                                    fy * FREQ_GAP,
                                    fz * FREQ_GAP,
                                );
                                let roll = unit_f32(hash3(seed ^ SALT_VEIN_BLOCK, wx, wy, wz));

                                if roll < 0.02 {
                                    dense[idx] = blocks.raw_iron_block;
                                } else {
                                    let ore_prob = lerp(
                                        0.10,
                                        0.30,
                                        ((abs_toggle - 0.4) / 0.6).clamp(0.0, 1.0),
                                    );
                                    if gap > -0.30 && roll < (0.02 + ore_prob) {
                                        let is_deep =
                                            current == blocks.deepslate || current == blocks.tuff;
                                        dense[idx] = if is_deep {
                                            blocks.deepslate_iron_ore
                                        } else {
                                            blocks.iron_ore
                                        };
                                    } else {
                                        dense[idx] = blocks.tuff;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Applies procedural 3D blob ores and large sinuous ore veins to a chunk's dense voxel array.
///
/// Modifies `dense` in-place by replacing host rock (`stone`, `deepslate`, and filler rocks)
/// while preserving caves, fluid bodies, soil strata, and surface flora.
#[inline]
pub fn apply_ores(
    seed: u64,
    chunk_pos: ChunkPos,
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    // 1. Evaluate 3D blob ores (coal, iron, copper, gold, redstone, lapis, diamond, emerald, pockets)
    apply_blob_ores(seed, chunk_pos, blocks, dense);

    // 2. Evaluate large sinuous ore veins (copper in [0, 128] with granite; iron in [-480, -64] with tuff)
    apply_large_ore_veins(seed, chunk_pos, blocks, dense);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_triangle_distribution_bounds() {
        let dist = HeightDistribution::Triangle {
            min_y: -100,
            max_y: 100,
        };

        // Midpoint must have maximum density 1.0
        assert_eq!(dist.sample_density(0), 1.0);
        // Bounds must have density 0.0
        assert_eq!(dist.sample_density(-100), 0.0);
        assert_eq!(dist.sample_density(100), 0.0);
        // Outside bounds must have density 0.0
        assert_eq!(dist.sample_density(-150), 0.0);
        assert_eq!(dist.sample_density(150), 0.0);
        // Symmetry test
        assert!((dist.sample_density(-50) - dist.sample_density(50)).abs() < 1e-6);
        assert!((dist.sample_density(50) - 0.5).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_uniform_distribution_bounds() {
        let dist = HeightDistribution::Uniform {
            min_y: 10,
            max_y: 50,
        };

        assert_eq!(dist.sample_density(9), 0.0);
        assert_eq!(dist.sample_density(10), 1.0);
        assert_eq!(dist.sample_density(30), 1.0);
        assert_eq!(dist.sample_density(50), 1.0);
        assert_eq!(dist.sample_density(51), 0.0);
    }

    #[test]
    fn test_ore_type_resolution() {
        let reg = telos_voxel::registry::BlockRegistry::standard();
        let blocks = ResolvedBlocks::resolve(&reg);

        let (stone_coal, deep_coal) = OreType::Coal.resolve_blocks(&blocks);
        assert_eq!(stone_coal, blocks.coal_ore);
        assert_eq!(deep_coal, blocks.deepslate_coal_ore);

        let (stone_dia, deep_dia) = OreType::Diamond.resolve_blocks(&blocks);
        assert_eq!(stone_dia, blocks.diamond_ore);
        assert_eq!(deep_dia, blocks.deepslate_diamond_ore);
    }

    #[test]
    fn test_air_exposure_discard_determinism() {
        let h1 = hash3(0x0042 ^ SALT_AIR_DISCARD, 10, -20, 30);
        let h2 = hash3(0x0042 ^ SALT_AIR_DISCARD, 10, -20, 30);
        assert_eq!(h1, h2);
        let roll = unit_f32(h1);
        assert!((0.0..1.0).contains(&roll));
    }
}
