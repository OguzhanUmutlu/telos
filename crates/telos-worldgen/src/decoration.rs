//! High-performance, zero-allocation procedural surface decoration and flora scatter engine.
//!
//! Evaluates deterministic biome-specific foliage placement (short grass, tall grass, ferns, dead bushes)
//! and dynamically clustered floral meadows (poppy, dandelion, cornflower, oxeye daisy) and shaded
//! mushroom groves (brown and red mushrooms) on terrain surface blocks.
//!
//! Implements a multi-scale Poisson-disk / jittered grid distribution that guarantees minimum separation
//! between cluster centroids without order-dependent dart-throwing or per-block heap allocations.

use crate::biome::BiomeId;
use crate::density::sample_density;
use crate::math::{hash3, unit_f32};
use crate::noise::noise2;
use crate::surface::ResolvedBlocks;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Salting constant for flower meadow cluster noise.
const SALT_CLUSTER: u64 = 0xD3C0_4477_1122_3344;
/// Salting constant for floral species selection noise.
const SALT_SPECIES: u64 = 0x5566_7788_99AA_BBCC;
/// Salting constant for mushroom grove noise.
const SALT_MUSHROOM: u64 = 0x7788_99AA_BBCC_DD00;
/// Salting constant for cell jitter offsets (Poisson-disk spacing).
const SALT_JITTER: u64 = 0x1A2B_3C4D_5E6F_7081;
/// Salting constant for per-voxel pseudo-random scatter rolls.
const SALT_SCATTER: u64 = 0xA1B2_C3D4_E5F6_0718;

/// Wildflower species supported in clustered meadows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowerKind {
    /// Red poppy.
    Poppy,
    /// Yellow dandelion.
    Dandelion,
    /// Blue cornflower.
    Cornflower,
    /// White oxeye daisy.
    OxeyeDaisy,
}

impl FlowerKind {
    /// Resolves the corresponding block state identifier.
    #[inline]
    #[must_use]
    pub const fn to_state(self, blocks: &ResolvedBlocks) -> BlockStateId {
        match self {
            Self::Poppy => blocks.poppy,
            Self::Dandelion => blocks.dandelion,
            Self::Cornflower => blocks.cornflower,
            Self::OxeyeDaisy => blocks.oxeye_daisy,
        }
    }
}

/// Evaluates surface flora decoration for a single chunk in-place on the dense voxel array.
///
/// Operates with **zero heap allocations** and guarantees boundary continuity across
/// cubic chunk seams by referencing global coordinate hashes and density boundaries.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub fn apply_surface_decorations(
    seed: u64,
    chunk_pos: ChunkPos,
    biomes: &[BiomeId; 64],
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let origin_x = chunk_pos.x() * 32;
    let origin_y = chunk_pos.y() * 32;
    let origin_z = chunk_pos.z() * 32;

    // Precompute 64 cell meadow and mushroom clusters (4x4 column blocks, matching the biomes grid).
    // Uses stack arrays with zero heap allocations for fast throughput (<= 15 µs per chunk).
    let mut cell_meadow = [false; 64];
    let mut cell_flower_kind = [FlowerKind::Poppy; 64];
    let mut cell_flower_chance = [0.0f32; 64];
    let mut cell_mushroom = [false; 64];
    let mut cell_jitter_x = [1u8; 64];
    let mut cell_jitter_z = [1u8; 64];

    for cz in 0usize..8 {
        let cell_z = origin_z + (cz as i32 * 4);
        let center_z = cell_z + 2;
        let z_f = center_z as f32;

        for cx in 0usize..8 {
            let cell_x = origin_x + (cx as i32 * 4);
            let center_x = cell_x + 2;
            let x_f = center_x as f32;

            let cell_idx = cz * 8 + cx;
            let biome = biomes[cell_idx];

            // Deterministic jitter offset (1..2) ensuring Poisson-disk spacing between cell centers
            let j_hash = hash3(seed.wrapping_add(SALT_JITTER), cell_x >> 2, 0, cell_z >> 2);
            cell_jitter_x[cell_idx] = 1 + ((j_hash & 1) as u8);
            cell_jitter_z[cell_idx] = 1 + (((j_hash >> 1) & 1) as u8);

            if biome == BiomeId::Plains || biome == BiomeId::Forest || biome == BiomeId::Mountains {
                let cluster_val = noise2(seed.wrapping_add(SALT_CLUSTER), x_f * 0.025, z_f * 0.025);
                if cluster_val > 0.32 {
                    cell_meadow[cell_idx] = true;
                    let species_val =
                        noise2(seed.wrapping_add(SALT_SPECIES), x_f * 0.012, z_f * 0.012);

                    cell_flower_kind[cell_idx] = if species_val < -0.20 {
                        FlowerKind::Poppy
                    } else if species_val < 0.05 {
                        FlowerKind::Dandelion
                    } else if species_val < 0.30 {
                        FlowerKind::Cornflower
                    } else {
                        FlowerKind::OxeyeDaisy
                    };

                    let meadow_intensity = (cluster_val - 0.32) * 3.0;
                    cell_flower_chance[cell_idx] =
                        (0.35f32 * meadow_intensity + 0.15f32).min(0.60f32);
                }
            }

            // Mushroom groves in forests and shaded lowlands
            if biome == BiomeId::Forest {
                let shroom_val = noise2(seed.wrapping_add(SALT_MUSHROOM), x_f * 0.035, z_f * 0.035);
                if shroom_val > 0.38 {
                    cell_mushroom[cell_idx] = true;
                }
            }
        }
    }

    for z in 0usize..32 {
        let wz = origin_z + z as i32;
        let cz = z >> 2;
        for x in 0usize..32 {
            let wx = origin_x + x as i32;
            let cx = x >> 2;
            let cell_idx = cz * 8 + cx;
            let biome = biomes[cell_idx];

            let in_meadow = cell_meadow[cell_idx];
            let flower_kind = cell_flower_kind[cell_idx];
            let flower_chance = cell_flower_chance[cell_idx];
            let in_mushroom_grove = cell_mushroom[cell_idx];
            let jx = usize::from(cell_jitter_x[cell_idx]);
            let jz = usize::from(cell_jitter_z[cell_idx]);

            // Squared distance to cell jitter anchor
            let lx = x & 3;
            let lz = z & 3;
            let dist_sq = (lx as i32 - jx as i32).pow(2) + (lz as i32 - jz as i32).pow(2);

            // 1. Scan y from 30 down to 0 for surface ground inside this chunk
            let mut decorated = false;
            for y in (0usize..31).rev() {
                let ground_idx = (y << 10) | (z << 5) | x;
                let above_idx = ((y + 1) << 10) | (z << 5) | x;

                let ground_block = dense[ground_idx];
                let is_valid_ground = if biome == BiomeId::Desert {
                    ground_block == blocks.sand
                } else {
                    ground_block == blocks.grass
                        || (in_mushroom_grove
                            && (ground_block == blocks.dirt || ground_block == blocks.stone))
                };

                if is_valid_ground && dense[above_idx] == blocks.air {
                    let wy = origin_y + y as i32;
                    let is_shaded = check_shaded(dense, x, y + 1, z, blocks);

                    let deco = evaluate_decoration_fast(
                        seed,
                        wx,
                        wy,
                        wz,
                        biome,
                        in_meadow,
                        flower_kind,
                        flower_chance,
                        in_mushroom_grove,
                        is_shaded,
                        dist_sq,
                        blocks,
                    );
                    if deco != BlockStateId::AIR {
                        dense[above_idx] = deco;
                    }
                    decorated = true;
                    break;
                }
            }

            // 2. Handle boundary continuity: check local y = 0
            // If no surface was decorated inside this chunk and local y = 0 is AIR,
            // test if the block directly beneath this chunk (origin_y - 1) is ground.
            if !decorated {
                let bottom_idx = (z << 5) | x;
                if dense[bottom_idx] == blocks.air {
                    let wy_below = origin_y - 1;
                    if wy_below > 0 && wy_below < 300 {
                        let density_below =
                            sample_density(seed, wx as f32, wy_below as f32, wz as f32);
                        if density_below > 0.0 {
                            let is_valid = biome == BiomeId::Desert
                                || !(biome == BiomeId::Mountains && wy_below >= 90);

                            if is_valid {
                                let is_shaded = check_shaded(dense, x, 0, z, blocks);
                                let deco = evaluate_decoration_fast(
                                    seed,
                                    wx,
                                    wy_below,
                                    wz,
                                    biome,
                                    in_meadow,
                                    flower_kind,
                                    flower_chance,
                                    in_mushroom_grove,
                                    is_shaded,
                                    dist_sq,
                                    blocks,
                                );
                                if deco != BlockStateId::AIR {
                                    dense[bottom_idx] = deco;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Helper checking if an air location has canopy/rock cover above it within 6 blocks.
#[inline]
fn check_shaded(
    dense: &[BlockStateId; CHUNK_VOLUME],
    x: usize,
    y: usize,
    z: usize,
    blocks: &ResolvedBlocks,
) -> bool {
    let max_y = (y + 6).min(31);
    for check_y in (y + 1)..=max_y {
        let block = dense[(check_y << 10) | (z << 5) | x];
        if block == blocks.oak_leaves
            || block == blocks.birch_leaves
            || block == blocks.spruce_leaves
            || block == blocks.stone
            || block == blocks.dirt
        {
            return true;
        }
    }
    false
}

/// Evaluates decoration flora block at world position `(wx, wy, wz)` using precomputed cell meadow state.
#[inline]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn evaluate_decoration_fast(
    seed: u64,
    wx: i32,
    wy: i32,
    wz: i32,
    biome: BiomeId,
    in_meadow: bool,
    flower_kind: FlowerKind,
    flower_chance: f32,
    in_mushroom_grove: bool,
    is_shaded: bool,
    dist_sq: i32,
    blocks: &ResolvedBlocks,
) -> BlockStateId {
    let h = hash3(seed.wrapping_add(SALT_SCATTER), wx, wy, wz);
    let roll = unit_f32(h);

    match biome {
        BiomeId::Plains => {
            if in_meadow {
                // Radial proximity bonus to cell jitter anchor
                let proximity_bonus = if dist_sq <= 2 { 0.15 } else { 0.0 };
                let effective_chance = (flower_chance + proximity_bonus).min(0.70);

                if roll < effective_chance {
                    flower_kind.to_state(blocks)
                } else if roll < 0.72 {
                    if roll < 0.58 {
                        blocks.short_grass
                    } else {
                        blocks.tall_grass
                    }
                } else {
                    BlockStateId::AIR
                }
            } else {
                // Plains: grass meadows with tall grass and rare wildflowers
                if roll < 0.20 {
                    blocks.short_grass
                } else if roll < 0.28 {
                    blocks.tall_grass
                } else if roll < 0.30 {
                    // Stray single wildflower
                    match (h >> 4) & 3 {
                        0 => blocks.poppy,
                        1 => blocks.dandelion,
                        2 => blocks.cornflower,
                        _ => blocks.oxeye_daisy,
                    }
                } else {
                    BlockStateId::AIR
                }
            }
        }
        BiomeId::Forest => {
            // Mushrooms in shaded groves or under canopy cover
            if is_shaded || in_mushroom_grove {
                let shroom_chance = if is_shaded && in_mushroom_grove {
                    0.20
                } else if is_shaded {
                    0.08
                } else {
                    0.04
                };

                if roll < shroom_chance {
                    return if (h & 1) == 0 {
                        blocks.brown_mushroom
                    } else {
                        blocks.red_mushroom
                    };
                }
            }

            if in_meadow {
                let effective_chance = flower_chance.min(0.50);
                if roll < effective_chance {
                    flower_kind.to_state(blocks)
                } else if roll < 0.68 {
                    if roll < 0.45 {
                        blocks.fern
                    } else if roll < 0.60 {
                        blocks.short_grass
                    } else {
                        blocks.tall_grass
                    }
                } else {
                    BlockStateId::AIR
                }
            } else {
                // Forest: ferns, short grass, tall grass, and rare forest flowers
                if roll < 0.18 {
                    blocks.short_grass
                } else if roll < 0.32 {
                    blocks.fern
                } else if roll < 0.38 {
                    blocks.tall_grass
                } else if roll < 0.40 {
                    if (h & 1) == 0 {
                        blocks.poppy
                    } else {
                        blocks.oxeye_daisy
                    }
                } else {
                    BlockStateId::AIR
                }
            }
        }
        BiomeId::Mountains => {
            if wy < 90 {
                if roll < 0.14 {
                    blocks.short_grass
                } else if roll < 0.18 {
                    blocks.tall_grass
                } else if roll < 0.21 {
                    if (h & 1) == 0 {
                        blocks.dandelion
                    } else {
                        blocks.poppy
                    }
                } else {
                    BlockStateId::AIR
                }
            } else {
                BlockStateId::AIR
            }
        }
        BiomeId::Desert => {
            if roll < 0.025 {
                blocks.dead_bush
            } else {
                BlockStateId::AIR
            }
        }
        BiomeId::Ocean => BlockStateId::AIR,
    }
}

/// Evaluates distance surface color blending vegetation cover into base terrain top color for LOD rendering.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn biome_expected_vegetation_color(biome: BiomeId, base_top_color: [u8; 4]) -> [u8; 4] {
    let (veg_color, density): ([u8; 4], f32) = match biome {
        BiomeId::Plains => ([106, 170, 64, 255], 0.30f32),
        BiomeId::Forest => ([80, 160, 50, 255], 0.35f32),
        BiomeId::Mountains => ([106, 170, 64, 255], 0.10f32),
        BiomeId::Desert | BiomeId::Ocean => return base_top_color,
    };

    let t = density.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    [
        (f32::from(base_top_color[0]) * inv + f32::from(veg_color[0]) * t) as u8,
        (f32::from(base_top_color[1]) * inv + f32::from(veg_color[1]) * t) as u8,
        (f32::from(base_top_color[2]) * inv + f32::from(veg_color[2]) * t) as u8,
        255,
    ]
}
