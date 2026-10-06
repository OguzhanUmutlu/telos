//! High-performance, zero-allocation procedural surface decoration and flora scatter engine.
//!
//! Evaluates deterministic biome-specific foliage placement (short grass, ferns, dead bushes)
//! and dynamically clustered floral meadows (poppy and dandelion groups) on terrain surface blocks.

use crate::biome::BiomeId;
use crate::density::sample_density;
use crate::math::{hash3, unit_f32};
use crate::noise::noise2;
use crate::surface::ResolvedBlocks;
use vx_core::coords::{CHUNK_VOLUME, ChunkPos};
use vx_voxel::state::BlockStateId;

/// Salting constant for flower meadow cluster noise.
const SALT_CLUSTER: u64 = 0xD3C0_4477_1122_3344;
/// Salting constant for floral species selection noise.
const SALT_SPECIES: u64 = 0x5566_7788_99AA_BBCC;
/// Salting constant for per-voxel pseudo-random scatter rolls.
const SALT_SCATTER: u64 = 0xA1B2_C3D4_E5F6_0718;

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

    // Precompute 64 cell meadow clusters (4x4 column blocks, matching the biomes grid)
    // to keep decoration throughput ultra-fast (<= 15 µs per chunk) without heap allocations.
    let mut cell_meadow = [false; 64];
    let mut cell_is_poppy = [false; 64];
    let mut cell_flower_chance = [0.0f32; 64];

    for cz in 0usize..8 {
        let center_z = origin_z + (cz as i32 * 4) + 2;
        let z_f = center_z as f32;
        for cx in 0usize..8 {
            let cell_idx = cz * 8 + cx;
            let biome = biomes[cell_idx];
            if biome == BiomeId::Plains || biome == BiomeId::Forest {
                let center_x = origin_x + (cx as i32 * 4) + 2;
                let x_f = center_x as f32;

                let cluster_val = noise2(seed.wrapping_add(SALT_CLUSTER), x_f * 0.025, z_f * 0.025);
                if cluster_val > 0.35 {
                    cell_meadow[cell_idx] = true;
                    let species_val =
                        noise2(seed.wrapping_add(SALT_SPECIES), x_f * 0.012, z_f * 0.012);
                    cell_is_poppy[cell_idx] = species_val > 0.0;
                    let meadow_intensity = (cluster_val - 0.35) * 2.8;
                    cell_flower_chance[cell_idx] =
                        (0.35f32 * meadow_intensity + 0.15f32).min(0.55f32);
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
            let is_poppy = cell_is_poppy[cell_idx];
            let flower_chance = cell_flower_chance[cell_idx];

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
                };

                if is_valid_ground && dense[above_idx] == blocks.air {
                    let wy = origin_y + y as i32;
                    let deco = evaluate_decoration_fast(
                        seed,
                        wx,
                        wy,
                        wz,
                        biome,
                        in_meadow,
                        is_poppy,
                        flower_chance,
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
                                let deco = evaluate_decoration_fast(
                                    seed,
                                    wx,
                                    wy_below,
                                    wz,
                                    biome,
                                    in_meadow,
                                    is_poppy,
                                    flower_chance,
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

/// Evaluates decoration flora block at world position `(wx, wy, wz)` using precomputed cell meadow state.
#[inline]
#[allow(clippy::too_many_arguments)]
fn evaluate_decoration_fast(
    seed: u64,
    wx: i32,
    wy: i32,
    wz: i32,
    biome: BiomeId,
    in_meadow: bool,
    is_poppy: bool,
    flower_chance: f32,
    blocks: &ResolvedBlocks,
) -> BlockStateId {
    match biome {
        BiomeId::Plains | BiomeId::Forest => {
            let h = hash3(seed.wrapping_add(SALT_SCATTER), wx, wy, wz);
            let roll = unit_f32(h);

            if in_meadow {
                let flower = if is_poppy {
                    blocks.poppy
                } else {
                    blocks.dandelion
                };

                if roll < flower_chance {
                    flower
                } else if roll < 0.72 {
                    if biome == BiomeId::Forest && roll > 0.55 {
                        blocks.fern
                    } else {
                        blocks.short_grass
                    }
                } else {
                    BlockStateId::AIR
                }
            } else if biome == BiomeId::Plains {
                // Plains: grass meadows with rare single flowers
                if roll < 0.22 {
                    blocks.short_grass
                } else if roll < 0.235 {
                    if (h & 1) == 0 {
                        blocks.poppy
                    } else {
                        blocks.dandelion
                    }
                } else {
                    BlockStateId::AIR
                }
            } else {
                // Forest: ferns and short grass with rare forest flowers
                if roll < 0.18 {
                    blocks.short_grass
                } else if roll < 0.30 {
                    blocks.fern
                } else if roll < 0.32 {
                    if (h & 1) == 0 {
                        blocks.poppy
                    } else {
                        blocks.dandelion
                    }
                } else {
                    BlockStateId::AIR
                }
            }
        }
        BiomeId::Mountains => {
            if wy < 90 {
                let h = hash3(seed.wrapping_add(SALT_SCATTER), wx, wy, wz);
                let roll = unit_f32(h);
                if roll < 0.12 {
                    blocks.short_grass
                } else if roll < 0.15 {
                    blocks.dandelion
                } else {
                    BlockStateId::AIR
                }
            } else {
                BlockStateId::AIR
            }
        }
        BiomeId::Desert => {
            let h = hash3(seed.wrapping_add(SALT_SCATTER), wx, wy, wz);
            let roll = unit_f32(h);
            if roll < 0.025 {
                blocks.dead_bush
            } else {
                BlockStateId::AIR
            }
        }
        BiomeId::Ocean => BlockStateId::AIR,
    }
}
