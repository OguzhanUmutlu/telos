//! High-performance, deterministic procedural tree generation and foliage canopies.
//!
//! Generates multi-stage tree structures (Oak, Branching Oak, Birch, Spruce, Pine, Mega Taiga)
//! using a pure-CPU world-aligned jittered grid and a deterministic pull evaluation pattern.
//! Guarantees 100% seamless continuity across 32³ cubic chunk seams without inter-chunk messaging.

use crate::biome::{BiomeId, lookup_biome};
use crate::climate::ClimatePoint;
use crate::density::{base_terrain_height, sample_density};
use crate::math::{hash3, unit_f32};
use crate::surface::ResolvedBlocks;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Salting key for tree jittered grid position hashing.
pub const SALT_TREE_JITTER: u64 = 0x7E3E_A1B2_C3D4_E5F6;
/// Salting key for tree spawn probability rolls.
pub const SALT_TREE_CHANCE: u64 = 0x9182_7364_5544_3322;
/// Salting key for tree species selection.
pub const SALT_TREE_SPECIES: u64 = 0xAABB_CCDD_EEFF_0011;
/// Salting key for tree morphology variations (height, branches).
pub const SALT_TREE_SHAPE: u64 = 0xFEDC_BA98_7654_3210;

/// Grid cell size for the world-aligned tree candidate lattice (in voxels).
pub const TREE_GRID_CELL_SIZE: i32 = 16;
/// Maximum horizontal reach of any tree canopy from its anchor coordinate (in voxels).
pub const TREE_MAX_HORIZONTAL_REACH: i32 = 5;
/// Maximum vertical height of any tree from its anchor coordinate (in voxels).
pub const TREE_MAX_HEIGHT: i32 = 24;

/// Distinct procedural tree archetypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeSpecies {
    /// Standard temperate oak with compact rounded canopy.
    StandardOak,
    /// Large branching oak with diagonal limbs and terminal leaf clusters.
    BranchingOak,
    /// Slender white birch with tall cylindrical canopy.
    Birch,
    /// Conical tiered spruce with concentric horizontal disc layers.
    Spruce,
    /// High-altitude pine with bare lower trunk and umbrella dome canopy.
    Pine,
    /// Massive dense conifer with 2×2 thick trunk and wide layered canopy.
    MegaTaiga,
}

/// Evaluates surface ground elevation $y$ for a tree anchor at world coordinates `(wx, wz)`.
///
/// Uses binary search around `base_terrain_height` to find surface ground elevation.
/// Returns `Some(y)` if a valid non-submerged surface ground block exists, or `None` if
/// the coordinate is open ocean, deep water, or above the sky boundary.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
pub fn find_surface_ground_y(seed: u64, wx: i32, wz: i32) -> Option<i32> {
    let climate = ClimatePoint::sample(seed, wx as f32, wz as f32);
    // Ocean biomes do not anchor land trees
    if climate.continentalness < -0.15 {
        return None;
    }

    let h_approx = base_terrain_height(seed, wx as f32, wz as f32, &climate).round() as i32;
    let softness = if climate.erosion < -0.3 && climate.peaks_and_valleys > 0.2 {
        24.0
    } else {
        12.0
    };

    // Fast initial guess with Newton step
    let mut y_cur = h_approx;
    let d0 = sample_density(seed, wx as f32, y_cur as f32, wz as f32);
    let step = (d0 * softness).round() as i32;
    y_cur += step;

    // Window around estimated surface
    let mut low = (y_cur - 4).max(-64);
    let mut high = (y_cur + 4).min(240);

    // Expand bounds if necessary to guarantee low is solid and high is air
    if sample_density(seed, wx as f32, low as f32, wz as f32) <= 0.0 {
        low = (low - 8).max(-64);
    }
    if sample_density(seed, wx as f32, high as f32, wz as f32) > 0.0 {
        high = (high + 8).min(240);
    }

    // Binary search for exact zero-crossing
    while low + 1 < high {
        let mid = i32::midpoint(low, high);
        if sample_density(seed, wx as f32, mid as f32, wz as f32) > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }

    // Verify low is solid and low + 1 is air
    if low < 1 || sample_density(seed, wx as f32, low as f32, wz as f32) <= 0.0 {
        return None;
    }

    Some(low)
}

/// Evaluates and stamps procedural trees intersecting the given 32³ cubic chunk.
///
/// Pure-pull evaluation: scans all candidate lattice cells whose reach bounding box
/// intersects the chunk bounds, generating trunks and foliage with zero inter-chunk communication.
#[allow(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub fn apply_trees(
    seed: u64,
    chunk_pos: ChunkPos,
    _biomes: &[BiomeId; 64],
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let chunk_min_x = chunk_pos.x() * 32;
    let chunk_max_x = chunk_min_x + 31;
    let chunk_min_y = chunk_pos.y() * 32;
    let chunk_max_y = chunk_min_y + 31;
    let chunk_min_z = chunk_pos.z() * 32;
    let chunk_max_z = chunk_min_z + 31;

    // Determine the range of 16x16 candidate grid cells that could reach into this chunk
    let cell_x_start = (chunk_min_x - TREE_MAX_HORIZONTAL_REACH).div_euclid(TREE_GRID_CELL_SIZE);
    let cell_x_end = (chunk_max_x + TREE_MAX_HORIZONTAL_REACH).div_euclid(TREE_GRID_CELL_SIZE);
    let cell_z_start = (chunk_min_z - TREE_MAX_HORIZONTAL_REACH).div_euclid(TREE_GRID_CELL_SIZE);
    let cell_z_end = (chunk_max_z + TREE_MAX_HORIZONTAL_REACH).div_euclid(TREE_GRID_CELL_SIZE);

    for cz in cell_z_start..=cell_z_end {
        for cx in cell_x_start..=cell_x_end {
            // Jittered anchor inside this 16x16 cell
            let h_pos = hash3(seed.wrapping_add(SALT_TREE_JITTER), cx, 0, cz);
            let jx = 2 + (h_pos & 0x7) as i32; // [2..=9]
            let jz = 2 + ((h_pos >> 3) & 0x7) as i32; // [2..=9]

            let wx = cx * TREE_GRID_CELL_SIZE + jx;
            let wz = cz * TREE_GRID_CELL_SIZE + jz;

            // Fast climate check and coarse elevation cull before density evaluation
            let climate = ClimatePoint::sample(seed, wx as f32, wz as f32);
            if climate.continentalness < -0.15 {
                continue;
            }
            let h_approx = base_terrain_height(seed, wx as f32, wz as f32, &climate).round() as i32;
            if (h_approx + TREE_MAX_HEIGHT + 14) < chunk_min_y || (h_approx - 18) > chunk_max_y {
                continue;
            }

            // Find surface ground elevation ry
            let Some(ry) = find_surface_ground_y(seed, wx, wz) else {
                continue;
            };

            // Bounding box vertical cull: tree spans [ry, ry + TREE_MAX_HEIGHT]
            if (ry + TREE_MAX_HEIGHT) < chunk_min_y || ry > chunk_max_y {
                continue;
            }

            // Sample climate and biome at the tree anchor
            let climate = ClimatePoint::sample(seed, wx as f32, wz as f32);
            let biome = lookup_biome(&climate);

            // Determine spawn probability per biome
            let chance_roll = unit_f32(hash3(seed.wrapping_add(SALT_TREE_CHANCE), wx, ry, wz));
            let spawn_prob = match biome {
                BiomeId::Forest => 0.85,
                BiomeId::Plains => 0.08,
                BiomeId::Mountains => {
                    if ry < 80 {
                        0.55
                    } else if ry < 140 {
                        0.30
                    } else {
                        0.05
                    }
                }
                BiomeId::Desert | BiomeId::Ocean => 0.0,
            };

            if chance_roll > spawn_prob {
                continue;
            }

            // Select tree species based on biome and species roll
            let species_roll = unit_f32(hash3(seed.wrapping_add(SALT_TREE_SPECIES), wx, ry, wz));
            let species = match biome {
                BiomeId::Forest => {
                    if species_roll < 0.45 {
                        TreeSpecies::StandardOak
                    } else if species_roll < 0.70 {
                        TreeSpecies::BranchingOak
                    } else {
                        TreeSpecies::Birch
                    }
                }
                BiomeId::Plains => {
                    if species_roll < 0.80 {
                        TreeSpecies::StandardOak
                    } else {
                        TreeSpecies::BranchingOak
                    }
                }
                BiomeId::Mountains => {
                    if species_roll < 0.50 {
                        TreeSpecies::Spruce
                    } else if species_roll < 0.85 {
                        TreeSpecies::Pine
                    } else {
                        TreeSpecies::MegaTaiga
                    }
                }
                BiomeId::Desert | BiomeId::Ocean => continue,
            };

            let shape_seed = hash3(seed.wrapping_add(SALT_TREE_SHAPE), wx, ry, wz);

            // Stamp tree blocks with clipping to the current chunk
            stamp_tree_blocks(
                species,
                wx,
                ry,
                wz,
                shape_seed,
                blocks,
                chunk_min_x,
                chunk_max_x,
                chunk_min_y,
                chunk_max_y,
                chunk_min_z,
                chunk_max_z,
                dense,
            );
        }
    }
}

/// Helper that stamps a single voxel if it lies inside the target chunk's bounding box.
#[inline]
#[allow(clippy::too_many_arguments)]
fn stamp_voxel(
    bx: i32,
    by: i32,
    bz: i32,
    block: BlockStateId,
    is_wood: bool,
    blocks: &ResolvedBlocks,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    min_z: i32,
    max_z: i32,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    if bx >= min_x && bx <= max_x && by >= min_y && by <= max_y && bz >= min_z && bz <= max_z {
        let lx = (bx - min_x) as usize;
        let ly = (by - min_y) as usize;
        let lz = (bz - min_z) as usize;
        let idx = (ly << 10) | (lz << 5) | lx;
        let current = dense[idx];

        if is_wood {
            // Wood logs replace air, leaves, and soft ground foliage
            if current == blocks.air
                || current == blocks.oak_leaves
                || current == blocks.birch_leaves
                || current == blocks.spruce_leaves
                || current == blocks.short_grass
                || current == blocks.fern
                || current == blocks.poppy
                || current == blocks.dandelion
                || current == blocks.dead_bush
            {
                dense[idx] = block;
            }
        } else {
            // Leaves replace ONLY air and soft ground foliage (never replace logs, dirt, grass, stone)
            if current == blocks.air
                || current == blocks.short_grass
                || current == blocks.fern
                || current == blocks.poppy
                || current == blocks.dandelion
                || current == blocks.dead_bush
            {
                dense[idx] = block;
            }
        }
    }
}

/// Stamps blocks for a specific tree archetype.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_wrap
)]
fn stamp_tree_blocks(
    species: TreeSpecies,
    wx: i32,
    ry: i32,
    wz: i32,
    shape_seed: u64,
    blocks: &ResolvedBlocks,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    min_z: i32,
    max_z: i32,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let mut stamp = |bx: i32, by: i32, bz: i32, block: BlockStateId, is_wood: bool| {
        stamp_voxel(
            bx, by, bz, block, is_wood, blocks, min_x, max_x, min_y, max_y, min_z, max_z, dense,
        );
    };

    match species {
        TreeSpecies::StandardOak => {
            let height = 4 + (shape_seed % 3) as i32; // [4..=6]
            let log = blocks.oak_log;
            let leaves = blocks.oak_leaves;

            // Trunk
            for y in 1..=height {
                stamp(wx, ry + y, wz, log, true);
            }

            // Canopy
            // Lower/mid canopy layers
            for dy in (height - 2)..=height {
                let radius: i32 = if dy == height { 1 } else { 2 };
                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        // Omit outer sharp corners
                        if dx.abs() == 2
                            && dz.abs() == 2
                            && shape_seed.wrapping_add((dx + dz) as u64).is_multiple_of(2)
                        {
                            continue;
                        }
                        stamp(wx + dx, ry + dy, wz + dz, leaves, false);
                    }
                }
            }

            // Canopy crown apex (radius 1 cross)
            let apex_y = ry + height + 1;
            for dz in -1i32..=1i32 {
                for dx in -1i32..=1i32 {
                    if dx * dx + dz * dz <= 1 {
                        stamp(wx + dx, apex_y, wz + dz, leaves, false);
                    }
                }
            }
        }

        TreeSpecies::BranchingOak => {
            let trunk_h = 5 + (shape_seed % 3) as i32; // [5..=7]
            let log = blocks.oak_log;
            let leaves = blocks.oak_leaves;

            // Main vertical trunk
            for y in 1..=trunk_h {
                stamp(wx, ry + y, wz, log, true);
            }

            // 2 to 3 branches slanting diagonally outward
            let branch_dirs: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            let num_branches = 2 + (shape_seed & 1) as usize; // 2 or 3 branches

            for i in 0..num_branches {
                let (dir_x, dir_z) = branch_dirs[(shape_seed as usize + i) % 4];
                let branch_start_y = ry + trunk_h - 2;

                // Branch log steps
                for step in 1..=2 {
                    stamp(
                        wx + dir_x * step,
                        branch_start_y + step,
                        wz + dir_z * step,
                        log,
                        true,
                    );
                }

                // Leaf cluster around branch tip
                let tip_x = wx + dir_x * 2;
                let tip_y = branch_start_y + 2;
                let tip_z = wz + dir_z * 2;

                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for dx in -1..=1 {
                            if dx * dx + dy * dy + dz * dz <= 2 {
                                stamp(tip_x + dx, tip_y + dy, tip_z + dz, leaves, false);
                            }
                        }
                    }
                }
            }

            // Leaf dome atop the main central trunk
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    for dx in -1i32..=1i32 {
                        if dx * dx + dy * dy + dz * dz <= 2 {
                            stamp(wx + dx, ry + trunk_h + dy, wz + dz, leaves, false);
                        }
                    }
                }
            }
        }

        TreeSpecies::Birch => {
            let height = 6 + (shape_seed % 3) as i32; // [6..=8]
            let log = blocks.birch_log;
            let leaves = blocks.birch_leaves;

            // Slender tall trunk
            for y in 1..=height {
                stamp(wx, ry + y, wz, log, true);
            }

            // Tall cylindrical canopy
            for dy in (height - 3)..=height {
                for dz in -2i32..=2i32 {
                    for dx in -2i32..=2i32 {
                        // Exclude corner voxels
                        if (dx == 2 || dx == -2) && (dz == 2 || dz == -2) {
                            continue;
                        }
                        stamp(wx + dx, ry + dy, wz + dz, leaves, false);
                    }
                }
            }

            // Cap at apex
            for dz in -1i32..=1i32 {
                for dx in -1i32..=1i32 {
                    if dx * dx + dz * dz <= 1 {
                        stamp(wx + dx, ry + height + 1, wz + dz, leaves, false);
                    }
                }
            }
        }

        TreeSpecies::Spruce => {
            let height = 7 + (shape_seed % 4) as i32; // [7..=10]
            let log = blocks.spruce_log;
            let leaves = blocks.spruce_leaves;

            // Central vertical trunk
            for y in 1..=height {
                stamp(wx, ry + y, wz, log, true);
            }

            // Conical tiered leaf layers decreasing toward apex
            // Apex tip
            stamp(wx, ry + height + 1, wz, leaves, false);
            // Height layer: radius 1 cross
            for dz in -1i32..=1i32 {
                for dx in -1i32..=1i32 {
                    if dx * dx + dz * dz <= 1 {
                        stamp(wx + dx, ry + height, wz + dz, leaves, false);
                    }
                }
            }

            // Tiered disc layers alternating in radius
            for y in (ry + 2)..(ry + height) {
                let dist_from_apex = (ry + height + 1) - y;
                let radius = match dist_from_apex % 2 {
                    0 => (dist_from_apex / 3).clamp(1, 3),
                    _ => ((dist_from_apex / 3) + 1).clamp(1, 3),
                };

                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dz * dz <= radius * radius {
                            stamp(wx + dx, y, wz + dz, leaves, false);
                        }
                    }
                }
            }
        }

        TreeSpecies::Pine => {
            let height = 11 + (shape_seed % 5) as i32; // [11..=15]
            let log = blocks.spruce_log;
            let leaves = blocks.spruce_leaves;

            // Slender tall trunk, bare at the bottom
            for y in 1..=height {
                stamp(wx, ry + y, wz, log, true);
            }

            // Umbrella dome canopy concentrated at crown
            // Apex cap
            stamp(wx, ry + height + 1, wz, leaves, false);

            // Level height: radius 2 dome
            for dz in -2..=2 {
                for dx in -2..=2 {
                    if dx * dx + dz * dz <= 4 {
                        stamp(wx + dx, ry + height, wz + dz, leaves, false);
                    }
                }
            }

            // Level height - 1: wide umbrella flange (radius 3)
            for dz in -3..=3 {
                for dx in -3..=3 {
                    if dx * dx + dz * dz <= 8 {
                        stamp(wx + dx, ry + height - 1, wz + dz, leaves, false);
                    }
                }
            }

            // Level height - 2: under-fringe (radius 2)
            for dz in -2i32..=2i32 {
                for dx in -2i32..=2i32 {
                    if dx * dx + dz * dz <= 4 {
                        stamp(wx + dx, ry + height - 2, wz + dz, leaves, false);
                    }
                }
            }
        }

        TreeSpecies::MegaTaiga => {
            let height = 14 + (shape_seed % 6) as i32; // [14..=19]
            let log = blocks.spruce_log;
            let leaves = blocks.spruce_leaves;

            // 2×2 thick log trunk
            for oz in 0..=1 {
                for ox in 0..=1 {
                    for y in 1..=height {
                        stamp(wx + ox, ry + y, wz + oz, log, true);
                    }
                }
            }

            // Apex cap over 2x2 trunk
            for oz in 0..=1 {
                for ox in 0..=1 {
                    stamp(wx + ox, ry + height + 1, wz + oz, leaves, false);
                }
            }

            // Tiered conical layers radiating outward from 2×2 center
            for y in (ry + 3)..=(ry + height) {
                let dist_from_apex = (ry + height + 1) - y;
                let radius = (1 + dist_from_apex / 3).clamp(1, 4);

                for dz in (-radius)..=(1 + radius) {
                    for dx in (-radius)..=(1 + radius) {
                        let center_dx = if dx <= 0 { dx } else { dx - 1 };
                        let center_dz = if dz <= 0 { dz } else { dz - 1 };

                        if center_dx * center_dx + center_dz * center_dz <= radius * radius {
                            stamp(wx + dx, y, wz + dz, leaves, false);
                        }
                    }
                }
            }
        }
    }
}
