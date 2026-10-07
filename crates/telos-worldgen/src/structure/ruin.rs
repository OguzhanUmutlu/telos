//! Modular procedural surface ruins (crumbling outposts, shrines & sunken cellars).

use super::bounding_box::StructureBoundingBox;
use crate::math::hash3;
use crate::surface::ResolvedBlocks;
use crate::tree::find_surface_ground_y;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Grid cell size for the surface ruin candidate lattice (in voxels).
pub const RUIN_CELL_SIZE: i32 = 64;
/// Maximum horizontal reach of a surface ruin from its anchor coordinate.
pub const RUIN_MAX_REACH: i32 = 8;
/// Salting key for surface ruin candidate generation.
pub const SALT_RUIN: u64 = 0x51A8_F011_BEEF_0049;

/// Distinct surface ruin architectural styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuinVariant {
    /// Ancient crumbling outpost with fortified perimeter walls and corner chest.
    AncientOutpost,
    /// Ceremonial stone shrine with upright pillars and an altar chest.
    StoneShrine,
    /// Sunken stone cellar/vault with subterranean chest alcove.
    SunkenCellar,
}

/// Evaluates and places procedural surface ruins intersecting the chunk at `pos`.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
pub fn apply_ruins(
    seed: u64,
    pos: ChunkPos,
    blocks: &ResolvedBlocks,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let chunk_x = pos.x();
    let chunk_y = pos.y();
    let chunk_z = pos.z();

    let c_min_x = chunk_x * 32;
    let c_max_x = c_min_x + 31;
    let c_min_y = chunk_y * 32;
    let c_max_y = c_min_y + 31;
    let c_min_z = chunk_z * 32;
    let c_max_z = c_min_z + 31;

    // Surface ruins only generate above sea level ($y \in [-5, 200]$)
    if c_max_y < 0 || c_min_y > 200 {
        return;
    }

    let min_cell_x = (c_min_x - RUIN_MAX_REACH).div_euclid(RUIN_CELL_SIZE);
    let max_cell_x = (c_max_x + RUIN_MAX_REACH).div_euclid(RUIN_CELL_SIZE);
    let min_cell_z = (c_min_z - RUIN_MAX_REACH).div_euclid(RUIN_CELL_SIZE);
    let max_cell_z = (c_max_z + RUIN_MAX_REACH).div_euclid(RUIN_CELL_SIZE);

    for cell_z in min_cell_z..=max_cell_z {
        for cell_x in min_cell_x..=max_cell_x {
            // 1. Roll spawn chance: ~1 in 10 surface cells (~10%)
            let cell_hash = hash3(seed ^ SALT_RUIN, cell_x, 0, cell_z);
            if !cell_hash.is_multiple_of(10) {
                continue;
            }

            // 2. Anchor coordinate within candidate cell
            let offset_x = (cell_hash % (RUIN_CELL_SIZE as u64 - 16)) as i32 + 8;
            let offset_z = ((cell_hash >> 16) % (RUIN_CELL_SIZE as u64 - 16)) as i32 + 8;
            let anchor_x = cell_x * RUIN_CELL_SIZE + offset_x;
            let anchor_z = cell_z * RUIN_CELL_SIZE + offset_z;

            // 3. Find ground elevation at anchor
            let Some(ground_y) = find_surface_ground_y(seed, anchor_x, anchor_z) else {
                continue;
            };
            if !(4..=180).contains(&ground_y) {
                continue;
            }

            // 4. Slope check across ruin footprint (reject if slope variance > 3)
            let check_y =
                |dx: i32, dz: i32| find_surface_ground_y(seed, anchor_x + dx, anchor_z + dz);
            let Some(y_nw) = check_y(-4, -4) else {
                continue;
            };
            let Some(y_ne) = check_y(4, -4) else { continue };
            let Some(y_sw) = check_y(-4, 4) else { continue };
            let Some(y_se) = check_y(4, 4) else { continue };

            let min_ground = y_nw.min(y_ne).min(y_sw).min(y_se);
            let max_ground = y_nw.max(y_ne).max(y_sw).max(y_se);
            if max_ground - min_ground > 3 {
                continue;
            }

            // 5. Variant selection
            let variant = match (cell_hash >> 32) % 3 {
                0 => RuinVariant::AncientOutpost,
                1 => RuinVariant::StoneShrine,
                _ => RuinVariant::SunkenCellar,
            };

            let ruin_bbox = match variant {
                RuinVariant::AncientOutpost => StructureBoundingBox::new(
                    anchor_x - 3,
                    ground_y - 1,
                    anchor_z - 3,
                    anchor_x + 3,
                    ground_y + 4,
                    anchor_z + 3,
                ),
                RuinVariant::StoneShrine => StructureBoundingBox::new(
                    anchor_x - 4,
                    ground_y,
                    anchor_z - 4,
                    anchor_x + 4,
                    ground_y + 5,
                    anchor_z + 4,
                ),
                RuinVariant::SunkenCellar => StructureBoundingBox::new(
                    anchor_x - 2,
                    ground_y - 3,
                    anchor_z - 2,
                    anchor_x + 2,
                    ground_y + 2,
                    anchor_z + 2,
                ),
            };

            if !ruin_bbox.intersects_chunk(pos) {
                continue;
            }

            let clamp = ruin_bbox.clamp_to_chunk(pos).unwrap();

            // 6. Slice and place ruin blocks into dense array
            for wy in clamp.min_y..=clamp.max_y {
                let ly = (wy - c_min_y) as usize;
                let y_offset = ly << 10;

                for wz in clamp.min_z..=clamp.max_z {
                    let lz = (wz - c_min_z) as usize;
                    let z_offset = y_offset | (lz << 5);

                    for wx in clamp.min_x..=clamp.max_x {
                        let lx = (wx - c_min_x) as usize;
                        let local_idx = z_offset | lx;

                        let block = evaluate_ruin_voxel(
                            cell_hash,
                            variant,
                            (wx, wy, wz),
                            (anchor_x, ground_y, anchor_z),
                            blocks,
                        );

                        if let Some(b) = block {
                            dense[local_idx] = b;
                        }
                    }
                }
            }
        }
    }
}

/// Evaluates a single voxel inside a surface ruin. Returns `Some(BlockStateId)` to place a block,
/// or `None` to leave existing terrain untouched.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn evaluate_ruin_voxel(
    cell_hash: u64,
    variant: RuinVariant,
    (wx, wy, wz): (i32, i32, i32),
    (ax, gy, az): (i32, i32, i32),
    blocks: &ResolvedBlocks,
) -> Option<BlockStateId> {
    let dx = wx - ax;
    let dy = wy - gy;
    let dz = wz - az;

    let v_hash = hash3(cell_hash, wx, wy, wz);

    match variant {
        RuinVariant::AncientOutpost => {
            // Foundation below ground level
            if dy == -1 {
                return Some(if v_hash.is_multiple_of(3) {
                    blocks.mossy_cobblestone
                } else {
                    blocks.cobblestone
                });
            }

            // Floor
            if dy == 0 {
                return if dx.abs() <= 2 && dz.abs() <= 2 {
                    Some(if v_hash.is_multiple_of(4) {
                        blocks.mossy_cobblestone
                    } else {
                        blocks.cobblestone
                    })
                } else {
                    None
                };
            }

            // Perimeter walls
            if dx.abs() == 3 || dz.abs() == 3 {
                // Break openings in walls randomly
                if (dx == 0 && dz == 3) || v_hash.is_multiple_of(5) {
                    return Some(BlockStateId::AIR);
                }

                // Variable wall height (crumbling ruin battlements)
                let wall_h = 2 + (v_hash % 3) as i32;
                if dy <= wall_h {
                    return Some(if v_hash.is_multiple_of(2) {
                        blocks.mossy_cobblestone
                    } else {
                        blocks.cobblestone
                    });
                }
            }

            // Chest in corner
            if dx == 2 && dz == 2 && dy == 1 {
                return Some(blocks.chest);
            }

            None
        }
        RuinVariant::StoneShrine => {
            // Stepped central altar floor
            if dy == 0 && dx.abs() <= 3 && dz.abs() <= 3 {
                return Some(if v_hash.is_multiple_of(3) {
                    blocks.mossy_cobblestone
                } else {
                    blocks.stone
                });
            }

            // 4 corner stone pillars
            if dx.abs() == 3 && dz.abs() == 3 && (1..=4).contains(&dy) {
                return Some(if dy == 4 && v_hash.is_multiple_of(3) {
                    blocks.mossy_cobblestone
                } else {
                    blocks.stone
                });
            }

            // Central altar pedestal and chest
            if dx == 0 && dz == 0 {
                if dy == 1 {
                    return Some(blocks.mossy_cobblestone);
                } else if dy == 2 {
                    return Some(blocks.chest);
                }
            }

            None
        }
        RuinVariant::SunkenCellar => {
            // Sunken chamber carved into ground: dy in [-3, -1]
            if (-3..=-1).contains(&dy) {
                let is_wall = dx.abs() == 2 || dz.abs() == 2;
                let is_floor = dy == -3;

                if is_floor || is_wall {
                    return Some(if v_hash.is_multiple_of(3) {
                        blocks.mossy_cobblestone
                    } else {
                        blocks.cobblestone
                    });
                }

                // Interior of sunken cellar
                if dx == 1 && dz == 1 && dy == -2 {
                    return Some(blocks.chest);
                }

                return Some(BlockStateId::AIR);
            }

            // Small entrance arch at ground level
            if (0..=2).contains(&dy) {
                if dx.abs() == 2 && dz == 0 {
                    return Some(blocks.cobblestone);
                }
                if dy == 2 && dx.abs() <= 1 && dz == 0 {
                    return Some(blocks.stone);
                }
            }

            None
        }
    }
}
