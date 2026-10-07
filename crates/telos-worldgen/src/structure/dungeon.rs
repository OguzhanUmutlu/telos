//! Procedural subterranean monster dungeons with mossy cobblestone, spawners & chests.

use super::bounding_box::StructureBoundingBox;
use crate::aquifer::AquiferSampler;
use crate::density::sample_density;
use crate::math::hash3;
use crate::surface::ResolvedBlocks;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Grid cell size for the subterranean dungeon candidate lattice (in voxels).
pub const DUNGEON_CELL_SIZE: i32 = 48;
/// Maximum horizontal half-reach of a dungeon from its anchor coordinate.
pub const DUNGEON_MAX_REACH: i32 = 6;
/// Salting key for dungeon candidate cell generation.
pub const SALT_DUNGEON: u64 = 0xD046_E011_CAFE_0049;

/// Evaluates and places subterranean dungeons intersecting the chunk at `pos`.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn apply_dungeons(
    seed: u64,
    pos: ChunkPos,
    blocks: &ResolvedBlocks,
    aquifer: &AquiferSampler,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let chunk_x = pos.x();
    let chunk_y = pos.y();
    let chunk_z = pos.z();

    // World voxel bounds of the chunk
    let c_min_x = chunk_x * 32;
    let c_max_x = c_min_x + 31;
    let c_min_y = chunk_y * 32;
    let c_max_y = c_min_y + 31;
    let c_min_z = chunk_z * 32;
    let c_max_z = c_min_z + 31;

    // Dungeons only generate in subterranean depths [-500, 50]
    if c_max_y < -500 || c_min_y > 50 {
        return;
    }

    // Determine overlapping 3D candidate cell coordinates
    let min_cell_x = (c_min_x - DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);
    let max_cell_x = (c_max_x + DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);
    let min_cell_y = (c_min_y - DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);
    let max_cell_y = (c_max_y + DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);
    let min_cell_z = (c_min_z - DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);
    let max_cell_z = (c_max_z + DUNGEON_MAX_REACH).div_euclid(DUNGEON_CELL_SIZE);

    for cell_y in min_cell_y..=max_cell_y {
        let base_y = cell_y * DUNGEON_CELL_SIZE;
        if !(-500..=50).contains(&base_y) {
            continue;
        }

        for cell_z in min_cell_z..=max_cell_z {
            for cell_x in min_cell_x..=max_cell_x {
                // 1. Roll spawn chance: ~1 in 28 3D cells
                let cell_hash = hash3(seed ^ SALT_DUNGEON, cell_x, cell_y, cell_z);
                if !cell_hash.is_multiple_of(28) {
                    continue;
                }

                // 2. Compute anchor coordinate inside cell
                let offset_x = (cell_hash % (DUNGEON_CELL_SIZE as u64 - 12)) as i32 + 6;
                let offset_y = ((cell_hash >> 16) % (DUNGEON_CELL_SIZE as u64 - 8)) as i32 + 2;
                let offset_z = ((cell_hash >> 32) % (DUNGEON_CELL_SIZE as u64 - 12)) as i32 + 6;

                let anchor_x = cell_x * DUNGEON_CELL_SIZE + offset_x;
                let anchor_y = cell_y * DUNGEON_CELL_SIZE + offset_y;
                let anchor_z = cell_z * DUNGEON_CELL_SIZE + offset_z;

                // 3. Room dimensions: odd widths (5, 7, or 9), height 4 or 5
                let dim_hash = hash3(cell_hash, anchor_x, anchor_y, anchor_z);
                let width = match dim_hash % 3 {
                    0 => 5,
                    1 => 7,
                    _ => 9,
                };
                let depth = match (dim_hash >> 8) % 3 {
                    0 => 5,
                    1 => 7,
                    _ => 9,
                };
                let height = if (dim_hash >> 16) & 1 == 0 { 4 } else { 5 };

                let min_x = anchor_x - width / 2;
                let max_x = min_x + width - 1;
                let min_y = anchor_y;
                let max_y = min_y + height - 1;
                let min_z = anchor_z - depth / 2;
                let max_z = min_z + depth - 1;

                let dungeon_bbox =
                    StructureBoundingBox::new(min_x, min_y, min_z, max_x, max_y, max_z);
                if !dungeon_bbox.intersects_chunk(pos) {
                    continue;
                }

                // 4. Validate cave cavity:
                // - Room center interior must be in cave air
                let center_density = sample_density(
                    seed,
                    anchor_x as f32,
                    (anchor_y + 1) as f32,
                    anchor_z as f32,
                );
                if center_density > 0.0 {
                    continue;
                }

                // - Floor base must be solid rock
                let floor_density = sample_density(
                    seed,
                    anchor_x as f32,
                    (anchor_y - 1) as f32,
                    anchor_z as f32,
                );
                if floor_density <= 0.0 {
                    continue;
                }

                // - Must not be flooded with liquid aquifer
                let fluid_sample = aquifer.sample_world(anchor_x, anchor_y, anchor_z);
                if matches!(fluid_sample, crate::aquifer::AquiferSample::Fluid(_)) {
                    continue;
                }

                // - Count cave wall openings (must have >= 1 opening to adjacent cave air)
                let wall_north = sample_density(
                    seed,
                    anchor_x as f32,
                    (anchor_y + 1) as f32,
                    (min_z - 1) as f32,
                ) <= 0.0;
                let wall_south = sample_density(
                    seed,
                    anchor_x as f32,
                    (anchor_y + 1) as f32,
                    (max_z + 1) as f32,
                ) <= 0.0;
                let wall_east = sample_density(
                    seed,
                    (max_x + 1) as f32,
                    (anchor_y + 1) as f32,
                    anchor_z as f32,
                ) <= 0.0;
                let wall_west = sample_density(
                    seed,
                    (min_x - 1) as f32,
                    (anchor_y + 1) as f32,
                    anchor_z as f32,
                ) <= 0.0;
                let openings = u8::from(wall_north)
                    + u8::from(wall_south)
                    + u8::from(wall_east)
                    + u8::from(wall_west);
                if openings == 0 {
                    continue;
                }

                // 5. Slice and place dungeon blocks into current chunk
                let clamp = dungeon_bbox.clamp_to_chunk(pos).unwrap();

                // Deterministic chest placement along walls
                let chest_1_pos = (min_x + 1, min_y + 1, anchor_z);
                let chest_2_pos = (max_x - 1, min_y + 1, anchor_z);
                let has_chest_2 = (dim_hash >> 24) & 1 == 1;

                for wy in clamp.min_y..=clamp.max_y {
                    let ly = (wy - c_min_y) as usize;
                    let y_offset = ly << 10;

                    for wz in clamp.min_z..=clamp.max_z {
                        let lz = (wz - c_min_z) as usize;
                        let z_offset = y_offset | (lz << 5);

                        for wx in clamp.min_x..=clamp.max_x {
                            let lx = (wx - c_min_x) as usize;
                            let local_idx = z_offset | lx;

                            let is_floor = wy == min_y;
                            let is_ceiling = wy == max_y;
                            let is_wall_x = wx == min_x || wx == max_x;
                            let is_wall_z = wz == min_z || wz == max_z;
                            let is_perimeter = is_wall_x || is_wall_z;

                            if is_floor {
                                // Floor: 75% cobblestone, 25% mossy cobblestone
                                let floor_h = hash3(cell_hash ^ 0x1111, wx, wy, wz);
                                let block = if floor_h.is_multiple_of(4) {
                                    blocks.mossy_cobblestone
                                } else {
                                    blocks.cobblestone
                                };
                                dense[local_idx] = block;
                            } else if is_ceiling {
                                // Ceiling: solid cobblestone
                                dense[local_idx] = blocks.cobblestone;
                            } else if is_perimeter {
                                // Doorway opening if this wall midpoint opens to cave air
                                let is_doorway = (is_wall_x
                                    && wz == anchor_z
                                    && ((wx == min_x && wall_west) || (wx == max_x && wall_east)))
                                    || (is_wall_z
                                        && wx == anchor_x
                                        && ((wz == min_z && wall_north)
                                            || (wz == max_z && wall_south)));

                                if is_doorway && wy <= min_y + 2 {
                                    dense[local_idx] = BlockStateId::AIR;
                                } else {
                                    // Walls: 60% cobblestone, 40% mossy cobblestone
                                    let wall_h = hash3(cell_hash ^ 0x2222, wx, wy, wz);
                                    let block = if wall_h % 5 < 2 {
                                        blocks.mossy_cobblestone
                                    } else {
                                        blocks.cobblestone
                                    };
                                    dense[local_idx] = block;
                                }
                            } else {
                                // Interior:
                                if wx == anchor_x && wy == min_y + 1 && wz == anchor_z {
                                    // Monster spawner in exact center
                                    dense[local_idx] = blocks.monster_spawner;
                                } else if (wx, wy, wz) == chest_1_pos
                                    || (has_chest_2 && (wx, wy, wz) == chest_2_pos)
                                {
                                    // Loot chest
                                    dense[local_idx] = blocks.chest;
                                } else {
                                    // Hollow interior cave air
                                    dense[local_idx] = BlockStateId::AIR;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
