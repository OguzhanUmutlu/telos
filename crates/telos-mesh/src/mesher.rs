//! High-throughput 64-bit binary greedy meshing entry points with smooth lighting.

use telos_voxel::{
    chunk::ChunkSnapshot,
    coords::LocalIdx,
    occupancy::Occupancy,
    registry::BlockRegistry,
    shape::BlockShape,
    state::{BlockStateId, StateFlags},
    storage::Blocks,
};

use crate::{
    bitwise::{NeighborSlices, extract_face_slice},
    greedy::greedy_merge_slice_with_light,
    light::{
        LightPatternTable, OccupancyLightSampler, VoxelLightSampler, VoxelNeighborhood,
        compute_face_pattern,
    },
    mesh::{QuadRange, T0Mesh},
    quad::{FaceDir, T0Quad},
};

/// Meshes a `ChunkSnapshot` using 64-bit binary greedy meshing with 6 optional neighbor snapshots.
#[must_use]
pub fn mesh_chunk_t0(chunk: &ChunkSnapshot, neighbors: &[Option<&ChunkSnapshot>; 6]) -> T0Mesh {
    // Fast path: completely uniform air chunk produces 0 quads
    if chunk.blocks().is_uniform()
        && chunk.blocks().get(LocalIdx::from_coords_unchecked(0, 0, 0)) == BlockStateId::AIR
    {
        return T0Mesh::empty();
    }

    let neighbor_slices = NeighborSlices::from_snapshots(neighbors);
    let neighborhood = VoxelNeighborhood::new(chunk, neighbors);
    mesh_blocks_with_sampler(
        chunk.blocks(),
        chunk.occupancy(),
        &neighbor_slices,
        &neighborhood,
    )
}

/// Meshes an arbitrary `Blocks` container given its occupancy and neighbor boundary bitboards.
#[must_use]
pub fn mesh_blocks_with_occupancy(
    blocks: &Blocks,
    occ: &Occupancy,
    neighbors: &NeighborSlices,
) -> T0Mesh {
    // Fast path: uniform air
    if blocks.is_uniform()
        && blocks.get(LocalIdx::from_coords_unchecked(0, 0, 0)) == BlockStateId::AIR
    {
        return T0Mesh::empty();
    }

    let sampler = OccupancyLightSampler::new(occ, neighbors);
    mesh_blocks_with_sampler(blocks, occ, neighbors, &sampler)
}

/// Meshes blocks and computes occupancy on the fly using the registry flags.
#[must_use]
pub fn mesh_blocks_t0(
    blocks: &Blocks,
    neighbors: &[Option<&Blocks>; 6],
    reg: &BlockRegistry,
) -> T0Mesh {
    let is_opaque = |state: BlockStateId| reg.flags(state).contains(StateFlags::OPAQUE_FULL);
    let occ = Occupancy::from_blocks(blocks, is_opaque);
    let neighbor_slices = NeighborSlices::from_blocks(neighbors, is_opaque);

    mesh_blocks_with_occupancy(blocks, &occ, &neighbor_slices)
}

fn mesh_blocks_with_sampler<S: VoxelLightSampler>(
    blocks: &Blocks,
    occ: &Occupancy,
    neighbors: &NeighborSlices,
    sampler: &S,
) -> T0Mesh {
    let mut buckets: [Vec<T0Quad>; 6] = [
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
    ];
    let mut pattern_table = LightPatternTable::new();

    let is_uniform = blocks.is_uniform();
    let uniform_mat = if is_uniform {
        blocks.get(LocalIdx::from_coords_unchecked(0, 0, 0)).0 as u16
    } else {
        0
    };

    for dir in FaceDir::ALL {
        let bucket = &mut buckets[dir as usize];

        for d in 0..32 {
            let slice = extract_face_slice(dir, d, occ, neighbors);
            if slice == [0; 32] {
                continue;
            }

            // Per-slice 4 KiB stack cache to evaluate each exposed cell's lighting at most once
            let mut cache = [0u32; 1024];

            let mut get_cell = |u: u32, v: u32| -> (u16, u16, bool, bool) {
                let cell_idx = ((v << 5) | u) as usize;
                let cached = cache[cell_idx];
                if (cached & 0x8000_0000) != 0 {
                    let mat = cached as u16;
                    let pat_idx = ((cached >> 16) & 0x3FFF) as u16;
                    let can_u = ((cached >> 30) & 1) != 0;
                    let can_v = ((cached >> 29) & 1) != 0;
                    return (mat, pat_idx, can_u, can_v);
                }

                let (x, y, z) = map_coords(dir, d as u32, u, v);
                let mat = if is_uniform {
                    uniform_mat
                } else {
                    let idx = LocalIdx::from_coords_unchecked(x, y, z);
                    blocks.get(idx).0 as u16
                };

                let pattern = compute_face_pattern(sampler, x, y, z, dir);
                let pat_idx = pattern_table.insert(pattern);
                let can_u = pattern.can_merge_u();
                let can_v = pattern.can_merge_v();

                let packed = u32::from(mat)
                    | ((u32::from(pat_idx) & 0x3FFF) << 16)
                    | (u32::from(can_v) << 29)
                    | (u32::from(can_u) << 30)
                    | 0x8000_0000;
                cache[cell_idx] = packed;

                (mat, pat_idx, can_u, can_v)
            };

            greedy_merge_slice_with_light(slice, &mut get_cell, |u, v, w, h, mat, pat_idx| {
                let (x, y, z) = map_coords(dir, d as u32, u, v);
                bucket.push(T0Quad::new(x, y, z, w, h, dir, mat, pat_idx));
            });
        }
    }

    assemble_t0_mesh(buckets, pattern_table)
}

fn assemble_t0_mesh(buckets: [Vec<T0Quad>; 6], pattern_table: LightPatternTable) -> T0Mesh {
    let mut all_quads = Vec::new();
    let mut ranges = [QuadRange::default(); 6];

    for (i, bucket) in buckets.into_iter().enumerate() {
        let offset = all_quads.len() as u32;
        let count = bucket.len() as u32;
        ranges[i] = QuadRange::new(offset, count);
        all_quads.extend(bucket);
    }

    T0Mesh {
        quads: all_quads,
        patterns: pattern_table.into_patterns(),
        ranges,
    }
}

/// Meshes all layers (opaque T0, cutout T0, translucent T0, and T1 sub-cubes) of a chunk snapshot.
#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
#[must_use]
pub fn mesh_chunk_multilayers(
    chunk: &ChunkSnapshot,
    neighbors: &[Option<&ChunkSnapshot>; 6],
    reg: &BlockRegistry,
) -> crate::mesh::ChunkMeshLayers {
    if chunk.blocks().is_uniform()
        && chunk.blocks().get(LocalIdx::from_coords_unchecked(0, 0, 0)) == BlockStateId::AIR
    {
        return crate::mesh::ChunkMeshLayers::empty();
    }

    let neighbor_slices = NeighborSlices::from_snapshots(neighbors);
    let neighborhood = VoxelNeighborhood::new(chunk, neighbors);

    // 1. Opaque T0 layer (using 64-bit binary greedy mesher)
    let opaque = mesh_blocks_with_sampler(
        chunk.blocks(),
        chunk.occupancy(),
        &neighbor_slices,
        &neighborhood,
    );

    // Helper to get block state at (bx, by, bz)
    let get_state_at = |bx: i32, by: i32, bz: i32| -> BlockStateId {
        if (0..32).contains(&bx) && (0..32).contains(&by) && (0..32).contains(&bz) {
            let idx = LocalIdx::from_coords_unchecked(bx as u32, by as u32, bz as u32);
            chunk.blocks().get(idx)
        } else {
            let (n_idx, nx, ny, nz) = if bx < 0 {
                (1, bx + 32, by, bz)
            } else if bx >= 32 {
                (0, bx - 32, by, bz)
            } else if by < 0 {
                (3, bx, by + 32, bz)
            } else if by >= 32 {
                (2, bx, by - 32, bz)
            } else if bz < 0 {
                (5, bx, by, bz + 32)
            } else {
                (4, bx, by, bz - 32)
            };

            if (0..32).contains(&nx) && (0..32).contains(&ny) && (0..32).contains(&nz) {
                if let Some(snap) = neighbors[n_idx] {
                    let idx = LocalIdx::from_coords_unchecked(nx as u32, ny as u32, nz as u32);
                    snap.blocks().get(idx)
                } else {
                    BlockStateId::AIR
                }
            } else {
                BlockStateId::AIR
            }
        }
    };

    // 2. Cutout T0 layer
    let mut cutout_buckets: [Vec<T0Quad>; 6] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut cutout_patterns = LightPatternTable::new();

    // 3. Translucent T0 layer (water)
    let mut trans_buckets: [Vec<T0Quad>; 6] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut trans_patterns = LightPatternTable::new();

    // 4. T2 Cutout and Translucent layers
    let mut t2_cutout_quads = Vec::new();
    let mut t2_trans_quads = Vec::new();

    let get_shape_fn =
        |x: i32, y: i32, z: i32| -> BlockShape { reg.shape(get_state_at(x, y, z)).clone() };

    let blocks = chunk.blocks();
    for bz in 0..32u32 {
        for by in 0..32u32 {
            for bx in 0..32u32 {
                let idx = LocalIdx::from_coords_unchecked(bx, by, bz);
                let state = blocks.get(idx);
                let flags = reg.flags(state);
                let shape = reg.shape(state);

                if flags.contains(StateFlags::CUTOUT) {
                    let mat = state.0 as u16;
                    match shape {
                        BlockShape::Cross => {
                            let sky = neighborhood.get_sky_light(bx as i32, by as i32, bz as i32);
                            let block =
                                neighborhood.get_block_light(bx as i32, by as i32, bz as i32);
                            crate::t2::mesh_cross_model(
                                bx,
                                by,
                                bz,
                                mat,
                                sky,
                                block,
                                &mut t2_cutout_quads,
                            );
                        }
                        BlockShape::Torch { wall } => {
                            crate::t2::mesh_torch_model(
                                bx,
                                by,
                                bz,
                                *wall,
                                mat,
                                &mut t2_cutout_quads,
                            );
                        }
                        _ => {
                            // T0 full-cube cutout (leaves, glass)
                            for dir in FaceDir::ALL {
                                let (nx, ny, nz) = neighbor_coords(bx, by, bz, dir);
                                let n_state = get_state_at(nx, ny, nz);
                                let n_flags = reg.flags(n_state);

                                // Opaque full block culls cutout face
                                if n_flags.contains(StateFlags::OPAQUE_FULL) {
                                    continue;
                                }
                                // Same glass block culls internal face
                                if n_state == state
                                    && flags.contains(StateFlags::CUTOUT)
                                    && !flags.contains(StateFlags::LIGHT_BLOCKING)
                                {
                                    continue;
                                }

                                let pattern = compute_face_pattern(&neighborhood, bx, by, bz, dir);
                                let pat_idx = cutout_patterns.insert(pattern);
                                cutout_buckets[dir as usize]
                                    .push(T0Quad::new(bx, by, bz, 1, 1, dir, mat, pat_idx));
                            }
                        }
                    }
                } else if flags.contains(StateFlags::TRANSLUCENT) {
                    let mat = state.0 as u16;
                    match shape {
                        BlockShape::Fluid { .. } => {
                            let sky = neighborhood.get_sky_light(bx as i32, by as i32, bz as i32);
                            let block =
                                neighborhood.get_block_light(bx as i32, by as i32, bz as i32);
                            crate::t2::mesh_fluid_cell(
                                bx,
                                by,
                                bz,
                                mat,
                                sky,
                                block,
                                &get_shape_fn,
                                &mut t2_trans_quads,
                            );
                        }
                        _ => {
                            // T0 full-cube translucent
                            for dir in FaceDir::ALL {
                                let (nx, ny, nz) = neighbor_coords(bx, by, bz, dir);
                                let n_state = get_state_at(nx, ny, nz);
                                let n_flags = reg.flags(n_state);

                                // Translucent-translucent internal culling
                                if n_flags.contains(StateFlags::TRANSLUCENT) {
                                    continue;
                                }
                                // Translucent against solid opaque block
                                if n_flags.contains(StateFlags::OPAQUE_FULL) {
                                    continue;
                                }

                                let pattern = compute_face_pattern(&neighborhood, bx, by, bz, dir);
                                let pat_idx = trans_patterns.insert(pattern);
                                trans_buckets[dir as usize]
                                    .push(T0Quad::new(bx, by, bz, 1, 1, dir, mat, pat_idx));
                            }
                        }
                    }
                }
            }
        }
    }

    let cutout = assemble_t0_mesh(cutout_buckets, cutout_patterns);
    let translucent = assemble_t0_mesh(trans_buckets, trans_patterns);

    // 5. T1 Sub-Cube layer
    let t1_opaque = crate::t1::mesh_chunk_t1(chunk.blocks(), neighbors, &neighborhood, reg);

    crate::mesh::ChunkMeshLayers {
        opaque,
        cutout,
        translucent,
        t1_opaque,
        t2_cutout: crate::t2::T2Mesh {
            quads: t2_cutout_quads,
        },
        t2_translucent: crate::t2::T2Mesh {
            quads: t2_trans_quads,
        },
    }
}

#[allow(clippy::cast_possible_wrap)]
#[inline]
const fn neighbor_coords(x: u32, y: u32, z: u32, dir: FaceDir) -> (i32, i32, i32) {
    let (ix, iy, iz) = (x as i32, y as i32, z as i32);
    match dir {
        FaceDir::PosX => (ix + 1, iy, iz),
        FaceDir::NegX => (ix - 1, iy, iz),
        FaceDir::PosY => (ix, iy + 1, iz),
        FaceDir::NegY => (ix, iy - 1, iz),
        FaceDir::PosZ => (ix, iy, iz + 1),
        FaceDir::NegZ => (ix, iy, iz - 1),
    }
}

/// Maps tangent frame `(d, u, v)` back to chunk coordinates `(x, y, z)`.
#[inline]
const fn map_coords(dir: FaceDir, d: u32, u: u32, v: u32) -> (u32, u32, u32) {
    match dir {
        FaceDir::PosX => (d, u, v),
        FaceDir::NegX => (d, v, u),
        FaceDir::PosY => (v, d, u),
        FaceDir::NegY => (u, d, v),
        FaceDir::PosZ => (u, v, d),
        FaceDir::NegZ => (v, u, d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use telos_core::coords::ChunkPos;
    use telos_voxel::chunk::Chunk;

    #[test]
    fn test_multilayers_meshing() {
        let reg = BlockRegistry::standard();
        let mut chunk = Chunk::new_uniform(ChunkPos::new(0, 0, 0), BlockStateId::AIR, false);

        // Place stone (opaque T0)
        let stone = BlockStateId::new(1);
        chunk.set(
            LocalIdx::from_coords_unchecked(0, 0, 0),
            stone,
            StateFlags::AIR,
            reg.flags(stone),
            1,
        );
        // Place oak leaves (cutout T0)
        let leaves = BlockStateId::new(8);
        chunk.set(
            LocalIdx::from_coords_unchecked(10, 10, 10),
            leaves,
            StateFlags::AIR,
            reg.flags(leaves),
            1,
        );
        // Place two adjacent water blocks (translucent T0)
        let water = BlockStateId::new(6);
        chunk.set(
            LocalIdx::from_coords_unchecked(20, 20, 20),
            water,
            StateFlags::AIR,
            reg.flags(water),
            1,
        );
        chunk.set(
            LocalIdx::from_coords_unchecked(21, 20, 20),
            water,
            StateFlags::AIR,
            reg.flags(water),
            1,
        );
        // Place a stone slab (T1 sub-cube)
        let slab = BlockStateId::new(10);
        chunk.set(
            LocalIdx::from_coords_unchecked(5, 5, 5),
            slab,
            StateFlags::AIR,
            reg.flags(slab),
            1,
        );

        // Place a poppy (T2 cross cutout)
        let poppy = BlockStateId::new(12); // poppy in standard registry
        chunk.set(
            LocalIdx::from_coords_unchecked(15, 15, 15),
            poppy,
            StateFlags::AIR,
            reg.flags(poppy),
            1,
        );

        let snap = chunk.publish_snapshot();
        let neighbors = [None; 6];
        let layers = mesh_chunk_multilayers(&snap, &neighbors, &reg);

        // Verify opaque T0 has stone faces
        assert!(!layers.opaque.is_empty());
        // Verify cutout T0 has leaf faces
        assert!(!layers.cutout.is_empty());
        // Verify T2 cutout has poppy cross quads (4 quads)
        assert_eq!(layers.t2_cutout.quad_count(), 4);
        // Verify T2 translucent has water faces (2 blocks with shared internal face culled: 10 quads)
        assert_eq!(layers.t2_translucent.quad_count(), 10);
        // Verify T1 has slab faces
        assert!(!layers.t1_opaque.is_empty());
    }
}
