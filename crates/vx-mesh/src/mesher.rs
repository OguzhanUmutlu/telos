//! High-throughput 64-bit binary greedy meshing entry points with smooth lighting.

use vx_voxel::{
    chunk::ChunkSnapshot,
    coords::LocalIdx,
    occupancy::Occupancy,
    registry::BlockRegistry,
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
