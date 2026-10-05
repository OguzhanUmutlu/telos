//! High-throughput 64-bit binary greedy meshing entry points.

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
    greedy::{greedy_merge_slice, greedy_merge_slice_uniform},
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
    mesh_blocks_internal(chunk.blocks(), chunk.occupancy(), &neighbor_slices)
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

    mesh_blocks_internal(blocks, occ, neighbors)
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

fn mesh_blocks_internal(blocks: &Blocks, occ: &Occupancy, neighbors: &NeighborSlices) -> T0Mesh {
    let mut buckets: [Vec<T0Quad>; 6] = [
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
        Vec::with_capacity(64),
    ];

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

            if is_uniform {
                greedy_merge_slice_uniform(slice, uniform_mat, |u, v, w, h, mat| {
                    let (x, y, z) = map_coords(dir, d as u32, u, v);
                    bucket.push(T0Quad::new(x, y, z, w, h, dir, mat, 0));
                });
            } else {
                let get_mat = |u: u32, v: u32| -> u16 {
                    let (x, y, z) = map_coords(dir, d as u32, u, v);
                    let idx = LocalIdx::from_coords_unchecked(x, y, z);
                    blocks.get(idx).0 as u16
                };

                greedy_merge_slice(slice, get_mat, |u, v, w, h, mat| {
                    let (x, y, z) = map_coords(dir, d as u32, u, v);
                    bucket.push(T0Quad::new(x, y, z, w, h, dir, mat, 0));
                });
            }
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
