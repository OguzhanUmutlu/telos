//! Baseline naive 6-face mesher used as a reference oracle for testing and benchmarks.

use vx_voxel::{coords::LocalIdx, occupancy::Occupancy, storage::Blocks};

use crate::{
    bitwise::NeighborSlices,
    mesh::{QuadRange, T0Mesh},
    quad::{FaceDir, T0Quad},
};

/// Reference 6-face unmerged mesher.
///
/// Iterates all 32,768 voxels, checks each of the 6 cardinal directions individually,
/// and emits unmerged 1x1 `T0Quad` primitives.
pub struct NaiveMesher;

impl NaiveMesher {
    /// Meshes a chunk into unmerged 1x1 quads grouped by direction.
    #[must_use]
    pub fn mesh(blocks: &Blocks, occ: &Occupancy, neighbors: &NeighborSlices) -> T0Mesh {
        let mut buckets = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        for y in 0..32i32 {
            for z in 0..32i32 {
                for x in 0..32i32 {
                    if !occ.is_solid(x as u32, y as u32, z as u32) {
                        continue;
                    }

                    let idx = LocalIdx::from_coords_unchecked(x as u32, y as u32, z as u32);
                    let material = blocks.get(idx).0 as u16;

                    // Check all 6 directions
                    for dir in FaceDir::ALL {
                        let normal = dir.normal();
                        let nx = x + normal.x;
                        let ny = y + normal.y;
                        let nz = z + normal.z;

                        let is_neighbor_solid = if (0..32).contains(&nx)
                            && (0..32).contains(&ny)
                            && (0..32).contains(&nz)
                        {
                            occ.is_solid(nx as u32, ny as u32, nz as u32)
                        } else {
                            // Check boundary slice
                            match dir {
                                FaceDir::PosX => (neighbors.pos_x[z as usize] >> y) & 1 == 1,
                                FaceDir::NegX => (neighbors.neg_x[y as usize] >> z) & 1 == 1,
                                FaceDir::PosY => (neighbors.pos_y[x as usize] >> z) & 1 == 1,
                                FaceDir::NegY => (neighbors.neg_y[z as usize] >> x) & 1 == 1,
                                FaceDir::PosZ => (neighbors.pos_z[y as usize] >> x) & 1 == 1,
                                FaceDir::NegZ => (neighbors.neg_z[x as usize] >> y) & 1 == 1,
                            }
                        };

                        if !is_neighbor_solid {
                            // Emits 1x1 unmerged quad
                            let quad =
                                T0Quad::new(x as u32, y as u32, z as u32, 1, 1, dir, material, 0);
                            buckets[dir as usize].push(quad);
                        }
                    }
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
}
