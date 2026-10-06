//! Bitwise face extraction and neighbor boundary management.

use telos_voxel::{chunk::ChunkSnapshot, occupancy::Occupancy, storage::Blocks};

use crate::quad::FaceDir;

/// Precomputed boundary occupancy slices from the 6 cardinal neighbors.
/// Each slice is a `[u32; 32]` bitboard matching the orientation of the adjacent face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborSlices {
    /// Neighbor at +X: its x=0 layer matching (row=z, bit=y)
    pub pos_x: [u32; 32],
    /// Neighbor at -X: its x=31 layer matching (row=y, bit=z)
    pub neg_x: [u32; 32],
    /// Neighbor at +Y: its y=0 layer matching (row=x, bit=z)
    pub pos_y: [u32; 32],
    /// Neighbor at -Y: its y=31 layer matching (row=z, bit=x)
    pub neg_y: [u32; 32],
    /// Neighbor at +Z: its z=0 layer matching (row=y, bit=x)
    pub pos_z: [u32; 32],
    /// Neighbor at -Z: its z=31 layer matching (row=x, bit=y)
    pub neg_z: [u32; 32],
}

impl Default for NeighborSlices {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl NeighborSlices {
    /// All neighbors are empty (air). Boundary faces will not be culled.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            pos_x: [0; 32],
            neg_x: [0; 32],
            pos_y: [0; 32],
            neg_y: [0; 32],
            pos_z: [0; 32],
            neg_z: [0; 32],
        }
    }

    /// All neighbors are solid. Boundary faces will be completely culled.
    #[inline]
    #[must_use]
    pub const fn solid() -> Self {
        Self {
            pos_x: [u32::MAX; 32],
            neg_x: [u32::MAX; 32],
            pos_y: [u32::MAX; 32],
            neg_y: [u32::MAX; 32],
            pos_z: [u32::MAX; 32],
            neg_z: [u32::MAX; 32],
        }
    }

    /// Extracts boundary slices from an array of 6 optional neighbor snapshots.
    /// Order matches `FaceDir::ALL`: [+X, -X, +Y, -Y, +Z, -Z].
    #[must_use]
    pub fn from_snapshots(neighbors: &[Option<&ChunkSnapshot>; 6]) -> Self {
        Self {
            pos_x: extract_slice_from_snapshot(neighbors[0], FaceDir::PosX),
            neg_x: extract_slice_from_snapshot(neighbors[1], FaceDir::NegX),
            pos_y: extract_slice_from_snapshot(neighbors[2], FaceDir::PosY),
            neg_y: extract_slice_from_snapshot(neighbors[3], FaceDir::NegY),
            pos_z: extract_slice_from_snapshot(neighbors[4], FaceDir::PosZ),
            neg_z: extract_slice_from_snapshot(neighbors[5], FaceDir::NegZ),
        }
    }

    /// Extracts boundary slices from an array of 6 optional block containers and `is_opaque` predicate.
    #[must_use]
    pub fn from_blocks(
        neighbors: &[Option<&Blocks>; 6],
        is_opaque: impl Fn(telos_voxel::state::BlockStateId) -> bool,
    ) -> Self {
        let mut slices = Self::empty();
        for (i, &neighbor) in neighbors.iter().enumerate() {
            if let Some(blocks) = neighbor {
                let occ = Occupancy::from_blocks(blocks, &is_opaque);
                let dir = FaceDir::ALL[i];
                let slice = extract_slice_from_occ(&occ, dir);
                match dir {
                    FaceDir::PosX => slices.pos_x = slice,
                    FaceDir::NegX => slices.neg_x = slice,
                    FaceDir::PosY => slices.pos_y = slice,
                    FaceDir::NegY => slices.neg_y = slice,
                    FaceDir::PosZ => slices.pos_z = slice,
                    FaceDir::NegZ => slices.neg_z = slice,
                }
            }
        }
        slices
    }
}

fn extract_slice_from_snapshot(snap: Option<&ChunkSnapshot>, dir: FaceDir) -> [u32; 32] {
    match snap {
        Some(s) => extract_slice_from_occ(s.occupancy(), dir),
        None => [0; 32],
    }
}

#[allow(clippy::needless_range_loop)]
fn extract_slice_from_occ(occ: &Occupancy, dir: FaceDir) -> [u32; 32] {
    let mut slice = [0u32; 32];
    match dir {
        // Neighbor is +X: we want its x=0 boundary, where row=z, bit=y
        FaceDir::PosX => {
            for z in 0..32 {
                slice[z] = occ.col_y[z << 5];
            }
        }
        // Neighbor is -X: we want its x=31 boundary, where row=y, bit=z
        FaceDir::NegX => {
            for y in 0..32 {
                slice[y] = occ.col_z[(y << 5) | 31];
            }
        }
        // Neighbor is +Y: we want its y=0 boundary, where row=x, bit=z
        FaceDir::PosY => {
            for x in 0..32 {
                slice[x] = occ.col_z[x];
            }
        }
        // Neighbor is -Y: we want its y=31 boundary, where row=z, bit=x
        FaceDir::NegY => {
            for z in 0..32 {
                slice[z] = occ.col_x[(31 << 5) | z];
            }
        }
        // Neighbor is +Z: we want its z=0 boundary, where row=y, bit=x
        FaceDir::PosZ => {
            for y in 0..32 {
                slice[y] = occ.col_x[y << 5];
            }
        }
        // Neighbor is -Z: we want its z=31 boundary, where row=x, bit=y
        FaceDir::NegZ => {
            for x in 0..32 {
                slice[x] = occ.col_y[(31 << 5) | x];
            }
        }
    }
    slice
}

/// Computes the 32x32 visible face bitboard for slice `d ∈ 0..31` along direction `dir`.
///
/// Uses branchless bitwise AND-NOT operations:
/// - Positive face (+): `current & !neighbor_above`
/// - Negative face (-): `current & !neighbor_below`
#[inline]
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn extract_face_slice(
    dir: FaceDir,
    d: usize,
    occ: &Occupancy,
    neighbors: &NeighborSlices,
) -> [u32; 32] {
    let mut face_slice = [0u32; 32];

    match dir {
        FaceDir::PosX => {
            // Axis X: d is x. Row is z, bit is y.
            // +X face: solid at x, empty at x + 1
            for z in 0..32 {
                let curr = occ.col_y[(z << 5) | d];
                let next = if d == 31 {
                    neighbors.pos_x[z]
                } else {
                    occ.col_y[(z << 5) | (d + 1)]
                };
                face_slice[z] = curr & !next;
            }
        }
        FaceDir::NegX => {
            // Axis X: d is x. Row is y, bit is z.
            // -X face: solid at x, empty at x - 1
            for y in 0..32 {
                let curr = occ.col_z[(y << 5) | d];
                let prev = if d == 0 {
                    neighbors.neg_x[y]
                } else {
                    occ.col_z[(y << 5) | (d - 1)]
                };
                face_slice[y] = curr & !prev;
            }
        }
        FaceDir::PosY => {
            // Axis Y: d is y. Row is x, bit is z.
            // +Y face: solid at y, empty at y + 1
            for x in 0..32 {
                let curr = occ.col_z[(d << 5) | x];
                let next = if d == 31 {
                    neighbors.pos_y[x]
                } else {
                    occ.col_z[((d + 1) << 5) | x]
                };
                face_slice[x] = curr & !next;
            }
        }
        FaceDir::NegY => {
            // Axis Y: d is y. Row is z, bit is x.
            // -Y face: solid at y, empty at y - 1
            for z in 0..32 {
                let curr = occ.col_x[(d << 5) | z];
                let prev = if d == 0 {
                    neighbors.neg_y[z]
                } else {
                    occ.col_x[((d - 1) << 5) | z]
                };
                face_slice[z] = curr & !prev;
            }
        }
        FaceDir::PosZ => {
            // Axis Z: d is z. Row is y, bit is x.
            // +Z face: solid at z, empty at z + 1
            for y in 0..32 {
                let curr = occ.col_x[(y << 5) | d];
                let next = if d == 31 {
                    neighbors.pos_z[y]
                } else {
                    occ.col_x[(y << 5) | (d + 1)]
                };
                face_slice[y] = curr & !next;
            }
        }
        FaceDir::NegZ => {
            // Axis Z: d is z. Row is x, bit is y.
            // -Z face: solid at z, empty at z - 1
            for x in 0..32 {
                let curr = occ.col_y[(d << 5) | x];
                let prev = if d == 0 {
                    neighbors.neg_z[x]
                } else {
                    occ.col_y[((d - 1) << 5) | x]
                };
                face_slice[x] = curr & !prev;
            }
        }
    }

    face_slice
}
