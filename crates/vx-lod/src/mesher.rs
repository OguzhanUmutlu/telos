//! 64-bit binary greedy mesher for 32³ LOD nodes producing 8-byte color LOD quads.

use hashbrown::HashMap;
use vx_core::coords::Face;

use crate::color::LodColorTable;
use crate::node::{LodNode, LodVoxelSource};
use crate::quad::LodQuad;

/// Mesh representation of a 32³ LOD node ready for GPU upload.
#[derive(Clone, Debug, Default)]
pub struct LodMesh {
    /// List of packed 8-byte color LOD quads.
    pub quads: Vec<LodQuad>,
    /// Deduplicated RGBA8 color palette stored directly after quads in GPU buffer.
    pub palette: Vec<[u8; 4]>,
}

impl LodMesh {
    /// Returns `true` if the mesh contains no quads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Serializes quads and palette into a single contiguous `u32` buffer for GPU BDA upload.
    ///
    /// Memory layout:
    /// - `[0 .. quads.len() * 2]`: 8-byte quads (2 `u32` words per quad)
    /// - `[quads.len() * 2 ..]`: 4-byte packed RGBA8 palette entries (1 `u32` word per entry)
    pub fn write_to_u32_buffer(&self, out: &mut Vec<u32>) {
        out.reserve(self.quads.len() * 2 + self.palette.len());
        for quad in &self.quads {
            out.push(quad.data0);
            out.push(quad.data1);
        }
        for color in &self.palette {
            let packed = u32::from_le_bytes(*color);
            out.push(packed);
        }
    }
}

/// Meshes an LOD node against 6 optional same-level neighbor nodes.
///
/// Neighbor order matches `[+X, -X, +Y, -Y, +Z, -Z]`.
/// Missing neighbors are treated as empty air (boundary faces emitted).
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn mesh_lod_node<S: LodVoxelSource>(
    node: &LodNode,
    neighbors: &[Option<&S>; 6],
    color_table: &LodColorTable,
) -> LodMesh {
    if node.occupancy.is_empty() {
        return LodMesh::default();
    }

    let mut quads = Vec::new();
    let mut palette = Vec::new();
    let mut palette_map: HashMap<[u8; 4], u16> = HashMap::new();

    let mut get_color_index = |rgba: [u8; 4]| -> u16 {
        if let Some(&idx) = palette_map.get(&rgba) {
            return idx;
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = palette.len() as u16;
        palette.push(rgba);
        palette_map.insert(rgba, idx);
        idx
    };

    // Mesh each of the 6 cardinal directions
    for face in Face::ALL {
        let (axis_u, axis_v, axis_d) = match face {
            Face::Down | Face::Up => (0, 2, 1),     // u=X, v=Z, d=Y
            Face::North | Face::South => (0, 1, 2), // u=X, v=Y, d=Z
            Face::West | Face::East => (2, 1, 0),   // u=Z, v=Y, d=X
        };

        let is_positive = matches!(face, Face::Up | Face::South | Face::East);
        let neighbor_idx = match face {
            Face::East => 0,  // +X
            Face::West => 1,  // -X
            Face::Up => 2,    // +Y
            Face::Down => 3,  // -Y
            Face::South => 4, // +Z
            Face::North => 5, // -Z
        };
        let neighbor = neighbors[neighbor_idx];

        for d in 0..32u32 {
            let mut mask = [0u32; 32];
            let mut key_grid = [0u16; 1024];

            // Build 32x32 face mask for slice d
            for v in 0..32u32 {
                for u in 0..32u32 {
                    let mut coords = [0u32; 3];
                    coords[axis_u] = u;
                    coords[axis_v] = v;
                    coords[axis_d] = d;

                    let (x, y, z) = (coords[0], coords[1], coords[2]);
                    if !node.occupancy.is_solid(x, y, z) {
                        continue;
                    }

                    // Check neighbor along normal
                    let is_face_visible = if is_positive {
                        if d + 1 < 32 {
                            let mut n_coords = coords;
                            n_coords[axis_d] = d + 1;
                            !node
                                .occupancy
                                .is_solid(n_coords[0], n_coords[1], n_coords[2])
                        } else {
                            // Boundary: check across node boundary
                            let mut n_coords = coords;
                            n_coords[axis_d] = 0;
                            neighbor.is_none_or(|nb| {
                                !nb.is_solid(n_coords[0], n_coords[1], n_coords[2])
                            })
                        }
                    } else if d > 0 {
                        let mut n_coords = coords;
                        n_coords[axis_d] = d - 1;
                        !node
                            .occupancy
                            .is_solid(n_coords[0], n_coords[1], n_coords[2])
                    } else {
                        // Boundary: check across node boundary
                        let mut n_coords = coords;
                        n_coords[axis_d] = 31;
                        neighbor
                            .is_none_or(|nb| !nb.is_solid(n_coords[0], n_coords[1], n_coords[2]))
                    };

                    if is_face_visible {
                        mask[v as usize] |= 1 << u;

                        let state = node.get_state(x, y, z);
                        let lod_color = color_table.get(state);
                        let rgba = lod_color.for_face(face);
                        let color_idx = get_color_index(rgba);
                        key_grid[(v * 32 + u) as usize] = color_idx;
                    }
                }
            }

            // Greedy merge coplanar faces sharing the same color key
            for v in 0..32u32 {
                while mask[v as usize] != 0 {
                    let u = mask[v as usize].trailing_zeros();
                    let color_idx = key_grid[(v * 32 + u) as usize];

                    // Extend width along row v
                    let mut width = 1;
                    while u + width < 32
                        && (mask[v as usize] & (1 << (u + width))) != 0
                        && key_grid[(v * 32 + u + width) as usize] == color_idx
                    {
                        width += 1;
                    }

                    let row_mask = if width == 32 {
                        !0u32
                    } else {
                        ((1 << width) - 1) << u
                    };

                    // Extend height across rows
                    let mut height = 1;
                    while v + height < 32 {
                        if (mask[(v + height) as usize] & row_mask) != row_mask {
                            break;
                        }
                        // Verify matching keys for entire row span
                        let mut matching = true;
                        for du in 0..width {
                            if key_grid[((v + height) * 32 + u + du) as usize] != color_idx {
                                matching = false;
                                break;
                            }
                        }
                        if !matching {
                            break;
                        }
                        height += 1;
                    }

                    // Clear merged span
                    for h in 0..height {
                        mask[(v + h) as usize] &= !row_mask;
                    }

                    // Compute local (x, y, z) of the quad's minimum corner
                    let mut quad_coords = [0u32; 3];
                    quad_coords[axis_u] = u;
                    quad_coords[axis_v] = v;
                    quad_coords[axis_d] = d;

                    let quad = LodQuad::new(
                        quad_coords[0],
                        quad_coords[1],
                        quad_coords[2],
                        width,
                        height,
                        face,
                        15, // Max sky light at LOD
                        color_idx,
                        [3, 3, 3, 3], // Fully lit AO at LOD distance
                        0,
                        0,
                    );
                    quads.push(quad);
                }
            }
        }
    }

    LodMesh { quads, palette }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::LodNodeKey;
    use vx_voxel::state::BlockStateId;

    #[test]
    fn test_empty_node_produces_no_quads() {
        let key = LodNodeKey::new(1, 0, 0, 0);
        let node = LodNode::empty(key);
        let neighbors: [Option<&LodNode>; 6] = [None; 6];
        let color_table = LodColorTable::standard();

        let mesh = mesh_lod_node(&node, &neighbors, &color_table);
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_produces_cube() {
        let key = LodNodeKey::new(1, 0, 0, 0);
        let mut node = LodNode::empty(key);
        node.set_state(10, 10, 10, BlockStateId::new(1));

        let neighbors: [Option<&LodNode>; 6] = [None; 6];
        let color_table = LodColorTable::standard();

        let mesh = mesh_lod_node(&node, &neighbors, &color_table);
        assert_eq!(mesh.quads.len(), 6, "Expected 6 faces for isolated voxel");
        assert_eq!(mesh.palette.len(), 1, "Expected 1 color in palette");
    }

    #[test]
    fn test_greedy_merging_full_layer() {
        let key = LodNodeKey::new(1, 0, 0, 0);
        let mut node = LodNode::empty(key);
        let stone = BlockStateId::new(1);

        // Fill an entire 32x32 plane at y=0
        for z in 0..32 {
            for x in 0..32 {
                node.set_state(x, 0, z, stone);
            }
        }

        let neighbors: [Option<&LodNode>; 6] = [None; 6];
        let color_table = LodColorTable::standard();

        let mesh = mesh_lod_node(&node, &neighbors, &color_table);
        // Top (+Y) face should merge into a single 32x32 quad!
        let top_quads: Vec<_> = mesh.quads.iter().filter(|q| q.face() == Face::Up).collect();
        assert_eq!(top_quads.len(), 1, "Expected single merged 32x32 top quad");
        assert_eq!(top_quads[0].width(), 32);
        assert_eq!(top_quads[0].height(), 32);
    }
}
