//! 6-connected BFS exterior-only flood fill algorithm to eliminate enclosed caves ($L \ge 2$).

use crate::node::LodNode;

#[inline]
fn is_marked(bits: &[u32; 1024], idx: usize) -> bool {
    (bits[idx >> 5] & (1 << (idx & 31))) != 0
}

#[inline]
fn set_marked(bits: &mut [u32; 1024], idx: usize) {
    bits[idx >> 5] |= 1 << (idx & 31);
}

/// Performs exterior-only flood fill culling on an LOD node.
///
/// Nodes with $L < 2$ are untouched. For $L \ge 2$, enclosed underground caves
/// that are unreachable from the surface are filled as solid, preventing face generation.
#[allow(clippy::cast_possible_wrap)]
pub fn exterior_flood_fill(node: &mut LodNode) {
    if node.key.level < 2 {
        return;
    }

    let mut exterior = [0u32; 1024]; // 32,768 bits
    let mut queue = Vec::with_capacity(2048);

    // 1. Seed exterior air from sky and node boundaries
    for z in 0..32u32 {
        for x in 0..32u32 {
            // Find top solid voxel
            let mut top_solid = None;
            for y in (0..32u32).rev() {
                if node.occupancy.is_solid(x, y, z) {
                    top_solid = Some(y);
                    break;
                }
            }

            // Air above top solid voxel is directly open to sky
            let start_y = top_solid.map_or(0, |y| y + 1);
            for y in start_y..32 {
                let idx = LodNode::local_index(x, y, z);
                if !is_marked(&exterior, idx) {
                    set_marked(&mut exterior, idx);
                    queue.push(idx as u16);
                }
            }

            // Boundary air at y=0, x=0,31, z=0,31 treated as exterior
            for y in 0..start_y {
                if (x == 0 || x == 31 || y == 0 || z == 0 || z == 31)
                    && !node.occupancy.is_solid(x, y, z)
                {
                    let idx = LodNode::local_index(x, y, z);
                    if !is_marked(&exterior, idx) {
                        set_marked(&mut exterior, idx);
                        queue.push(idx as u16);
                    }
                }
            }
        }
    }

    // 2. 6-connected BFS through connected air voxels
    let mut head = 0;
    while head < queue.len() {
        let idx = queue[head] as usize;
        head += 1;

        let x = (idx & 0x1F) as i32;
        let z = ((idx >> 5) & 0x1F) as i32;
        let y = ((idx >> 10) & 0x1F) as i32;

        let neighbors = [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ];

        for (nx, ny, nz) in neighbors {
            if (0..32).contains(&nx) && (0..32).contains(&ny) && (0..32).contains(&nz) {
                let n_idx = LodNode::local_index(nx as u32, ny as u32, nz as u32);
                if !node.occupancy.is_solid(nx as u32, ny as u32, nz as u32)
                    && !is_marked(&exterior, n_idx)
                {
                    set_marked(&mut exterior, n_idx);
                    queue.push(n_idx as u16);
                }
            }
        }
    }

    // 3. Mark non-exterior enclosed air as solid to cull cave walls
    for y in 0..32u32 {
        for z in 0..32u32 {
            for x in 0..32u32 {
                let idx = LodNode::local_index(x, y, z);
                if !node.occupancy.is_solid(x, y, z) && !is_marked(&exterior, idx) {
                    node.occupancy.set_solid(x, y, z, true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::LodNodeKey;
    use vx_voxel::state::BlockStateId;

    #[test]
    fn test_cave_elimination_at_level_2() {
        let key = LodNodeKey::new(2, 0, 0, 0);
        let stone = BlockStateId::new(1);
        let mut node = LodNode::uniform_solid(key, stone);

        // Carve an isolated 4x4x4 cave in the center (14..18)
        for y in 14..18 {
            for z in 14..18 {
                for x in 14..18 {
                    node.set_state(x, y, z, BlockStateId::AIR);
                }
            }
        }

        // Before flood fill: cave voxels are air (non-solid)
        assert!(!node.occupancy.is_solid(15, 15, 15));

        exterior_flood_fill(&mut node);

        // After flood fill: isolated cave voxels are sealed as solid
        assert!(
            node.occupancy.is_solid(15, 15, 15),
            "Enclosed cave should be marked solid to eliminate quads"
        );
    }

    #[test]
    fn test_caves_preserved_at_level_1() {
        let key = LodNodeKey::new(1, 0, 0, 0);
        let stone = BlockStateId::new(1);
        let mut node = LodNode::uniform_solid(key, stone);

        // Carve cave
        node.set_state(15, 15, 15, BlockStateId::AIR);

        exterior_flood_fill(&mut node);

        // Level 1 keeps all caves
        assert!(!node.occupancy.is_solid(15, 15, 15));
    }
}
