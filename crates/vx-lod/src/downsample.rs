//! Server-side downsampling from level $L-1$ to level $L$ (2×2×2 → 1 voxel).

use hashbrown::HashMap;
use vx_voxel::state::BlockStateId;

use crate::coords::LodNodeKey;
use crate::node::{LodNode, LodVoxelSource};

/// Downsamples 8 child octants of level $L-1$ into a single level $L$ `LodNode`.
///
/// Children are provided as an array of 8 optional references indexed by octant:
/// `octant = ox | (oy << 1) | (oz << 2)`.
/// Missing child nodes are treated as empty air.
#[must_use]
#[allow(clippy::similar_names)]
pub fn downsample_octants<S: LodVoxelSource>(
    parent_key: LodNodeKey,
    children: &[Option<&S>; 8],
) -> LodNode {
    let mut node = LodNode::empty(parent_key);

    for vz in 0..32u32 {
        let oz = usize::from(vz >= 16);
        let base_cz = (vz & 15) * 2;

        for vy in 0..32u32 {
            let oy = usize::from(vy >= 16);
            let base_cy = (vy & 15) * 2;

            for vx in 0..32u32 {
                let ox = usize::from(vx >= 16);
                let base_cx = (vx & 15) * 2;

                let octant = ox | (oy << 1) | (oz << 2);
                let Some(child) = children[octant] else {
                    continue;
                };

                let state = vote_representative(child, base_cx, base_cy, base_cz);
                if !state.is_air() {
                    node.set_state(vx, vy, vz, state);
                }
            }
        }
    }

    node
}

/// Evaluates 8 child voxels in a 2×2×2 cube and votes for the representative `BlockStateId`.
#[inline]
fn vote_representative<S: LodVoxelSource>(child: &S, cx: u32, cy: u32, cz: u32) -> BlockStateId {
    let mut any_solid = false;
    let mut weights: HashMap<BlockStateId, f32> = HashMap::new();
    let mut counts: HashMap<BlockStateId, u32> = HashMap::new();

    for dz in 0..=1u32 {
        for dy in 0..=1u32 {
            for dx in 0..=1u32 {
                let x = cx + dx;
                let y = cy + dy;
                let z = cz + dz;

                if !child.is_solid(x, y, z) {
                    continue;
                }

                any_solid = true;
                let state = child.get_state(x, y, z);
                *counts.entry(state).or_insert(0) += 1;

                // Evaluate directional exposures to air
                let mut voxel_weight = 0.0f32;

                // Up (+Y)
                if y + 1 >= 32 || !child.is_solid(x, y + 1, z) {
                    voxel_weight += 4.0;
                }
                // Down (-Y)
                if y == 0 || !child.is_solid(x, y - 1, z) {
                    voxel_weight += 0.25;
                }
                // Sides (+X, -X, +Z, -Z)
                if x + 1 >= 32 || !child.is_solid(x + 1, y, z) {
                    voxel_weight += 1.0;
                }
                if x == 0 || !child.is_solid(x - 1, y, z) {
                    voxel_weight += 1.0;
                }
                if z + 1 >= 32 || !child.is_solid(x, y, z + 1) {
                    voxel_weight += 1.0;
                }
                if z == 0 || !child.is_solid(x, y, z - 1) {
                    voxel_weight += 1.0;
                }

                if voxel_weight > 0.0 {
                    *weights.entry(state).or_insert(0.0) += voxel_weight;
                }
            }
        }
    }

    if !any_solid {
        return BlockStateId::AIR;
    }

    // 1. Pick maximum weighted exposed state
    if let Some((&best_state, _)) = weights.iter().max_by(|(s_a, w_a), (s_b, w_b)| {
        w_a.partial_cmp(w_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| s_b.0.cmp(&s_a.0)) // Ties break to lower BlockStateId
    }) {
        return best_state;
    }

    // 2. Fallback: most frequent solid state
    counts
        .into_iter()
        .max_by(|(s_a, c_a), (s_b, c_b)| c_a.cmp(c_b).then_with(|| s_b.0.cmp(&s_a.0)))
        .map_or(BlockStateId::AIR, |(s, _)| s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservative_occupancy() {
        let child_key = LodNodeKey::new(0, 0, 0, 0);
        let mut child = LodNode::empty(child_key);

        // Place a single solid stone block in child octant 0 at (0, 0, 0)
        let stone = BlockStateId::new(1);
        child.set_state(0, 0, 0, stone);

        let parent_key = LodNodeKey::new(1, 0, 0, 0);
        let children: [Option<&LodNode>; 8] =
            [Some(&child), None, None, None, None, None, None, None];

        let parent = downsample_octants(parent_key, &children);
        assert!(parent.occupancy.is_solid(0, 0, 0));
        assert_eq!(parent.get_state(0, 0, 0), stone);

        // Surrounding parent voxels must remain air
        assert!(!parent.occupancy.is_solid(1, 0, 0));
        assert!(!parent.occupancy.is_solid(0, 1, 0));
        assert!(!parent.occupancy.is_solid(0, 0, 1));
    }

    #[test]
    fn test_top_face_weight_precedence() {
        let child_key = LodNodeKey::new(0, 0, 0, 0);
        let mut child = LodNode::empty(child_key);

        let dirt = BlockStateId::new(2);
        let grass = BlockStateId::new(3);

        // Top voxel at (0, 1, 0) is grass (exposed to top), bottom at (0, 0, 0) is dirt
        child.set_state(0, 0, 0, dirt);
        child.set_state(0, 1, 0, grass);

        let parent_key = LodNodeKey::new(1, 0, 0, 0);
        let children: [Option<&LodNode>; 8] =
            [Some(&child), None, None, None, None, None, None, None];

        let parent = downsample_octants(parent_key, &children);
        // Grass has higher top-facing weight (4.0) than side dirt
        assert_eq!(parent.get_state(0, 0, 0), grass);
    }
}
