//! Property-based tests for LOD coordinates, quad bit-packing, and downsampling invariants.

use proptest::prelude::*;
use telos_core::coords::Face;
use telos_lod::coords::LodNodeKey;
use telos_lod::downsample::downsample_octants;
use telos_lod::node::LodNode;
use telos_lod::quad::LodQuad;
use telos_voxel::state::BlockStateId;

proptest! {
    #[test]
    fn prop_parent_child_roundtrip(level in 0u8..6, x in -1000..1000, y in -100..100, z in -1000..1000) {
        let node = LodNodeKey::new(level, x, y, z);
        let parent = node.parent();
        prop_assert_eq!(parent.level, level + 1);

        let octant = node.octant();
        let reconstructed = parent.child(octant);
        prop_assert_eq!(reconstructed, node);
    }

    #[test]
    fn prop_lod_quad_roundtrip(
        x in 0u32..32,
        y in 0u32..32,
        z in 0u32..32,
        width in 1u32..=32,
        height in 1u32..=32,
        dir in 0u8..6,
        sky in 0u8..16,
        color in 0u16..4096,
        ao0 in 0u8..4,
        ao1 in 0u8..4,
        ao2 in 0u8..4,
        ao3 in 0u8..4,
        block in 0u8..16,
        flags in 0u8..16,
    ) {
        let face = Face::ALL[dir as usize];
        let quad = LodQuad::new(
            x, y, z, width, height, face, sky, color, [ao0, ao1, ao2, ao3], block, flags,
        );

        prop_assert_eq!(quad.x(), x);
        prop_assert_eq!(quad.y(), y);
        prop_assert_eq!(quad.z(), z);
        prop_assert_eq!(quad.width(), width);
        prop_assert_eq!(quad.height(), height);
        prop_assert_eq!(quad.face(), face);
        prop_assert_eq!(quad.sky_light(), sky);
        prop_assert_eq!(quad.color_index(), color);
        prop_assert_eq!(quad.ao(), [ao0, ao1, ao2, ao3]);
        prop_assert_eq!(quad.block_light(), block);
        prop_assert_eq!(quad.flags(), flags);
    }

    #[test]
    fn prop_conservative_downsampling_invariant(
        child_octant in 0usize..8,
        cx in 0u32..16,
        cy in 0u32..16,
        cz in 0u32..16,
        state_id in 1u32..100,
    ) {
        let child_key = LodNodeKey::new(0, 0, 0, 0);
        let mut child = LodNode::empty(child_key);

        let solid_state = BlockStateId::new(state_id);
        child.set_state(cx * 2, cy * 2, cz * 2, solid_state);

        let parent_key = LodNodeKey::new(1, 0, 0, 0);
        let mut children: [Option<&LodNode>; 8] = [None; 8];
        children[child_octant] = Some(&child);

        let parent = downsample_octants(parent_key, &children);

        let ox = (child_octant & 1) as u32;
        let oy = ((child_octant >> 1) & 1) as u32;
        let oz = ((child_octant >> 2) & 1) as u32;

        let px = ox * 16 + cx;
        let py = oy * 16 + cy;
        let pz = oz * 16 + cz;

        prop_assert!(
            parent.occupancy.is_solid(px, py, pz),
            "Parent voxel must be solid if constituent child is solid"
        );
        prop_assert_eq!(parent.get_state(px, py, pz), solid_state);
    }
}
