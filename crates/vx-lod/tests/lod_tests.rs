//! Integration tests for LOD hierarchy, downsampling, exterior cave culling, and binary meshing.

use vx_core::coords::Face;
use vx_lod::color::LodColorTable;
use vx_lod::coords::LodNodeKey;
use vx_lod::downsample::downsample_octants;
use vx_lod::exterior::exterior_flood_fill;
use vx_lod::mesher::mesh_lod_node;
use vx_lod::node::LodNode;
use vx_voxel::state::BlockStateId;

#[test]
fn test_end_to_end_downsample_and_mesh() {
    let child_key = LodNodeKey::new(0, 0, 0, 0);
    let mut child = LodNode::empty(child_key);

    let stone = BlockStateId::new(1);
    let dirt = BlockStateId::new(2);

    // Build a small hill in child octant 0: base is dirt, top is stone
    for x in 0..8 {
        for z in 0..8 {
            child.set_state(x, 0, z, dirt);
            child.set_state(x, 1, z, stone);
        }
    }

    let parent_key = LodNodeKey::new(1, 0, 0, 0);
    let children: [Option<&LodNode>; 8] = [Some(&child), None, None, None, None, None, None, None];

    // Downsample L0 -> L1
    let mut parent = downsample_octants(parent_key, &children);

    // Check that parent voxel (0..4, 0, 0..4) is solid
    for x in 0..4 {
        for z in 0..4 {
            assert!(
                parent.occupancy.is_solid(x, 0, z),
                "Expected downsampled voxel at ({x}, 0, {z}) to be solid"
            );
            // Top stone should have won the vote
            assert_eq!(parent.get_state(x, 0, z), stone);
        }
    }

    // Run exterior flood fill
    exterior_flood_fill(&mut parent);

    // Mesh L1 node
    let color_table = LodColorTable::standard();
    let neighbors: [Option<&LodNode>; 6] = [None; 6];
    let mesh = mesh_lod_node(&parent, &neighbors, &color_table);

    assert!(
        !mesh.is_empty(),
        "Expected non-empty mesh for downsampled hill"
    );

    // Check greedy merging: 4x4 top quads should be merged
    let top_quads: Vec<_> = mesh.quads.iter().filter(|q| q.face() == Face::Up).collect();
    assert_eq!(
        top_quads.len(),
        1,
        "Expected single 4x4 merged quad for top surface"
    );
    assert_eq!(top_quads[0].width(), 4);
    assert_eq!(top_quads[0].height(), 4);

    // Verify buffer serialization
    let mut buffer = Vec::new();
    mesh.write_to_u32_buffer(&mut buffer);
    assert_eq!(
        buffer.len(),
        mesh.quads.len() * 2 + mesh.palette.len(),
        "Buffer length must equal 2 words per quad + 1 word per palette entry"
    );
}
