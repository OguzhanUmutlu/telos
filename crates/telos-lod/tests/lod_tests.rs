//! Integration tests for LOD hierarchy, downsampling, exterior cave culling, and binary meshing.

use telos_core::coords::Face;
use telos_lod::color::LodColorTable;
use telos_lod::coords::LodNodeKey;
use telos_lod::downsample::downsample_octants;
use telos_lod::exterior::exterior_flood_fill;
use telos_lod::mesher::mesh_lod_node;
use telos_lod::node::LodNode;
use telos_voxel::state::BlockStateId;

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

#[test]
fn test_flora_lod_colors_and_blending() {
    let lod_table = LodColorTable::standard();

    // Verify flora colors are properly registered (not fallback gray)
    let poppy_color = lod_table.get(BlockStateId::new(12));
    assert_eq!(poppy_color.top, [220, 40, 40, 255]);
    let dandelion_color = lod_table.get(BlockStateId::new(13));
    assert_eq!(dandelion_color.top, [255, 230, 40, 255]);
    let short_grass_color = lod_table.get(BlockStateId::new(16));
    assert_eq!(short_grass_color.top, [106, 170, 64, 255]);
    let fern_color = lod_table.get(BlockStateId::new(17));
    assert_eq!(fern_color.top, [80, 160, 50, 255]);
    let dead_bush_color = lod_table.get(BlockStateId::new(18));
    assert_eq!(dead_bush_color.top, [140, 110, 70, 255]);
    let tall_grass_color = lod_table.get(BlockStateId::new(85));
    assert_eq!(tall_grass_color.top, [100, 180, 60, 255]);
    let cornflower_color = lod_table.get(BlockStateId::new(86));
    assert_eq!(cornflower_color.top, [70, 120, 240, 255]);
    let daisy_color = lod_table.get(BlockStateId::new(87));
    assert_eq!(daisy_color.top, [240, 240, 230, 255]);
    let brown_shroom = lod_table.get(BlockStateId::new(88));
    assert_eq!(brown_shroom.top, [150, 110, 80, 255]);
    let red_shroom = lod_table.get(BlockStateId::new(89));
    assert_eq!(red_shroom.top, [200, 40, 40, 255]);

    // Test blend_vegetation_surface helper
    let base = [100, 100, 100, 255];
    let veg = [200, 200, 200, 255];
    let blended = LodColorTable::blend_vegetation_surface(base, veg, 0.5);
    assert_eq!(blended, [150, 150, 150, 255]);
}
