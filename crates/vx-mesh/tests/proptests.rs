//! Property-based correctness tests for vx-mesh binary greedy mesher.

#![allow(clippy::large_stack_arrays)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use proptest::prelude::*;
use std::collections::HashSet;
use vx_voxel::{
    coords::CHUNK_VOLUME,
    occupancy::Occupancy,
    registry::BlockRegistry,
    state::{BlockStateId, StateFlags},
    storage::{Blocks, from_dense},
};

use vx_mesh::{
    bitwise::NeighborSlices,
    mesher::mesh_blocks_with_occupancy,
    naive::NaiveMesher,
    quad::{FaceDir, T0Quad},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_quad_encoding_round_trip(
        x in 0u32..32,
        y in 0u32..32,
        z in 0u32..32,
        w in 1u32..=32,
        h in 1u32..=32,
        dir_idx in 0u8..6,
        material in 0u16..=1000,
        light_pattern in 0u16..1000,
    ) {
        let dir = FaceDir::from_u8(dir_idx).unwrap();
        let quad = T0Quad::new(x, y, z, w, h, dir, material, light_pattern);

        prop_assert_eq!(quad.x(), x);
        prop_assert_eq!(quad.y(), y);
        prop_assert_eq!(quad.z(), z);
        prop_assert_eq!(quad.w(), w);
        prop_assert_eq!(quad.h(), h);
        prop_assert_eq!(quad.dir(), dir);
        prop_assert_eq!(quad.material(), material);
        prop_assert_eq!(quad.light_pattern(), light_pattern);
    }

    #[test]
    fn prop_greedy_matches_naive_surface(
        seed in any::<u64>(),
        density in 0.05f64..0.5f64,
    ) {
        let reg = BlockRegistry::standard();
        let is_opaque = |s: BlockStateId| reg.flags(s).contains(StateFlags::OPAQUE_FULL);

        let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
        let mut rng_state = seed;

        // Populate pseudo-random voxels with Stone and Dirt
        for i in 0..CHUNK_VOLUME {
            rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let val = (rng_state >> 32) as f64 / (u32::MAX as f64);
            if val < density {
                let mat = if (rng_state & 1) == 0 { 1 } else { 2 };
                dense[i] = BlockStateId::new(mat);
            }
        }

        let blocks = from_dense(&dense);
        let occ = Occupancy::from_blocks(&blocks, is_opaque);
        let neighbors = NeighborSlices::empty();

        // 1. Run naive mesher
        let naive_mesh = NaiveMesher::mesh(&blocks, &occ, &neighbors);

        // 2. Run binary greedy mesher
        let greedy_mesh = mesh_blocks_with_occupancy(&blocks, &occ, &neighbors);

        // Verify that greedy mesh quads have <= total quads than naive
        prop_assert!(greedy_mesh.total_quads() <= naive_mesh.total_quads());

        // 3. Deconstruct greedy quads into individual 1x1 faces and assert exact match with naive
        // Key: (x, y, z, dir as u8, material)
        let mut naive_faces = HashSet::new();
        for quad in &naive_mesh.quads {
            let key = (quad.x(), quad.y(), quad.z(), quad.dir() as u8, quad.material());
            prop_assert!(naive_faces.insert(key), "Naive mesher emitted duplicate face: {:?}", key);
        }

        let mut greedy_faces = HashSet::new();
        for dir in FaceDir::ALL {
            let bucket = greedy_mesh.bucket_quads(dir);
            for quad in bucket {
                prop_assert_eq!(quad.dir(), dir, "Quad in wrong directional bucket!");

                let (u_dir, v_dir) = dir.tangent_frame();
                let qx = quad.x();
                let qy = quad.y();
                let qz = quad.z();

                for du in 0..quad.w() {
                    for dv in 0..quad.h() {
                        let fx = qx as i32 + (u_dir.x * du as i32) + (v_dir.x * dv as i32);
                        let fy = qy as i32 + (u_dir.y * du as i32) + (v_dir.y * dv as i32);
                        let fz = qz as i32 + (u_dir.z * du as i32) + (v_dir.z * dv as i32);

                        let key = (fx as u32, fy as u32, fz as u32, dir as u8, quad.material());
                        prop_assert!(greedy_faces.insert(key), "Greedy mesher emitted overlapping quad area: {:?}", key);
                    }
                }
            }
        }

        // Exact 1:1 face equivalence
        prop_assert_eq!(greedy_faces, naive_faces, "Greedy mesher surface does not match naive mesher ground truth!");
    }
}

#[test]
fn test_solid_cube_culling() {
    let stone = BlockStateId::new(1);
    let blocks = Blocks::Uniform(stone);
    let occ = Occupancy::solid();

    // With solid neighbors: fully enclosed -> 0 quads!
    let mesh_closed = mesh_blocks_with_occupancy(&blocks, &occ, &NeighborSlices::solid());
    assert_eq!(mesh_closed.total_quads(), 0);

    // With empty neighbors: exactly 6 quads (one 32x32 quad per face)!
    let mesh_open = mesh_blocks_with_occupancy(&blocks, &occ, &NeighborSlices::empty());
    assert_eq!(mesh_open.total_quads(), 6);
    for dir in FaceDir::ALL {
        let bucket = mesh_open.bucket_quads(dir);
        assert_eq!(bucket.len(), 1);
        let quad = bucket[0];
        assert_eq!(quad.w(), 32);
        assert_eq!(quad.h(), 32);
        assert_eq!(quad.material(), 1);
    }
}
