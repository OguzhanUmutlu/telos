//! Benchmarks for telos-mesh binary greedy mesher.

#![allow(missing_docs)]
#![allow(clippy::large_stack_arrays)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use telos_mesh::{bitwise::NeighborSlices, mesher::mesh_blocks_with_occupancy, naive::NaiveMesher};
use telos_voxel::{
    coords::{CHUNK_VOLUME, LocalIdx},
    occupancy::Occupancy,
    registry::BlockRegistry,
    state::{BlockStateId, StateFlags},
    storage::{Blocks, from_dense},
};

fn bench_meshing_scenarios(c: &mut Criterion) {
    let reg = BlockRegistry::standard();
    let is_opaque = |s: BlockStateId| reg.flags(s).contains(StateFlags::OPAQUE_FULL);
    let neighbors = NeighborSlices::empty();

    // 1. Uniform Air Chunk (Empty)
    let air_blocks = Blocks::Uniform(BlockStateId::AIR);
    let air_occ = Occupancy::empty();

    c.bench_function("meshing_empty_air_chunk", |b| {
        b.iter(|| {
            mesh_blocks_with_occupancy(
                black_box(&air_blocks),
                black_box(&air_occ),
                black_box(&neighbors),
            )
        });
    });

    // 2. Uniform Solid Chunk (32x32x32 Stone)
    let stone_blocks = Blocks::Uniform(BlockStateId::new(1));
    let stone_occ = Occupancy::solid();

    c.bench_function("meshing_full_solid_chunk", |b| {
        b.iter(|| {
            mesh_blocks_with_occupancy(
                black_box(&stone_blocks),
                black_box(&stone_occ),
                black_box(&neighbors),
            )
        });
    });

    // 3. Realistic Terrain Chunk (surface layer: stone base, dirt layer, grass surface, air above)
    let mut terrain_dense = [BlockStateId::AIR; CHUNK_VOLUME];
    let stone = BlockStateId::new(1);
    let dirt = BlockStateId::new(2);
    let grass = BlockStateId::new(3);

    for y in 0..16 {
        for z in 0..32 {
            for x in 0..32 {
                let idx = LocalIdx::from_coords_unchecked(x, y, z).as_usize();
                terrain_dense[idx] = if y < 12 {
                    stone
                } else if y < 15 {
                    dirt
                } else {
                    grass
                };
            }
        }
    }

    let terrain_blocks = from_dense(&terrain_dense);
    let terrain_occ = Occupancy::from_blocks(&terrain_blocks, is_opaque);

    c.bench_function("meshing_terrain_surface_binary_greedy", |b| {
        b.iter(|| {
            mesh_blocks_with_occupancy(
                black_box(&terrain_blocks),
                black_box(&terrain_occ),
                black_box(&neighbors),
            )
        });
    });

    // Naive baseline comparison on terrain
    c.bench_function("meshing_terrain_surface_naive_baseline", |b| {
        b.iter(|| {
            NaiveMesher::mesh(
                black_box(&terrain_blocks),
                black_box(&terrain_occ),
                black_box(&neighbors),
            )
        });
    });

    // Print comparison metrics
    let greedy_mesh = mesh_blocks_with_occupancy(&terrain_blocks, &terrain_occ, &neighbors);
    let naive_mesh = NaiveMesher::mesh(&terrain_blocks, &terrain_occ, &neighbors);
    println!(
        "\n--- Terrain Surface Chunk Mesh Statistics ---\n\
         Naive mesher quads:        {}\n\
         Binary greedy quads:       {}\n\
         Quad reduction ratio:      {:.2}x\n\
         ----------------------------------------------\n",
        naive_mesh.total_quads(),
        greedy_mesh.total_quads(),
        naive_mesh.total_quads() as f64 / greedy_mesh.total_quads() as f64
    );

    // 4. Worst-case 3D Checkerboard Chunk
    let mut checker_dense = [BlockStateId::AIR; CHUNK_VOLUME];
    for y in 0..32 {
        for z in 0..32 {
            for x in 0..32 {
                if (x + y + z) % 2 == 0 {
                    let idx = LocalIdx::from_coords_unchecked(x, y, z).as_usize();
                    checker_dense[idx] = stone;
                }
            }
        }
    }
    let checker_blocks = from_dense(&checker_dense);
    let checker_occ = Occupancy::from_blocks(&checker_blocks, is_opaque);

    c.bench_function("meshing_worst_case_checkerboard", |b| {
        b.iter(|| {
            mesh_blocks_with_occupancy(
                black_box(&checker_blocks),
                black_box(&checker_occ),
                black_box(&neighbors),
            )
        });
    });
}

criterion_group!(benches, bench_meshing_scenarios);
criterion_main!(benches);
