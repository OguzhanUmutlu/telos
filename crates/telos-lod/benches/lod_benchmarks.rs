//! Criterion benchmarks for telos-lod downsampling, exterior flood fill, and greedy meshing.

#![allow(missing_docs)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::cast_possible_wrap)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use telos_core::coords::ChunkPos;
use telos_lod::{
    color::LodColorTable, coords::LodNodeKey, downsample::downsample_octants,
    exterior::exterior_flood_fill, mesher::mesh_lod_node, node::LodNode,
};
use telos_voxel::{
    chunk::ChunkSnapshot,
    coords::{CHUNK_VOLUME, LocalIdx},
    occupancy::Occupancy,
    registry::BlockRegistry,
    state::{BlockStateId, StateFlags},
    storage::from_dense,
};

fn create_surface_chunk(y_offset: i32) -> Arc<ChunkSnapshot> {
    let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
    let stone = BlockStateId::new(1);
    let dirt = BlockStateId::new(2);
    let grass = BlockStateId::new(3);

    for y in 0..32 {
        let global_y = y_offset * 32 + y as i32;
        for z in 0..32 {
            for x in 0..32 {
                let idx = LocalIdx::from_coords_unchecked(x, y, z).as_usize();
                dense[idx] = if global_y < 12 {
                    stone
                } else if global_y < 15 {
                    dirt
                } else if global_y == 15 {
                    grass
                } else {
                    BlockStateId::AIR
                };
            }
        }
    }

    let blocks = from_dense(&dense);
    let reg = BlockRegistry::standard();
    let occ = Occupancy::from_blocks(&blocks, |s| reg.flags(s).contains(StateFlags::OPAQUE_FULL));

    Arc::new(ChunkSnapshot::from_parts(
        ChunkPos::new(0, y_offset, 0),
        blocks,
        occ,
        None,
        1,
        [0; 6],
        1,
    ))
}

fn bench_lod_pipeline(c: &mut Criterion) {
    let color_table = LodColorTable::standard();
    let key = LodNodeKey::new(1, 0, 0, 0);

    // 8 children chunks (octants)
    let c0 = create_surface_chunk(0);
    let c1 = create_surface_chunk(0);
    let c2 = create_surface_chunk(0);
    let c3 = create_surface_chunk(0);
    let c4 = create_surface_chunk(1);
    let c5 = create_surface_chunk(1);
    let c6 = create_surface_chunk(1);
    let c7 = create_surface_chunk(1);

    let children: [Option<&Arc<ChunkSnapshot>>; 8] = [
        Some(&c0),
        Some(&c1),
        Some(&c2),
        Some(&c3),
        Some(&c4),
        Some(&c5),
        Some(&c6),
        Some(&c7),
    ];

    // 1. Downsampling benchmark (target: <= 100 µs)
    c.bench_function("lod_downsample_8_octants", |b| {
        b.iter(|| downsample_octants(black_box(key), black_box(&children)));
    });

    let mut lod_node = downsample_octants(key, &children);

    // 2. Exterior culling BFS benchmark (target: <= 30 µs)
    c.bench_function("lod_exterior_cull_bfs", |b| {
        b.iter_batched(
            || lod_node.clone(),
            |mut node| {
                exterior_flood_fill(black_box(&mut node));
                node
            },
            criterion::BatchSize::SmallInput,
        );
    });

    exterior_flood_fill(&mut lod_node);

    // 3. Binary greedy meshing benchmark (target: <= 50 µs)
    let neighbors: [Option<&LodNode>; 6] = [None; 6];
    c.bench_function("lod_binary_greedy_meshing", |b| {
        b.iter(|| {
            mesh_lod_node(
                black_box(&lod_node),
                black_box(&neighbors),
                black_box(&color_table),
            )
        });
    });

    let mesh = mesh_lod_node(&lod_node, &neighbors, &color_table);
    let mut initial_buffer = Vec::new();
    mesh.write_to_u32_buffer(&mut initial_buffer);
    println!(
        "\n--- LOD Benchmark Stats ---\n\
         LOD Quad count:   {}\n\
         Palette entries:  {}\n\
         Buffer words:     {}\n\
         Buffer byte size: {} bytes\n\
         ---------------------------\n",
        mesh.quads.len(),
        mesh.palette.len(),
        initial_buffer.len(),
        initial_buffer.len() * 4,
    );

    // 4. Single-buffer serialization benchmark
    let mut out_buf = Vec::with_capacity(initial_buffer.len());
    c.bench_function("lod_mesh_write_u32_buffer", |b| {
        b.iter(|| {
            out_buf.clear();
            black_box(&mesh).write_to_u32_buffer(black_box(&mut out_buf));
        });
    });
}

criterion_group!(benches, bench_lod_pipeline);
criterion_main!(benches);
