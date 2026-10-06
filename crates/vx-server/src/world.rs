//! Server-authoritative world management, terrain generation, and lighting caching.

use hashbrown::HashMap;
use std::sync::Arc;
use vx_core::coords::ChunkPos;
use vx_lod::{
    color::LodColorTable,
    coords::LodNodeKey,
    mesher::{LodMesh, mesh_lod_node},
    node::LodNode,
    pyramid::LodPyramid,
};
use vx_voxel::{
    chunk::ChunkSnapshot,
    light::{ChunkHeightmap, ChunkLight, ColumnHeights, LightBfs},
    registry::BlockRegistry,
};
use vx_worldgen::WorldGenerator;

/// Server-side world storage, procedural generator, lighting cache, and LOD pyramid.
pub struct ServerWorld {
    generator: WorldGenerator,
    registry: BlockRegistry,
    chunks: HashMap<ChunkPos, Arc<ChunkSnapshot>>,
    columns: HashMap<(i32, i32), ColumnHeights>,
    light_bfs: LightBfs,
    pyramid: LodPyramid,
    lod_color_table: LodColorTable,
    lod_meshes: HashMap<LodNodeKey, Arc<LodMesh>>,
}

impl ServerWorld {
    /// Creates a new `ServerWorld` with the given seed and block registry.
    #[must_use]
    pub fn new(seed: u64, registry: BlockRegistry) -> Self {
        let generator = WorldGenerator::new(seed, &registry);
        Self {
            generator,
            registry,
            chunks: HashMap::new(),
            columns: HashMap::new(),
            light_bfs: LightBfs::new(),
            pyramid: LodPyramid::new(),
            lod_color_table: LodColorTable::standard(),
            lod_meshes: HashMap::new(),
        }
    }

    /// Accesses the server's block registry.
    #[must_use]
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }

    /// Retrieves an existing chunk snapshot or generates, lights, and caches it.
    pub fn get_or_generate_chunk(&mut self, pos: ChunkPos) -> Arc<ChunkSnapshot> {
        if let Some(snapshot) = self.chunks.get(&pos) {
            return snapshot.clone();
        }

        // 1. Generate chunk terrain
        let mut chunk = self.generator.generate_chunk(pos);

        // 2. Update column heightmap
        let col_key = (pos.x(), pos.z());
        let col_heights = self.columns.entry(col_key).or_default();
        let local_h = ChunkHeightmap::from_occupancy(chunk.occupancy());
        col_heights.update_chunk(pos.y(), &local_h);

        // 3. Compute initial sky light
        let mut chunk_light = ChunkLight::default();
        let is_opaque = |idx: usize| -> u8 {
            let x = (idx & 0x1F) as u32;
            let z = ((idx >> 5) & 0x1F) as u32;
            let y = ((idx >> 10) & 0x1F) as u32;
            if chunk.occupancy().is_solid(x, y, z) {
                15
            } else {
                0
            }
        };

        self.light_bfs.compute_initial_sky_light(
            pos.y(),
            col_heights,
            &mut chunk_light.sky,
            is_opaque,
        );
        chunk_light.try_collapse();
        chunk.set_light(Some(chunk_light));

        // 4. Publish snapshot and cache
        let snapshot = chunk.publish_snapshot();
        self.chunks.insert(pos, snapshot.clone());

        snapshot
    }

    /// Returns the number of currently cached chunks.
    #[must_use]
    pub fn cached_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the number of currently cached LOD nodes.
    #[must_use]
    pub fn cached_lod_count(&self) -> usize {
        self.pyramid.len()
    }

    /// Ensures that an LOD node is downsampled in the pyramid, generating chunks/children recursively as needed.
    pub fn ensure_lod_node(&mut self, key: LodNodeKey) -> Arc<LodNode> {
        if let Some(node) = self.pyramid.get(&key) {
            return node;
        }

        if key.level == 1 {
            // Level 1 children are Level 0 chunks
            let mut snapshots = Vec::with_capacity(8);
            for dy in 0..2 {
                for dz in 0..2 {
                    for dx in 0..2 {
                        let child_pos =
                            ChunkPos::new(key.x * 2 + dx, key.y * 2 + dy, key.z * 2 + dz);
                        snapshots.push(self.get_or_generate_chunk(child_pos));
                    }
                }
            }
            let children: [Option<&Arc<ChunkSnapshot>>; 8] = [
                Some(&snapshots[0]),
                Some(&snapshots[1]),
                Some(&snapshots[2]),
                Some(&snapshots[3]),
                Some(&snapshots[4]),
                Some(&snapshots[5]),
                Some(&snapshots[6]),
                Some(&snapshots[7]),
            ];
            self.pyramid.downsample_and_store(key, &children)
        } else {
            // Level L >= 2 children are Level L-1 LodNodes
            let mut children_nodes = Vec::with_capacity(8);
            for dy in 0..2 {
                for dz in 0..2 {
                    for dx in 0..2 {
                        let child_key = LodNodeKey::new(
                            key.level - 1,
                            key.x * 2 + dx,
                            key.y * 2 + dy,
                            key.z * 2 + dz,
                        );
                        children_nodes.push(self.ensure_lod_node(child_key));
                    }
                }
            }
            let children: [Option<&Arc<LodNode>>; 8] = [
                Some(&children_nodes[0]),
                Some(&children_nodes[1]),
                Some(&children_nodes[2]),
                Some(&children_nodes[3]),
                Some(&children_nodes[4]),
                Some(&children_nodes[5]),
                Some(&children_nodes[6]),
                Some(&children_nodes[7]),
            ];
            self.pyramid.downsample_and_store(key, &children)
        }
    }

    /// Gets or builds a greedy-meshed LOD mesh for a given LOD node key.
    pub fn get_or_mesh_lod_node(&mut self, key: LodNodeKey) -> Arc<LodMesh> {
        if let Some(mesh) = self.lod_meshes.get(&key) {
            return mesh.clone();
        }

        let node = self.ensure_lod_node(key);
        let neighbors: [Option<&LodNode>; 6] = [None; 6];
        let mesh = mesh_lod_node(node.as_ref(), &neighbors, &self.lod_color_table);
        let arc = Arc::new(mesh);
        self.lod_meshes.insert(key, arc.clone());
        arc
    }
}
