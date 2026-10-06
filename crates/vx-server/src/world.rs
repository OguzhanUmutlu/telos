//! Server-authoritative world management, terrain generation, and lighting caching.

use hashbrown::HashMap;
use std::sync::Arc;
use vx_core::coords::ChunkPos;
use vx_voxel::{
    chunk::ChunkSnapshot,
    light::{ChunkHeightmap, ChunkLight, ColumnHeights, LightBfs},
    registry::BlockRegistry,
};
use vx_worldgen::WorldGenerator;

/// Server-side world storage, procedural generator, and lighting cache.
pub struct ServerWorld {
    generator: WorldGenerator,
    registry: BlockRegistry,
    chunks: HashMap<ChunkPos, Arc<ChunkSnapshot>>,
    columns: HashMap<(i32, i32), ColumnHeights>,
    light_bfs: LightBfs,
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
}
