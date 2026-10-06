//! Server-authoritative world management, terrain generation, lighting caching, and persistence.

use hashbrown::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use telos_core::coords::{BlockPos, ChunkPos};
use telos_lod::{
    color::LodColorTable,
    coords::LodNodeKey,
    mesher::{LodMesh, mesh_lod_node},
    node::LodNode,
    pyramid::LodPyramid,
};
use telos_voxel::{
    chunk::{Chunk, ChunkSnapshot},
    coords::split_block_pos,
    light::{ChunkHeightmap, ChunkLight, ColumnHeights, LightBfs},
    registry::BlockRegistry,
    state::{BlockStateId, StateFlags},
};
use telos_worldgen::{GeneratorKind, WorldGenerator};

use crate::multi_world::WorldConfig;
use crate::storage::WorldStorage;

/// Actively tracked chunk containing mutable block storage and its latest published immutable snapshot.
pub struct ServerChunk {
    /// Mutable chunk storage.
    pub chunk: Chunk,
    /// Latest published immutable snapshot shared across network and mesher threads.
    pub snapshot: Arc<ChunkSnapshot>,
}

/// Server-side world storage, procedural generator, lighting cache, and LOD pyramid.
pub struct ServerWorld {
    name: String,
    generator: WorldGenerator,
    registry: BlockRegistry,
    chunks: HashMap<ChunkPos, ServerChunk>,
    columns: HashMap<(i32, i32), ColumnHeights>,
    light_bfs: LightBfs,
    pyramid: LodPyramid,
    lod_color_table: LodColorTable,
    lod_meshes: HashMap<LodNodeKey, Arc<LodMesh>>,
    storage: Option<WorldStorage>,
    dirty_chunks: HashSet<ChunkPos>,
}

impl ServerWorld {
    /// Creates a new in-memory `ServerWorld` with the given seed and block registry.
    #[must_use]
    pub fn new(seed: u64, registry: BlockRegistry) -> Self {
        Self::with_generator("overworld", seed, registry, GeneratorKind::Standard)
    }

    /// Creates a world with a specific name, seed, registry, and generator kind.
    #[must_use]
    pub fn with_generator(
        name: impl Into<String>,
        seed: u64,
        registry: BlockRegistry,
        kind: GeneratorKind,
    ) -> Self {
        let generator = WorldGenerator::with_kind(seed, &registry, kind);
        Self {
            name: name.into(),
            generator,
            registry,
            chunks: HashMap::new(),
            columns: HashMap::new(),
            light_bfs: LightBfs::new(),
            pyramid: LodPyramid::new(),
            lod_color_table: LodColorTable::standard(),
            lod_meshes: HashMap::new(),
            storage: None,
            dirty_chunks: HashSet::new(),
        }
    }

    /// Creates a persistent `ServerWorld` with `.tlr` region container saving enabled.
    pub fn with_storage(
        seed: u64,
        registry: BlockRegistry,
        save_dir: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let storage = WorldStorage::new(save_dir)?;
        let generator = WorldGenerator::new(seed, &registry);
        Ok(Self {
            name: "overworld".to_string(),
            generator,
            registry,
            chunks: HashMap::new(),
            columns: HashMap::new(),
            light_bfs: LightBfs::new(),
            pyramid: LodPyramid::new(),
            lod_color_table: LodColorTable::standard(),
            lod_meshes: HashMap::new(),
            storage: Some(storage),
            dirty_chunks: HashSet::new(),
        })
    }

    /// Creates a persistent `ServerWorld` with `.tlr` region container saving and full save-ID mapping.
    pub fn with_content_storage(
        seed: u64,
        registries: &telos_content::FrozenRegistries,
        save_dir: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let storage = WorldStorage::open_or_create(save_dir, registries)?;
        let registry = registries.block_registry().clone();
        let generator = WorldGenerator::new(seed, &registry);
        Ok(Self {
            name: "overworld".to_string(),
            generator,
            registry,
            chunks: HashMap::new(),
            columns: HashMap::new(),
            light_bfs: LightBfs::new(),
            pyramid: LodPyramid::new(),
            lod_color_table: LodColorTable::standard(),
            lod_meshes: HashMap::new(),
            storage: Some(storage),
            dirty_chunks: HashSet::new(),
        })
    }

    /// Creates a `ServerWorld` configured from a `WorldConfig`.
    pub fn with_config(
        config: &WorldConfig,
        registries: &telos_content::FrozenRegistries,
    ) -> std::io::Result<Self> {
        let registry = registries.block_registry().clone();
        let generator = WorldGenerator::with_kind(config.seed, &registry, config.generator);
        let storage = if let Some(ref dir) = config.save_directory {
            Some(WorldStorage::open_or_create(dir, registries)?)
        } else {
            None
        };

        Ok(Self {
            name: config.name.clone(),
            generator,
            registry,
            chunks: HashMap::new(),
            columns: HashMap::new(),
            light_bfs: LightBfs::new(),
            pyramid: LodPyramid::new(),
            lod_color_table: LodColorTable::standard(),
            lod_meshes: HashMap::new(),
            storage,
            dirty_chunks: HashSet::new(),
        })
    }

    /// Returns the world name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the active generator algorithm kind.
    #[must_use]
    pub const fn generator_kind(&self) -> GeneratorKind {
        self.generator.kind()
    }

    /// Returns the world generation seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.generator.seed()
    }

    /// Accesses the server's block registry.
    #[must_use]
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }

    /// Retrieves an existing chunk snapshot or generates, lights, and caches it.
    pub fn get_or_generate_chunk(&mut self, pos: ChunkPos) -> Arc<ChunkSnapshot> {
        if let Some(sc) = self.chunks.get(&pos) {
            return sc.snapshot.clone();
        }

        // 1. Try loading from persistent storage if configured
        if let Some(storage) = &mut self.storage
            && let Ok(Some(mut chunk)) = storage.load_chunk(pos, &self.registry)
        {
            let col_key = (pos.x(), pos.z());
            let col_heights = self.columns.entry(col_key).or_default();
            let local_h = ChunkHeightmap::from_occupancy(chunk.occupancy());
            col_heights.update_chunk(pos.y(), &local_h);

            if chunk.light().is_none() {
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
                Self::diffuse_cross_chunk_boundary(
                    &self.chunks,
                    pos,
                    &mut chunk_light,
                    &mut self.light_bfs,
                    is_opaque,
                );
                chunk_light.try_collapse();
                chunk.set_light(Some(chunk_light));
            }

            let snapshot = chunk.publish_snapshot();
            self.chunks.insert(
                pos,
                ServerChunk {
                    chunk,
                    snapshot: snapshot.clone(),
                },
            );
            return snapshot;
        }

        // 2. Generate chunk terrain
        let mut chunk = self.generator.generate_chunk(pos);

        // 3. Update column heightmap
        let col_key = (pos.x(), pos.z());
        let col_heights = self.columns.entry(col_key).or_default();
        let local_h = ChunkHeightmap::from_occupancy(chunk.occupancy());
        col_heights.update_chunk(pos.y(), &local_h);

        // 4. Compute initial sky light
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
        Self::diffuse_cross_chunk_boundary(
            &self.chunks,
            pos,
            &mut chunk_light,
            &mut self.light_bfs,
            is_opaque,
        );
        chunk_light.try_collapse();
        chunk.set_light(Some(chunk_light));

        // 5. Publish snapshot and cache
        let snapshot = chunk.publish_snapshot();
        if self.storage.is_some() {
            self.dirty_chunks.insert(pos);
        }
        self.chunks.insert(
            pos,
            ServerChunk {
                chunk,
                snapshot: snapshot.clone(),
            },
        );

        snapshot
    }

    /// Queries the block state at a world `BlockPos`, generating the chunk if missing.
    pub fn get_block(&mut self, pos: BlockPos) -> BlockStateId {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        let snapshot = self.get_or_generate_chunk(chunk_pos);
        snapshot.blocks().get(local_idx)
    }

    /// Sets a block in the world, updating chunk storage, heightmaps, and lighting.
    ///
    /// Invalidates affected LOD pyramid levels and flags the chunk dirty for save.
    /// Returns `Some((snapshot, version))` if state changed, or `None` if unchanged or out of bounds.
    pub fn set_block(
        &mut self,
        pos: BlockPos,
        new_state: BlockStateId,
    ) -> Option<(Arc<ChunkSnapshot>, u64)> {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        self.get_or_generate_chunk(chunk_pos);

        let registry = &self.registry;
        let sc = self.chunks.get_mut(&chunk_pos)?;

        let old_state = sc.chunk.get(local_idx);
        if old_state == new_state {
            return None;
        }

        let old_flags = registry.flags(old_state);
        let new_flags = registry.flags(new_state);

        let changed = sc.chunk.set(local_idx, new_state, old_flags, new_flags, 1);
        if !changed {
            return None;
        }

        // Update column heightmap
        let col_key = (chunk_pos.x(), chunk_pos.z());
        let col_heights = self.columns.entry(col_key).or_default();
        let local_h = ChunkHeightmap::from_occupancy(sc.chunk.occupancy());
        col_heights.update_chunk(chunk_pos.y(), &local_h);

        // Update lighting
        let mut chunk_light = sc.chunk.light().cloned().unwrap_or_default();
        let is_opaque = |idx: usize| -> u8 {
            let x = (idx & 0x1F) as u32;
            let z = ((idx >> 5) & 0x1F) as u32;
            let y = ((idx >> 10) & 0x1F) as u32;
            if sc.chunk.occupancy().is_solid(x, y, z) {
                15
            } else {
                0
            }
        };

        let was_opaque = old_flags.contains(StateFlags::OPAQUE_FULL);
        let is_now_opaque = new_flags.contains(StateFlags::OPAQUE_FULL);

        if was_opaque != is_now_opaque {
            self.light_bfs.compute_initial_sky_light(
                chunk_pos.y(),
                col_heights,
                &mut chunk_light.sky,
                is_opaque,
            );
        }

        let lx = local_idx.x();
        let ly = local_idx.y();
        let lz = local_idx.z();

        if new_flags.contains(StateFlags::EMITS_LIGHT) {
            self.light_bfs
                .add_source(&mut chunk_light.block, lx, ly, lz, 14);
            self.light_bfs
                .propagate_block_add(&mut chunk_light.block, is_opaque);
        } else if old_flags.contains(StateFlags::EMITS_LIGHT) {
            self.light_bfs
                .remove_source(&mut chunk_light.block, lx, ly, lz);
            self.light_bfs
                .propagate_block_remove(&mut chunk_light.block, is_opaque);
        }

        chunk_light.try_collapse();
        sc.chunk.set_light(Some(chunk_light));

        let snapshot = sc.chunk.publish_snapshot();
        let version = snapshot.content_version();
        sc.snapshot = snapshot.clone();

        self.dirty_chunks.insert(chunk_pos);
        self.invalidate_lod_hierarchy(chunk_pos);

        Some((snapshot, version))
    }

    /// Invalidates LOD pyramid nodes and cached meshes containing the given chunk.
    pub fn invalidate_lod_hierarchy(&mut self, chunk_pos: ChunkPos) {
        let mut key = LodNodeKey::from_chunk(chunk_pos).parent();
        for _ in 1..=4 {
            self.pyramid.remove(&key);
            self.lod_meshes.remove(&key);
            key = key.parent();
        }
    }

    /// Saves all dirty chunks to `.tlr` region files if storage is configured.
    /// Returns the number of chunks saved.
    pub fn save_dirty_chunks(&mut self) -> Result<usize, telos_storage::StorageError> {
        let Some(storage) = &mut self.storage else {
            self.dirty_chunks.clear();
            return Ok(0);
        };

        if self.dirty_chunks.is_empty() {
            return Ok(0);
        }

        let chunks_to_save: Vec<(ChunkPos, &Chunk)> = self
            .dirty_chunks
            .iter()
            .filter_map(|pos| self.chunks.get(pos).map(|sc| (*pos, &sc.chunk)))
            .collect();

        let count = storage.save_chunks(chunks_to_save)?;
        self.dirty_chunks.clear();
        Ok(count)
    }

    /// Flushes all open region files to disk.
    pub fn flush_storage(&mut self) -> Result<(), telos_storage::StorageError> {
        if let Some(storage) = &mut self.storage {
            storage.flush()?;
        }
        Ok(())
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

    /// Returns the world Y coordinate of the highest solid block at (x, z), or 64 as fallback.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_surface_y(&self, x: i32, z: i32) -> i32 {
        let chunk_x = x.div_euclid(32);
        let chunk_z = z.div_euclid(32);
        let local_x = x.rem_euclid(32) as u32;
        let local_z = z.rem_euclid(32) as u32;
        if let Some(col) = self.columns.get(&(chunk_x, chunk_z)) {
            let top = col.get_top(local_x, local_z);
            if top > i16::MIN {
                return i32::from(top);
            }
        }
        64
    }

    /// Returns (`sky_light`, `block_light`) at pos.
    #[must_use]
    pub fn get_light(&self, pos: BlockPos) -> (u8, u8) {
        let (cpos, lpos) = split_block_pos(pos);
        if let Some(sc) = self.chunks.get(&cpos)
            && let Some(light) = sc.snapshot.light()
        {
            return (
                light.get_sky(lpos.as_usize()),
                light.get_block(lpos.as_usize()),
            );
        }
        (15, 0)
    }

    /// Propagates sky light from already generated cardinal neighbor chunks across boundaries.
    #[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
    fn diffuse_cross_chunk_boundary<F>(
        chunks: &HashMap<ChunkPos, ServerChunk>,
        pos: ChunkPos,
        chunk_light: &mut ChunkLight,
        light_bfs: &mut LightBfs,
        mut is_opaque: F,
    ) where
        F: FnMut(usize) -> u8,
    {
        let neighbors = [
            (ChunkPos::new(pos.x() - 1, pos.y(), pos.z()), -1, 0, 0),
            (ChunkPos::new(pos.x() + 1, pos.y(), pos.z()), 1, 0, 0),
            (ChunkPos::new(pos.x(), pos.y() - 1, pos.z()), 0, -1, 0),
            (ChunkPos::new(pos.x(), pos.y() + 1, pos.z()), 0, 1, 0),
            (ChunkPos::new(pos.x(), pos.y(), pos.z() - 1), 0, 0, -1),
            (ChunkPos::new(pos.x(), pos.y(), pos.z() + 1), 0, 0, 1),
        ];

        let mut seeded = false;
        for (n_pos, dx, dy, dz) in neighbors {
            if let Some(sc) = chunks.get(&n_pos)
                && let Some(nl) = sc.chunk.light()
            {
                if dx == -1 {
                    for y in 0..32u32 {
                        for z in 0..32u32 {
                            let n_idx = ((y << 10) | (z << 5) | 31) as usize;
                            let c_idx = ((y << 10) | (z << 5)) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(0, y as i32, z as i32);
                                    seeded = true;
                                }
                            }
                        }
                    }
                } else if dx == 1 {
                    for y in 0..32u32 {
                        for z in 0..32u32 {
                            let n_idx = ((y << 10) | (z << 5)) as usize;
                            let c_idx = ((y << 10) | (z << 5) | 31) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(31, y as i32, z as i32);
                                    seeded = true;
                                }
                            }
                        }
                    }
                } else if dy == -1 {
                    for x in 0..32u32 {
                        for z in 0..32u32 {
                            let n_idx = ((31 << 10) | (z << 5) | x) as usize;
                            let c_idx = ((z << 5) | x) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(x as i32, 0, z as i32);
                                    seeded = true;
                                }
                            }
                        }
                    }
                } else if dy == 1 {
                    for x in 0..32u32 {
                        for z in 0..32u32 {
                            let n_idx = ((z << 5) | x) as usize;
                            let c_idx = ((31 << 10) | (z << 5) | x) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(x as i32, 31, z as i32);
                                    seeded = true;
                                }
                            }
                        }
                    }
                } else if dz == -1 {
                    for y in 0..32u32 {
                        for x in 0..32u32 {
                            let n_idx = ((y << 10) | (31 << 5) | x) as usize;
                            let c_idx = ((y << 10) | x) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(x as i32, y as i32, 0);
                                    seeded = true;
                                }
                            }
                        }
                    }
                } else if dz == 1 {
                    for y in 0..32u32 {
                        for x in 0..32u32 {
                            let n_idx = ((y << 10) | x) as usize;
                            let c_idx = ((y << 10) | (31 << 5) | x) as usize;
                            let n_sky = nl.sky.get(n_idx);
                            if n_sky > 1 && is_opaque(c_idx) == 0 {
                                let target = n_sky - 1;
                                if chunk_light.sky.get(c_idx) < target {
                                    chunk_light.sky.set(c_idx, target);
                                    light_bfs.enqueue_add(x as i32, y as i32, 31);
                                    seeded = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if seeded {
            light_bfs.propagate_block_add(&mut chunk_light.sky, is_opaque);
        }
    }
}
