//! Server world persistence using `.tlr` region files via `telos-storage`.

use hashbrown::HashMap;
use std::fs::{OpenOptions, create_dir_all};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use telos_content::{FrozenRegistries, RegistryRemap, WorldRegistryMap};
use telos_core::coords::ChunkPos;
use telos_storage::format::header::CodecId;
use telos_storage::format::section::{ChunkPayload, ChunkStatus};
use telos_storage::io::{FileRegionIo, RegionIo};
use telos_storage::region::{RegionFile, RegionPos};
use telos_voxel::chunk::Chunk;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;
use tracing::{debug, error};

/// Server storage backend managing cubic `.tlr` region containers.
pub struct WorldStorage {
    regions_dir: PathBuf,
    regions: HashMap<RegionPos, RegionFile<FileRegionIo>>,
    save_map: Option<WorldRegistryMap>,
    remap: Option<RegistryRemap>,
}

impl WorldStorage {
    /// Initializes `WorldStorage` at the given root directory (e.g. `saves/world`) without save-ID remapping.
    pub fn new(root_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let regions_dir = root_dir.as_ref().join("regions");
        create_dir_all(&regions_dir)?;
        Ok(Self {
            regions_dir,
            regions: HashMap::new(),
            save_map: None,
            remap: None,
        })
    }

    /// Initializes `WorldStorage` at root directory with full save-ID mapping and palette remapping.
    pub fn open_or_create(
        root_dir: impl AsRef<Path>,
        registries: &FrozenRegistries,
    ) -> std::io::Result<Self> {
        let root = root_dir.as_ref();
        let regions_dir = root.join("regions");
        create_dir_all(&regions_dir)?;

        let save_map = WorldRegistryMap::open_or_create(root, registries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let remap = save_map.build_remap(registries);

        Ok(Self {
            regions_dir,
            regions: HashMap::new(),
            save_map: Some(save_map),
            remap: Some(remap),
        })
    }

    /// Returns a reference to the active `WorldRegistryMap`, if initialized.
    #[must_use]
    pub fn save_map(&self) -> Option<&WorldRegistryMap> {
        self.save_map.as_ref()
    }

    /// Retrieves an open region file or opens/creates it.
    fn get_or_open_region(
        &mut self,
        rpos: RegionPos,
    ) -> Result<&mut RegionFile<FileRegionIo>, telos_storage::StorageError> {
        if !self.regions.contains_key(&rpos) {
            let filename = rpos.filename();
            let path = self.regions_dir.join(filename);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .map_err(telos_storage::StorageError::Io)?;
            let io = FileRegionIo::new(file);
            let region = RegionFile::open(io, rpos.x, rpos.y, rpos.z)?;
            self.regions.insert(rpos, region);
        }
        Ok(self.regions.get_mut(&rpos).unwrap())
    }

    /// Attempts to load a chunk from disk.
    /// Returns `Ok(Some(chunk))` if found, or `Ok(None)` if chunk is not yet saved.
    pub fn load_chunk(
        &mut self,
        pos: ChunkPos,
        registry: &BlockRegistry,
    ) -> Result<Option<Chunk>, telos_storage::StorageError> {
        let rpos = RegionPos::from_chunk(pos);
        let region = self.get_or_open_region(rpos)?;
        let maybe_payload = region.read_chunk(pos)?;
        let Some(payload) = maybe_payload else {
            return Ok(None);
        };

        let blocks = if let Some(ref remap) = self.remap {
            remap.remap_blocks_save_to_runtime(payload.blocks)
        } else {
            payload.blocks
        };

        let is_opaque = |id: BlockStateId| {
            registry
                .flags(id)
                .contains(telos_voxel::state::StateFlags::OPAQUE_FULL)
        };
        let chunk = Chunk::from_blocks(pos, blocks, is_opaque);
        Ok(Some(chunk))
    }

    /// Commits a collection of modified chunks to their respective region files on disk.
    pub fn save_chunks<'a, I>(&mut self, chunks: I) -> Result<usize, telos_storage::StorageError>
    where
        I: IntoIterator<Item = (ChunkPos, &'a Chunk)>,
    {
        let mut grouped: HashMap<RegionPos, Vec<(ChunkPos, Option<ChunkPayload>)>> = HashMap::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as u32);

        let mut count = 0;
        for (pos, chunk) in chunks {
            let rpos = RegionPos::from_chunk(pos);
            let raw_blocks = chunk.to_blocks();
            let disk_blocks = if let Some(ref remap) = self.remap {
                remap.remap_blocks_runtime_to_save(&raw_blocks)
            } else {
                raw_blocks
            };
            let payload = ChunkPayload::new(disk_blocks, ChunkStatus::default());
            grouped.entry(rpos).or_default().push((pos, Some(payload)));
            count += 1;
        }

        for (rpos, chunk_list) in grouped {
            let region = self.get_or_open_region(rpos)?;
            region.commit_chunks(&chunk_list, CodecId::Zstd, timestamp)?;
            debug!("Committed {} chunks to region {:?}", chunk_list.len(), rpos);
        }

        Ok(count)
    }

    /// Flushes all open region files to durable storage via `fsync`.
    pub fn flush(&mut self) -> Result<(), telos_storage::StorageError> {
        for (rpos, region) in &mut self.regions {
            if let Err(err) = region.io_mut().sync_data() {
                error!("Failed to sync region {:?}: {err}", rpos);
                return Err(err);
            }
        }
        Ok(())
    }
}
