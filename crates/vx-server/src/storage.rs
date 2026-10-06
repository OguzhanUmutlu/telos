//! Server world persistence using `.vxr` region files via `vx-storage`.

use hashbrown::HashMap;
use std::fs::{OpenOptions, create_dir_all};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error};
use vx_core::coords::ChunkPos;
use vx_storage::format::header::CodecId;
use vx_storage::format::section::{ChunkPayload, ChunkStatus};
use vx_storage::io::{FileRegionIo, RegionIo};
use vx_storage::region::{RegionFile, RegionPos};
use vx_voxel::chunk::Chunk;
use vx_voxel::registry::BlockRegistry;
use vx_voxel::state::BlockStateId;

/// Server storage backend managing cubic `.vxr` region containers.
pub struct WorldStorage {
    regions_dir: PathBuf,
    regions: HashMap<RegionPos, RegionFile<FileRegionIo>>,
}

impl WorldStorage {
    /// Initializes `WorldStorage` at the given root directory (e.g. `saves/world`).
    pub fn new(root_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let regions_dir = root_dir.as_ref().join("regions");
        create_dir_all(&regions_dir)?;
        Ok(Self {
            regions_dir,
            regions: HashMap::new(),
        })
    }

    /// Retrieves an open region file or opens/creates it.
    fn get_or_open_region(
        &mut self,
        rpos: RegionPos,
    ) -> Result<&mut RegionFile<FileRegionIo>, vx_storage::StorageError> {
        if !self.regions.contains_key(&rpos) {
            let filename = format!("r.{}.{}.{}.vxr", rpos.x, rpos.y, rpos.z);
            let path = self.regions_dir.join(filename);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .map_err(vx_storage::StorageError::Io)?;
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
    ) -> Result<Option<Chunk>, vx_storage::StorageError> {
        let rpos = RegionPos::from_chunk(pos);
        let region = self.get_or_open_region(rpos)?;
        let maybe_payload = region.read_chunk(pos)?;
        let Some(payload) = maybe_payload else {
            return Ok(None);
        };

        let is_opaque = |id: BlockStateId| {
            registry
                .flags(id)
                .contains(vx_voxel::state::StateFlags::OPAQUE_FULL)
        };
        let chunk = Chunk::from_blocks(pos, payload.blocks, is_opaque);
        Ok(Some(chunk))
    }

    /// Commits a collection of modified chunks to their respective region files on disk.
    pub fn save_chunks<'a, I>(&mut self, chunks: I) -> Result<usize, vx_storage::StorageError>
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
            let payload = ChunkPayload::new(chunk.to_blocks(), ChunkStatus::default());
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
    pub fn flush(&mut self) -> Result<(), vx_storage::StorageError> {
        for (rpos, region) in &mut self.regions {
            if let Err(err) = region.io_mut().sync_data() {
                error!("Failed to sync region {:?}: {err}", rpos);
                return Err(err);
            }
        }
        Ok(())
    }
}
