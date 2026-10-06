//! Storage IO abstractions for region containers and crash simulation.

use crate::error::{Result, StorageError};
use std::sync::{Arc, RwLock};

/// Storage IO interface for positional reads, writes, and sync barriers.
pub trait RegionIo: Send + Sync {
    /// Read exact number of bytes into `buf` starting at `offset`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Write all bytes from `buf` starting at `offset`.
    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()>;

    /// Synchronize modified data to durable storage.
    fn sync_data(&self) -> Result<()>;

    /// Returns the current byte length of the container.
    fn len(&self) -> Result<u64>;

    /// Truncates or extends the container to `size` bytes.
    fn set_len(&self, size: u64) -> Result<()>;

    /// Checks if the container is empty (0 bytes).
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// OS filesystem implementation using positional `pread` / `pwrite`.
pub struct FileRegionIo {
    file: std::fs::File,
}

impl FileRegionIo {
    /// Creates a new `FileRegionIo` wrapping an open `std::fs::File`.
    #[must_use]
    pub fn new(file: std::fs::File) -> Self {
        Self { file }
    }

    /// Access the underlying `std::fs::File`.
    #[must_use]
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }
}

#[cfg(unix)]
impl RegionIo for FileRegionIo {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        self.file
            .read_exact_at(buf, offset)
            .map_err(StorageError::Io)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        self.file
            .write_all_at(buf, offset)
            .map_err(StorageError::Io)
    }

    fn sync_data(&self) -> Result<()> {
        self.file.sync_data().map_err(StorageError::Io)
    }

    fn len(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|m| m.len())
            .map_err(StorageError::Io)
    }

    fn set_len(&self, size: u64) -> Result<()> {
        self.file.set_len(size).map_err(StorageError::Io)
    }
}

#[cfg(not(unix))]
impl RegionIo for FileRegionIo {
    fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        compile_error!("Non-unix platforms require platform-specific FileExt implementation");
    }

    fn write_at(&self, _offset: u64, _buf: &[u8]) -> Result<()> {
        compile_error!("Non-unix platforms require platform-specific FileExt implementation");
    }

    fn sync_data(&self) -> Result<()> {
        self.file.sync_data().map_err(StorageError::Io)
    }

    fn len(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|m| m.len())
            .map_err(StorageError::Io)
    }

    fn set_len(&self, size: u64) -> Result<()> {
        self.file.set_len(size).map_err(StorageError::Io)
    }
}

/// Inner state of `SimFs` representing in-memory storage and crash state.
#[derive(Debug, Clone)]
struct SimFsState {
    /// In-memory bytes currently modified by writes.
    data: Vec<u8>,
    /// Durable snapshot as of the last successful `sync_data`.
    synced_data: Vec<u8>,
    /// Number of operations remaining before an intentional crash/failure.
    ops_until_failure: Option<usize>,
}

/// In-memory storage implementation for tests, benchmarks, and crash simulation.
#[derive(Debug, Clone)]
pub struct SimFs {
    state: Arc<RwLock<SimFsState>>,
}

impl Default for SimFs {
    fn default() -> Self {
        Self::new()
    }
}

impl SimFs {
    /// Creates a new empty in-memory simulation container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(SimFsState {
                data: Vec::new(),
                synced_data: Vec::new(),
                ops_until_failure: None,
            })),
        }
    }

    /// Creates a simulation container with initial byte content.
    #[must_use]
    pub fn with_bytes(bytes: &[u8]) -> Self {
        Self {
            state: Arc::new(RwLock::new(SimFsState {
                data: bytes.to_vec(),
                synced_data: bytes.to_vec(),
                ops_until_failure: None,
            })),
        }
    }

    /// Injects an upcoming crash/error after `ops` number of write or sync operations.
    pub fn fail_after_ops(&self, ops: usize) {
        let mut guard = self.state.write().expect("lock poisoned");
        guard.ops_until_failure = Some(ops);
    }

    /// Simulates sudden power loss or process termination:
    /// Reverts all uncommitted (unsynced) writes to the durable snapshot.
    pub fn simulate_crash(&self) {
        let mut guard = self.state.write().expect("lock poisoned");
        let guard = &mut *guard;
        guard.data.clone_from(&guard.synced_data);
        guard.ops_until_failure = None;
    }

    /// Returns a copy of the current in-memory bytes (including unsynced writes).
    #[must_use]
    pub fn dump_current_bytes(&self) -> Vec<u8> {
        let guard = self.state.read().expect("lock poisoned");
        guard.data.clone()
    }

    /// Returns a copy of the synced (durable) bytes.
    #[must_use]
    pub fn dump_synced_bytes(&self) -> Vec<u8> {
        let guard = self.state.read().expect("lock poisoned");
        guard.synced_data.clone()
    }

    /// Corrupts or truncates the underlying data directly (for fuzz/recovery tests).
    pub fn corrupt_bytes(&self, offset: usize, corrupt_data: &[u8]) {
        let mut guard = self.state.write().expect("lock poisoned");
        let guard = &mut *guard;
        let end = offset + corrupt_data.len();
        if guard.data.len() < end {
            guard.data.resize(end, 0);
        }
        guard.data[offset..end].copy_from_slice(corrupt_data);
        guard.synced_data.clone_from(&guard.data);
    }
}

impl RegionIo for SimFs {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let guard = self.state.read().expect("lock poisoned");
        let offset = usize::try_from(offset)
            .map_err(|_| StorageError::Io(std::io::Error::other("Offset overflow")))?;
        let end = offset + buf.len();

        if offset > guard.data.len() || end > guard.data.len() {
            return Err(StorageError::Truncated {
                actual: guard.data.len() as u64,
                expected: end as u64,
            });
        }

        buf.copy_from_slice(&guard.data[offset..end]);
        Ok(())
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        let mut guard = self.state.write().expect("lock poisoned");

        if let Some(ref mut ops) = guard.ops_until_failure {
            if *ops == 0 {
                return Err(StorageError::Io(std::io::Error::other(
                    "Simulated IO crash during write",
                )));
            }
            *ops -= 1;
        }

        let offset = usize::try_from(offset)
            .map_err(|_| StorageError::Io(std::io::Error::other("Offset overflow")))?;
        let end = offset + buf.len();

        if guard.data.len() < end {
            guard.data.resize(end, 0);
        }
        guard.data[offset..end].copy_from_slice(buf);
        Ok(())
    }

    fn sync_data(&self) -> Result<()> {
        let mut guard = self.state.write().expect("lock poisoned");

        if let Some(ref mut ops) = guard.ops_until_failure {
            if *ops == 0 {
                return Err(StorageError::Io(std::io::Error::other(
                    "Simulated IO crash during sync",
                )));
            }
            *ops -= 1;
        }

        let guard = &mut *guard;
        guard.synced_data.clone_from(&guard.data);
        Ok(())
    }

    fn len(&self) -> Result<u64> {
        let guard = self.state.read().expect("lock poisoned");
        Ok(guard.data.len() as u64)
    }

    fn set_len(&self, size: u64) -> Result<()> {
        let mut guard = self.state.write().expect("lock poisoned");
        let size = usize::try_from(size)
            .map_err(|_| StorageError::Io(std::io::Error::other("Size overflow")))?;
        guard.data.resize(size, 0);
        Ok(())
    }
}
