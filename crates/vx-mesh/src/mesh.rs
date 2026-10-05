//! Mesh containers with directional draw bucket ranges.

use crate::quad::{FaceDir, T0Quad};

/// Slice range of quads within a contiguous quad buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuadRange {
    /// Offset in quads from start of buffer.
    pub offset: u32,
    /// Number of quads in this range.
    pub count: u32,
}

impl QuadRange {
    /// Creates a new `QuadRange`.
    #[inline]
    #[must_use]
    pub const fn new(offset: u32, count: u32) -> Self {
        Self { offset, count }
    }

    /// Whether this range contains 0 quads.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Contiguous mesh buffer containing packed `T0Quad` entries organized by cardinal direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T0Mesh {
    /// Contiguous array of all quads in the chunk mesh.
    pub quads: Vec<T0Quad>,
    /// Sub-ranges corresponding to each cardinal direction (+X, -X, +Y, -Y, +Z, -Z).
    pub ranges: [QuadRange; 6],
}

impl T0Mesh {
    /// Creates an empty mesh with 0 quads.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            quads: Vec::new(),
            ranges: [QuadRange {
                offset: 0,
                count: 0,
            }; 6],
        }
    }

    /// Total number of quads across all directions.
    #[inline]
    #[must_use]
    pub fn total_quads(&self) -> usize {
        self.quads.len()
    }

    /// Whether the mesh has 0 quads.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Returns the quad range for a specific cardinal direction.
    #[inline]
    #[must_use]
    pub fn range(&self, dir: FaceDir) -> QuadRange {
        self.ranges[dir as usize]
    }

    /// Returns a slice of all quads for a specific cardinal direction.
    #[inline]
    #[must_use]
    pub fn bucket_quads(&self, dir: FaceDir) -> &[T0Quad] {
        let range = self.ranges[dir as usize];
        let start = range.offset as usize;
        let end = start + (range.count as usize);
        &self.quads[start..end]
    }

    /// Total size in bytes of the mesh quads in GPU memory.
    #[inline]
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.quads.len() * size_of::<T0Quad>()
    }
}
