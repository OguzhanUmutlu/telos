//! Mesh containers with directional draw bucket ranges.

use crate::{
    light::LightPattern,
    quad::{FaceDir, T0Quad},
};

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

/// Contiguous mesh buffer containing packed `T0Quad` entries organized by cardinal direction,
/// followed by deduplicated `LightPattern` entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T0Mesh {
    /// Contiguous array of all quads in the chunk mesh.
    pub quads: Vec<T0Quad>,
    /// Deduplicated light pattern table referenced by `quads`.
    pub patterns: Vec<LightPattern>,
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
            patterns: Vec::new(),
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

    /// Total number of deduplicated light patterns.
    #[inline]
    #[must_use]
    pub fn total_patterns(&self) -> usize {
        self.patterns.len()
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

    /// Total size in bytes of the mesh quads and patterns in GPU memory.
    #[inline]
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.quads.len() * size_of::<T0Quad>() + self.patterns.len() * size_of::<LightPattern>()
    }

    /// Writes quads and light patterns into a contiguous `u32` buffer for GPU upload.
    ///
    /// Quads occupy indices `0..quads.len()*2`, patterns occupy `quads.len()*2..`.
    pub fn write_to_u32_buffer(&self, dst: &mut Vec<u32>) {
        dst.reserve(self.quads.len() * 2 + self.patterns.len() * 2);
        for q in &self.quads {
            dst.extend_from_slice(&q.raw());
        }
        for p in &self.patterns {
            dst.extend_from_slice(&p.raw);
        }
    }
}
