//! Spatial coordinate primitives and chunk geometry for voxel worlds.

use glam::IVec3;
use std::fmt;

/// The edge length of a cubic chunk in blocks.
pub const CHUNK_EDGE: i32 = 32;

/// Bit shift corresponding to `log2(CHUNK_EDGE)`.
pub const CHUNK_SHIFT: i32 = 5;

/// Bit mask for local coordinate extraction: `CHUNK_EDGE - 1 = 31`.
pub const CHUNK_MASK: i32 = 31;

/// Total number of voxels in a 32³ cubic chunk: 32,768.
pub const CHUNK_VOLUME: usize = 32 * 32 * 32;

/// The edge length of a region file in chunks (ADR-17: 8×8×8 chunks = 256³ blocks).
pub const REGION_EDGE_CHUNKS: i32 = 8;

/// Bit shift for chunk-to-region conversions (`log2(8) = 3`).
pub const REGION_SHIFT: i32 = 3;

/// Bit mask for chunk-in-region index: `REGION_EDGE_CHUNKS - 1 = 7`.
pub const REGION_MASK: i32 = 7;

/// Total number of chunks in a region: 8×8×8 = 512 chunks.
pub const REGION_VOLUME_CHUNKS: usize = 8 * 8 * 8;

/// Cardinal face directions in 3D voxel space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Face {
    /// -Y direction (bottom)
    Down = 0,
    /// +Y direction (top)
    Up = 1,
    /// -Z direction (north in standard Classic Voxel/OpenGL orientation)
    North = 2,
    /// +Z direction (south)
    South = 3,
    /// -X direction (west)
    West = 4,
    /// +X direction (east)
    East = 5,
}

impl Face {
    /// All six cardinal directions in canonical order.
    pub const ALL: [Face; 6] = [
        Face::Down,
        Face::Up,
        Face::North,
        Face::South,
        Face::West,
        Face::East,
    ];

    /// Returns the opposite face.
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Face::Down => Face::Up,
            Face::Up => Face::Down,
            Face::North => Face::South,
            Face::South => Face::North,
            Face::West => Face::East,
            Face::East => Face::West,
        }
    }

    /// Returns the unit normal vector as `IVec3`.
    #[inline]
    #[must_use]
    pub const fn normal_ivec(self) -> IVec3 {
        match self {
            Face::Down => IVec3::new(0, -1, 0),
            Face::Up => IVec3::new(0, 1, 0),
            Face::North => IVec3::new(0, 0, -1),
            Face::South => IVec3::new(0, 0, 1),
            Face::West => IVec3::new(-1, 0, 0),
            Face::East => IVec3::new(1, 0, 0),
        }
    }

    /// Returns the unit normal vector as `(i32, i32, i32)`.
    #[inline]
    #[must_use]
    pub const fn normal(self) -> (i32, i32, i32) {
        match self {
            Face::Down => (0, -1, 0),
            Face::Up => (0, 1, 0),
            Face::North => (0, 0, -1),
            Face::South => (0, 0, 1),
            Face::West => (-1, 0, 0),
            Face::East => (1, 0, 0),
        }
    }

    /// Returns the axis index: 0 for X, 1 for Y, 2 for Z.
    #[inline]
    #[must_use]
    pub const fn axis_index(self) -> usize {
        match self {
            Face::West | Face::East => 0,
            Face::Down | Face::Up => 1,
            Face::North | Face::South => 2,
        }
    }

    /// Returns true if this face points in the positive axis direction (+X, +Y, +Z).
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        match self {
            Face::Up | Face::South | Face::East => true,
            Face::Down | Face::North | Face::West => false,
        }
    }
}

/// A 3D world-space block coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct BlockPos(pub IVec3);

impl BlockPos {
    /// Creates a new `BlockPos` from integer coordinates.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// Returns the X coordinate.
    #[inline]
    pub const fn x(self) -> i32 {
        self.0.x
    }

    /// Returns the Y coordinate.
    #[inline]
    pub const fn y(self) -> i32 {
        self.0.y
    }

    /// Returns the Z coordinate.
    #[inline]
    pub const fn z(self) -> i32 {
        self.0.z
    }

    /// Offsets this position by a given direction.
    #[inline]
    #[must_use]
    pub fn offset(self, face: Face) -> Self {
        Self(self.0 + face.normal_ivec())
    }

    /// Decomposes this block position into its containing `ChunkPos` and chunk-local `LocalPos`.
    ///
    /// This uses arithmetic right shift (`>> 5`), which correctly handles negative numbers:
    /// `-1 >> 5 == -1`, and `-1 & 31 == 31`.
    #[inline]
    #[must_use]
    pub fn to_chunk_and_local(self) -> (ChunkPos, LocalPos) {
        let cx = self.0.x >> CHUNK_SHIFT;
        let cy = self.0.y >> CHUNK_SHIFT;
        let cz = self.0.z >> CHUNK_SHIFT;

        let lx = (self.0.x & CHUNK_MASK) as u8;
        let ly = (self.0.y & CHUNK_MASK) as u8;
        let lz = (self.0.z & CHUNK_MASK) as u8;

        (ChunkPos::new(cx, cy, cz), LocalPos::from_xyz(lx, ly, lz))
    }

    /// Returns the containing `ChunkPos`.
    #[inline]
    #[must_use]
    pub fn chunk(self) -> ChunkPos {
        ChunkPos::new(
            self.0.x >> CHUNK_SHIFT,
            self.0.y >> CHUNK_SHIFT,
            self.0.z >> CHUNK_SHIFT,
        )
    }

    /// Returns the chunk-local `LocalPos`.
    #[inline]
    #[must_use]
    pub fn local(self) -> LocalPos {
        LocalPos::from_xyz(
            (self.0.x & CHUNK_MASK) as u8,
            (self.0.y & CHUNK_MASK) as u8,
            (self.0.z & CHUNK_MASK) as u8,
        )
    }
}

impl fmt::Display for BlockPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

impl PartialOrd for BlockPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockPos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .x
            .cmp(&other.0.x)
            .then_with(|| self.0.y.cmp(&other.0.y))
            .then_with(|| self.0.z.cmp(&other.0.z))
    }
}

/// A 3D chunk coordinate representing a 32³ cubic chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct ChunkPos(pub IVec3);

impl PartialOrd for ChunkPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkPos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .x
            .cmp(&other.0.x)
            .then_with(|| self.0.y.cmp(&other.0.y))
            .then_with(|| self.0.z.cmp(&other.0.z))
    }
}

impl ChunkPos {
    /// Creates a new `ChunkPos` from integer chunk coordinates.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// Returns the chunk X coordinate.
    #[inline]
    pub const fn x(self) -> i32 {
        self.0.x
    }

    /// Returns the chunk Y coordinate.
    #[inline]
    pub const fn y(self) -> i32 {
        self.0.y
    }

    /// Returns the chunk Z coordinate.
    #[inline]
    pub const fn z(self) -> i32 {
        self.0.z
    }

    /// Computes the world-space `BlockPos` for the minimum corner (0, 0, 0) of this chunk.
    #[inline]
    #[must_use]
    pub fn min_block_pos(self) -> BlockPos {
        BlockPos::new(
            self.0.x << CHUNK_SHIFT,
            self.0.y << CHUNK_SHIFT,
            self.0.z << CHUNK_SHIFT,
        )
    }

    /// Reconstructs the world-space `BlockPos` from this chunk and a chunk-local position.
    #[inline]
    #[must_use]
    pub fn block_pos(self, local: LocalPos) -> BlockPos {
        let (lx, ly, lz) = local.to_xyz();
        BlockPos::new(
            (self.0.x << CHUNK_SHIFT) | i32::from(lx),
            (self.0.y << CHUNK_SHIFT) | i32::from(ly),
            (self.0.z << CHUNK_SHIFT) | i32::from(lz),
        )
    }

    /// Decomposes this chunk position into a `RegionPos` and a chunk index in `[0, 511]`.
    #[inline]
    #[must_use]
    pub fn to_region_and_index(self) -> (RegionPos, u16) {
        let rx = self.0.x >> REGION_SHIFT;
        let ry = self.0.y >> REGION_SHIFT;
        let rz = self.0.z >> REGION_SHIFT;

        let lx = (self.0.x & REGION_MASK) as u16;
        let ly = (self.0.y & REGION_MASK) as u16;
        let lz = (self.0.z & REGION_MASK) as u16;

        let index = (ly << 6) | (lz << 3) | lx;
        (RegionPos::new(rx, ry, rz), index)
    }

    /// Returns the neighbor chunk in the given cardinal direction.
    #[inline]
    #[must_use]
    pub fn offset(self, face: Face) -> Self {
        Self(self.0 + face.normal_ivec())
    }

    /// Chebyshev distance to another chunk (maximum coordinate difference).
    #[inline]
    #[must_use]
    pub fn chebyshev_distance(self, other: Self) -> i32 {
        let d = (self.0 - other.0).abs();
        d.x.max(d.y).max(d.z)
    }
}

impl fmt::Display for ChunkPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chunk({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// A 15-bit packed chunk-local block position: `(y << 10) | (z << 5) | x`.
///
/// $x, y, z \in [0, 31]$. The index is in $[0, 32{,}767]$, fitting inside a `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct LocalPos(pub u16);

impl LocalPos {
    /// Creates a `LocalPos` from chunk-local coordinates in `0..32`.
    ///
    /// # Panics
    /// In debug mode, asserts that `x < 32`, `y < 32`, and `z < 32`.
    #[inline]
    #[must_use]
    pub const fn from_xyz(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(x < 32 && y < 32 && z < 32, "Local coordinate out of range");
        let idx = ((y as u16) << 10) | ((z as u16) << 5) | (x as u16);
        Self(idx)
    }

    /// Creates a `LocalPos` directly from a raw 15-bit index.
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        debug_assert!(index < CHUNK_VOLUME, "Index exceeds chunk volume");
        Self(index as u16)
    }

    /// Returns the raw 15-bit index in `0..32768`.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the (x, y, z) components as `(u8, u8, u8)` in `0..32`.
    #[inline]
    pub const fn to_xyz(self) -> (u8, u8, u8) {
        let x = (self.0 & 31) as u8;
        let z = ((self.0 >> 5) & 31) as u8;
        let y = ((self.0 >> 10) & 31) as u8;
        (x, y, z)
    }

    /// Returns the local X coordinate (0..32).
    #[inline]
    pub const fn x(self) -> u8 {
        (self.0 & 31) as u8
    }

    /// Returns the local Y coordinate (0..32).
    #[inline]
    pub const fn y(self) -> u8 {
        ((self.0 >> 10) & 31) as u8
    }

    /// Returns the local Z coordinate (0..32).
    #[inline]
    pub const fn z(self) -> u8 {
        ((self.0 >> 5) & 31) as u8
    }
}

impl fmt::Display for LocalPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y, z) = self.to_xyz();
        write!(f, "Local({x}, {y}, {z})")
    }
}

/// A 3D region coordinate representing an 8×8×8 chunk storage container (256³ blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct RegionPos(pub IVec3);

impl PartialOrd for RegionPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionPos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .x
            .cmp(&other.0.x)
            .then_with(|| self.0.y.cmp(&other.0.y))
            .then_with(|| self.0.z.cmp(&other.0.z))
    }
}

impl RegionPos {
    /// Creates a new `RegionPos`.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// Returns the region X coordinate.
    #[inline]
    pub const fn x(self) -> i32 {
        self.0.x
    }

    /// Returns the region Y coordinate.
    #[inline]
    pub const fn y(self) -> i32 {
        self.0.y
    }

    /// Returns the region Z coordinate.
    #[inline]
    pub const fn z(self) -> i32 {
        self.0.z
    }

    /// Returns the min chunk pos of this region.
    #[inline]
    #[must_use]
    pub fn min_chunk_pos(self) -> ChunkPos {
        ChunkPos::new(
            self.0.x << REGION_SHIFT,
            self.0.y << REGION_SHIFT,
            self.0.z << REGION_SHIFT,
        )
    }
}

impl fmt::Display for RegionPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r.{}.{}.{}", self.0.x, self.0.y, self.0.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_opposite() {
        for face in Face::ALL {
            assert_eq!(face.opposite().opposite(), face);
        }
    }

    #[test]
    fn test_local_pos_round_trip() {
        for y in 0..32 {
            for z in 0..32 {
                for x in 0..32 {
                    let local = LocalPos::from_xyz(x, y, z);
                    assert_eq!(local.to_xyz(), (x, y, z));
                    assert_eq!(local.x(), x);
                    assert_eq!(local.y(), y);
                    assert_eq!(local.z(), z);
                }
            }
        }
    }

    #[test]
    fn test_negative_coordinate_shifts() {
        let p = BlockPos::new(-1, -1, -1);
        let (c, l) = p.to_chunk_and_local();
        assert_eq!(c, ChunkPos::new(-1, -1, -1));
        assert_eq!(l, LocalPos::from_xyz(31, 31, 31));

        let p2 = BlockPos::new(-32, -32, -32);
        let (c2, l2) = p2.to_chunk_and_local();
        assert_eq!(c2, ChunkPos::new(-1, -1, -1));
        assert_eq!(l2, LocalPos::from_xyz(0, 0, 0));

        let p3 = BlockPos::new(-33, -33, -33);
        let (c3, l3) = p3.to_chunk_and_local();
        assert_eq!(c3, ChunkPos::new(-2, -2, -2));
        assert_eq!(l3, LocalPos::from_xyz(31, 31, 31));
    }

    #[test]
    fn test_chunk_block_round_trip() {
        let p = BlockPos::new(-127, 45, 1023);
        let (c, l) = p.to_chunk_and_local();
        assert_eq!(c.block_pos(l), p);
    }
}
