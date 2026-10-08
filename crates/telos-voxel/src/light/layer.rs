//! Compact 4-bit light layer storage with zero-allocation uniform chunk elision.

use crate::coords::CHUNK_VOLUME;

/// Number of bytes required to pack 32³ 4-bit nibbles (32,768 / 2 = 16,384 bytes = 16 KiB).
pub const LIGHT_LAYER_BYTES: usize = CHUNK_VOLUME / 2;

/// A 4-bit lighting layer (0..=15) over a 32³ cubic chunk.
///
/// Implements ADR-07 and ADR-11:
/// - `Uniform(u8)`: zero-allocation elision for common cases (e.g. 15 for open sky, 0 for deep stone).
/// - `Nibbles(Box<[u8; 16384]>)`: 16 KiB heap allocation when light levels vary.
#[derive(Clone, PartialEq, Eq)]
pub enum LightLayer {
    /// All voxels in the chunk share the same light level (0..=15).
    Uniform(u8),
    /// Two 4-bit light levels packed per byte: even index in low nibble, odd index in high nibble.
    Nibbles(Box<[u8; LIGHT_LAYER_BYTES]>),
}

impl LightLayer {
    /// Creates an empty light layer initialized to 0.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::Uniform(0)
    }

    /// Creates a full uniform light layer initialized to `level` (clamped to 0..=15).
    #[inline]
    #[must_use]
    pub const fn uniform(level: u8) -> Self {
        Self::Uniform(level & 0x0F)
    }

    /// Retrieves the 4-bit light level (0..=15) at linear voxel index `idx`.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < CHUNK_VOLUME, "Voxel index out of bounds: {idx}");
        match self {
            Self::Uniform(val) => *val,
            Self::Nibbles(bytes) => {
                let byte = bytes[idx >> 1];
                if (idx & 1) == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                }
            }
        }
    }

    /// Sets the 4-bit light level at linear voxel index `idx`.
    ///
    /// Promotes from `Uniform` to `Nibbles` if setting a value different from the uniform level.
    #[inline]
    pub fn set(&mut self, idx: usize, val: u8) {
        debug_assert!(idx < CHUNK_VOLUME, "Voxel index out of bounds: {idx}");
        let val = val & 0x0F;

        match self {
            Self::Uniform(curr) => {
                if *curr == val {
                    return;
                }
                let mut bytes = Box::new([(*curr) | ((*curr) << 4); LIGHT_LAYER_BYTES]);
                let byte_idx = idx >> 1;
                if (idx & 1) == 0 {
                    bytes[byte_idx] = (bytes[byte_idx] & 0xF0) | val;
                } else {
                    bytes[byte_idx] = (bytes[byte_idx] & 0x0F) | (val << 4);
                }
                *self = Self::Nibbles(bytes);
            }
            Self::Nibbles(bytes) => {
                let byte_idx = idx >> 1;
                if (idx & 1) == 0 {
                    bytes[byte_idx] = (bytes[byte_idx] & 0xF0) | val;
                } else {
                    bytes[byte_idx] = (bytes[byte_idx] & 0x0F) | (val << 4);
                }
            }
        }
    }

    /// Fills the entire layer with a single uniform level (0..=15).
    #[inline]
    pub fn fill(&mut self, level: u8) {
        *self = Self::Uniform(level & 0x0F);
    }

    /// Returns `true` if this layer is stored as a uniform scalar without heap memory.
    #[inline]
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform(_))
    }

    /// Returns the uniform value if uniform.
    #[inline]
    #[must_use]
    pub const fn uniform_value(&self) -> Option<u8> {
        match self {
            Self::Uniform(v) => Some(*v),
            Self::Nibbles(_) => None,
        }
    }

    /// Scans the nibble buffer and collapses to `Uniform(val)` if all voxels share the same level.
    ///
    /// Scans the nibble buffer and collapses to `Uniform(val)` if all voxels share the same level.
    ///
    /// Accelerated with AVX2 vectorized comparisons when supported (~80 ns scan time).
    pub fn try_collapse(&mut self) -> bool {
        let bytes = match self {
            Self::Uniform(_) => return true,
            Self::Nibbles(b) => b,
        };

        if let Some(level) = crate::simd::light::try_collapse_simd(bytes) {
            *self = Self::Uniform(level);
            true
        } else {
            false
        }
    }
}

impl Default for LightLayer {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Debug for LightLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform(val) => write!(f, "LightLayer::Uniform({val})"),
            Self::Nibbles(_) => write!(f, "LightLayer::Nibbles(16384 bytes)"),
        }
    }
}

/// Dual-layer chunk lighting containing independent sky light and block light layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkLight {
    /// Sky light layer (attenuated from sunlight above heightmaps).
    pub sky: LightLayer,
    /// Block light layer (emitted from light sources such as torches and lava).
    pub block: LightLayer,
}

impl ChunkLight {
    /// Creates a `ChunkLight` with uniform levels for sky and block light.
    #[inline]
    #[must_use]
    pub const fn new_uniform(sky: u8, block: u8) -> Self {
        Self {
            sky: LightLayer::uniform(sky),
            block: LightLayer::uniform(block),
        }
    }

    /// Retrieves sky light level (0..=15) at local voxel index `idx`.
    #[inline]
    #[must_use]
    pub fn get_sky(&self, idx: usize) -> u8 {
        self.sky.get(idx)
    }

    /// Sets sky light level at local voxel index `idx`.
    #[inline]
    pub fn set_sky(&mut self, idx: usize, val: u8) {
        self.sky.set(idx, val);
    }

    /// Retrieves block light level (0..=15) at local voxel index `idx`.
    #[inline]
    #[must_use]
    pub fn get_block(&self, idx: usize) -> u8 {
        self.block.get(idx)
    }

    /// Sets block light level at local voxel index `idx`.
    #[inline]
    pub fn set_block(&mut self, idx: usize, val: u8) {
        self.block.set(idx, val);
    }

    /// Attempts to collapse both sky and block light layers to uniform scalars if homogeneous.
    pub fn try_collapse(&mut self) {
        self.sky.try_collapse();
        self.block.try_collapse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_get_set() {
        let mut layer = LightLayer::uniform(15);
        assert_eq!(layer.get(0), 15);
        assert_eq!(layer.get(100), 15);
        assert!(layer.is_uniform());

        // Setting same value doesn't promote
        layer.set(100, 15);
        assert!(layer.is_uniform());

        // Setting different value promotes to Nibbles
        layer.set(100, 7);
        assert!(!layer.is_uniform());
        assert_eq!(layer.get(100), 7);
        assert_eq!(layer.get(99), 15);
        assert_eq!(layer.get(101), 15);

        // Setting back and collapsing
        layer.set(100, 15);
        assert!(layer.try_collapse());
        assert!(layer.is_uniform());
        assert_eq!(layer.uniform_value(), Some(15));
    }

    #[test]
    fn test_nibble_packing_roundtrip() {
        let mut layer = LightLayer::zero();
        for i in 0..CHUNK_VOLUME {
            let val = ((i * 7 + 3) % 16) as u8;
            layer.set(i, val);
        }

        for i in 0..CHUNK_VOLUME {
            let expected = ((i * 7 + 3) % 16) as u8;
            assert_eq!(layer.get(i), expected, "Mismatch at index {i}");
        }
    }
}
