//! Block state identifier and fast state property bitflags.

use bitflags::bitflags;

/// A numeric runtime identifier for a distinct block state.
///
/// Guaranteed to be dense and synced between server and clients.
/// `BlockStateId(0)` is reserved for air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct BlockStateId(pub u32);

impl BlockStateId {
    /// Canonical air block state identifier (0).
    pub const AIR: Self = Self(0);

    /// Creates a new `BlockStateId`.
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw numeric index as `u32`.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the raw numeric index as `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns `true` if this state represents air.
    #[inline]
    #[must_use]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }
}

bitflags! {
    /// Fast block state properties packed into bitflags for $O(1)$ meshing and physics queries.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct StateFlags: u16 {
        /// Fully opaque cube filling the 1×1×1 voxel (T0 occluder).
        const OPAQUE_FULL = 1 << 0;
        /// Blocks sunlight and artificial block light propagation.
        const LIGHT_BLOCKING = 1 << 1;
        /// Non-empty voxel (any solid, cutout, or translucent geometry).
        const NON_EMPTY = 1 << 2;
        /// Receives random game ticks (e.g. crop growth, leaf decay).
        const TICKABLE = 1 << 3;
        /// Associated with an external block entity (e.g. chest, sign).
        const HAS_BLOCK_ENTITY = 1 << 4;
        /// Emits artificial block light (light level > 0).
        const EMITS_LIGHT = 1 << 5;
        /// Translucent material requiring depth-sorted or blended passes (e.g. water, stained glass).
        const TRANSLUCENT = 1 << 6;
        /// Alpha cutout material with sharp binary discard (e.g. leaves, glass, saplings).
        const CUTOUT = 1 << 7;
        /// Interacts with the deterministic logic and signal propagation engine.
        const LOGIC_COMPONENT = 1 << 8;
        /// Currently in an active/powered state (emits signal or visual power).
        const LOGIC_POWERED = 1 << 9;
        /// Fluid material (water, lava) simulating real-time cellular automata flow.
        const FLUID = 1 << 10;
    }
}

impl StateFlags {
    /// Alias for light-emitting blocks (torches, glowstone).
    pub const EMISSIVE: Self = Self::EMITS_LIGHT;

    /// Default flags for air (all flags clear).
    pub const AIR: Self = Self::empty();

    /// Default flags for a standard solid opaque full cube (stone, dirt).
    pub const OPAQUE_CUBE: Self = Self::from_bits_truncate(
        Self::OPAQUE_FULL.bits() | Self::LIGHT_BLOCKING.bits() | Self::NON_EMPTY.bits(),
    );

    /// Default flags for an alpha cutout full cube (glass, leaves).
    pub const CUTOUT_CUBE: Self =
        Self::from_bits_truncate(Self::CUTOUT.bits() | Self::NON_EMPTY.bits());
}
