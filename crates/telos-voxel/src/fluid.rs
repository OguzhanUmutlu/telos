//! Fluid types, fluid state representation, and level classification.

/// Classification of fluid type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FluidKind {
    /// Water fluid.
    Water,
    /// Lava fluid (viscous, emissive).
    Lava,
}

/// Evaluated fluid properties for a block state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidState {
    /// Fluid kind (water or lava).
    pub kind: FluidKind,
    /// Fluid decay level: 0 = full source, 1..=7 = flowing.
    pub level: u8,
    /// Whether the fluid block represents a vertical falling column.
    pub falling: bool,
}

impl FluidState {
    /// Creates a new `FluidState`.
    #[inline]
    #[must_use]
    pub const fn new(kind: FluidKind, level: u8, falling: bool) -> Self {
        Self {
            kind,
            level,
            falling,
        }
    }

    /// Returns `true` if this fluid state is a source block.
    #[inline]
    #[must_use]
    pub const fn is_source(self) -> bool {
        self.level == 0 && !self.falling
    }
}
