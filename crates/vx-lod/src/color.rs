//! Per-block average linear RGBA colors for far-field LOD rendering.

use hashbrown::HashMap;
use vx_core::coords::Face;
use vx_voxel::state::BlockStateId;

/// Directional average colors for a block state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodColor {
    /// Average color of the top face (+Y).
    pub top: [u8; 4],
    /// Average color of side faces (±X, ±Z).
    pub side: [u8; 4],
    /// Average color of the bottom face (-Y).
    pub bottom: [u8; 4],
}

impl LodColor {
    /// Creates a uniform color for all faces.
    #[must_use]
    pub const fn uniform(rgba: [u8; 4]) -> Self {
        Self {
            top: rgba,
            side: rgba,
            bottom: rgba,
        }
    }

    /// Selects face color based on normal direction.
    #[must_use]
    pub const fn for_face(&self, face: Face) -> [u8; 4] {
        match face {
            Face::Up => self.top,
            Face::Down => self.bottom,
            _ => self.side,
        }
    }
}

/// Lookup table mapping `BlockStateId` to directional `LodColor`.
#[derive(Clone, Debug)]
pub struct LodColorTable {
    colors: HashMap<BlockStateId, LodColor>,
    fallback: LodColor,
}

impl Default for LodColorTable {
    fn default() -> Self {
        Self::standard()
    }
}

impl LodColorTable {
    /// Creates a table initialized with standard baseline colors.
    #[must_use]
    pub fn standard() -> Self {
        let mut colors = HashMap::new();

        // Stone: #808080
        colors.insert(
            BlockStateId::new(1),
            LodColor::uniform([128, 128, 128, 255]),
        );
        // Dirt: #866043
        colors.insert(BlockStateId::new(2), LodColor::uniform([134, 96, 67, 255]));
        // Grass: top #6AAA40, side #786E41, bottom #866043
        colors.insert(
            BlockStateId::new(3),
            LodColor {
                top: [106, 170, 64, 255],
                side: [120, 110, 65, 255],
                bottom: [134, 96, 67, 255],
            },
        );
        // Bedrock: #505050
        colors.insert(BlockStateId::new(4), LodColor::uniform([80, 80, 80, 255]));
        // Sand: #DBCFA3
        colors.insert(
            BlockStateId::new(5),
            LodColor::uniform([219, 207, 163, 255]),
        );
        // Water: #4064C8
        colors.insert(BlockStateId::new(6), LodColor::uniform([64, 100, 200, 255]));

        Self {
            colors,
            fallback: LodColor::uniform([150, 150, 150, 255]),
        }
    }

    /// Registers or updates the color for a `BlockStateId`.
    pub fn register(&mut self, state: BlockStateId, color: LodColor) {
        self.colors.insert(state, color);
    }

    /// Retrieves the `LodColor` for a `BlockStateId`.
    #[must_use]
    pub fn get(&self, state: BlockStateId) -> LodColor {
        self.colors.get(&state).copied().unwrap_or(self.fallback)
    }
}
