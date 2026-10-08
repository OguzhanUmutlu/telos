//! Per-block average linear RGBA colors for far-field LOD rendering.

use hashbrown::HashMap;
use telos_core::coords::Face;
use telos_voxel::state::BlockStateId;

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
        // Oak Leaves: #489120
        colors.insert(BlockStateId::new(8), LodColor::uniform([72, 145, 32, 255]));
        // Oak Log: top/bottom #8C734B, side #675231
        colors.insert(
            BlockStateId::new(74),
            LodColor {
                top: [140, 115, 75, 255],
                side: [103, 82, 49, 255],
                bottom: [140, 115, 75, 255],
            },
        );
        // Birch Log: top/bottom #B4A07D, side #D7D7D2
        colors.insert(
            BlockStateId::new(75),
            LodColor {
                top: [180, 160, 125, 255],
                side: [215, 215, 210, 255],
                bottom: [180, 160, 125, 255],
            },
        );
        // Spruce Log: top/bottom #695032, side #2D1E0F
        colors.insert(
            BlockStateId::new(76),
            LodColor {
                top: [105, 80, 50, 255],
                side: [45, 30, 15, 255],
                bottom: [105, 80, 50, 255],
            },
        );
        // Birch Leaves: #80A755
        colors.insert(
            BlockStateId::new(77),
            LodColor::uniform([128, 167, 85, 255]),
        );
        // Spruce Leaves: #619961
        colors.insert(BlockStateId::new(78), LodColor::uniform([97, 153, 97, 255]));

        // Surface Vegetation & Flora (Phase 61)
        // Poppy: #DC2828
        colors.insert(BlockStateId::new(12), LodColor::uniform([220, 40, 40, 255]));
        // Dandelion: #FFE628
        colors.insert(
            BlockStateId::new(13),
            LodColor::uniform([255, 230, 40, 255]),
        );
        // Short Grass: #6AAA40
        colors.insert(
            BlockStateId::new(16),
            LodColor::uniform([106, 170, 64, 255]),
        );
        // Fern: #50A032
        colors.insert(BlockStateId::new(17), LodColor::uniform([80, 160, 50, 255]));
        // Dead Bush: #8C6E46
        colors.insert(
            BlockStateId::new(18),
            LodColor::uniform([140, 110, 70, 255]),
        );
        // Tall Grass: #64B43C
        colors.insert(
            BlockStateId::new(85),
            LodColor::uniform([100, 180, 60, 255]),
        );
        // Cornflower: #4678F0
        colors.insert(
            BlockStateId::new(86),
            LodColor::uniform([70, 120, 240, 255]),
        );
        // Oxeye Daisy: #F0F0E6
        colors.insert(
            BlockStateId::new(87),
            LodColor::uniform([240, 240, 230, 255]),
        );
        // Brown Mushroom: #966E50
        colors.insert(
            BlockStateId::new(88),
            LodColor::uniform([150, 110, 80, 255]),
        );
        // Red Mushroom: #C82828
        colors.insert(BlockStateId::new(89), LodColor::uniform([200, 40, 40, 255]));

        Self {
            colors,
            fallback: LodColor::uniform([150, 150, 150, 255]),
        }
    }

    /// Evaluates distance surface color blending vegetation cover into base terrain top color.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn blend_vegetation_surface(
        base_top_color: [u8; 4],
        vegetation_color: [u8; 4],
        vegetation_density: f32,
    ) -> [u8; 4] {
        let t = vegetation_density.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        [
            (f32::from(base_top_color[0]) * inv + f32::from(vegetation_color[0]) * t) as u8,
            (f32::from(base_top_color[1]) * inv + f32::from(vegetation_color[1]) * t) as u8,
            (f32::from(base_top_color[2]) * inv + f32::from(vegetation_color[2]) * t) as u8,
            255,
        ]
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
