//! Resource pack animation metadata (.mcmeta) parsing and frame resolution.

use serde::{Deserialize, Serialize};

const fn default_frametime() -> u32 {
    1
}

/// Metadata definition describing texture animation parameters from `.png.mcmeta`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnimationDef {
    /// Duration of each frame in ticks (default: 1).
    #[serde(default = "default_frametime")]
    pub frametime: u32,
    /// Whether to generate interpolated intermediate frames.
    #[serde(default)]
    pub interpolate: bool,
    /// Explicit sequence of frames. If `None` or empty, sequential `0..frame_count` is assumed.
    #[serde(default)]
    pub frames: Option<Vec<AnimationFrameDef>>,
}

impl Default for AnimationDef {
    fn default() -> Self {
        Self {
            frametime: 1,
            interpolate: false,
            frames: None,
        }
    }
}

/// A single frame entry within an animation definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum AnimationFrameDef {
    /// Direct frame index integer.
    Index(u32),
    /// Detailed frame entry specifying frame index and custom frame duration.
    Detailed {
        /// Physical frame index within the texture strip.
        index: u32,
        /// Duration of this frame in ticks.
        #[serde(default = "default_frametime")]
        time: u32,
    },
}

impl AnimationFrameDef {
    /// The physical frame index.
    #[must_use]
    pub const fn index(&self) -> u32 {
        match *self {
            Self::Index(i) => i,
            Self::Detailed { index, .. } => index,
        }
    }

    /// The duration of this frame in ticks, defaulting to the global `frametime` if not specified.
    #[must_use]
    pub const fn time(&self, default_time: u32) -> u32 {
        match *self {
            Self::Index(_) => default_time,
            Self::Detailed { time, .. } => time,
        }
    }
}

/// Scaling definition for a GUI sprite from `.mcmeta`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GuiScaling {
    /// Stretch mode: quad UV stretch across target bounds.
    #[default]
    Stretch,
    /// Tile mode: repeated across target area with fixed `width` and `height`.
    Tile {
        /// Fixed width of each tile in pixels.
        width: u32,
        /// Fixed height of each tile in pixels.
        height: u32,
    },
    /// Nine-slice mode: fixed edge borders with scalable interior.
    NineSlice {
        /// Logical width of the sprite in pixels.
        width: u32,
        /// Logical height of the sprite in pixels.
        height: u32,
        /// Border widths for the 9 slices.
        border: NineSliceBorder,
        /// Whether the center slice should stretch rather than tile.
        #[serde(default)]
        stretch_inner: bool,
    },
}

/// Border specification for 9-slice GUI scaling.
/// Supports both integer shorthand (`"border": 3`) and detailed edge definitions (`"border": { ... }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NineSliceBorder {
    /// Uniform border applied to all 4 edges.
    Uniform(u32),
    /// Detailed per-edge borders.
    Detailed {
        /// Left edge border width in pixels.
        #[serde(default)]
        left: u32,
        /// Top edge border width in pixels.
        #[serde(default)]
        top: u32,
        /// Right edge border width in pixels.
        #[serde(default)]
        right: u32,
        /// Bottom edge border width in pixels.
        #[serde(default)]
        bottom: u32,
    },
}

impl NineSliceBorder {
    /// Left border in source pixels.
    #[must_use]
    pub const fn left(&self) -> u32 {
        match *self {
            Self::Uniform(b) => b,
            Self::Detailed { left, .. } => left,
        }
    }

    /// Top border in source pixels.
    #[must_use]
    pub const fn top(&self) -> u32 {
        match *self {
            Self::Uniform(b) => b,
            Self::Detailed { top, .. } => top,
        }
    }

    /// Right border in source pixels.
    #[must_use]
    pub const fn right(&self) -> u32 {
        match *self {
            Self::Uniform(b) => b,
            Self::Detailed { right, .. } => right,
        }
    }

    /// Bottom border in source pixels.
    #[must_use]
    pub const fn bottom(&self) -> u32 {
        match *self {
            Self::Uniform(b) => b,
            Self::Detailed { bottom, .. } => bottom,
        }
    }
}

/// GUI metadata container within `.mcmeta`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GuiMetaDef {
    /// GUI scaling mode configuration.
    #[serde(default)]
    pub scaling: GuiScaling,
}

impl GuiMetaDef {
    /// Parses a `GuiMetaDef` from a JSON string.
    ///
    /// Supports both root `{ "gui": { ... } }` and direct `{ "scaling": ... }`.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        if let Ok(meta) = serde_json::from_str::<TextureMetaDef>(json)
            && let Some(gui) = meta.gui
        {
            return Ok(gui);
        }
        serde_json::from_str::<Self>(json)
    }
}

/// Wrapper for root `.png.mcmeta` JSON files.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TextureMetaDef {
    /// Animation configuration if present.
    #[serde(default)]
    pub animation: Option<AnimationDef>,
    /// GUI sprite configuration if present.
    #[serde(default)]
    pub gui: Option<GuiMetaDef>,
}

impl TextureMetaDef {
    /// Parses a root `TextureMetaDef` from a JSON string.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str::<Self>(json)
    }
}

impl AnimationDef {
    /// Parses an `AnimationDef` from a JSON string.
    ///
    /// Supports both root `"animation": { ... }` wrappers and direct `{ "frametime": ... }` objects.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        // Try parsing standard wrapped { "animation": { ... } }
        if let Ok(meta) = serde_json::from_str::<TextureMetaDef>(json)
            && let Some(anim) = meta.animation
        {
            return Ok(anim);
        }
        // Fall back to direct animation definition object
        serde_json::from_str::<Self>(json)
    }

    /// Resolves the sequence of physical frame indices for a texture with `total_physical_frames`.
    ///
    /// If `frames` is unspecified or empty, returns `(0..total_physical_frames).collect()`.
    /// Validates and clamps frame indices to valid bounds `0..total_physical_frames`.
    #[must_use]
    pub fn resolve_frames(&self, total_physical_frames: u32) -> Vec<u32> {
        if total_physical_frames == 0 {
            return Vec::new();
        }
        match &self.frames {
            Some(defs) if !defs.is_empty() => defs
                .iter()
                .map(|f| f.index() % total_physical_frames)
                .collect(),
            _ => (0..total_physical_frames).collect(),
        }
    }
}
