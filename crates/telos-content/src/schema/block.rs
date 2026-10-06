//! Data schema for block definitions (`data/<ns>/block/<name>.ron` or `.json`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use telos_core::coords::Face;
use telos_voxel::shape::BlockShape;
use telos_voxel::state::StateFlags;

/// Visual render layer classification for meshing and pipeline pass routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RenderLayerDef {
    /// Fully solid, opaque quads.
    #[default]
    Opaque,
    /// Alpha-tested cutout quads (e.g. leaves, glass, flowers).
    Cutout,
    /// Alpha-blended translucent quads (e.g. water, stained glass).
    Translucent,
}

/// Light transmission and occlusion opacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpacityDef {
    /// Fully blocks all light and occludes neighbor faces.
    #[default]
    Opaque,
    /// Fully transparent to light (e.g. air, glass).
    Transparent,
    /// Partially filters light by a fixed level reduction (e.g. water).
    Filter(u8),
}

/// Simplified shape tier for data-driven blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BlockShapeDef {
    /// Tier 0 full 1x1x1 cube.
    #[default]
    FullCube,
    /// Tier 1 axis-aligned half-slab.
    Slab {
        /// Whether slab sits on the bottom half of the block.
        #[serde(default = "default_true")]
        bottom: bool,
    },
    /// Tier 1 stairs.
    Stairs {
        /// Facing direction of stairs ("north", "south", "east", "west").
        #[serde(default = "default_north")]
        facing: String,
        /// Top or bottom half ("bottom", "top").
        #[serde(default = "default_bottom")]
        half: String,
    },
    /// Empty shape without geometry (e.g. Air).
    Empty,
    /// Fluid block shape with height level and falling flag.
    Fluid {
        /// Fluid level (0..=7).
        #[serde(default)]
        level: u8,
        /// Whether fluid is falling from above.
        #[serde(default)]
        falling: bool,
    },
    /// Tier 2 diagonal cross shape (flowers, saplings, tall grass).
    Cross,
    /// Tier 2 upright or wall-mounted torch.
    Torch {
        /// Optional wall facing ("north", "south", "east", "west").
        #[serde(default)]
        wall: Option<String>,
    },
    /// Tier 1 flat horizontal plate (wire, repeater, diode).
    FlatPlate,
    /// Tier 1 small toggleable switch / lever sub-box.
    Lever {
        /// Whether the lever is toggled on.
        #[serde(default)]
        powered: bool,
    },
    /// Tier 1 small vertical post (logic inverter torch).
    Post,
}

fn default_true() -> bool {
    true
}

fn default_north() -> String {
    "north".to_string()
}

fn default_bottom() -> String {
    "bottom".to_string()
}

/// Policy for generating or linking a corresponding inventory item for this block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BlockItemPolicy {
    /// Automatically register a corresponding block item (`<namespace>:<block_name>`).
    #[default]
    Auto,
    /// Explicitly link an item identifier.
    Explicit(String),
    /// No item exists for this block (e.g. Air, Water).
    None,
}

/// Property definition for multi-state blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyDef {
    /// Boolean state (true/false).
    Bool,
    /// Integer range (min..=max).
    Int {
        /// Minimum value.
        min: i32,
        /// Maximum value.
        max: i32,
    },
    /// Named string enum values.
    Enum(Vec<String>),
}

/// Data-driven definition of a block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Block")]
pub struct BlockDef {
    /// State properties.
    #[serde(default)]
    pub properties: HashMap<String, PropertyDef>,
    /// Geometric shape tier.
    #[serde(default)]
    pub shape: BlockShapeDef,
    /// Rendering layer.
    #[serde(default)]
    pub render_layer: RenderLayerDef,
    /// Light opacity.
    #[serde(default)]
    pub opacity: OpacityDef,
    /// Light emission (0..=15).
    #[serde(default)]
    pub light_emission: u8,
    /// Mining hardness (seconds to break by hand).
    #[serde(default = "default_hardness")]
    pub hardness: f32,
    /// Explosion resistance.
    #[serde(default = "default_blast_resistance")]
    pub blast_resistance: f32,
    /// Required or preferred tool category (e.g. "telos:pickaxe").
    #[serde(default)]
    pub tool: Option<String>,
    /// Sound category (e.g. "telos:stone").
    #[serde(default)]
    pub sound: Option<String>,
    /// Item registration policy.
    #[serde(default)]
    pub item: BlockItemPolicy,
    /// Material texture index in the texture array layer.
    #[serde(default)]
    pub material_texture_index: Option<u16>,
    /// Whether this block is a logic component.
    #[serde(default)]
    pub logic_component: bool,
    /// Whether this block is actively powered.
    #[serde(default)]
    pub logic_powered: bool,
}

fn default_hardness() -> f32 {
    1.5
}

fn default_blast_resistance() -> f32 {
    6.0
}

impl Default for BlockDef {
    fn default() -> Self {
        Self {
            properties: HashMap::new(),
            shape: BlockShapeDef::FullCube,
            render_layer: RenderLayerDef::Opaque,
            opacity: OpacityDef::Opaque,
            light_emission: 0,
            hardness: default_hardness(),
            blast_resistance: default_blast_resistance(),
            tool: None,
            sound: None,
            item: BlockItemPolicy::Auto,
            material_texture_index: None,
            logic_component: false,
            logic_powered: false,
        }
    }
}

impl BlockDef {
    /// Converts the data-driven block definition into voxel `StateFlags`.
    #[must_use]
    pub fn compute_flags(&self) -> StateFlags {
        if matches!(self.shape, BlockShapeDef::Empty) {
            return StateFlags::AIR;
        }

        let mut flags = StateFlags::NON_EMPTY;

        match self.render_layer {
            RenderLayerDef::Opaque => {
                if matches!(self.shape, BlockShapeDef::FullCube) {
                    flags |= StateFlags::OPAQUE_FULL;
                }
            }
            RenderLayerDef::Cutout => {
                flags |= StateFlags::CUTOUT;
            }
            RenderLayerDef::Translucent => {
                flags |= StateFlags::TRANSLUCENT;
            }
        }

        if matches!(self.opacity, OpacityDef::Opaque) {
            flags |= StateFlags::LIGHT_BLOCKING;
        }

        if self.light_emission > 0 {
            flags |= StateFlags::EMITS_LIGHT;
        }

        if self.logic_component {
            flags |= StateFlags::LOGIC_COMPONENT;
        }

        if self.logic_powered {
            flags |= StateFlags::LOGIC_POWERED;
        }

        flags
    }

    /// Converts the data-driven shape into a `BlockShape`.
    #[must_use]
    pub fn compute_shape(&self) -> BlockShape {
        match &self.shape {
            BlockShapeDef::FullCube => BlockShape::Cube,
            BlockShapeDef::Empty => BlockShape::Empty,
            BlockShapeDef::Slab { bottom } => {
                if *bottom {
                    BlockShape::bottom_slab()
                } else {
                    BlockShape::top_slab()
                }
            }
            BlockShapeDef::Stairs { facing, half } => {
                let face = match facing.to_lowercase().as_str() {
                    "south" => Face::South,
                    "east" => Face::East,
                    "west" => Face::West,
                    _ => Face::North,
                };
                let is_upside_down = half.to_lowercase() == "top";
                BlockShape::stairs(face, is_upside_down)
            }
            BlockShapeDef::Fluid { level, falling } => BlockShape::Fluid {
                level: *level,
                falling: *falling,
            },
            BlockShapeDef::Cross => BlockShape::Cross,
            BlockShapeDef::Torch { wall } => {
                let wall_face = wall.as_deref().map(|w| match w.to_lowercase().as_str() {
                    "south" => Face::South,
                    "east" => Face::East,
                    "west" => Face::West,
                    _ => Face::North,
                });
                BlockShape::Torch { wall: wall_face }
            }
            BlockShapeDef::FlatPlate => BlockShape::flat_plate(),
            BlockShapeDef::Lever { powered } => BlockShape::lever(*powered),
            BlockShapeDef::Post => BlockShape::post(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ron_deserialization() {
        let ron_str = r#"
            Block(
                shape: FullCube,
                render_layer: Opaque,
                opacity: Opaque,
                light_emission: 12,
                hardness: 3.0,
                blast_resistance: 9.0,
                tool: Some("telos:pickaxe"),
                sound: Some("telos:stone"),
                item: Auto,
            )
        "#;
        let def: BlockDef = ron::from_str(ron_str).expect("Valid RON BlockDef");
        assert_eq!(def.light_emission, 12);
        assert!((def.hardness - 3.0).abs() < f32::EPSILON);
        assert_eq!(def.tool, Some("telos:pickaxe".to_string()));
        let flags = def.compute_flags();
        assert!(flags.contains(StateFlags::OPAQUE_FULL));
        assert!(flags.contains(StateFlags::LIGHT_BLOCKING));
        assert_eq!(def.compute_shape(), BlockShape::Cube);
    }
}
