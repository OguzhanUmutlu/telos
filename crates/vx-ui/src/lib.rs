//! Retained UI framework, taffy layout, bitmap font engine, and in-game HUD.
//!
//! This crate contains pure CPU UI logic with zero Vulkan or windowing dependencies,
//! ensuring complete headless testability. It produces GPU-ready 48-byte `UiQuad`
//! instances for single-draw-call instanced rendering.

pub mod font;
pub mod frame;
pub mod hud;
pub mod quad;
pub mod scale;
pub mod tree;

pub use font::{BitmapFont, GlyphMetrics};
pub use frame::UiFrame;
pub use hud::{HudState, UiLayers, render_hud};
pub use quad::{QuadKind, UiQuad};
pub use scale::{compute_gui_scale, snap_to_physical, to_physical_pixels};
pub use tree::{DirtyFlags, NodeId, UiTree, WidgetKind, WidgetNode};
