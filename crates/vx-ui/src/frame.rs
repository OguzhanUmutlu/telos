//! Render frame output holding collected UI quads and viewport metadata.

use crate::quad::UiQuad;

/// Output of a UI layout/paint pass ready to be uploaded to the GPU.
#[derive(Debug, Clone, Default)]
pub struct UiFrame {
    /// List of GPU-ready instanced quad descriptors.
    pub quads: Vec<UiQuad>,
    /// Physical viewport dimensions `[width, height]`.
    pub viewport_size: [u32; 2],
    /// GUI scale used to lay out and snap the quads.
    pub gui_scale: u32,
}

impl UiFrame {
    /// Creates a new empty `UiFrame`.
    #[must_use]
    pub fn new(viewport_size: [u32; 2], gui_scale: u32) -> Self {
        Self {
            quads: Vec::new(),
            viewport_size,
            gui_scale,
        }
    }

    /// Returns the number of quads to render.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.quads.len()
    }

    /// Clears the quads while retaining capacity.
    pub fn clear(&mut self) {
        self.quads.clear();
    }
}
