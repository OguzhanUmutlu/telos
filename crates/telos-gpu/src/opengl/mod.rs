//! OpenGL 4.5 / WebGL 2 fallback rendering subsystem.

pub mod buffer;
pub mod context;
pub mod program;
pub mod texture;

pub use buffer::GlBuffer;
pub use context::GlContext;
pub use program::GlProgram;
pub use texture::{GlTexture2d, GlTextureArray};
