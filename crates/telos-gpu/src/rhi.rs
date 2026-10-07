//! # Render Hardware Interface (RHI) Abstractions
//!
//! Provides common types, backend enumerations, and capability descriptors for
//! multi-backend rendering (Vulkan 1.3 primary and OpenGL 4.5 fallback).

/// Supported graphics rendering backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RenderBackendType {
    /// Vulkan 1.3 with dynamic rendering, sync2, and timeline semaphores (default high performance).
    #[default]
    Vulkan,
    /// OpenGL 4.5 / WebGL 2 fallback backend via `glow`.
    OpenGl,
}

impl RenderBackendType {
    /// Human-readable name of the backend.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan 1.3",
            Self::OpenGl => "OpenGL 4.5",
        }
    }
}

/// Device capabilities queryable across all graphics backends.
#[derive(Debug, Clone)]
pub struct RenderCaps {
    /// Active rendering backend.
    pub backend: RenderBackendType,
    /// Device / GPU adapter name.
    pub device_name: String,
    /// Driver information or version.
    pub driver_info: String,
    /// Maximum supported 2D texture array layer count.
    pub max_texture_array_layers: u32,
    /// Whether Buffer Device Address (BDA) is supported.
    pub supports_bda: bool,
    /// Whether `VK_EXT_mesh_shader` or mesh shaders are supported.
    pub supports_mesh_shaders: bool,
    /// Whether compute shaders are supported.
    pub supports_compute: bool,
}

impl RenderCaps {
    /// Creates basic capability descriptor for the OpenGL backend.
    #[must_use]
    pub fn opengl_fallback(vendor: &str, renderer: &str, version: &str) -> Self {
        Self {
            backend: RenderBackendType::OpenGl,
            device_name: format!("{vendor} - {renderer}"),
            driver_info: version.to_string(),
            max_texture_array_layers: 2048,
            supports_bda: false,
            supports_mesh_shaders: false,
            supports_compute: false,
        }
    }
}
