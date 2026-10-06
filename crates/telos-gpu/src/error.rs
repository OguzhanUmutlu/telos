//! Error types for the Vulkan RHI.

use ash::vk;
use thiserror::Error;

/// An error that occurred within the GPU rendering subsystem.
#[derive(Debug, Error)]
pub enum GpuError {
    /// Failed to load the Vulkan library or entry points.
    #[error("Vulkan loading error: {0}")]
    Loading(#[from] ash::LoadingError),

    /// A Vulkan API call returned an error code.
    #[error("Vulkan API error: {0}")]
    Vk(#[from] vk::Result),

    /// No suitable physical device was found matching the engine's requirements.
    #[error("no suitable physical device found: {0}")]
    NoSuitableDevice(&'static str),

    /// Failed to query or create window surface.
    #[error("surface creation error: {0}")]
    Surface(String),

    /// Swapchain recreation or acquisition failure.
    #[error("swapchain error: {0}")]
    Swapchain(String),

    /// GPU memory allocation failure.
    #[error("GPU allocator error: {0}")]
    Allocation(String),

    /// Failed to compile or create shader module.
    #[error("shader error: {0}")]
    Shader(String),

    /// Dynamic rendering or graphics pipeline creation failure.
    #[error("pipeline creation error: {0}")]
    Pipeline(String),
}
