//! # vx-gpu
//!
//! Thin, high-performance Vulkan 1.3 Render Hardware Interface (RHI) for the voxel engine.
//!
//! Exposes dynamic rendering, synchronization2, timeline semaphores, and GPU memory allocation.

pub mod allocator;
pub mod context;
pub mod device;
pub mod error;
pub mod frame;
pub mod instance;
pub mod pipeline;
pub mod swapchain;

pub use allocator::GpuAllocator;
pub use context::GpuContext;
pub use device::{Device, QueueFamilyIndices};
pub use error::GpuError;
pub use frame::{FRAMES_IN_FLIGHT, FrameManager, FrameResources};
pub use instance::Instance;
pub use pipeline::{GraphicsPipeline, ShaderModule};
pub use swapchain::Swapchain;

// Re-export ash vk for shader/command ergonomics
pub use ash::vk;
