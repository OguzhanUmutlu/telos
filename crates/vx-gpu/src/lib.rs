//! # vx-gpu
//!
//! Thin, high-performance Vulkan 1.3 Render Hardware Interface (RHI) for the voxel engine.
//!
//! Exposes dynamic rendering, synchronization2, timeline semaphores, and GPU memory allocation.

pub mod allocator;
pub mod buffer;
pub mod compute;
pub mod context;
pub mod depth;
pub mod device;
pub mod error;
pub mod frame;
pub mod hiz;
pub mod instance;
pub mod pipeline;
pub mod swapchain;
pub mod texture;

pub use allocator::GpuAllocator;
pub use buffer::GpuBuffer;
pub use compute::ComputePipeline;
pub use context::GpuContext;
pub use depth::DepthBuffer;
pub use device::{Device, QueueFamilyIndices};
pub use error::GpuError;
pub use frame::{FRAMES_IN_FLIGHT, FrameManager, FrameResources};
pub use hiz::HiZPyramid;
pub use instance::Instance;
pub use pipeline::{GraphicsPipeline, ShaderModule};
pub use swapchain::Swapchain;
pub use texture::{GpuTextureArray, TextureMipRegion};

// Re-export ash and gpu_allocator for ergonomics
pub use ash;
pub use ash::vk;
pub use gpu_allocator;
pub use gpu_allocator::MemoryLocation;
