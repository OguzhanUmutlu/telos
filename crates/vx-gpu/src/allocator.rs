//! Memory allocation wrapper around `gpu-allocator`.

use ash::vk;
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::error::GpuError;

/// Manages dedicated and sub-allocated GPU memory.
pub struct GpuAllocator {
    allocator: Arc<Mutex<Allocator>>,
}

impl GpuAllocator {
    /// Creates a new GPU memory allocator.
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, GpuError> {
        let desc = AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        };

        let allocator = Allocator::new(&desc).map_err(|e| {
            GpuError::Allocation(format!("Failed to initialize gpu-allocator: {e}"))
        })?;

        info!("GPU memory allocator initialized with buffer device address support");

        Ok(Self {
            allocator: Arc::new(Mutex::new(allocator)),
        })
    }

    /// Allocates device memory for a buffer or image.
    pub fn allocate(
        &self,
        name: &'static str,
        requirements: vk::MemoryRequirements,
        location: gpu_allocator::MemoryLocation,
        linear: bool,
    ) -> Result<Allocation, GpuError> {
        let desc = AllocationCreateDesc {
            name,
            requirements,
            location,
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let mut alloc = self
            .allocator
            .lock()
            .map_err(|_| GpuError::Allocation("Allocator lock poisoned".into()))?;

        alloc
            .allocate(&desc)
            .map_err(|e| GpuError::Allocation(format!("Allocation failed for '{name}': {e}")))
    }

    /// Frees an existing GPU memory allocation.
    pub fn free(&self, allocation: Allocation) -> Result<(), GpuError> {
        let mut alloc = self
            .allocator
            .lock()
            .map_err(|_| GpuError::Allocation("Allocator lock poisoned".into()))?;

        alloc
            .free(allocation)
            .map_err(|e| GpuError::Allocation(format!("Free allocation failed: {e}")))
    }
}
