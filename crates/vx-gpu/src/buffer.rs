//! GPU buffer allocation with Buffer Device Address (BDA) support and staging upload.

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use tracing::info;

use crate::{allocator::GpuAllocator, error::GpuError};

/// A GPU-allocated buffer with optional 64-bit device address for BDA vertex pulling.
pub struct GpuBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    device_address: vk::DeviceAddress,
    size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Creates and allocates a new GPU buffer.
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        name: &'static str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating buffer on valid device
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // SAFETY: Querying memory requirements for the newly created buffer
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.allocate(name, requirements, location, true)?;

        // SAFETY: Binding allocated GPU memory to the buffer
        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let device_address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
            // SAFETY: Device supports buffer device address (verified in feature chain)
            unsafe { device.get_buffer_device_address(&addr_info) }
        } else {
            0
        };

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            device_address,
            size,
        })
    }

    /// Uploads arbitrary slice data to a newly allocated device-local GPU buffer via a temporary staging buffer.
    pub fn from_data<T: Copy>(
        device: &ash::Device,
        allocator: &GpuAllocator,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        name: &'static str,
        data: &[T],
        extra_usage: vk::BufferUsageFlags,
    ) -> Result<Self, GpuError> {
        let byte_size = std::mem::size_of_val(data) as vk::DeviceSize;
        if byte_size == 0 {
            return Self::new(
                device,
                allocator,
                name,
                size_of::<T>() as vk::DeviceSize,
                extra_usage
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            );
        }

        // 1. Create staging buffer
        let mut staging = Self::new(
            device,
            allocator,
            "staging_buffer",
            byte_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;

        // 2. Copy data into staging buffer
        if let Some(alloc) = &staging.allocation {
            let ptr = alloc
                .mapped_ptr()
                .ok_or_else(|| GpuError::Allocation("Staging buffer not host-mapped".into()))?
                .as_ptr()
                .cast::<T>();

            // SAFETY: ptr is valid for byte_size bytes
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }

        // 3. Create destination GPU-only buffer
        let dst_usage = extra_usage
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let dst = Self::new(
            device,
            allocator,
            name,
            byte_size,
            dst_usage,
            MemoryLocation::GpuOnly,
        )?;

        // 4. One-time copy command submission
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: Allocating single-use command buffer
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: Recording transfer copy commands
        unsafe {
            device.begin_command_buffer(cmd, &begin_info)?;

            let copy_region = vk::BufferCopy::default().size(byte_size);
            device.cmd_copy_buffer(cmd, staging.raw(), dst.raw(), &[copy_region]);

            device.end_command_buffer(cmd)?;

            let cmd_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;

            device.free_command_buffers(command_pool, &[cmd]);
        }

        // Destroy temporary staging buffer
        staging.destroy(device, allocator);

        info!(
            name = name,
            bytes = byte_size,
            address = dst.device_address(),
            "GPU buffer created and initialized with data"
        );

        Ok(dst)
    }

    /// Raw `vk::Buffer` handle.
    #[inline]
    #[must_use]
    pub fn raw(&self) -> vk::Buffer {
        self.buffer
    }

    /// 64-bit Buffer Device Address.
    #[inline]
    #[must_use]
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Total size of the buffer in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Destroys the buffer and releases memory back to `GpuAllocator`.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        if self.buffer != vk::Buffer::null() {
            // SAFETY: Destroying buffer with valid device
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            self.buffer = vk::Buffer::null();
        }

        if let Some(alloc) = self.allocation.take()
            && let Err(e) = allocator.free(alloc)
        {
            tracing::error!("Failed to free GPU buffer allocation: {e}");
        }
    }
}
