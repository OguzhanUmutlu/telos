//! High-level GPU context orchestrating Vulkan device, swapchain, allocator, and frames.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::info;

use crate::{
    allocator::GpuAllocator,
    buffer::GpuBuffer,
    depth::DepthBuffer,
    device::{Device, create_device},
    error::GpuError,
    frame::FrameManager,
    instance::Instance,
    swapchain::Swapchain,
    texture::{GpuTextureArray, TextureMipRegion},
};

/// The primary Vulkan rendering context.
pub struct GpuContext {
    frame_manager: FrameManager,
    swapchain: Swapchain,
    allocator: GpuAllocator,
    device: Device,
    instance: Instance,
    current_extent: vk::Extent2D,
}

impl GpuContext {
    /// Creates a complete Vulkan 1.3 GPU context bound to the given window.
    pub fn new<W: HasWindowHandle + HasDisplayHandle>(
        window: &W,
        width: u32,
        height: u32,
        enable_validation: bool,
    ) -> Result<Self, GpuError> {
        let instance = Instance::new(window, enable_validation)?;

        // We need surface first to create device with presentation support
        let raw_display = window
            .display_handle()
            .map_err(|e| GpuError::Surface(e.to_string()))?
            .as_raw();
        let raw_window = window
            .window_handle()
            .map_err(|e| GpuError::Surface(e.to_string()))?
            .as_raw();

        // SAFETY: Surface creation on valid instance
        let surface = unsafe {
            ash_window::create_surface(
                instance.entry(),
                instance.raw(),
                raw_display,
                raw_window,
                None,
            )?
        };
        let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.raw());

        let device = create_device(instance.raw(), surface, &surface_loader)?;

        let swapchain = Swapchain::new(
            instance.raw(),
            device.raw(),
            device.physical_device(),
            surface,
            surface_loader,
            width,
            height,
        )?;

        let current_extent = swapchain.extent();

        let allocator = GpuAllocator::new(
            instance.raw().clone(),
            device.raw().clone(),
            device.physical_device(),
        )?;

        let frame_manager =
            FrameManager::new(device.raw(), device.queue_families().graphics_family)?;

        info!("GpuContext successfully initialized");

        Ok(Self {
            frame_manager,
            swapchain,
            allocator,
            device,
            instance,
            current_extent,
        })
    }

    /// Resizes swapchain and internal buffers.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        self.device.wait_idle()?;
        self.swapchain.resize(
            self.device.raw(),
            self.device.physical_device(),
            width,
            height,
        )?;
        self.current_extent = self.swapchain.extent();
        Ok(())
    }

    /// Begins a frame: acquires next swapchain image, waits for timeline semaphore,
    /// and begins recording command buffer.
    pub fn begin_frame(&mut self) -> Result<Option<(vk::CommandBuffer, u32)>, GpuError> {
        let frame = self.frame_manager.current_frame();

        let (image_index, is_suboptimal) = match self
            .swapchain
            .acquire_next_image(frame.image_available_semaphore(), u64::MAX)
        {
            Ok(res) => res,
            Err(GpuError::Vk(vk::Result::ERROR_OUT_OF_DATE_KHR)) => {
                let extent = self.current_extent;
                self.resize(extent.width, extent.height)?;
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        if is_suboptimal {
            let extent = self.current_extent;
            self.resize(extent.width, extent.height)?;
        }

        let frame_res = self.frame_manager.begin_frame(self.device.raw())?;
        let cmd = frame_res.command_buffer();
        let swapchain_image = self.swapchain.image(image_index as usize);

        // Transition swapchain image from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL via sync2 barrier
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(swapchain_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);

        // SAFETY: Synchronization2 pipeline barrier on active command buffer
        unsafe {
            self.device
                .raw()
                .cmd_pipeline_barrier2(cmd, &dependency_info);
        }

        Ok(Some((cmd, image_index)))
    }

    /// Ends frame: transitions swapchain image to `PRESENT_SRC_KHR`, submits command buffer,
    /// and presents.
    pub fn end_frame(&mut self, image_index: u32) -> Result<(), GpuError> {
        let cmd = self.frame_manager.current_frame().command_buffer();
        let render_finished = self
            .frame_manager
            .current_frame()
            .render_finished_semaphore();
        let swapchain_image = self.swapchain.image(image_index as usize);

        // Transition swapchain image from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);

        // SAFETY: Synchronization2 barrier on recorded command buffer
        unsafe {
            self.device
                .raw()
                .cmd_pipeline_barrier2(cmd, &dependency_info);
        }

        self.frame_manager
            .submit_and_advance(self.device.raw(), self.device.graphics_queue())?;

        let _ = self
            .swapchain
            .present(self.device.graphics_queue(), image_index, render_finished);

        Ok(())
    }

    /// Reference to Vulkan instance.
    #[must_use]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Reference to logical device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reference to swapchain.
    #[must_use]
    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }

    /// Reference to memory allocator.
    #[must_use]
    pub fn allocator(&self) -> &GpuAllocator {
        &self.allocator
    }

    /// Current swapchain extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.current_extent
    }

    /// Creates a GPU-local storage buffer and uploads initial data via a staging transfer buffer.
    pub fn create_buffer_with_data<T: Copy>(
        &self,
        name: &'static str,
        data: &[T],
        extra_usage: vk::BufferUsageFlags,
    ) -> Result<GpuBuffer, GpuError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.queue_families().graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        // SAFETY: Creating transient command pool for one-time copy submission
        let pool = unsafe { self.device.raw().create_command_pool(&pool_info, None)? };

        let res = GpuBuffer::from_data(
            self.device.raw(),
            &self.allocator,
            self.device.graphics_queue(),
            pool,
            name,
            data,
            extra_usage,
        );

        // SAFETY: Destroying transient transfer command pool
        unsafe {
            self.device.raw().destroy_command_pool(pool, None);
        }

        res
    }

    /// Creates a 2D depth attachment buffer matching the current swapchain extent.
    pub fn create_depth_buffer(&self) -> Result<DepthBuffer, GpuError> {
        DepthBuffer::new(self.device.raw(), &self.allocator, self.current_extent)
    }

    /// Creates a GPU 2D texture array and uploads raw mip slices via a staging transfer buffer.
    pub fn create_texture_array(
        &self,
        resolution: u32,
        layer_count: u32,
        mip_levels: u32,
        pixel_data: &[u8],
        copy_regions: &[TextureMipRegion],
    ) -> Result<GpuTextureArray, GpuError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.queue_families().graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        // SAFETY: Creating transient command pool for one-time copy submission
        let pool = unsafe { self.device.raw().create_command_pool(&pool_info, None)? };

        let res = GpuTextureArray::from_raw_mips(
            self.device.raw(),
            &self.allocator,
            self.device.graphics_queue(),
            pool,
            resolution,
            layer_count,
            mip_levels,
            pixel_data,
            copy_regions,
        );

        // SAFETY: Destroying transient command pool
        unsafe {
            self.device.raw().destroy_command_pool(pool, None);
        }

        res
    }

    /// Waits for all GPU queues to finish operations.
    pub fn wait_idle(&self) -> Result<(), GpuError> {
        self.device.wait_idle()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();
        self.frame_manager.destroy(self.device.raw());
        self.swapchain.destroy(self.device.raw());
    }
}
