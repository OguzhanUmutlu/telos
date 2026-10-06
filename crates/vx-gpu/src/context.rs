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

    /// Creates an allocated GPU buffer with specified usage and memory location.
    pub fn create_buffer(
        &self,
        name: &'static str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: gpu_allocator::MemoryLocation,
    ) -> Result<GpuBuffer, GpuError> {
        GpuBuffer::new(
            self.device.raw(),
            &self.allocator,
            name,
            size,
            usage,
            location,
        )
    }

    /// Creates a 2D depth attachment buffer matching the current swapchain extent.
    pub fn create_depth_buffer(&self) -> Result<DepthBuffer, GpuError> {
        DepthBuffer::new(self.device.raw(), &self.allocator, self.current_extent)
    }

    /// Creates a power-of-two Hi-Z depth pyramid matching the current render extent.
    pub fn create_hiz_pyramid(&self) -> Result<crate::hiz::HiZPyramid, GpuError> {
        crate::hiz::HiZPyramid::new(self.device.raw(), &self.allocator, self.current_extent)
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
        self.create_texture_array_2d(
            resolution,
            resolution,
            layer_count,
            mip_levels,
            pixel_data,
            copy_regions,
        )
    }

    /// Creates a GPU 2D texture array with custom width/height and uploads raw mip slices via a staging transfer buffer.
    pub fn create_texture_array_2d(
        &self,
        width: u32,
        height: u32,
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

        let res = GpuTextureArray::from_raw_mips_2d(
            self.device.raw(),
            &self.allocator,
            self.device.graphics_queue(),
            pool,
            width,
            height,
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

    /// Captures the specified swapchain image into host memory and returns `(width, height, RGBA8 bytes)`.
    #[allow(clippy::too_many_lines)]
    pub fn capture_screenshot(
        &mut self,
        image_index: u32,
    ) -> Result<(u32, u32, Vec<u8>), GpuError> {
        self.device.wait_idle()?;

        let width = self.current_extent.width;
        let height = self.current_extent.height;
        let byte_size = u64::from(width) * u64::from(height) * 4;
        let swapchain_image = self.swapchain.image(image_index as usize);

        // 1. Allocate host-visible readback buffer
        let mut staging_buf = GpuBuffer::new(
            self.device.raw(),
            &self.allocator,
            "screenshot_staging",
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
        )?;

        // 2. Transient command pool and command buffer for readback
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.queue_families().graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        // SAFETY: Creating transient command pool for screenshot readback
        let pool = unsafe { self.device.raw().create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: Allocating one-time command buffer
        let cmd = unsafe { self.device.raw().allocate_command_buffers(&alloc_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: Recording transfer commands to copy swapchain image into staging buffer
        unsafe {
            self.device.raw().begin_command_buffer(cmd, &begin_info)?;

            // Transition swapchain image from PRESENT_SRC_KHR to TRANSFER_SRC_OPTIMAL
            let barrier_to_src = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(swapchain_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier_to_src));
            self.device.raw().cmd_pipeline_barrier2(cmd, &dep_info);

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            self.device.raw().cmd_copy_image_to_buffer(
                cmd,
                swapchain_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buf.raw(),
                std::slice::from_ref(&copy_region),
            );

            // Transition back to PRESENT_SRC_KHR
            let barrier_back = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::empty())
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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

            let dep_info_back = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier_back));
            self.device.raw().cmd_pipeline_barrier2(cmd, &dep_info_back);

            self.device.raw().end_command_buffer(cmd)?;

            let cmd_bufs = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            self.device.raw().queue_submit(
                self.device.graphics_queue(),
                &[submit_info],
                vk::Fence::null(),
            )?;
            self.device
                .raw()
                .queue_wait_idle(self.device.graphics_queue())?;

            self.device.raw().destroy_command_pool(pool, None);
        }

        // 3. Read back pixels and convert to RGBA
        let num_bytes = usize::try_from(byte_size)
            .map_err(|_| GpuError::Allocation("Screenshot byte size overflow".into()))?;
        let mut raw_bytes = vec![0u8; num_bytes];
        staging_buf.read_bytes(&mut raw_bytes)?;

        let mut pixels = vec![0u8; num_bytes];
        let is_bgr = self.swapchain.format() == vk::Format::B8G8R8A8_SRGB
            || self.swapchain.format() == vk::Format::B8G8R8A8_UNORM;
        if is_bgr {
            for (src, dst) in raw_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(pixels.as_chunks_mut::<4>().0.iter_mut())
            {
                dst[0] = src[2]; // R
                dst[1] = src[1]; // G
                dst[2] = src[0]; // B
                dst[3] = src[3]; // A
            }
        } else {
            pixels.copy_from_slice(&raw_bytes);
        }

        staging_buf.destroy(self.device.raw(), &self.allocator);

        Ok((width, height, pixels))
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
