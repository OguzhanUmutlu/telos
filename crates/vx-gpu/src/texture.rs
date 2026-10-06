//! Vulkan 2D texture array image, view, and sampler management.

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use tracing::info;

use crate::{allocator::GpuAllocator, buffer::GpuBuffer, error::GpuError};

/// Region specification for uploading a mip slice to a 2D texture array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureMipRegion {
    /// Byte offset into the staging data buffer.
    pub buffer_offset: u64,
    /// Target array slice / layer index.
    pub layer: u32,
    /// Target mipmap level.
    pub mip_level: u32,
    /// Width of this mip slice in pixels.
    pub width: u32,
    /// Height of this mip slice in pixels.
    pub height: u32,
}

/// A GPU-allocated Vulkan 2D texture array (`VkImage`) with combined image sampler.
pub struct GpuTextureArray {
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
    sampler: vk::Sampler,
    extent: vk::Extent2D,
    layer_count: u32,
    mip_levels: u32,
}

impl GpuTextureArray {
    /// Uploads pixel bytes and mip slices into a device-local 2D texture array via staging buffer.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn from_raw_mips(
        device: &ash::Device,
        allocator: &GpuAllocator,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        resolution: u32,
        layer_count: u32,
        mip_levels: u32,
        pixel_data: &[u8],
        copy_regions: &[TextureMipRegion],
    ) -> Result<Self, GpuError> {
        let extent = vk::Extent2D {
            width: resolution,
            height: resolution,
        };
        let format = vk::Format::R8G8B8A8_SRGB;

        // 1. Create staging buffer with all pixel data
        let mut staging = GpuBuffer::new(
            device,
            allocator,
            "texture_array_staging",
            pixel_data.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;

        staging.write_bytes(pixel_data)?;

        // 2. Create target 2D array image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(layer_count)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating image on valid device
        let image = unsafe { device.create_image(&image_info, None)? };
        // SAFETY: Querying memory requirements for image
        let reqs = unsafe { device.get_image_memory_requirements(image) };

        let allocation =
            allocator.allocate("texture_array", reqs, MemoryLocation::GpuOnly, false)?;

        // SAFETY: Binding image memory
        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        // 3. Record upload commands
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: Allocating single-use transfer command buffer
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: Recording barriers and copy commands
        unsafe {
            device.begin_command_buffer(cmd, &begin_info)?;

            // Transition all layers and mips: UNDEFINED -> TRANSFER_DST_OPTIMAL
            let to_transfer_dst = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count,
                });

            let dst_barriers = [to_transfer_dst];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&dst_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            // Copy each mip slice
            let mut vk_regions = Vec::with_capacity(copy_regions.len());
            for r in copy_regions {
                vk_regions.push(vk::BufferImageCopy {
                    buffer_offset: r.buffer_offset,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: r.mip_level,
                        base_array_layer: r.layer,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: r.width,
                        height: r.height,
                        depth: 1,
                    },
                });
            }

            device.cmd_copy_buffer_to_image(
                cmd,
                staging.raw(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk_regions,
            );

            // Transition: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let to_shader_read = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count,
                });

            let read_barriers = [to_shader_read];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&read_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            device.end_command_buffer(cmd)?;

            let cmd_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;

            device.free_command_buffers(command_pool, &[cmd]);
        }

        // 4. Destroy staging buffer
        staging.destroy(device, allocator);

        // 5. Create 2D Array Image View
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count,
            });

        // SAFETY: Creating image view for 2D array
        let view = unsafe { device.create_image_view(&view_info, None)? };

        // 6. Create Sampler with Repeat and Anisotropic Trilinear Filtering
        #[allow(clippy::cast_precision_loss)]
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);

        // SAFETY: Creating texture sampler
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        info!(
            width = resolution,
            height = resolution,
            layers = layer_count,
            mips = mip_levels,
            "GPU 2D texture array initialized"
        );

        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            sampler,
            extent,
            layer_count,
            mip_levels,
        })
    }

    /// Image view handle configured as `TYPE_2D_ARRAY`.
    #[inline]
    #[must_use]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Sampler configured for repeating texture array sampling.
    #[inline]
    #[must_use]
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Layer resolution extent.
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Total number of 2D layers in the array.
    #[inline]
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Number of mip levels.
    #[inline]
    #[must_use]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Destroys sampler, view, image, and frees GPU allocation.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        // SAFETY: Destroying Vulkan objects with valid device handle
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }

            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
                self.view = vk::ImageView::null();
            }

            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }
        }

        if let Some(alloc) = self.allocation.take()
            && let Err(e) = allocator.free(alloc)
        {
            tracing::error!("Failed to free texture array allocation: {e}");
        }
    }
}

/// A GPU-allocated Vulkan 2D texture (`VkImage`) with combined image sampler.
pub struct GpuTexture2d {
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
    sampler: vk::Sampler,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl GpuTexture2d {
    /// Creates an empty 2D texture image, view, and sampler.
    pub fn new_empty(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        sampler_filter: vk::Filter,
    ) -> Result<Self, GpuError> {
        let extent = vk::Extent2D { width, height };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating image on valid device
        let image = unsafe { device.create_image(&image_info, None)? };
        // SAFETY: Querying memory requirements for image
        let reqs = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator.allocate("texture_2d", reqs, MemoryLocation::GpuOnly, false)?;

        // SAFETY: Binding image memory
        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Creating image view for 2D texture
        let view = unsafe { device.create_image_view(&view_info, None)? };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(sampler_filter)
            .min_filter(sampler_filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);

        // SAFETY: Creating sampler for 2D texture
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            sampler,
            extent,
            format,
        })
    }

    /// Image handle.
    #[inline]
    #[must_use]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Image view handle.
    #[inline]
    #[must_use]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Combined sampler handle.
    #[inline]
    #[must_use]
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Texture extent in pixels.
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Texture pixel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Destroys sampler, view, image, and frees GPU allocation.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        // SAFETY: Destroying Vulkan objects with valid device handle
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }

            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
                self.view = vk::ImageView::null();
            }

            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }
        }

        if let Some(alloc) = self.allocation.take()
            && let Err(e) = allocator.free(alloc)
        {
            tracing::error!("Failed to free texture_2d allocation: {e}");
        }
    }
}
