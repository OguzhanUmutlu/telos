//! Hierarchical-Z (Hi-Z) depth pyramid generation and MIN-reduction sampling.
//!
//! Provides conservative depth bounds for two-phase GPU occlusion culling.

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use tracing::info;

use crate::{allocator::GpuAllocator, error::GpuError};

/// Hierarchical-Z depth pyramid storing a power-of-two mip chain of minimum depth.
pub struct HiZPyramid {
    image: vk::Image,
    allocation: Option<Allocation>,
    full_view: vk::ImageView,
    mip_views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    extent: vk::Extent2D,
    mip_levels: u32,
}

impl HiZPyramid {
    /// Creates a new power-of-two Hi-Z pyramid for the given depth render extent.
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        render_extent: vk::Extent2D,
    ) -> Result<Self, GpuError> {
        let width: u32 = if render_extent.width > 1 {
            1 << render_extent.width.ilog2()
        } else {
            1
        };
        let height: u32 = if render_extent.height > 1 {
            1 << render_extent.height.ilog2()
        } else {
            1
        };

        let mip_levels = width.max(height).ilog2() + 1;
        let format = vk::Format::R32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating Hi-Z image on valid device
        let image = unsafe { device.create_image(&image_info, None)? };

        // SAFETY: Querying memory requirements for image
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation =
            allocator.allocate("hiz_pyramid", requirements, MemoryLocation::GpuOnly, false)?;

        // SAFETY: Binding image memory to allocation
        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        // Full mip-chain image view for sampling with reduction sampler
        let full_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        // SAFETY: Creating full mip image view
        let full_view = unsafe { device.create_image_view(&full_view_info, None)? };

        // Per-mip image views for storage image writes in compute shader
        let mut mip_views = Vec::with_capacity(mip_levels as usize);
        for i in 0..mip_levels {
            let mip_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // SAFETY: Creating individual storage image view for mip level
            let mip_view = unsafe { device.create_image_view(&mip_view_info, None)? };
            mip_views.push(mip_view);
        }

        // MIN reduction sampler for conservative occlusion testing
        let mut reduction_info = vk::SamplerReductionModeCreateInfo::default()
            .reduction_mode(vk::SamplerReductionMode::MIN);

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .push_next(&mut reduction_info);

        // SAFETY: Creating reduction sampler on device supporting sampler_filter_minmax
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        info!(
            width,
            height, mip_levels, "Hi-Z pyramid and MIN-reduction sampler initialized"
        );

        Ok(Self {
            image,
            allocation: Some(allocation),
            full_view,
            mip_views,
            sampler,
            extent: vk::Extent2D { width, height },
            mip_levels,
        })
    }

    /// Extent of Hi-Z level 0.
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Number of mip levels in pyramid.
    #[inline]
    #[must_use]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Raw `vk::Image` handle.
    #[inline]
    #[must_use]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Full image view sampling across all mip levels.
    #[inline]
    #[must_use]
    pub fn full_view(&self) -> vk::ImageView {
        self.full_view
    }

    /// Storage image view for specific mip level.
    #[inline]
    #[must_use]
    pub fn mip_view(&self, level: usize) -> vk::ImageView {
        self.mip_views[level]
    }

    /// MIN reduction sampler.
    #[inline]
    #[must_use]
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Destroys all views, samplers, images, and frees GPU allocation.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        // SAFETY: Destroying sampler
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
            for view in self.mip_views.drain(..) {
                if view != vk::ImageView::null() {
                    device.destroy_image_view(view, None);
                }
            }
            if self.full_view != vk::ImageView::null() {
                device.destroy_image_view(self.full_view, None);
                self.full_view = vk::ImageView::null();
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }
        }

        if let Some(alloc) = self.allocation.take()
            && let Err(e) = allocator.free(alloc)
        {
            tracing::error!("Failed to free HiZ pyramid allocation: {e}");
        }
    }
}
