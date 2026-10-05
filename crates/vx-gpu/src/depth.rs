//! Depth buffer allocation and image view management for dynamic rendering.

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use tracing::info;

use crate::{allocator::GpuAllocator, error::GpuError};

/// Manages a GPU-allocated depth attachment buffer.
pub struct DepthBuffer {
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl DepthBuffer {
    /// Creates a 2D depth attachment buffer with format `D32_SFLOAT`.
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        extent: vk::Extent2D,
    ) -> Result<Self, GpuError> {
        let format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating image on valid device
        let image = unsafe { device.create_image(&image_info, None)? };

        // SAFETY: Querying memory requirements for the depth image
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation =
            allocator.allocate("depth_buffer", requirements, MemoryLocation::GpuOnly, false)?;

        // SAFETY: Binding image memory
        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // SAFETY: Creating image view
        let view = unsafe { device.create_image_view(&view_info, None)? };

        info!(
            width = extent.width,
            height = extent.height,
            "Depth buffer initialized"
        );

        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent,
        })
    }

    /// Depth image format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Raw `vk::Image` handle.
    #[inline]
    #[must_use]
    pub fn raw(&self) -> vk::Image {
        self.image
    }

    /// Depth image view.
    #[inline]
    #[must_use]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Depth buffer resolution.
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Destroys view, image, and frees GPU allocation.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        if self.view != vk::ImageView::null() {
            // SAFETY: Destroying image view
            unsafe {
                device.destroy_image_view(self.view, None);
            }
            self.view = vk::ImageView::null();
        }

        if self.image != vk::Image::null() {
            // SAFETY: Destroying image
            unsafe {
                device.destroy_image(self.image, None);
            }
            self.image = vk::Image::null();
        }

        if let Some(alloc) = self.allocation.take()
            && let Err(e) = allocator.free(alloc)
        {
            tracing::error!("Failed to free depth buffer allocation: {e}");
        }
    }
}
