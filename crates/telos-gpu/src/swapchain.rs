//! Vulkan window surface and Swapchain management.

use ash::{
    khr::{surface, swapchain},
    vk,
};
use tracing::info;

use crate::error::GpuError;

type SwapchainBundle = (
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::SurfaceFormatKHR,
    vk::Extent2D,
);

/// Manages the Vulkan window surface, swapchain images, and presentation.
#[allow(clippy::struct_field_names)]
pub struct Swapchain {
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
}

impl Swapchain {
    /// Creates the swapchain for an existing window surface.
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: surface::Instance,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuError> {
        let swapchain_loader = swapchain::Device::new(instance, device);

        let (swapchain, images, image_views, format, extent) = Self::create_swapchain_internal(
            &surface_loader,
            surface,
            &swapchain_loader,
            device,
            physical_device,
            width,
            height,
            vk::SwapchainKHR::null(),
        )?;

        Ok(Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format,
            extent,
        })
    }

    /// Recreates the swapchain with the new window dimensions.
    pub fn resize(
        &mut self,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        if width == 0 || height == 0 {
            return Ok(()); // Window minimized, skip recreation until restored
        }

        let (new_swapchain, images, image_views, format, extent) = Self::create_swapchain_internal(
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            device,
            physical_device,
            width,
            height,
            self.swapchain,
        )?;

        self.cleanup_swapchain_resources(device);

        self.swapchain = new_swapchain;
        self.images = images;
        self.image_views = image_views;
        self.format = format;
        self.extent = extent;

        Ok(())
    }

    /// Acquires the next available swapchain image index.
    pub fn acquire_next_image(
        &self,
        semaphore: vk::Semaphore,
        timeout_ns: u64,
    ) -> Result<(u32, bool), GpuError> {
        // SAFETY: Calling vkAcquireNextImageKHR on valid swapchain
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                timeout_ns,
                semaphore,
                vk::Fence::null(),
            )
        };

        match result {
            Ok((index, is_suboptimal)) => Ok((index, is_suboptimal)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok((0, true)),
            Err(e) => Err(GpuError::Vk(e)),
        }
    }

    /// Presents an image to the window surface.
    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, GpuError> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // SAFETY: Queue present call on verified graphics/present queue
        let result = unsafe { self.swapchain_loader.queue_present(queue, &present_info) };

        match result {
            Ok(is_suboptimal) => Ok(is_suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(e) => Err(GpuError::Vk(e)),
        }
    }

    /// Surface handle getter.
    #[must_use]
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// Surface loader getter.
    #[must_use]
    pub fn surface_loader(&self) -> &surface::Instance {
        &self.surface_loader
    }

    /// Current extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Chosen surface format.
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    /// All swapchain image views.
    #[must_use]
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    /// Swapchain image view for a specific index.
    #[must_use]
    pub fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// Swapchain image handle for a specific index.
    #[must_use]
    pub fn image(&self, index: usize) -> vk::Image {
        self.images[index]
    }

    fn cleanup_swapchain_resources(&mut self, device: &ash::Device) {
        // SAFETY: Cleaning up image views and old swapchain handle
        unsafe {
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_swapchain_internal(
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
        swapchain_loader: &swapchain::Device,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<SwapchainBundle, GpuError> {
        // SAFETY: Querying surface capabilities and formats on physical device
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        // Choose SRGB surface format
        let format = formats
            .iter()
            .copied()
            .find(|f| {
                (f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::R8G8B8A8_SRGB)
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(formats[0]);

        // Choose present mode: prefer MAILBOX (uncapped, no tearing) or IMMEDIATE (uncapped, high refresh 144Hz+) over FIFO (60Hz vsync lock)
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        };
        tracing::info!("Selected swapchain present mode: {present_mode:?}");

        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        } else {
            caps.current_extent
        };

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | (caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC);

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        // SAFETY: Calling vkCreateSwapchainKHR with valid parameters
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let mut image_views = Vec::with_capacity(images.len());
        for &img in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            // SAFETY: Creating image view for each swapchain image
            let view = unsafe { device.create_image_view(&view_info, None)? };
            image_views.push(view);
        }

        info!(
            width = extent.width,
            height = extent.height,
            format = ?format.format,
            present_mode = ?present_mode,
            images = images.len(),
            "Vulkan swapchain initialized"
        );

        Ok((swapchain, images, image_views, format, extent))
    }

    /// Destroys all swapchain and surface resources.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.cleanup_swapchain_resources(device);
        // SAFETY: Destroying Vulkan surface
        unsafe {
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }
        }
    }
}
