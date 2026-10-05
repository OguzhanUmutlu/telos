//! Physical device selection, scoring, feature chaining, and logical device management.

use ash::{khr::swapchain, vk};
use std::{env, ffi::CStr};
use tracing::{info, warn};

use crate::error::GpuError;

/// Queue family indices identified on the physical device.
#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    /// Queue family supporting graphics and surface presentation.
    pub graphics_family: u32,
    /// Queue family supporting compute (may be same as graphics).
    pub compute_family: u32,
    /// Queue family supporting transfer (may be same as graphics).
    pub transfer_family: u32,
}

/// Logical device wrapper with queue handles.
#[allow(clippy::struct_field_names)]
pub struct Device {
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    queue_families: QueueFamilyIndices,
}

impl Device {
    /// Returns the raw `ash::Device`.
    #[must_use]
    pub fn raw(&self) -> &ash::Device {
        &self.device
    }

    /// Returns the chosen physical device handle.
    #[must_use]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the primary graphics queue.
    #[must_use]
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Returns the queue family indices.
    #[must_use]
    pub fn queue_families(&self) -> QueueFamilyIndices {
        self.queue_families
    }

    /// Waits for all device operations to idle before destruction or resizing.
    pub fn wait_idle(&self) -> Result<(), GpuError> {
        // SAFETY: Calling vkDeviceWaitIdle with valid logical device
        unsafe {
            self.device.device_wait_idle()?;
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: Waiting idle and destroying logical device
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            info!("Vulkan logical device destroyed");
        }
    }
}

/// Selects the best physical device and creates the Vulkan 1.3 logical device.
pub fn create_device(
    instance: &ash::Instance,
    surface_khr: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Result<Device, GpuError> {
    // SAFETY: Enumerating physical devices on initialized instance
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.is_empty() {
        return Err(GpuError::NoSuitableDevice(
            "no Vulkan physical devices found on system",
        ));
    }

    let env_override = env::var("VOXEL_GPU").ok();

    let mut scored_devices: Vec<(
        i32,
        vk::PhysicalDevice,
        vk::PhysicalDeviceProperties,
        QueueFamilyIndices,
    )> = Vec::new();

    for &pdevice in &physical_devices {
        // SAFETY: Querying properties on valid physical device
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_str()
                .unwrap_or("Unknown")
        };

        if let Some(queue_indices) =
            find_queue_families(instance, pdevice, surface_khr, surface_loader)
        {
            let mut score = score_physical_device(&props);

            // Apply environment override preference
            if let Some(ref override_name) = env_override
                && name.to_lowercase().contains(&override_name.to_lowercase())
            {
                score += 1_000_000;
            }

            info!(
                gpu_name = name,
                score = score,
                device_type = ?props.device_type,
                api_version = format!("{}.{}.{}", vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version), vk::api_version_patch(props.api_version)),
                "Evaluated physical device"
            );

            scored_devices.push((score, pdevice, props, queue_indices));
        } else {
            warn!(
                gpu_name = name,
                "Device skipped: Missing required graphics or presentation queue"
            );
        }
    }

    scored_devices.sort_by_key(|a| std::cmp::Reverse(a.0));

    let (_, chosen_device, chosen_props, queue_indices) =
        scored_devices
            .into_iter()
            .next()
            .ok_or(GpuError::NoSuitableDevice(
                "no physical device satisfied queue and extension requirements",
            ))?;

    let chosen_name = unsafe {
        CStr::from_ptr(chosen_props.device_name.as_ptr())
            .to_str()
            .unwrap_or("Unknown")
    };
    info!(gpu_name = chosen_name, "Selected primary GPU for rendering");

    let priorities = [1.0f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_indices.graphics_family)
        .queue_priorities(&priorities)];

    let enabled_extensions = [swapchain::NAME.as_ptr()];

    // Baseline features for Vulkan 1.3 (ADR-02)
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .runtime_descriptor_array(true);

    let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&enabled_extensions)
        .enabled_features(&features)
        .push_next(&mut features12)
        .push_next(&mut features13);

    // SAFETY: Creating logical device with verified feature chain and extension pointers
    let device = unsafe { instance.create_device(chosen_device, &device_create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(queue_indices.graphics_family, 0) };

    info!("Vulkan logical device and graphics queue created");

    Ok(Device {
        device,
        physical_device: chosen_device,
        graphics_queue,
        queue_families: queue_indices,
    })
}

fn score_physical_device(props: &vk::PhysicalDeviceProperties) -> i32 {
    let mut score = 0;

    match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 10_000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 1_000,
        vk::PhysicalDeviceType::VIRTUAL_GPU => score += 500,
        vk::PhysicalDeviceType::CPU => score += 100,
        _ => {}
    }

    // Require at least Vulkan 1.3
    if props.api_version < vk::API_VERSION_1_3 {
        return -1_000_000;
    }

    score
}

fn find_queue_families(
    instance: &ash::Instance,
    pdevice: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Option<QueueFamilyIndices> {
    // SAFETY: Querying queue families on valid physical device
    let families = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };

    let mut graphics = None;
    let mut compute = None;
    let mut transfer = None;

    for (index, family) in families.iter().enumerate() {
        let i = index as u32;

        let supports_surface = unsafe {
            surface_loader
                .get_physical_device_surface_support(pdevice, i, surface)
                .unwrap_or(false)
        };

        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && supports_surface {
            graphics = Some(i);
        }

        if family.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute.is_none() {
            compute = Some(i);
        }

        if family.queue_flags.contains(vk::QueueFlags::TRANSFER) && transfer.is_none() {
            transfer = Some(i);
        }
    }

    let g = graphics?;
    Some(QueueFamilyIndices {
        graphics_family: g,
        compute_family: compute.unwrap_or(g),
        transfer_family: transfer.unwrap_or(g),
    })
}
