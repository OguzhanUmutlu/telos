//! Vulkan 1.3 Entry, Instance, and Debug Utils messenger management.

use ash::{ext::debug_utils, vk};
use raw_window_handle::HasDisplayHandle;
use std::ffi::{CStr, c_char, c_void};
use tracing::{debug, error, info, warn};

use crate::error::GpuError;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHR_validation";

/// Wraps `ash::Entry`, `ash::Instance`, and the debug utils messenger.
#[allow(clippy::struct_field_names)]
pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Instance {
    /// Creates a new Vulkan 1.3 instance configured for modern dynamic rendering.
    ///
    /// Checks display handle for required windowing surface extensions and enables
    /// validation layers if available on the system.
    pub fn new<D: HasDisplayHandle>(
        display_handle: &D,
        enable_validation: bool,
    ) -> Result<Self, GpuError> {
        // SAFETY: Entry::load loads the system Vulkan dynamic library libvulkan.so.1
        let entry = unsafe { ash::Entry::load()? };

        let app_name = c"Voxel";
        let engine_name = c"VoxelEngine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let raw_display = display_handle
            .display_handle()
            .map_err(|e| GpuError::Surface(e.to_string()))?
            .as_raw();

        let window_extensions = ash_window::enumerate_required_extensions(raw_display)
            .map_err(|e| GpuError::Surface(e.to_string()))?;

        let mut extension_names: Vec<*const c_char> = window_extensions.to_vec();

        // Check validation layer support
        let mut layer_names: Vec<*const c_char> = Vec::new();
        let mut debug_utils_enabled = false;

        if enable_validation {
            // SAFETY: Calling Vulkan API to enumerate available instance layers
            let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
            let has_validation_layer = available_layers.iter().any(|layer| {
                // SAFETY: layer_name is a null-terminated C string array in VkLayerProperties
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                name == VALIDATION_LAYER
            });

            if has_validation_layer {
                info!("Enabling Vulkan validation layer: VK_LAYER_KHR_validation");
                layer_names.push(VALIDATION_LAYER.as_ptr());
                extension_names.push(debug_utils::NAME.as_ptr());
                debug_utils_enabled = true;
            } else {
                warn!(
                    "Vulkan validation layers requested but 'VK_LAYER_KHR_validation' not found on system. Install vulkan-validationlayers."
                );
            }
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        // SAFETY: Creating Vulkan instance with verified extension and layer pointers
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        info!("Vulkan 1.3 instance created successfully");

        let debug_utils_data = if debug_utils_enabled {
            let debug_loader = debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            // SAFETY: Registering debug messenger with verified debug_utils instance
            let messenger =
                unsafe { debug_loader.create_debug_utils_messenger(&messenger_info, None)? };
            Some((debug_loader, messenger))
        } else {
            None
        };

        Ok(Self {
            entry,
            instance,
            debug_utils: debug_utils_data,
        })
    }

    /// Returns a reference to the `ash::Entry`.
    #[must_use]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// Returns a reference to the `ash::Instance`.
    #[must_use]
    pub fn raw(&self) -> &ash::Instance {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: Cleaning up debug messenger before destroying instance
        unsafe {
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
            debug!("Vulkan instance destroyed");
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = if p_callback_data.is_null() {
        "Unknown Vulkan message (null callback data)"
    } else {
        // SAFETY: p_callback_data is provided by Vulkan runtime and guaranteed valid
        let data = unsafe { &*p_callback_data };
        if data.p_message.is_null() {
            "Unknown Vulkan message"
        } else {
            // SAFETY: p_message is null-terminated UTF-8 string provided by loader
            unsafe {
                CStr::from_ptr(data.p_message)
                    .to_str()
                    .unwrap_or("Non-UTF8 message")
            }
        }
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(target: "vulkan", type_flags = ?message_type, "{message}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: "vulkan", type_flags = ?message_type, "{message}");
    } else {
        debug!(target: "vulkan", type_flags = ?message_type, "{message}");
    }

    vk::FALSE
}
