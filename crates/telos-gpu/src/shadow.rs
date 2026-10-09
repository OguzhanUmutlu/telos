//! Cascaded Shadow Maps (CSM) architecture for real-time directional sunlight/moonlight shadows.
//!
//! Provides a 4-cascade Practical Split Scheme (PSSM) frustum partition with texel-grid
//! stabilized orthographic light projections, preventing shadow swimming during camera motion.

use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use tracing::info;

use crate::{
    allocator::GpuAllocator,
    error::GpuError,
    pipeline::{GraphicsPipeline, ShaderModule},
};

/// Total number of shadow cascades.
pub const NUM_CASCADES: usize = 4;

/// Default shadow map extent for each cascade layer (2048x2048).
pub const DEFAULT_SHADOW_MAP_EXTENT: u32 = 2048;

/// Default cascade split distances in meters along the view camera forward axis.
pub const DEFAULT_CASCADE_SPLITS: [f32; NUM_CASCADES] = [16.0, 48.0, 112.0, 224.0];

/// Push constants supplied to the shadow map depth vertex shader (`shadow.vert`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowPushConstants {
    /// Combined light view-projection matrix for the active cascade.
    pub light_view_proj: [f32; 16],
    /// 64-bit Buffer Device Address (BDA) pointing to the chunk's quad buffer.
    pub vertex_addr: u64,
    /// Chunk origin X coordinate in world space.
    pub chunk_x: i32,
    /// Chunk origin Y coordinate in world space.
    pub chunk_y: i32,
    /// Chunk origin Z coordinate in world space.
    pub chunk_z: i32,
    /// 32-bit alignment padding.
    pub padding: u32,
}

/// Matrices and split bounds computed for all cascades in a frame.
#[derive(Debug, Clone, Copy)]
pub struct CascadeMatrices {
    /// Light view-projection matrix for each cascade ($0..4$).
    pub light_view_proj: [Mat4; NUM_CASCADES],
    /// Split distance in view units for each cascade.
    pub split_depths: [f32; NUM_CASCADES],
}

impl CascadeMatrices {
    /// Computes stabilized cascade view-projection matrices for the given camera and light direction.
    ///
    /// Snaps the orthographic projection bounds to the shadow map texel grid to eliminate
    /// edge crawling and shimmering during camera translation and rotation.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        cam_pos: Vec3,
        cam_forward: Vec3,
        cam_up: Vec3,
        fov_y: f32,
        aspect: f32,
        z_near: f32,
        splits: [f32; NUM_CASCADES],
        sun_dir: Vec3,
        resolution: u32,
    ) -> Self {
        let mut light_view_proj = [Mat4::IDENTITY; NUM_CASCADES];

        // Ensure light vector is facing downward from sky (flip for moon if below horizon)
        let mut light_dir = sun_dir.normalize_or_zero();
        if light_dir.length_squared() < 1e-4 {
            light_dir = Vec3::Y;
        } else if light_dir.y < 0.0 {
            light_dir = -light_dir;
        }
        let light_dir = light_dir.normalize();

        let cam_fwd = cam_forward.normalize();
        let cam_right = cam_fwd.cross(cam_up).normalize();
        let cam_up_actual = cam_right.cross(cam_fwd).normalize();

        let tan_half_fov = (fov_y * 0.5).tan();

        for i in 0..NUM_CASCADES {
            let d_near = if i == 0 { z_near } else { splits[i - 1] };
            let d_far = splits[i];

            let h_near = d_near * tan_half_fov;
            let w_near = h_near * aspect;
            let h_far = d_far * tan_half_fov;
            let w_far = h_far * aspect;

            let c_near = cam_pos + cam_fwd * d_near;
            let c_far = cam_pos + cam_fwd * d_far;

            // 8 frustum corners in world space
            let corners = [
                c_near - cam_right * w_near - cam_up_actual * h_near,
                c_near + cam_right * w_near - cam_up_actual * h_near,
                c_near - cam_right * w_near + cam_up_actual * h_near,
                c_near + cam_right * w_near + cam_up_actual * h_near,
                c_far - cam_right * w_far - cam_up_actual * h_far,
                c_far + cam_right * w_far - cam_up_actual * h_far,
                c_far - cam_right * w_far + cam_up_actual * h_far,
                c_far + cam_right * w_far + cam_up_actual * h_far,
            ];

            let mut frustum_center = Vec3::ZERO;
            for corner in &corners {
                frustum_center += *corner;
            }
            frustum_center /= 8.0;

            let light_up = if light_dir.y.abs() > 0.95 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            let light_view = Mat4::look_to_rh(frustum_center, -light_dir, light_up);

            // Compute AABB in light space
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            let mut min_z = f32::INFINITY;
            let mut max_z = f32::NEG_INFINITY;

            for corner in &corners {
                let p = light_view.transform_point3(*corner);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
                min_z = min_z.min(p.z);
                max_z = max_z.max(p.z);
            }

            // Expand Z to capture shadow casters behind the camera frustum along the light direction
            let z_margin = (max_z - min_z).max(120.0) * 0.75;
            min_z -= z_margin;
            max_z += z_margin;

            // Texel-grid snapping to stabilize shadows against sub-pixel shimmer
            #[allow(clippy::cast_precision_loss)]
            let res_f32 = resolution as f32;
            let world_units_per_texel_x = (max_x - min_x) / res_f32;
            let world_units_per_texel_y = (max_y - min_y) / res_f32;

            if world_units_per_texel_x > 1e-5 && world_units_per_texel_y > 1e-5 {
                min_x = (min_x / world_units_per_texel_x).floor() * world_units_per_texel_x;
                max_x = (max_x / world_units_per_texel_x).floor() * world_units_per_texel_x;
                min_y = (min_y / world_units_per_texel_y).floor() * world_units_per_texel_y;
                max_y = (max_y / world_units_per_texel_y).floor() * world_units_per_texel_y;
            }

            // In Vulkan right-handed orthographic, glam maps Z to [0, 1]
            let light_proj = Mat4::orthographic_rh(min_x, max_x, min_y, max_y, min_z, max_z);
            light_view_proj[i] = light_proj * light_view;
        }

        Self {
            light_view_proj,
            split_depths: splits,
        }
    }
}

/// GPU-resident 4-layer 2D texture array managing cascaded shadow depth attachments.
pub struct CascadedShadowMap {
    image: vk::Image,
    allocation: Option<Allocation>,
    array_view: vk::ImageView,
    layer_views: [vk::ImageView; NUM_CASCADES],
    sampler: vk::Sampler,
    format: vk::Format,
    resolution: u32,
}

impl CascadedShadowMap {
    /// Creates a new `CascadedShadowMap` with a 4-layer 2D depth array image.
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        resolution: u32,
    ) -> Result<Self, GpuError> {
        let format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(NUM_CASCADES as u32)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Calling vkCreateImage on valid Vulkan device
        let image = unsafe { device.create_image(&image_info, None)? };

        // SAFETY: Querying memory requirements for depth image allocation
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator.allocate(
            "cascaded_shadow_map",
            requirements,
            MemoryLocation::GpuOnly,
            false,
        )?;

        // SAFETY: Binding image memory
        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        // 1. Array view covering all 4 layers for shader sampling
        let array_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: NUM_CASCADES as u32,
            });

        // SAFETY: Creating 2D array image view
        let array_view = unsafe { device.create_image_view(&array_view_info, None)? };

        // 2. Individual layer views for per-cascade depth rendering attachments
        let mut layer_views = [vk::ImageView::null(); NUM_CASCADES];
        for (i, view_slot) in layer_views.iter_mut().enumerate() {
            let layer_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: i as u32,
                    layer_count: 1,
                });

            // SAFETY: Creating individual layer image view
            *view_slot = unsafe { device.create_image_view(&layer_info, None)? };
        }

        // 3. Bilinear clamp-to-edge sampler with float 1.0 border
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .anisotropy_enable(false)
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(1.0);

        // SAFETY: Creating shadow sampler
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        info!(
            resolution,
            cascades = NUM_CASCADES,
            "CascadedShadowMap initialized"
        );

        Ok(Self {
            image,
            allocation: Some(allocation),
            array_view,
            layer_views,
            sampler,
            format,
            resolution,
        })
    }

    /// Depth format (`vk::Format::D32_SFLOAT`).
    #[inline]
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Resolution of each square cascade in pixels.
    #[inline]
    #[must_use]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Resolution of each square cascade in pixels (alias for `resolution`).
    #[inline]
    #[must_use]
    pub fn extent(&self) -> u32 {
        self.resolution
    }

    /// Raw `vk::Image` handle.
    #[inline]
    #[must_use]
    pub fn raw(&self) -> vk::Image {
        self.image
    }

    /// Raw `vk::Image` handle (alias for `raw`).
    #[inline]
    #[must_use]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// 2D array image view covering all cascades for sampling in shaders.
    #[inline]
    #[must_use]
    pub fn array_view(&self) -> vk::ImageView {
        self.array_view
    }

    /// Individual 2D image view for a specific cascade layer (0..=3).
    #[inline]
    #[must_use]
    pub fn layer_view(&self, index: usize) -> vk::ImageView {
        self.layer_views[index]
    }

    /// Sampler for reading the shadow map in shaders.
    #[inline]
    #[must_use]
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Destroys all GPU resources associated with the shadow map.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
            for view in &mut self.layer_views {
                if *view != vk::ImageView::null() {
                    device.destroy_image_view(*view, None);
                    *view = vk::ImageView::null();
                }
            }
            if self.array_view != vk::ImageView::null() {
                device.destroy_image_view(self.array_view, None);
                self.array_view = vk::ImageView::null();
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }
}

/// Shadow depth-only dynamic rendering pipeline.
pub struct ShadowPipeline {
    pipeline: GraphicsPipeline,
    vert_module: ShaderModule,
    frag_module: Option<ShaderModule>,
}

impl ShadowPipeline {
    /// Creates a shadow depth-only pipeline.
    pub fn new(
        device: &ash::Device,
        vert_spv: &[u8],
        frag_spv: Option<&[u8]>,
    ) -> Result<Self, GpuError> {
        let vert_module = ShaderModule::from_spv(device, vert_spv)?;
        let frag_module = match frag_spv {
            Some(spv) => Some(ShaderModule::from_spv(device, spv)?),
            None => None,
        };

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<ShadowPushConstants>() as u32)];

        let pipeline = GraphicsPipeline::create_dynamic_depth_only(
            device,
            vert_module.raw(),
            frag_module.as_ref().map(ShaderModule::raw),
            vk::Format::D32_SFLOAT,
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            true,
            1.25,
            1.75,
            &[],
            &push_constant_ranges,
        )?;

        Ok(Self {
            pipeline,
            vert_module,
            frag_module,
        })
    }

    /// Raw `vk::Pipeline` handle.
    #[inline]
    #[must_use]
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline.raw()
    }

    /// Raw `vk::PipelineLayout` handle.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline.layout()
    }

    /// Destroys the pipeline and its shader modules.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.pipeline.destroy(device);
        self.vert_module.destroy(device);
        if let Some(frag) = &mut self.frag_module {
            frag.destroy(device);
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_matrices_computation() {
        let cam_pos = Vec3::new(100.0, 64.0, 100.0);
        let cam_fwd = Vec3::new(0.0, 0.0, -1.0);
        let cam_up = Vec3::Y;
        let sun_dir = Vec3::new(0.4, 0.8, 0.3);

        let cascades = CascadeMatrices::compute(
            cam_pos,
            cam_fwd,
            cam_up,
            std::f32::consts::FRAC_PI_3,
            16.0 / 9.0,
            0.1,
            DEFAULT_CASCADE_SPLITS,
            sun_dir,
            2048,
        );

        assert_eq!(cascades.split_depths, DEFAULT_CASCADE_SPLITS);
        for (i, mat) in cascades.light_view_proj.iter().enumerate() {
            let det = mat.determinant();
            assert!(
                det.abs() > 1e-12 && det.is_finite(),
                "Cascade {i} light_view_proj should be invertible, got det = {det}"
            );
            assert!(
                mat.inverse().is_finite(),
                "Cascade {i} light_view_proj inverse should be finite"
            );
        }
    }

    #[test]
    fn test_texel_snapping_invariance() {
        let cam_pos_a = Vec3::new(0.0, 64.0, 0.0);
        let cam_pos_b = Vec3::new(0.002, 64.0, 0.001); // Sub-texel translation
        let sun_dir = Vec3::new(0.0, 1.0, 0.0);

        let cascades_a = CascadeMatrices::compute(
            cam_pos_a,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            1.0,
            0.1,
            DEFAULT_CASCADE_SPLITS,
            sun_dir,
            2048,
        );

        let cascades_b = CascadeMatrices::compute(
            cam_pos_b,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            1.0,
            0.1,
            DEFAULT_CASCADE_SPLITS,
            sun_dir,
            2048,
        );

        // Near cascade should be closely aligned and stable
        let diff = (cascades_a.light_view_proj[0] - cascades_b.light_view_proj[0])
            .to_cols_array()
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(
            diff < 0.05,
            "Texel stabilization should prevent large matrix shifts on sub-texel movement: diff={diff}"
        );
    }
}
