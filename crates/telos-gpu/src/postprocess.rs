//! Extensible post-processing framework with SSAO, atmospheric horizon fog, and ACES tonemapping.

use ash::vk;
use std::path::Path;
use tracing::info;

use crate::{
    allocator::GpuAllocator,
    error::GpuError,
    pipeline::{GraphicsPipeline, ShaderModule},
    reloader::ShaderCompiler,
    texture::GpuTexture2d,
};

/// Push constants for the Screen-Space Ambient Occlusion (SSAO) pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoPushConstants {
    /// Inverse projection matrix to reconstruct view-space positions.
    pub inv_proj: [f32; 16],
    /// Projection matrix to project view-space sample points to clip space.
    pub proj: [f32; 16],
    /// Screen dimensions (width, height) in pixels.
    pub screen_size: [f32; 2],
    /// Sampling hemisphere radius in view units (meters).
    pub radius: f32,
    /// Depth comparison bias to avoid self-occlusion artifacts.
    pub bias: f32,
}

/// Push constants for the Fast Approximate Anti-Aliasing (FXAA 3.11 Quality) pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FxaaPushConstants {
    /// Texel size: (1.0 / width, 1.0 / height).
    pub texel_size: [f32; 2],
    /// Enabled feature flags: bit 0 = FXAA active (1 = on, 0 = passthrough).
    pub flags: u32,
    /// Subpixel antialiasing quality / sharpness (default: 0.75).
    pub subpix: f32,
    /// Edge detection contrast threshold (default: 0.125).
    pub edge_threshold: f32,
    /// Minimum edge detection threshold for dark areas (default: 0.0312).
    pub edge_threshold_min: f32,
    /// Alignment padding to 32 bytes (std430 layout).
    pub padding: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<FxaaPushConstants>() == 32);

impl Default for FxaaPushConstants {
    fn default() -> Self {
        Self {
            texel_size: [0.0, 0.0],
            flags: 1,
            subpix: 0.75,
            edge_threshold: 0.125,
            edge_threshold_min: 0.0312,
            padding: [0.0; 2],
        }
    }
}

/// Push constants for the composite pass with atmospheric fog, SSAO blending, and tonemapping.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostCompositePushConstants {
    /// Inverse view-projection matrix to reconstruct world-space positions.
    pub inv_view_proj: [f32; 16],
    /// World-space camera eye position.
    pub cam_pos: [f32; 3],
    /// World time of day in ticks (0..=24000).
    pub time_of_day: f32,
    /// Normalized direction toward the sun.
    pub sun_dir: [f32; 3],
    /// Maximum view distance in chunks.
    pub view_distance: f32,
    /// Fog density multiplier.
    pub fog_density: f32,
    /// Enabled feature flags: bit 0 = SSAO, bit 1 = Fog, bit 2 = Tonemapping, bit 3 = Vignette.
    pub flags: u32,
    /// Screen dimensions (width, height) in pixels.
    pub screen_size: [f32; 2],
}

/// Uniform buffer data for cascaded shadow mapping in the composite pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeUniforms {
    /// Light view-projection matrix for each cascade (0..=3).
    pub light_view_proj: [[f32; 16]; 4],
    /// Split distance along view Z for each cascade.
    pub cascade_splits: [f32; 4],
    /// Depth bias to prevent self-shadowing acne.
    pub shadow_bias: f32,
    /// Normal bias along surface normal.
    pub normal_bias: f32,
    /// 64-bit alignment padding.
    pub padding: [f32; 2],
}

impl Default for CascadeUniforms {
    fn default() -> Self {
        Self {
            light_view_proj: [[0.0; 16]; 4],
            cascade_splits: [16.0, 48.0, 112.0, 224.0],
            shadow_bias: 0.0015,
            normal_bias: 0.04,
            padding: [0.0; 2],
        }
    }
}

impl CascadeUniforms {
    /// Constructs `CascadeUniforms` from calculated `CascadeMatrices`.
    #[must_use]
    pub fn from_cascade_matrices(
        cascades: &crate::shadow::CascadeMatrices,
        shadow_bias: f32,
        normal_bias: f32,
    ) -> Self {
        let mut light_view_proj = [[0.0; 16]; 4];
        for (i, mat) in cascades.light_view_proj.iter().enumerate() {
            light_view_proj[i] = mat.to_cols_array();
        }
        Self {
            light_view_proj,
            cascade_splits: cascades.split_depths,
            shadow_bias,
            normal_bias,
            padding: [0.0; 2],
        }
    }
}

/// Post-processing frame graph runner managing offscreen targets, descriptors, and pipelines.
pub struct PostProcessFrameGraph {
    scene_color: GpuTexture2d,
    ssao_target: GpuTexture2d,
    composite_target: GpuTexture2d,
    cascade_ubo: crate::buffer::GpuBuffer,
    extent: vk::Extent2D,
    color_format: vk::Format,

    sampler_linear: vk::Sampler,
    sampler_nearest: vk::Sampler,

    ssao_descriptor_set_layout: vk::DescriptorSetLayout,
    ssao_descriptor_pool: vk::DescriptorPool,
    ssao_descriptor_set: vk::DescriptorSet,

    composite_descriptor_set_layout: vk::DescriptorSetLayout,
    composite_descriptor_pool: vk::DescriptorPool,
    composite_descriptor_set: vk::DescriptorSet,

    fxaa_descriptor_set_layout: vk::DescriptorSetLayout,
    fxaa_descriptor_pool: vk::DescriptorPool,
    fxaa_descriptor_set: vk::DescriptorSet,

    ssao_pipeline: GraphicsPipeline,
    ssao_vert_module: ShaderModule,
    ssao_frag_module: ShaderModule,

    composite_pipeline: GraphicsPipeline,
    composite_vert_module: ShaderModule,
    composite_frag_module: ShaderModule,

    fxaa_pipeline: GraphicsPipeline,
    fxaa_vert_module: ShaderModule,
    fxaa_frag_module: ShaderModule,
}

impl PostProcessFrameGraph {
    /// Creates a new `PostProcessFrameGraph` with offscreen targets matching the render extent.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        extent: vk::Extent2D,
        color_format: vk::Format,
        depth_view: vk::ImageView,
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        fullscreen_vert_spv: &[u8],
        ssao_frag_spv: &[u8],
        composite_frag_spv: &[u8],
        fxaa_frag_spv: &[u8],
    ) -> Result<Self, GpuError> {
        let scene_color = GpuTexture2d::new_empty(
            device,
            allocator,
            extent.width,
            extent.height,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        let ssao_target = GpuTexture2d::new_empty(
            device,
            allocator,
            extent.width,
            extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        let composite_target = GpuTexture2d::new_empty(
            device,
            allocator,
            extent.width,
            extent.height,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        let mut cascade_ubo = crate::buffer::GpuBuffer::new(
            device,
            allocator,
            "cascade_ubo",
            std::mem::size_of::<CascadeUniforms>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        let default_uniforms = CascadeUniforms::default();
        cascade_ubo.write_bytes(bytemuck::bytes_of(&default_uniforms))?;

        // Samplers
        let sampler_linear_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler_linear = unsafe { device.create_sampler(&sampler_linear_info, None)? };

        let sampler_nearest_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler_nearest = unsafe { device.create_sampler(&sampler_nearest_info, None)? };

        // 1. SSAO Descriptors (binding 0: depth)
        let ssao_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let ssao_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&ssao_binding));
        let ssao_descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&ssao_layout_info, None)? };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(4);
        let ssao_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let ssao_descriptor_pool = unsafe { device.create_descriptor_pool(&ssao_pool_info, None)? };

        let ssao_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(ssao_descriptor_pool)
            .set_layouts(std::slice::from_ref(&ssao_descriptor_set_layout));
        let ssao_descriptor_set = unsafe { device.allocate_descriptor_sets(&ssao_alloc_info)?[0] };

        // 2. Composite Descriptors (binding 0: color, binding 1: depth, binding 2: ssao, binding 3: shadow map, binding 4: cascade ubo)
        let comp_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let comp_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&comp_bindings);
        let composite_descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&comp_layout_info, None)? };

        let comp_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(6),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2),
        ];
        let comp_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&comp_pool_sizes);
        let composite_descriptor_pool =
            unsafe { device.create_descriptor_pool(&comp_pool_info, None)? };

        let comp_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(composite_descriptor_pool)
            .set_layouts(std::slice::from_ref(&composite_descriptor_set_layout));
        let composite_descriptor_set =
            unsafe { device.allocate_descriptor_sets(&comp_alloc_info)?[0] };

        // 3. FXAA Descriptors (binding 0: composite target)
        let fxaa_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let fxaa_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&fxaa_binding));
        let fxaa_descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&fxaa_layout_info, None)? };

        let fxaa_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2);
        let fxaa_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&fxaa_pool_size));
        let fxaa_descriptor_pool = unsafe { device.create_descriptor_pool(&fxaa_pool_info, None)? };

        let fxaa_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(fxaa_descriptor_pool)
            .set_layouts(std::slice::from_ref(&fxaa_descriptor_set_layout));
        let fxaa_descriptor_set = unsafe { device.allocate_descriptor_sets(&fxaa_alloc_info)?[0] };

        // Write Initial Descriptors
        Self::write_descriptors_internal(
            device,
            ssao_descriptor_set,
            composite_descriptor_set,
            fxaa_descriptor_set,
            scene_color.view(),
            depth_view,
            ssao_target.view(),
            shadow_view,
            shadow_sampler,
            cascade_ubo.raw(),
            composite_target.view(),
            sampler_linear,
            sampler_nearest,
        );

        // 4. SSAO Pipeline
        let ssao_vert_module = ShaderModule::from_spv(device, fullscreen_vert_spv)?;
        let ssao_frag_module = ShaderModule::from_spv(device, ssao_frag_spv)?;

        let ssao_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SsaoPushConstants>() as u32);

        let ssao_pipeline = GraphicsPipeline::create_dynamic(
            device,
            ssao_vert_module.raw(),
            ssao_frag_module.raw(),
            vk::Format::R8_UNORM,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[ssao_descriptor_set_layout],
            &[ssao_pc_range],
        )?;

        // 5. Composite Pipeline
        let composite_vert_module = ShaderModule::from_spv(device, fullscreen_vert_spv)?;
        let composite_frag_module = ShaderModule::from_spv(device, composite_frag_spv)?;

        let comp_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PostCompositePushConstants>() as u32);

        let composite_pipeline = GraphicsPipeline::create_dynamic(
            device,
            composite_vert_module.raw(),
            composite_frag_module.raw(),
            color_format,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[composite_descriptor_set_layout],
            &[comp_pc_range],
        )?;

        // 6. FXAA Pipeline
        let fxaa_vert_module = ShaderModule::from_spv(device, fullscreen_vert_spv)?;
        let fxaa_frag_module = ShaderModule::from_spv(device, fxaa_frag_spv)?;

        let fxaa_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<FxaaPushConstants>() as u32);

        let fxaa_pipeline = GraphicsPipeline::create_dynamic(
            device,
            fxaa_vert_module.raw(),
            fxaa_frag_module.raw(),
            color_format,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[fxaa_descriptor_set_layout],
            &[fxaa_pc_range],
        )?;

        info!(
            width = extent.width,
            height = extent.height,
            "Post-processing frame graph initialized with FXAA"
        );

        Ok(Self {
            scene_color,
            ssao_target,
            composite_target,
            cascade_ubo,
            extent,
            color_format,
            sampler_linear,
            sampler_nearest,
            ssao_descriptor_set_layout,
            ssao_descriptor_pool,
            ssao_descriptor_set,
            composite_descriptor_set_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            fxaa_descriptor_set_layout,
            fxaa_descriptor_pool,
            fxaa_descriptor_set,
            ssao_pipeline,
            ssao_vert_module,
            ssao_frag_module,
            composite_pipeline,
            composite_vert_module,
            composite_frag_module,
            fxaa_pipeline,
            fxaa_vert_module,
            fxaa_frag_module,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_descriptors_internal(
        device: &ash::Device,
        ssao_set: vk::DescriptorSet,
        comp_set: vk::DescriptorSet,
        fxaa_set: vk::DescriptorSet,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        ssao_view: vk::ImageView,
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        cascade_buffer: vk::Buffer,
        composite_view: vk::ImageView,
        linear_sampler: vk::Sampler,
        nearest_sampler: vk::Sampler,
    ) {
        let depth_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .image_view(depth_view)
            .sampler(nearest_sampler)];

        let color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(color_view)
            .sampler(linear_sampler)];

        let ssao_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ssao_view)
            .sampler(linear_sampler)];

        let shadow_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow_view)
            .sampler(shadow_sampler)];

        let cascade_buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(cascade_buffer)
            .offset(0)
            .range(std::mem::size_of::<CascadeUniforms>() as vk::DeviceSize)];

        let fxaa_color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(composite_view)
            .sampler(linear_sampler)];

        let writes = [
            // SSAO binding 0: depth
            vk::WriteDescriptorSet::default()
                .dst_set(ssao_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            // Composite binding 0: scene color
            vk::WriteDescriptorSet::default()
                .dst_set(comp_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&color_info),
            // Composite binding 1: depth
            vk::WriteDescriptorSet::default()
                .dst_set(comp_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            // Composite binding 2: ssao
            vk::WriteDescriptorSet::default()
                .dst_set(comp_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&ssao_info),
            // Composite binding 3: shadow map
            vk::WriteDescriptorSet::default()
                .dst_set(comp_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_info),
            // Composite binding 4: cascade ubo
            vk::WriteDescriptorSet::default()
                .dst_set(comp_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&cascade_buf_info),
            // FXAA binding 0: composite target
            vk::WriteDescriptorSet::default()
                .dst_set(fxaa_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&fxaa_color_info),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Returns the scene color texture image view for initial 3D rasterization passes.
    #[must_use]
    pub fn scene_color_view(&self) -> vk::ImageView {
        self.scene_color.view()
    }

    /// Returns the scene color raw `VkImage` handle.
    #[must_use]
    pub fn scene_color_image(&self) -> vk::Image {
        self.scene_color.image()
    }

    /// Resizes offscreen render targets and updates descriptor sets when the swapchain changes.
    pub fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        new_extent: vk::Extent2D,
        depth_view: vk::ImageView,
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
    ) -> Result<(), GpuError> {
        if self.extent == new_extent {
            return Ok(());
        }

        self.scene_color.destroy(device, allocator);
        self.ssao_target.destroy(device, allocator);
        self.composite_target.destroy(device, allocator);

        self.scene_color = GpuTexture2d::new_empty(
            device,
            allocator,
            new_extent.width,
            new_extent.height,
            self.color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        self.ssao_target = GpuTexture2d::new_empty(
            device,
            allocator,
            new_extent.width,
            new_extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        self.composite_target = GpuTexture2d::new_empty(
            device,
            allocator,
            new_extent.width,
            new_extent.height,
            self.color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Filter::LINEAR,
        )?;

        self.extent = new_extent;

        Self::write_descriptors_internal(
            device,
            self.ssao_descriptor_set,
            self.composite_descriptor_set,
            self.fxaa_descriptor_set,
            self.scene_color.view(),
            depth_view,
            self.ssao_target.view(),
            shadow_view,
            shadow_sampler,
            self.cascade_ubo.raw(),
            self.composite_target.view(),
            self.sampler_linear,
            self.sampler_nearest,
        );

        Ok(())
    }

    /// Uploads cascade matrices and split depths into the GPU uniform buffer for the composite shader.
    pub fn update_cascade_uniforms(&mut self, uniforms: &CascadeUniforms) -> Result<(), GpuError> {
        self.cascade_ubo.write_bytes(bytemuck::bytes_of(uniforms))
    }

    /// Recompiles shaders from disk and recreates post-processing pipelines on the fly.
    pub fn reload_shaders(
        &mut self,
        device: &ash::Device,
        shaders_dir: &Path,
    ) -> Result<(), GpuError> {
        info!("Hot-reloading post-processing shaders from disk...");

        let vert_path = shaders_dir.join("post_fullscreen.vert");
        let ssao_path = shaders_dir.join("post_ssao.frag");
        let comp_path = shaders_dir.join("post_composite.frag");
        let fxaa_path = shaders_dir.join("post_fxaa.frag");

        let new_vert = ShaderCompiler::compile_module(device, &vert_path)?;
        let new_ssao = ShaderCompiler::compile_module(device, &ssao_path)?;
        let new_comp = ShaderCompiler::compile_module(device, &comp_path)?;
        let new_fxaa = ShaderCompiler::compile_module(device, &fxaa_path)?;

        let ssao_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SsaoPushConstants>() as u32);

        let new_ssao_pipeline = GraphicsPipeline::create_dynamic(
            device,
            new_vert.raw(),
            new_ssao.raw(),
            vk::Format::R8_UNORM,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[self.ssao_descriptor_set_layout],
            &[ssao_pc_range],
        )?;

        let comp_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PostCompositePushConstants>() as u32);

        let new_comp_pipeline = GraphicsPipeline::create_dynamic(
            device,
            new_vert.raw(),
            new_comp.raw(),
            self.color_format,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[self.composite_descriptor_set_layout],
            &[comp_pc_range],
        )?;

        let fxaa_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<FxaaPushConstants>() as u32);

        let new_fxaa_pipeline = GraphicsPipeline::create_dynamic(
            device,
            new_vert.raw(),
            new_fxaa.raw(),
            self.color_format,
            None,
            vk::CullModeFlags::NONE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[self.fxaa_descriptor_set_layout],
            &[fxaa_pc_range],
        )?;

        // Destroy previous pipelines and modules
        self.ssao_pipeline.destroy(device);
        self.ssao_vert_module.destroy(device);
        self.ssao_frag_module.destroy(device);

        self.composite_pipeline.destroy(device);
        self.composite_vert_module.destroy(device);
        self.composite_frag_module.destroy(device);

        self.fxaa_pipeline.destroy(device);
        self.fxaa_vert_module.destroy(device);
        self.fxaa_frag_module.destroy(device);

        self.ssao_pipeline = new_ssao_pipeline;
        self.ssao_vert_module = new_vert;
        self.ssao_frag_module = new_ssao;

        let comp_vert = ShaderCompiler::compile_module(device, &vert_path)?;
        self.composite_pipeline = new_comp_pipeline;
        self.composite_vert_module = comp_vert;
        self.composite_frag_module = new_comp;

        let fxaa_vert = ShaderCompiler::compile_module(device, &vert_path)?;
        self.fxaa_pipeline = new_fxaa_pipeline;
        self.fxaa_vert_module = fxaa_vert;
        self.fxaa_frag_module = new_fxaa;

        info!("Post-processing shaders reloaded successfully");
        Ok(())
    }

    /// Records the SSAO pass, composite pass, and optional FXAA pass into the command buffer.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn record_postprocess(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        target_image: vk::Image,
        target_view: vk::ImageView,
        depth_image: vk::Image,
        ssao_pc: &SsaoPushConstants,
        composite_pc: &PostCompositePushConstants,
        fxaa_pc: Option<&FxaaPushConstants>,
    ) {
        let use_fxaa = fxaa_pc.is_some();

        // 1. Barrier: transition scene_color -> SHADER_READ_ONLY_OPTIMAL, depth -> DEPTH_READ_ONLY_OPTIMAL,
        // ssao_target -> COLOR_ATTACHMENT_OPTIMAL, and if FXAA is active, composite_target -> COLOR_ATTACHMENT_OPTIMAL.
        let mut barriers_before = vec![
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.scene_color.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                .image(depth_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.ssao_target.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];

        if use_fxaa {
            barriers_before.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(self.composite_target.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
            );
        }

        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers_before);
        unsafe {
            device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        // 2. Render SSAO Pass into ssao_target
        let ssao_color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.ssao_target.view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);

        let ssao_rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&ssao_color_attachment));

        unsafe {
            device.cmd_begin_rendering(cmd, &ssao_rendering_info);

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(self.extent.width as f32)
                .height(self.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ssao_pipeline.raw(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ssao_pipeline.layout(),
                0,
                &[self.ssao_descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.ssao_pipeline.layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(ssao_pc),
            );

            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }

        // 3. Transition ssao_target -> SHADER_READ_ONLY_OPTIMAL
        // If FXAA is disabled, also transition target_image -> COLOR_ATTACHMENT_OPTIMAL
        let mut barriers_mid = vec![
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.ssao_target.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];

        if !use_fxaa {
            barriers_mid.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(target_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
            );
        }

        let dep_mid = vk::DependencyInfo::default().image_memory_barriers(&barriers_mid);
        unsafe {
            device.cmd_pipeline_barrier2(cmd, &dep_mid);
        }

        // 4. Render Composite Pass (either into composite_target or target_view)
        let comp_target_view = if use_fxaa {
            self.composite_target.view()
        } else {
            target_view
        };

        let comp_color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(comp_target_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);

        let comp_rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&comp_color_attachment));

        unsafe {
            device.cmd_begin_rendering(cmd, &comp_rendering_info);

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(self.extent.width as f32)
                .height(self.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.composite_pipeline.raw(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.composite_pipeline.layout(),
                0,
                &[self.composite_descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.composite_pipeline.layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(composite_pc),
            );

            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }

        // 5. If FXAA is enabled, run the FXAA pass
        if let Some(fxaa_pc) = fxaa_pc {
            // Barrier: transition composite_target -> SHADER_READ_ONLY_OPTIMAL and target_image -> COLOR_ATTACHMENT_OPTIMAL
            let barriers_fxaa = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.composite_target.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(target_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
            ];

            let dep_fxaa = vk::DependencyInfo::default().image_memory_barriers(&barriers_fxaa);
            unsafe {
                device.cmd_pipeline_barrier2(cmd, &dep_fxaa);
            }

            // FXAA rendering into swapchain target
            let fxaa_color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(target_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);

            let fxaa_rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&fxaa_color_attachment));

            unsafe {
                device.cmd_begin_rendering(cmd, &fxaa_rendering_info);

                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(self.extent.width as f32)
                    .height(self.extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                device.cmd_set_viewport(cmd, 0, &[viewport]);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                };
                device.cmd_set_scissor(cmd, 0, &[scissor]);

                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.fxaa_pipeline.raw(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.fxaa_pipeline.layout(),
                    0,
                    &[self.fxaa_descriptor_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    self.fxaa_pipeline.layout(),
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(fxaa_pc),
                );

                device.cmd_draw(cmd, 3, 1, 0, 0);
                device.cmd_end_rendering(cmd);
            }
        }

        // 6. Transition scene_color back to COLOR_ATTACHMENT_OPTIMAL and depth back to DEPTH_ATTACHMENT_OPTIMAL for next frame
        let barriers_after = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.scene_color.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(depth_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];

        let dep_after = vk::DependencyInfo::default().image_memory_barriers(&barriers_after);
        unsafe {
            device.cmd_pipeline_barrier2(cmd, &dep_after);
        }
    }

    /// Destroys all GPU allocations, views, samplers, descriptors, and pipelines.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &GpuAllocator) {
        self.ssao_pipeline.destroy(device);
        self.ssao_vert_module.destroy(device);
        self.ssao_frag_module.destroy(device);

        self.composite_pipeline.destroy(device);
        self.composite_vert_module.destroy(device);
        self.composite_frag_module.destroy(device);

        self.fxaa_pipeline.destroy(device);
        self.fxaa_vert_module.destroy(device);
        self.fxaa_frag_module.destroy(device);

        unsafe {
            if self.ssao_descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.ssao_descriptor_pool, None);
            }
            if self.ssao_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.ssao_descriptor_set_layout, None);
            }
            if self.composite_descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.composite_descriptor_pool, None);
            }
            if self.composite_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.composite_descriptor_set_layout, None);
            }
            if self.fxaa_descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.fxaa_descriptor_pool, None);
            }
            if self.fxaa_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.fxaa_descriptor_set_layout, None);
            }
            if self.sampler_linear != vk::Sampler::null() {
                device.destroy_sampler(self.sampler_linear, None);
            }
            if self.sampler_nearest != vk::Sampler::null() {
                device.destroy_sampler(self.sampler_nearest, None);
            }
        }

        self.cascade_ubo.destroy(device, allocator);
        self.scene_color.destroy(device, allocator);
        self.ssao_target.destroy(device, allocator);
        self.composite_target.destroy(device, allocator);
    }
}
