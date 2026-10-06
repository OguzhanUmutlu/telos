//! Dynamic rendering graphics pipeline and shader module utilities.

use ash::vk;
use tracing::info;

use crate::error::GpuError;

/// A compiled SPIR-V shader module wrapper.
pub struct ShaderModule {
    module: vk::ShaderModule,
}

impl ShaderModule {
    /// Creates a shader module from raw SPIR-V u32 words or aligned bytes.
    pub fn from_spv(device: &ash::Device, spv_bytes: &[u8]) -> Result<Self, GpuError> {
        if !spv_bytes.len().is_multiple_of(4) {
            return Err(GpuError::Shader(
                "SPIR-V byte length must be multiple of 4".into(),
            ));
        }

        let (chunks, _) = spv_bytes.as_chunks::<4>();
        let words: Vec<u32> = chunks.iter().copied().map(u32::from_le_bytes).collect();

        let create_info = vk::ShaderModuleCreateInfo::default().code(&words);

        // SAFETY: Calling vkCreateShaderModule with verified SPIR-V bytecode
        let module = unsafe { device.create_shader_module(&create_info, None)? };

        Ok(Self { module })
    }

    /// Returns the raw `vk::ShaderModule` handle.
    #[must_use]
    pub fn raw(&self) -> vk::ShaderModule {
        self.module
    }

    /// Destroys the shader module.
    pub fn destroy(&mut self, device: &ash::Device) {
        if self.module != vk::ShaderModule::null() {
            // SAFETY: Destroying shader module with valid device handle
            unsafe {
                device.destroy_shader_module(self.module, None);
            }
            self.module = vk::ShaderModule::null();
        }
    }
}

/// A graphics pipeline configured for modern Vulkan 1.3 dynamic rendering.
pub struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    /// Builds a dynamic rendering graphics pipeline using `VkPipelineRenderingCreateInfo`.
    #[allow(clippy::too_many_arguments)]
    pub fn create_dynamic(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        // Empty vertex input (vertex pulling or hardcoded vertices)
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport and scissor configured dynamically
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(front_face);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(depth_format.is_some())
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // Vulkan 1.3 Dynamic Rendering info
        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth_fmt) = depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_fmt);
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Builds a dynamic rendering graphics pipeline specifically for line rendering (`LINE_LIST`).
    pub fn create_dynamic_lines(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::LINE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[])
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth_fmt) = depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_fmt);
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering line graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Builds a dynamic rendering graphics pipeline for the sky/celestial pass (empty vertex buffer, depth-tested at reversed-Z z=0, no depth write).
    #[allow(clippy::too_many_arguments)]
    pub fn create_dynamic_sky(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        // Empty vertex input - fullscreen triangle generated from gl_VertexIndex
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth_fmt) = depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_fmt);
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering sky graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Builds a dynamic rendering graphics pipeline for translucent surfaces (depth-tested, non-writing, alpha blended).
    #[allow(clippy::too_many_arguments)]
    pub fn create_dynamic_translucent(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        cull_mode: vk::CullModeFlags,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        // Translucent: depth test enabled, depth write DISABLED
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Alpha blend: SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth_fmt) = depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_fmt);
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering translucent graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Builds a dynamic rendering graphics pipeline for cutout foliage and alpha-test geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn create_dynamic_cutout(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        cull_mode: vk::CullModeFlags,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        // Cutout: depth test enabled, depth write ENABLED
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(depth_format.is_some())
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth_fmt) = depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_fmt);
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering cutout graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Builds a dynamic rendering graphics pipeline for UI quads (no depth test, alpha blending, dynamic viewport/scissor).
    pub fn create_dynamic_ui(
        device: &ash::Device,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
        color_format: vk::Format,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Alpha blending: SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        // SAFETY: Creating graphics pipeline with dynamic rendering extension chain
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Dynamic rendering UI graphics pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Returns the raw `vk::Pipeline` handle.
    #[must_use]
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Returns the raw `vk::PipelineLayout` handle.
    #[must_use]
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// Destroys pipeline and layout.
    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: Destroying pipeline and layout with valid handles
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.layout, None);
                self.layout = vk::PipelineLayout::null();
            }
        }
    }
}
