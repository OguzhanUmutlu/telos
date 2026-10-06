//! Compute pipeline abstraction for GPU compute shaders.

use ash::vk;
use tracing::info;

use crate::error::GpuError;

/// A compiled compute pipeline and its layout.
pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl ComputePipeline {
    /// Builds a compute pipeline from a shader module.
    pub fn new(
        device: &ash::Device,
        shader: vk::ShaderModule,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, GpuError> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: Creating compute pipeline layout
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        // SAFETY: Creating compute pipeline
        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| GpuError::Vk(err))?
        };

        info!("Compute pipeline compiled successfully");

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Raw `vk::Pipeline` handle.
    #[inline]
    #[must_use]
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Pipeline layout handle.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// Destroys the pipeline and pipeline layout.
    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: Destroying pipeline and layout with valid device handle
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
