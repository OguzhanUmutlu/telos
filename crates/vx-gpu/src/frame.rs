//! Frames-in-flight synchronization and command buffer pacing using timeline semaphores.

use ash::vk;
use tracing::debug;

use crate::error::GpuError;

/// Number of frames rendered concurrently on CPU and GPU (double-buffering).
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame resources including command buffers and binary synchronization semaphores.
pub struct FrameResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
}

impl FrameResources {
    /// Returns the primary command buffer for this frame.
    #[must_use]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Binary semaphore signaled when swapchain image is acquired.
    #[must_use]
    pub fn image_available_semaphore(&self) -> vk::Semaphore {
        self.image_available_semaphore
    }

    /// Binary semaphore signaled when command buffer rendering completes.
    #[must_use]
    pub fn render_finished_semaphore(&self) -> vk::Semaphore {
        self.render_finished_semaphore
    }
}

/// Manages double-buffered frames-in-flight and timeline semaphore pacing.
pub struct FrameManager {
    frames: [FrameResources; FRAMES_IN_FLIGHT],
    timeline_semaphore: vk::Semaphore,
    current_frame_index: usize,
    timeline_value: u64,
}

impl FrameManager {
    /// Initializes frames-in-flight resources and timeline semaphore.
    pub fn new(device: &ash::Device, graphics_queue_family: u32) -> Result<Self, GpuError> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let timeline_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        // SAFETY: Creating Vulkan 1.3 timeline semaphore
        let timeline_semaphore = unsafe { device.create_semaphore(&timeline_create_info, None)? };

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);

        for _ in 0..FRAMES_IN_FLIGHT {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            // SAFETY: Creating command pool and allocating command buffer
            let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
            let command_buffer = command_buffers[0];

            let binary_info = vk::SemaphoreCreateInfo::default();
            let image_available_semaphore = unsafe { device.create_semaphore(&binary_info, None)? };
            let render_finished_semaphore = unsafe { device.create_semaphore(&binary_info, None)? };

            frames.push(FrameResources {
                command_pool,
                command_buffer,
                image_available_semaphore,
                render_finished_semaphore,
            });
        }

        let frames_array: [FrameResources; FRAMES_IN_FLIGHT] = frames.try_into().map_err(|_| {
            GpuError::NoSuitableDevice("Failed to initialize frames in flight array")
        })?;

        debug!("FrameManager initialized with double-buffering and timeline semaphores");

        Ok(Self {
            frames: frames_array,
            timeline_semaphore,
            current_frame_index: 0,
            timeline_value: 0,
        })
    }

    /// Waits for the frame's timeline semaphore and begins recording the command buffer.
    pub fn begin_frame(&mut self, device: &ash::Device) -> Result<&FrameResources, GpuError> {
        let frame_index = self.current_frame_index;

        // If we have rendered at least FRAMES_IN_FLIGHT frames, wait for the frame that was recorded 2 turns ago
        if self.timeline_value >= FRAMES_IN_FLIGHT as u64 {
            let wait_value = self.timeline_value - (FRAMES_IN_FLIGHT as u64) + 1;
            let semaphores = [self.timeline_semaphore];
            let values = [wait_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);

            // SAFETY: Waiting for timeline semaphore to bound CPU-GPU pacing
            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX)?;
            }
        }

        let frame = &self.frames[frame_index];

        // SAFETY: Resetting command buffer and beginning recording
        unsafe {
            device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(frame.command_buffer, &begin_info)?;
        }

        Ok(frame)
    }

    /// Submits the recorded frame command buffer to the graphics queue and advances timeline.
    pub fn submit_and_advance(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
    ) -> Result<u64, GpuError> {
        self.timeline_value += 1;
        let signal_timeline_value = self.timeline_value;
        let frame = &self.frames[self.current_frame_index];

        // SAFETY: Ending command buffer recording
        unsafe {
            device.end_command_buffer(frame.command_buffer)?;
        }

        let wait_semaphores = [frame.image_available_semaphore];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let signal_semaphores = [frame.render_finished_semaphore, self.timeline_semaphore];
        let signal_values = [0, signal_timeline_value];

        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let command_buffers = [frame.command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);

        // SAFETY: Queue submit with timeline semaphore signaling
        unsafe {
            device.queue_submit(graphics_queue, &[submit_info], vk::Fence::null())?;
        }

        self.current_frame_index = (self.current_frame_index + 1) % FRAMES_IN_FLIGHT;

        Ok(signal_timeline_value)
    }

    /// Current frame resources reference.
    #[must_use]
    pub fn current_frame(&self) -> &FrameResources {
        &self.frames[self.current_frame_index]
    }

    /// Destroys all semaphores and command pools.
    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: Cleaning up sync objects and command pools
        unsafe {
            let _ = device.device_wait_idle();

            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available_semaphore, None);
                device.destroy_semaphore(frame.render_finished_semaphore, None);
                device.destroy_command_pool(frame.command_pool, None);
            }

            device.destroy_semaphore(self.timeline_semaphore, None);
        }
    }
}
