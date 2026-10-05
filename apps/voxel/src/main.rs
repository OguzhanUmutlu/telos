//! Client application executable for the voxel engine.

pub mod camera;

use std::time::{Duration, Instant};

use anyhow::Result;
use camera::{Camera, FlyController, Frustum};
use clap::Parser;
use glam::Vec3;
use mimalloc::MiMalloc;
use tracing::info;
use vx_core::{TelemetryConfig, coords::ChunkPos, ident::Identifier, init_telemetry};
use vx_gpu::{DepthBuffer, GpuBuffer, GpuContext, GraphicsPipeline, ShaderModule, vk};
use vx_voxel::{
    chunk::Chunk,
    coords::LocalIdx,
    registry::BlockRegistry,
    state::{BlockStateId, StateFlags},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Voxel engine client.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log level filter.
    #[arg(long, default_value = "info")]
    log: String,

    /// Enable Vulkan validation layers.
    #[arg(long, default_value_t = false)]
    validation: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChunkPushConstants {
    view_proj: glam::Mat4,
    chunk_pos: [i32; 3],
    _pad: u32,
    quad_buffer_address: u64,
}

struct GpuChunkMesh {
    pos: [i32; 3],
    buffer: GpuBuffer,
    quad_count: u32,
    min_aabb: Vec3,
    max_aabb: Vec3,
}

struct App {
    validation: bool,
    window: Option<Window>,
    gpu_context: Option<GpuContext>,
    depth_buffer: Option<DepthBuffer>,
    pipeline: Option<GraphicsPipeline>,
    vert_shader: Option<ShaderModule>,
    frag_shader: Option<ShaderModule>,
    chunk_meshes: Vec<GpuChunkMesh>,
    camera: Camera,
    controller: FlyController,
    last_frame_time: Instant,
    last_fps_time: Instant,
    frame_counter: u32,
    visible_chunks_last: usize,
}

impl App {
    fn new(validation: bool) -> Self {
        let mut camera = Camera::new(Vec3::new(128.0, 45.0, 160.0));
        camera.pitch = -0.3;

        Self {
            validation,
            window: None,
            gpu_context: None,
            depth_buffer: None,
            pipeline: None,
            vert_shader: None,
            frag_shader: None,
            chunk_meshes: Vec::new(),
            camera,
            controller: FlyController::default(),
            last_frame_time: Instant::now(),
            last_fps_time: Instant::now(),
            frame_counter: 0,
            visible_chunks_last: 0,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;

        self.controller.update(&mut self.camera, dt);

        self.frame_counter += 1;
        if now.duration_since(self.last_fps_time) >= Duration::from_secs(1) {
            let fps = self.frame_counter;
            self.frame_counter = 0;
            self.last_fps_time = now;
            info!(
                fps,
                visible = self.visible_chunks_last,
                total_chunks = self.chunk_meshes.len(),
                pos = ?self.camera.position,
                "Performance metrics"
            );
        }

        let (Some(gpu_context), Some(pipeline), Some(depth_buffer)) = (
            &mut self.gpu_context,
            &self.pipeline,
            &mut self.depth_buffer,
        ) else {
            return;
        };

        let frame_data = match gpu_context.begin_frame() {
            Ok(Some(data)) => data,
            Ok(None) => return,
            Err(err) => {
                tracing::error!("begin_frame failed: {err}");
                return;
            }
        };

        let (cmd, image_index) = frame_data;
        let swapchain_extent = gpu_context.extent();
        let image_view = gpu_context.swapchain().image_view(image_index as usize);

        #[allow(clippy::cast_precision_loss)]
        let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        let frustum = Frustum::from_view_proj(&view_proj);

        let device = gpu_context.device().raw();

        // SAFETY: Recording render barriers and drawing commands into active command buffer
        unsafe {
            // Transition depth image to DEPTH_ATTACHMENT_OPTIMAL
            let depth_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(depth_buffer.raw())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let depth_barriers = [depth_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&depth_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep_info);

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.52, 0.73, 0.95, 1.0], // Sky blue clear color
                    },
                });

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_buffer.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);

            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.raw());

            // Negative-height viewport: Vulkan clip space (+Y down) maps CCW to CCW
            #[allow(clippy::cast_precision_loss)]
            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(swapchain_extent.height as f32)
                .width(swapchain_extent.width as f32)
                .height(-(swapchain_extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Frustum culling and BDA vertex pulling draws
            let mut visible_chunks = 0;

            for mesh in &self.chunk_meshes {
                if !frustum.intersects_aabb(mesh.min_aabb, mesh.max_aabb) {
                    continue;
                }

                visible_chunks += 1;

                let pc = ChunkPushConstants {
                    view_proj,
                    chunk_pos: mesh.pos,
                    _pad: 0,
                    quad_buffer_address: mesh.buffer.device_address(),
                };

                let pc_bytes = std::slice::from_raw_parts(
                    std::ptr::from_ref(&pc).cast::<u8>(),
                    size_of::<ChunkPushConstants>(),
                );

                device.cmd_push_constants(
                    cmd,
                    pipeline.layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    pc_bytes,
                );

                device.cmd_draw(cmd, mesh.quad_count * 6, 1, 0, 0);
            }

            self.visible_chunks_last = visible_chunks;

            device.cmd_end_rendering(cmd);
        }

        if let Err(err) = gpu_context.end_frame(image_index) {
            tracing::error!("end_frame failed: {err}");
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Voxel Engine - Vulkan 1.3 Chunk Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = match event_loop.create_window(attributes) {
            Ok(w) => w,
            Err(err) => {
                tracing::error!("Failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        let gpu_context = match GpuContext::new(&window, size.width, size.height, self.validation) {
            Ok(ctx) => ctx,
            Err(err) => {
                tracing::error!("Failed to initialize GpuContext: {err}");
                event_loop.exit();
                return;
            }
        };

        let depth_buffer = match gpu_context.create_depth_buffer() {
            Ok(d) => d,
            Err(err) => {
                tracing::error!("Failed to create DepthBuffer: {err}");
                event_loop.exit();
                return;
            }
        };

        // Load SPIR-V bytecode generated by build script / xtask
        let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.vert.spv"));
        let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv"));

        let vert_module = match ShaderModule::from_spv(gpu_context.device().raw(), vert_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create vertex shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        let frag_module = match ShaderModule::from_spv(gpu_context.device().raw(), frag_spv) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to create fragment shader module: {err}");
                event_loop.exit();
                return;
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<ChunkPushConstants>() as u32);

        let pipeline = match GraphicsPipeline::create_dynamic(
            gpu_context.device().raw(),
            vert_module.raw(),
            frag_module.raw(),
            gpu_context.swapchain().format(),
            Some(vk::Format::D32_SFLOAT),
            vk::CullModeFlags::BACK,
            vk::FrontFace::COUNTER_CLOCKWISE,
            &[push_constant_range],
        ) {
            Ok(p) => p,
            Err(err) => {
                tracing::error!("Failed to create dynamic graphics pipeline: {err}");
                event_loop.exit();
                return;
            }
        };

        info!("Generating 8x8 chunk landscape test grid...");
        let chunk_meshes = match generate_test_chunks(&gpu_context) {
            Ok(m) => m,
            Err(err) => {
                tracing::error!("Failed to generate and upload test chunks: {err}");
                event_loop.exit();
                return;
            }
        };
        info!(
            chunk_count = chunk_meshes.len(),
            "Uploaded chunk meshes to GPU memory"
        );

        info!("Renderer and window loop successfully initialized");

        self.pipeline = Some(pipeline);
        self.vert_shader = Some(vert_module);
        self.frag_shader = Some(frag_module);
        self.depth_buffer = Some(depth_buffer);
        self.chunk_meshes = chunk_meshes;
        self.gpu_context = Some(gpu_context);
        self.window = Some(window);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.controller.on_mouse_move(&mut self.camera, dx, dy);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };
        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested, exiting application");
                event_loop.exit();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !self.controller.mouse_captured {
                    self.controller.mouse_captured = true;
                    let _ = window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .or_else(|_| {
                            window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        });
                    window.set_cursor_visible(false);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                match code {
                    KeyCode::KeyW => self.controller.forward = pressed,
                    KeyCode::KeyS => self.controller.backward = pressed,
                    KeyCode::KeyA => self.controller.left = pressed,
                    KeyCode::KeyD => self.controller.right = pressed,
                    KeyCode::Space => self.controller.up = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.controller.down = pressed,
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.controller.sprint = pressed;
                    }
                    KeyCode::Escape if pressed => {
                        if self.controller.mouse_captured {
                            self.controller.mouse_captured = false;
                            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        } else {
                            info!("Escape pressed, exiting application");
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0
                    && physical_size.height > 0
                    && let Some(gpu_context) = &mut self.gpu_context
                {
                    if let Err(err) = gpu_context.resize(physical_size.width, physical_size.height)
                    {
                        tracing::error!("Swapchain resize failed: {err}");
                    }
                    if let Some(mut old_depth) = self.depth_buffer.take() {
                        old_depth.destroy(gpu_context.device().raw(), gpu_context.allocator());
                    }
                    match gpu_context.create_depth_buffer() {
                        Ok(d) => self.depth_buffer = Some(d),
                        Err(err) => tracing::error!("Failed to recreate depth buffer: {err}"),
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu_context) = &mut self.gpu_context {
            let _ = gpu_context.wait_idle();
            for mesh in &mut self.chunk_meshes {
                mesh.buffer
                    .destroy(gpu_context.device().raw(), gpu_context.allocator());
            }
            self.chunk_meshes.clear();
            if let Some(mut depth) = self.depth_buffer.take() {
                depth.destroy(gpu_context.device().raw(), gpu_context.allocator());
            }
            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.destroy(gpu_context.device().raw());
            }
            if let Some(mut vert) = self.vert_shader.take() {
                vert.destroy(gpu_context.device().raw());
            }
            if let Some(mut frag) = self.frag_shader.take() {
                frag.destroy(gpu_context.device().raw());
            }
        }
    }
}

fn generate_test_chunks(gpu_context: &GpuContext) -> Result<Vec<GpuChunkMesh>> {
    const GRID_SIZE: i32 = 8;

    let mut reg = BlockRegistry::new();
    let _stone = reg.register(Identifier::classic("stone")?, StateFlags::OPAQUE_FULL);
    let _dirt = reg.register(Identifier::classic("dirt")?, StateFlags::OPAQUE_FULL);
    let _grass = reg.register(
        Identifier::classic("grass_block")?,
        StateFlags::OPAQUE_FULL,
    );
    let _bedrock = reg.register(Identifier::classic("bedrock")?, StateFlags::OPAQUE_FULL);
    reg.freeze();

    let mut snapshots = Vec::with_capacity(GRID_SIZE as usize);

    for cz in 0..GRID_SIZE {
        let mut row = Vec::with_capacity(GRID_SIZE as usize);
        for cx in 0..GRID_SIZE {
            let mut chunk = Chunk::new_uniform(ChunkPos::new(cx, 0, cz), BlockStateId::AIR, false);
            for z in 0..32 {
                for x in 0..32 {
                    #[allow(clippy::cast_precision_loss)]
                    let wx = (cx * 32 + x) as f32;
                    #[allow(clippy::cast_precision_loss)]
                    let wz = (cz * 32 + z) as f32;

                    // Smooth rolling hills landscape
                    #[allow(clippy::cast_possible_truncation)]
                    let height =
                        (14.0 + 8.0 * (wx * 0.04).sin() * (wz * 0.04).cos()).round() as i32;
                    let height = height.clamp(1, 31);

                    for y in 0..=height {
                        let state = if y == 0 {
                            BlockStateId::new(4) // Bedrock
                        } else if y < height - 3 {
                            BlockStateId::new(1) // Stone
                        } else if y < height {
                            BlockStateId::new(2) // Dirt
                        } else {
                            BlockStateId::new(3) // Grass
                        };

                        #[allow(clippy::cast_sign_loss)]
                        chunk.set(
                            LocalIdx::from_coords_unchecked(x as u32, y as u32, z as u32),
                            state,
                            StateFlags::empty(),
                            StateFlags::OPAQUE_FULL,
                            0,
                        );
                    }
                }
            }
            row.push(chunk.publish_snapshot());
        }
        snapshots.push(row);
    }

    let mut meshes = Vec::new();
    for cz in 0..GRID_SIZE {
        for cx in 0..GRID_SIZE {
            let chunk = &snapshots[cz as usize][cx as usize];
            let pos_x = if cx + 1 < GRID_SIZE {
                Some(snapshots[cz as usize][(cx + 1) as usize].as_ref())
            } else {
                None
            };
            let neg_x = if cx > 0 {
                Some(snapshots[cz as usize][(cx - 1) as usize].as_ref())
            } else {
                None
            };
            let pos_z = if cz + 1 < GRID_SIZE {
                Some(snapshots[(cz + 1) as usize][cx as usize].as_ref())
            } else {
                None
            };
            let neg_z = if cz > 0 {
                Some(snapshots[(cz - 1) as usize][cx as usize].as_ref())
            } else {
                None
            };

            let neighbors = [pos_x, neg_x, None, None, pos_z, neg_z];
            let mesh = vx_mesh::mesher::mesh_chunk_t0(chunk, &neighbors);

            if !mesh.is_empty() {
                let buffer = gpu_context.create_buffer_with_data(
                    "chunk_mesh",
                    &mesh.quads,
                    vk::BufferUsageFlags::empty(),
                )?;

                #[allow(clippy::cast_precision_loss)]
                let min_aabb = Vec3::new((cx * 32) as f32, 0.0, (cz * 32) as f32);
                #[allow(clippy::cast_precision_loss)]
                let max_aabb = Vec3::new(((cx + 1) * 32) as f32, 32.0, ((cz + 1) * 32) as f32);

                #[allow(clippy::cast_possible_truncation)]
                meshes.push(GpuChunkMesh {
                    pos: [cx * 32, 0, cz * 32],
                    buffer,
                    quad_count: mesh.quads.len() as u32,
                    min_aabb,
                    max_aabb,
                });
            }
        }
    }

    Ok(meshes)
}

fn main() -> Result<()> {
    let args = Args::parse();

    init_telemetry(&TelemetryConfig {
        app_name: "voxel-client",
        default_filter: args.log.into(),
        ansi_colors: true,
    });

    info!(version = env!("CARGO_PKG_VERSION"), "Starting Voxel client");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(args.validation);
    event_loop.run_app(&mut app)?;

    Ok(())
}
