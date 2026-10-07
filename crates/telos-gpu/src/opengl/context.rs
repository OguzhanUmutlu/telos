//! OpenGL 4.5 / WebGL 2 graphics context initialization and frame management.

use glow::HasContext;
use std::sync::Arc;

use crate::error::GpuError;
use crate::rhi::RenderCaps;

#[cfg(not(target_arch = "wasm32"))]
use glutin::prelude::*;

/// Primary OpenGL graphics context managing device state and surface swapping.
pub struct GlContext {
    gl: Arc<glow::Context>,
    caps: RenderCaps,
    #[cfg(not(target_arch = "wasm32"))]
    surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    #[cfg(not(target_arch = "wasm32"))]
    context: glutin::context::PossiblyCurrentContext,
    width: u32,
    height: u32,
}

impl GlContext {
    /// Creates a `GlContext` directly wrapping an existing `glow::Context` (e.g. for WebGL2 or custom loaders).
    #[must_use]
    pub fn from_glow(gl: Arc<glow::Context>, width: u32, height: u32) -> Self {
        let vendor = unsafe { gl.get_parameter_string(glow::VENDOR) };
        let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        let caps = RenderCaps::opengl_fallback(&vendor, &renderer, &version);

        Self {
            gl,
            caps,
            #[cfg(not(target_arch = "wasm32"))]
            surface: unsafe { std::mem::zeroed() }, // only used if from_glow is called outside desktop
            #[cfg(not(target_arch = "wasm32"))]
            context: unsafe { std::mem::zeroed() },
            width,
            height,
        }
    }

    /// Creates an OpenGL 4.5 desktop context from a window builder and active event loop.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_desktop(
        window_builder: winit::window::WindowAttributes,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> Result<(Self, winit::window::Window), GpuError> {
        use glutin::config::ConfigTemplateBuilder;
        use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
        use glutin::display::GetGlDisplay;
        use glutin::surface::SurfaceAttributesBuilder;
        use glutin_winit::{DisplayBuilder, GlWindow};
        use raw_window_handle::HasWindowHandle;
        use std::ffi::CString;

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .prefer_hardware_accelerated(Some(true));

        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_builder));

        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .max_by_key(glutin::config::GlConfig::num_samples)
                    .expect("No suitable OpenGL config found")
            })
            .map_err(|e| GpuError::OpenGl(format!("Failed to build GL display: {e}")))?;

        let window = window
            .ok_or_else(|| GpuError::OpenGl("Window creation deferred unexpectedly".to_string()))?;

        let gl_display = gl_config.display();
        let raw_window_handle = window
            .window_handle()
            .map_err(|e| GpuError::OpenGl(format!("Failed to get raw window handle: {e}")))?
            .as_raw();

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 5))))
            .build(Some(raw_window_handle));

        let not_current_gl_context = unsafe {
            gl_display
                .create_context(&gl_config, &context_attributes)
                .map_err(|e| GpuError::OpenGl(format!("Failed to create OpenGL context: {e}")))?
        };

        let attrs = window
            .build_surface_attributes(SurfaceAttributesBuilder::default())
            .map_err(|e| GpuError::OpenGl(format!("Failed to build surface attrs: {e}")))?;

        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &attrs)
                .map_err(|e| GpuError::OpenGl(format!("Failed to create window surface: {e}")))?
        };

        let current_gl_context = not_current_gl_context
            .make_current(&gl_surface)
            .map_err(|e| GpuError::OpenGl(format!("Failed to make GL context current: {e}")))?;

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                let c_str = CString::new(name).unwrap();
                gl_display.get_proc_address(&c_str)
            })
        };

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let vendor = unsafe { gl.get_parameter_string(glow::VENDOR) };
        let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        let caps = RenderCaps::opengl_fallback(&vendor, &renderer, &version);

        let ctx = Self {
            gl: Arc::new(gl),
            caps,
            surface: gl_surface,
            context: current_gl_context,
            width,
            height,
        };

        Ok((ctx, window))
    }

    /// Access the underlying `glow::Context`.
    #[must_use]
    pub fn gl(&self) -> &Arc<glow::Context> {
        &self.gl
    }

    /// Returns device capability descriptor.
    #[must_use]
    pub const fn caps(&self) -> &RenderCaps {
        &self.caps
    }

    /// Updates viewport dimensions on window resize.
    #[allow(clippy::cast_possible_wrap)]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::num::NonZeroU32;
            if let (Some(w), Some(h)) = (NonZeroU32::new(self.width), NonZeroU32::new(self.height))
            {
                self.surface.resize(&self.context, w, h);
            }
        }
        unsafe {
            self.gl
                .viewport(0, 0, self.width as i32, self.height as i32);
        }
    }

    /// Swaps front and back buffers to present the rendered frame.
    pub fn swap_buffers(&self) -> Result<(), GpuError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.surface
                .swap_buffers(&self.context)
                .map_err(|e| GpuError::OpenGl(format!("OpenGL buffer swap failed: {e}")))?;
        }
        Ok(())
    }

    /// Viewport width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Viewport height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}
