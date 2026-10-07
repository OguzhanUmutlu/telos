//! OpenGL buffer wrapper for vertex, index, and uniform buffer objects.

use glow::HasContext;
use std::sync::Arc;

use crate::error::GpuError;

/// An OpenGL buffer object.
pub struct GlBuffer {
    gl: Arc<glow::Context>,
    buffer: glow::Buffer,
    target: u32,
    size_bytes: usize,
}

impl GlBuffer {
    /// Creates a new buffer with the specified target and byte capacity.
    ///
    /// Target can be `glow::ARRAY_BUFFER`, `glow::ELEMENT_ARRAY_BUFFER`, or `glow::UNIFORM_BUFFER`.
    #[allow(clippy::cast_possible_wrap)]
    pub fn new(
        gl: Arc<glow::Context>,
        target: u32,
        size_bytes: usize,
        usage: u32,
    ) -> Result<Self, GpuError> {
        let buffer = unsafe {
            gl.create_buffer()
                .map_err(|e| GpuError::Generic(format!("Failed to create OpenGL buffer: {e}")))?
        };

        unsafe {
            gl.bind_buffer(target, Some(buffer));
            gl.buffer_data_size(target, size_bytes as i32, usage);
            gl.bind_buffer(target, None);
        }

        Ok(Self {
            gl,
            buffer,
            target,
            size_bytes,
        })
    }

    /// Uploads an entire slice of pod data into the buffer.
    pub fn upload<T: bytemuck::Pod>(&mut self, data: &[T], usage: u32) {
        let bytes = bytemuck::cast_slice(data);
        self.size_bytes = bytes.len();
        unsafe {
            self.gl.bind_buffer(self.target, Some(self.buffer));
            self.gl.buffer_data_u8_slice(self.target, bytes, usage);
            self.gl.bind_buffer(self.target, None);
        }
    }

    /// Updates a sub-region of the buffer with new data.
    #[allow(clippy::cast_possible_wrap)]
    pub fn upload_sub_data<T: bytemuck::Pod>(&self, offset_bytes: usize, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        unsafe {
            self.gl.bind_buffer(self.target, Some(self.buffer));
            self.gl
                .buffer_sub_data_u8_slice(self.target, offset_bytes as i32, bytes);
            self.gl.bind_buffer(self.target, None);
        }
    }

    /// Binds this buffer to its default target.
    pub fn bind(&self) {
        unsafe {
            self.gl.bind_buffer(self.target, Some(self.buffer));
        }
    }

    /// Unbinds the target.
    pub fn unbind(&self) {
        unsafe {
            self.gl.bind_buffer(self.target, None);
        }
    }

    /// Binds this buffer as a Uniform Buffer Object (UBO) at the specified binding index.
    pub fn bind_base(&self, index: u32) {
        unsafe {
            self.gl
                .bind_buffer_base(glow::UNIFORM_BUFFER, index, Some(self.buffer));
        }
    }

    /// Raw OpenGL buffer handle.
    #[must_use]
    pub const fn raw(&self) -> glow::Buffer {
        self.buffer
    }

    /// Allocated buffer capacity in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl Drop for GlBuffer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_buffer(self.buffer);
        }
    }
}
