//! OpenGL 2D texture array and 2D texture wrappers.

use glow::HasContext;
use std::sync::Arc;

use crate::error::GpuError;

/// A 2D OpenGL texture array (`GL_TEXTURE_2D_ARRAY`) used for block faces and GUI sprites.
pub struct GlTextureArray {
    gl: Arc<glow::Context>,
    texture: glow::Texture,
    width: u32,
    height: u32,
    layers: u32,
    mip_levels: u32,
}

impl GlTextureArray {
    /// Creates a new 2D texture array with the given dimensions and mip count.
    #[allow(clippy::cast_possible_wrap)]
    pub fn new(
        gl: Arc<glow::Context>,
        width: u32,
        height: u32,
        layers: u32,
        mip_levels: u32,
    ) -> Result<Self, GpuError> {
        let texture = unsafe {
            gl.create_texture()
                .map_err(|e| GpuError::Generic(format!("Failed to create GL texture array: {e}")))?
        };

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(texture));

            // Allocate immutable storage for all mip levels and layers
            gl.tex_storage_3d(
                glow::TEXTURE_2D_ARRAY,
                mip_levels as i32,
                glow::RGBA8,
                width as i32,
                height as i32,
                layers as i32,
            );

            // Set default texture filtering
            gl.tex_parameter_i32(
                glow::TEXTURE_2D_ARRAY,
                glow::TEXTURE_MIN_FILTER,
                if mip_levels > 1 {
                    glow::NEAREST_MIPMAP_LINEAR as i32
                } else {
                    glow::NEAREST as i32
                },
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D_ARRAY,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D_ARRAY,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D_ARRAY,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );

            gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
        }

        Ok(Self {
            gl,
            texture,
            width,
            height,
            layers,
            mip_levels,
        })
    }

    /// Uploads pixel data to a specific mip level and array slice layer.
    #[allow(clippy::cast_possible_wrap)]
    pub fn upload_mip_region(&self, mip: u32, layer: u32, width: u32, height: u32, data: &[u8]) {
        unsafe {
            self.gl
                .bind_texture(glow::TEXTURE_2D_ARRAY, Some(self.texture));
            self.gl.tex_sub_image_3d(
                glow::TEXTURE_2D_ARRAY,
                mip as i32,
                0,
                0,
                layer as i32,
                width as i32,
                height as i32,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
            self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
        }
    }

    /// Binds the texture array to the specified texture unit.
    pub fn bind(&self, unit: u32) {
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl
                .bind_texture(glow::TEXTURE_2D_ARRAY, Some(self.texture));
        }
    }

    /// Texture array width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Texture array height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of array slice layers.
    #[must_use]
    pub const fn layers(&self) -> u32 {
        self.layers
    }

    /// Number of mipmap levels allocated.
    #[must_use]
    pub const fn mip_levels(&self) -> u32 {
        self.mip_levels
    }
}

impl Drop for GlTextureArray {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_texture(self.texture);
        }
    }
}

/// A standard 2D OpenGL texture (`GL_TEXTURE_2D`), primarily used for the 16x16 dynamic lightmap LUT.
pub struct GlTexture2d {
    gl: Arc<glow::Context>,
    texture: glow::Texture,
    width: u32,
    height: u32,
}

impl GlTexture2d {
    /// Creates a 2D texture with linear filtering (enabling smooth bilinear light interpolation).
    #[allow(clippy::cast_possible_wrap)]
    pub fn new(gl: Arc<glow::Context>, width: u32, height: u32) -> Result<Self, GpuError> {
        let texture = unsafe {
            gl.create_texture()
                .map_err(|e| GpuError::Generic(format!("Failed to create GL texture 2d: {e}")))?
        };

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_storage_2d(
                glow::TEXTURE_2D,
                1,
                glow::RGBA8,
                width as i32,
                height as i32,
            );

            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );

            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        Ok(Self {
            gl,
            texture,
            width,
            height,
        })
    }

    /// Uploads an entire RGBA8 buffer to the 2D texture.
    #[allow(clippy::cast_possible_wrap)]
    pub fn upload_rgba8(&self, data: &[u8]) {
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                self.width as i32,
                self.height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
            self.gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    /// Binds the 2D texture to the specified texture unit.
    pub fn bind(&self, unit: u32) {
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        }
    }
}

impl Drop for GlTexture2d {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_texture(self.texture);
        }
    }
}
