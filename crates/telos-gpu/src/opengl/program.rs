//! OpenGL shader program compilation and uniform management.

use glow::HasContext;
use std::sync::Arc;

use crate::error::GpuError;

/// An OpenGL shader program composed of a vertex and fragment shader.
pub struct GlProgram {
    gl: Arc<glow::Context>,
    program: glow::Program,
}

impl GlProgram {
    /// Compiles and links a shader program from vertex and fragment GLSL source code.
    pub fn new(
        gl: Arc<glow::Context>,
        vert_source: &str,
        frag_source: &str,
    ) -> Result<Self, GpuError> {
        let vert_shader = Self::compile_shader(&gl, glow::VERTEX_SHADER, vert_source)?;
        let frag_shader = Self::compile_shader(&gl, glow::FRAGMENT_SHADER, frag_source)?;

        let program = unsafe {
            let prog = gl
                .create_program()
                .map_err(|e| GpuError::Generic(format!("Failed to create GL program: {e}")))?;

            gl.attach_shader(prog, vert_shader);
            gl.attach_shader(prog, frag_shader);
            gl.link_program(prog);

            if !gl.get_program_link_status(prog) {
                let log = gl.get_program_info_log(prog);
                gl.delete_shader(vert_shader);
                gl.delete_shader(frag_shader);
                gl.delete_program(prog);
                return Err(GpuError::Generic(format!("GL program link failed: {log}")));
            }

            // Shaders can be detached and flagged for deletion after successful link
            gl.detach_shader(prog, vert_shader);
            gl.detach_shader(prog, frag_shader);
            gl.delete_shader(vert_shader);
            gl.delete_shader(frag_shader);

            prog
        };

        Ok(Self { gl, program })
    }

    fn compile_shader(
        gl: &glow::Context,
        shader_type: u32,
        source: &str,
    ) -> Result<glow::Shader, GpuError> {
        unsafe {
            let shader = gl
                .create_shader(shader_type)
                .map_err(|e| GpuError::Generic(format!("Failed to create GL shader: {e}")))?;

            gl.shader_source(shader, source);
            gl.compile_shader(shader);

            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                let kind_str = if shader_type == glow::VERTEX_SHADER {
                    "vertex"
                } else {
                    "fragment"
                };
                return Err(GpuError::Generic(format!(
                    "GL {kind_str} shader compilation failed: {log}"
                )));
            }

            Ok(shader)
        }
    }

    /// Binds this program for subsequent draw calls.
    pub fn bind(&self) {
        unsafe {
            self.gl.use_program(Some(self.program));
        }
    }

    /// Unbinds any active program.
    pub fn unbind(&self) {
        unsafe {
            self.gl.use_program(None);
        }
    }

    /// Retrieves the location of a named uniform variable.
    #[must_use]
    pub fn get_uniform_location(&self, name: &str) -> Option<glow::UniformLocation> {
        unsafe { self.gl.get_uniform_location(self.program, name) }
    }

    /// Sets a 4x4 matrix uniform.
    pub fn set_mat4(&self, loc: &glow::UniformLocation, mat: &[f32; 16]) {
        unsafe {
            self.gl
                .uniform_matrix_4_f32_slice(Some(loc), false, mat.as_slice());
        }
    }

    /// Sets a 2-component float vector uniform.
    pub fn set_vec2(&self, loc: &glow::UniformLocation, x: f32, y: f32) {
        unsafe {
            self.gl.uniform_2_f32(Some(loc), x, y);
        }
    }

    /// Sets a 3-component float vector uniform.
    pub fn set_vec3(&self, loc: &glow::UniformLocation, x: f32, y: f32, z: f32) {
        unsafe {
            self.gl.uniform_3_f32(Some(loc), x, y, z);
        }
    }

    /// Sets a 4-component float vector uniform.
    pub fn set_vec4(&self, loc: &glow::UniformLocation, x: f32, y: f32, z: f32, w: f32) {
        unsafe {
            self.gl.uniform_4_f32(Some(loc), x, y, z, w);
        }
    }

    /// Sets a scalar float uniform.
    pub fn set_float(&self, loc: &glow::UniformLocation, val: f32) {
        unsafe {
            self.gl.uniform_1_f32(Some(loc), val);
        }
    }

    /// Sets a scalar integer uniform (e.g., texture sampler unit).
    pub fn set_int(&self, loc: &glow::UniformLocation, val: i32) {
        unsafe {
            self.gl.uniform_1_i32(Some(loc), val);
        }
    }

    /// Raw OpenGL program handle.
    #[must_use]
    pub const fn raw(&self) -> glow::Program {
        self.program
    }
}

impl Drop for GlProgram {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.program);
        }
    }
}
