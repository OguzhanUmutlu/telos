//! OpenGL 4.5 fallback desktop renderer for the Telos game client.
//!
//! Provides direct forward rendering of voxel terrain chunks, wireframe block highlights,
//! celestial sky backgrounds, and retained 2D UI quads on systems without Vulkan 1.3.

use glam::{Mat4, Vec3};
use glow::HasContext;
use hashbrown::HashMap;
use std::sync::Arc;

use telos_core::coords::ChunkPos;
use telos_gpu::GpuError;
use telos_gpu::opengl::{GlBuffer, GlContext, GlProgram, GlTexture2d, GlTextureArray};
use telos_mesh::mesh::ChunkMeshLayers;
use telos_mesh::quad::{FaceDir, T0Quad};
use telos_mesh::t1::T1Quad;
use telos_mesh::t2::T2Quad;

/// Vertex structure for OpenGL chunk terrain rendering (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlChunkVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Unit normal vector.
    pub normal: [f32; 3],
    /// Texture UV coordinates.
    pub uv: [f32; 2],
    /// 2D texture array layer index.
    pub layer: f32,
    /// Light levels: [ao (0..3), sky (0..15), block (0..15)].
    pub light: [f32; 3],
    /// RGBA tint multiplier.
    pub tint: [f32; 4],
}

/// Vertex structure for OpenGL 2D UI rendering (40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlUiVertex {
    /// Screen-space pixel position.
    pub position: [f32; 2],
    /// Texture UV coordinates.
    pub uv: [f32; 2],
    /// RGBA vertex color / tint.
    pub color: [f32; 4],
    /// Render mode (0.0 = solid, 1.0 = texture).
    pub mode: f32,
    /// Texture array layer index.
    pub layer: f32,
}

struct ChunkGlMesh {
    vao: glow::VertexArray,
    _vbo: GlBuffer,
    opaque_count: i32,
    cutout_offset: i32,
    cutout_count: i32,
    translucent_offset: i32,
    translucent_count: i32,
}

impl ChunkGlMesh {
    fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.vao);
        }
    }
}

/// Fallback OpenGL 4.5 forward renderer.
pub struct OpenGlRenderer {
    gl: Arc<glow::Context>,
    chunk_program: GlProgram,
    highlight_program: GlProgram,
    ui_program: GlProgram,
    sky_program: GlProgram,

    terrain_textures: Option<GlTextureArray>,
    lightmap: Option<GlTexture2d>,
    ui_textures: Option<GlTextureArray>,
    custom_layers: HashMap<u16, f32>,
    custom_tints: HashMap<u16, [f32; 4]>,

    highlight_vao: glow::VertexArray,
    _highlight_vbo: GlBuffer,

    ui_vao: glow::VertexArray,
    ui_vbo: GlBuffer,

    sky_vao: glow::VertexArray,

    meshes: HashMap<ChunkPos, ChunkGlMesh>,
}

impl OpenGlRenderer {
    /// Initializes all shaders, vertex array objects, and default buffers for OpenGL rendering.
    #[allow(clippy::similar_names, clippy::cast_possible_wrap)]
    pub fn new(context: &GlContext) -> Result<Self, GpuError> {
        let gl = context.gl().clone();

        let chunk_vert = include_str!("../../../shaders/gl/chunk.vert");
        let chunk_frag = include_str!("../../../shaders/gl/chunk.frag");
        let chunk_program = GlProgram::new(gl.clone(), chunk_vert, chunk_frag)?;

        let hl_vert = include_str!("../../../shaders/gl/highlight.vert");
        let hl_frag = include_str!("../../../shaders/gl/highlight.frag");
        let highlight_program = GlProgram::new(gl.clone(), hl_vert, hl_frag)?;

        let ui_vert = include_str!("../../../shaders/gl/ui.vert");
        let ui_frag = include_str!("../../../shaders/gl/ui.frag");
        let ui_program = GlProgram::new(gl.clone(), ui_vert, ui_frag)?;

        let sky_vert = include_str!("../../../shaders/gl/sky.vert");
        let sky_frag = include_str!("../../../shaders/gl/sky.frag");
        let sky_program = GlProgram::new(gl.clone(), sky_vert, sky_frag)?;

        // Highlight unit cube wireframe buffer
        let highlight_lines: [f32; 72] = [
            // Bottom 4 lines
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Top 4 lines
            0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, // Vertical 4 pillars
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
        ];

        let mut highlight_vbo = GlBuffer::new(
            gl.clone(),
            glow::ARRAY_BUFFER,
            std::mem::size_of_val(&highlight_lines),
            glow::STATIC_DRAW,
        )?;
        highlight_vbo.upload(&highlight_lines, glow::STATIC_DRAW);

        let highlight_vao = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| GpuError::OpenGl(format!("Failed to create VAO: {e}")))?;
            gl.bind_vertex_array(Some(vao));
            highlight_vbo.bind();
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);
            vao
        };

        // UI VAO and dynamic buffer (64 KiB initial)
        let ui_vbo = GlBuffer::new(gl.clone(), glow::ARRAY_BUFFER, 65536, glow::DYNAMIC_DRAW)?;
        let ui_vao = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| GpuError::OpenGl(format!("Failed to create VAO: {e}")))?;
            gl.bind_vertex_array(Some(vao));
            ui_vbo.bind();
            let stride = std::mem::size_of::<GlUiVertex>() as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 8);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 4, glow::FLOAT, false, stride, 16);
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 32);
            gl.enable_vertex_attrib_array(4);
            gl.vertex_attrib_pointer_f32(4, 1, glow::FLOAT, false, stride, 36);
            gl.bind_vertex_array(None);
            vao
        };

        // Sky fullscreen VAO (no buffers needed, generated from vertex index)
        let sky_vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| GpuError::OpenGl(format!("Failed to create sky VAO: {e}")))?
        };

        Ok(Self {
            gl,
            chunk_program,
            highlight_program,
            ui_program,
            sky_program,
            terrain_textures: None,
            lightmap: None,
            ui_textures: None,
            custom_layers: HashMap::new(),
            custom_tints: HashMap::new(),
            highlight_vao,
            _highlight_vbo: highlight_vbo,
            ui_vao,
            ui_vbo,
            sky_vao,
            meshes: HashMap::new(),
        })
    }

    /// Returns a reference to the underlying glow Context.
    pub fn gl(&self) -> &Arc<glow::Context> {
        &self.gl
    }

    /// Registers a custom block material with its texture array layer and RGBA tint color.
    pub fn register_custom_material(&mut self, mat_id: u16, layer: f32, tint: [f32; 4]) {
        self.custom_layers.insert(mat_id, layer);
        self.custom_tints.insert(mat_id, tint);
    }

    /// Sets the active terrain texture array.
    pub fn set_terrain_textures(&mut self, textures: GlTextureArray) {
        self.terrain_textures = Some(textures);
    }

    /// Sets the active 2D dynamic lighting lookup table.
    pub fn set_lightmap(&mut self, lightmap: GlTexture2d) {
        self.lightmap = Some(lightmap);
    }

    /// Sets the active UI sprite texture array.
    pub fn set_ui_textures(&mut self, textures: GlTextureArray) {
        self.ui_textures = Some(textures);
    }

    /// Updates dynamic lightmap LUT data from the CPU calculation.
    pub fn update_lightmap(&mut self, data: &[u8]) {
        if let Some(lm) = &self.lightmap {
            lm.upload_rgba8(data);
        }
    }

    /// Removes a chunk's OpenGL mesh and frees its VBO/VAO resources.
    pub fn remove_chunk_mesh(&mut self, pos: &ChunkPos) {
        if let Some(mesh) = self.meshes.remove(pos) {
            mesh.destroy(&self.gl);
        }
    }

    /// Updates or uploads an entire chunk's mesh geometry to OpenGL VBOs.
    #[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
    pub fn update_chunk_mesh(&mut self, pos: ChunkPos, layers: &ChunkMeshLayers) {
        let chunk_origin = Vec3::new(
            pos.x() as f32 * 32.0,
            pos.y() as f32 * 32.0,
            pos.z() as f32 * 32.0,
        );

        let mut opaque_verts = Vec::new();
        let mut cutout_verts = Vec::new();
        let mut translucent_verts = Vec::new();

        // 1. Opaque layers (T0 cubes + T1 sub-boxes)
        for q in &layers.opaque.quads {
            unpack_t0_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut opaque_verts,
            );
        }
        for q in &layers.t1_opaque.quads {
            unpack_t1_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut opaque_verts,
            );
        }

        // 2. Cutout foliage, glass, flowers, torches
        for q in &layers.cutout.quads {
            unpack_t0_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut cutout_verts,
            );
        }
        for q in &layers.t2_cutout.quads {
            unpack_t2_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut cutout_verts,
            );
        }

        // 3. Translucent water and sloped fluids
        for q in &layers.translucent.quads {
            unpack_t0_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut translucent_verts,
            );
        }
        for q in &layers.t2_translucent.quads {
            unpack_t2_quad(
                q,
                chunk_origin,
                &self.custom_layers,
                &self.custom_tints,
                &mut translucent_verts,
            );
        }

        let total_verts_count = opaque_verts.len() + cutout_verts.len() + translucent_verts.len();
        if total_verts_count == 0 {
            self.remove_chunk_mesh(&pos);
            return;
        }

        let opaque_count = opaque_verts.len() as i32;
        let cutout_count = cutout_verts.len() as i32;
        let translucent_count = translucent_verts.len() as i32;

        let cutout_offset = opaque_count;
        let translucent_offset = opaque_count + cutout_count;

        let mut all_verts = opaque_verts;
        all_verts.extend_from_slice(&cutout_verts);
        all_verts.extend_from_slice(&translucent_verts);

        let size_bytes = all_verts.len() * std::mem::size_of::<GlChunkVertex>();

        // Remove old mesh if exists
        self.remove_chunk_mesh(&pos);

        let vbo = match GlBuffer::new(
            self.gl.clone(),
            glow::ARRAY_BUFFER,
            size_bytes,
            glow::STATIC_DRAW,
        ) {
            Ok(mut buf) => {
                buf.upload(&all_verts, glow::STATIC_DRAW);
                buf
            }
            Err(e) => {
                tracing::error!("Failed to allocate GL chunk VBO: {e}");
                return;
            }
        };

        let vao = unsafe {
            let vao = match self.gl.create_vertex_array() {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Failed to create GL chunk VAO: {e}");
                    return;
                }
            };
            self.gl.bind_vertex_array(Some(vao));
            vbo.bind();

            let stride = std::mem::size_of::<GlChunkVertex>() as i32;
            self.gl.enable_vertex_attrib_array(0);
            self.gl
                .vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            self.gl.enable_vertex_attrib_array(1);
            self.gl
                .vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 12);
            self.gl.enable_vertex_attrib_array(2);
            self.gl
                .vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, 24);
            self.gl.enable_vertex_attrib_array(3);
            self.gl
                .vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 32);
            self.gl.enable_vertex_attrib_array(4);
            self.gl
                .vertex_attrib_pointer_f32(4, 3, glow::FLOAT, false, stride, 36);
            self.gl.enable_vertex_attrib_array(5);
            self.gl
                .vertex_attrib_pointer_f32(5, 4, glow::FLOAT, false, stride, 48);

            self.gl.bind_vertex_array(None);
            vao
        };

        self.meshes.insert(
            pos,
            ChunkGlMesh {
                vao,
                _vbo: vbo,
                opaque_count,
                cutout_offset,
                cutout_count,
                translucent_offset,
                translucent_count,
            },
        );
    }

    /// Renders a full frame forward through OpenGL.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_possible_wrap
    )]
    pub fn render_frame(
        &mut self,
        view_proj: &Mat4,
        inv_view_proj: &Mat4,
        camera_pos: Vec3,
        sun_dir: Vec3,
        daylight: f32,
        sunset: f32,
        highlight: Option<(Vec3, Vec3)>,
        ui_quads: &[GlUiVertex],
        width: u32,
        height: u32,
    ) {
        let gl = &self.gl;

        unsafe {
            gl.viewport(0, 0, width as i32, height as i32);
            gl.clear_color(0.1, 0.1, 0.15, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            // 1. Render celestial sky background
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            self.sky_program.bind();
            if let Some(loc) = self.sky_program.get_uniform_location("u_inv_view_proj") {
                self.sky_program
                    .set_mat4(&loc, &inv_view_proj.to_cols_array());
            }
            if let Some(loc) = self.sky_program.get_uniform_location("u_sun_dir") {
                self.sky_program
                    .set_vec3(&loc, sun_dir.x, sun_dir.y, sun_dir.z);
            }
            if let Some(loc) = self.sky_program.get_uniform_location("u_daylight") {
                self.sky_program.set_float(&loc, daylight);
            }
            if let Some(loc) = self.sky_program.get_uniform_location("u_sunset") {
                self.sky_program.set_float(&loc, sunset);
            }
            gl.bind_vertex_array(Some(self.sky_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);

            // 2. Enable depth testing for 3D world geometry
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.depth_func(glow::LEQUAL);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);

            // Bind textures
            if let Some(tex) = &self.terrain_textures {
                tex.bind(0);
            }
            if let Some(lm) = &self.lightmap {
                lm.bind(1);
            }

            self.chunk_program.bind();
            if let Some(loc) = self.chunk_program.get_uniform_location("u_view_proj") {
                self.chunk_program
                    .set_mat4(&loc, &view_proj.to_cols_array());
            }
            if let Some(loc) = self.chunk_program.get_uniform_location("u_camera_pos") {
                self.chunk_program
                    .set_vec3(&loc, camera_pos.x, camera_pos.y, camera_pos.z);
            }
            if let Some(loc) = self.chunk_program.get_uniform_location("u_sun_dir") {
                self.chunk_program
                    .set_vec3(&loc, sun_dir.x, sun_dir.y, sun_dir.z);
            }
            if let Some(loc) = self.chunk_program.get_uniform_location("u_daylight") {
                self.chunk_program.set_float(&loc, daylight);
            }
            if let Some(loc) = self
                .chunk_program
                .get_uniform_location("u_terrain_textures")
            {
                self.chunk_program.set_int(&loc, 0);
            }
            if let Some(loc) = self.chunk_program.get_uniform_location("u_lightmap") {
                self.chunk_program.set_int(&loc, 1);
            }
            if let Some(loc) = self.chunk_program.get_uniform_location("u_use_textures") {
                self.chunk_program
                    .set_int(&loc, i32::from(self.terrain_textures.is_some()));
            }

            // Draw opaque layers
            for mesh in self.meshes.values() {
                if mesh.opaque_count > 0 {
                    gl.bind_vertex_array(Some(mesh.vao));
                    gl.draw_arrays(glow::TRIANGLES, 0, mesh.opaque_count);
                }
            }

            // Draw cutout layers (alpha test in shader, no backface culling for double-sided foliage)
            gl.disable(glow::CULL_FACE);
            for mesh in self.meshes.values() {
                if mesh.cutout_count > 0 {
                    gl.bind_vertex_array(Some(mesh.vao));
                    gl.draw_arrays(glow::TRIANGLES, mesh.cutout_offset, mesh.cutout_count);
                }
            }

            // Draw translucent layers (blended)
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.depth_mask(false);
            for mesh in self.meshes.values() {
                if mesh.translucent_count > 0 {
                    gl.bind_vertex_array(Some(mesh.vao));
                    gl.draw_arrays(
                        glow::TRIANGLES,
                        mesh.translucent_offset,
                        mesh.translucent_count,
                    );
                }
            }
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.enable(glow::CULL_FACE);

            // 3. Selection wireframe highlight
            if let Some((min_pos, max_pos)) = highlight {
                gl.disable(glow::CULL_FACE);
                self.highlight_program.bind();
                if let Some(loc) = self.highlight_program.get_uniform_location("u_view_proj") {
                    self.highlight_program
                        .set_mat4(&loc, &view_proj.to_cols_array());
                }
                if let Some(loc) = self.highlight_program.get_uniform_location("u_min") {
                    self.highlight_program
                        .set_vec3(&loc, min_pos.x, min_pos.y, min_pos.z);
                }
                if let Some(loc) = self.highlight_program.get_uniform_location("u_max") {
                    self.highlight_program
                        .set_vec3(&loc, max_pos.x, max_pos.y, max_pos.z);
                }
                if let Some(loc) = self.highlight_program.get_uniform_location("u_color") {
                    self.highlight_program.set_vec4(&loc, 0.0, 0.0, 0.0, 0.85);
                }

                gl.bind_vertex_array(Some(self.highlight_vao));
                gl.draw_arrays(glow::LINES, 0, 24);
                gl.bind_vertex_array(None);
            }

            // 4. UI overlay pass
            if !ui_quads.is_empty() {
                gl.disable(glow::DEPTH_TEST);
                gl.disable(glow::CULL_FACE);
                gl.enable(glow::BLEND);
                gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

                if let Some(tex) = &self.ui_textures {
                    tex.bind(0);
                }

                self.ui_program.bind();
                if let Some(loc) = self.ui_program.get_uniform_location("u_viewport_size") {
                    self.ui_program.set_vec2(&loc, width as f32, height as f32);
                }
                if let Some(loc) = self.ui_program.get_uniform_location("u_ui_textures") {
                    self.ui_program.set_int(&loc, 0);
                }
                if let Some(loc) = self.ui_program.get_uniform_location("u_has_textures") {
                    self.ui_program
                        .set_int(&loc, i32::from(self.ui_textures.is_some()));
                }

                self.ui_vbo.upload(ui_quads, glow::STREAM_DRAW);

                gl.bind_vertex_array(Some(self.ui_vao));
                gl.draw_arrays(glow::TRIANGLES, 0, ui_quads.len() as i32);
                gl.bind_vertex_array(None);

                gl.disable(glow::BLEND);
            }
        }
    }
}

impl Drop for OpenGlRenderer {
    fn drop(&mut self) {
        for mesh in self.meshes.values() {
            mesh.destroy(&self.gl);
        }
        unsafe {
            self.gl.delete_vertex_array(self.highlight_vao);
            self.gl.delete_vertex_array(self.ui_vao);
            self.gl.delete_vertex_array(self.sky_vao);
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn unpack_t0_quad(
    quad: &T0Quad,
    origin: Vec3,
    custom_layers: &HashMap<u16, f32>,
    custom_tints: &HashMap<u16, [f32; 4]>,
    out: &mut Vec<GlChunkVertex>,
) {
    let x = quad.x() as f32;
    let y = quad.y() as f32;
    let z = quad.z() as f32;
    let w = quad.w() as f32;
    let h = quad.h() as f32;
    let dir = quad.dir();
    let mat = quad.material();

    let layer = get_gl_texture_layer(mat, dir, custom_layers);
    let tint = get_material_tint(mat, dir, custom_tints);

    let (norm, u_dir, v_dir, plane_offset) = match dir {
        FaceDir::PosX => (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        ),
        FaceDir::NegX => (
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::ZERO,
        ),
        FaceDir::PosY => (
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        FaceDir::NegY => (
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::ZERO,
        ),
        FaceDir::PosZ => (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        FaceDir::NegZ => (
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
        ),
    };

    let p0 = origin + Vec3::new(x, y, z) + plane_offset;
    let c0 = p0;
    let c1 = p0 + u_dir * w;
    let c2 = p0 + u_dir * w + v_dir * h;
    let c3 = p0 + v_dir * h;

    let norm_arr = [norm.x, norm.y, norm.z];
    let light = [0.0, 15.0, 0.0];

    let v0 = GlChunkVertex {
        position: [c0.x, c0.y, c0.z],
        normal: norm_arr,
        uv: [0.0, 0.0],
        layer,
        light,
        tint,
    };
    let v1 = GlChunkVertex {
        position: [c1.x, c1.y, c1.z],
        normal: norm_arr,
        uv: [w, 0.0],
        layer,
        light,
        tint,
    };
    let v2 = GlChunkVertex {
        position: [c2.x, c2.y, c2.z],
        normal: norm_arr,
        uv: [w, h],
        layer,
        light,
        tint,
    };
    let v3 = GlChunkVertex {
        position: [c3.x, c3.y, c3.z],
        normal: norm_arr,
        uv: [0.0, h],
        layer,
        light,
        tint,
    };

    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn unpack_t1_quad(
    quad: &T1Quad,
    origin: Vec3,
    custom_layers: &HashMap<u16, f32>,
    custom_tints: &HashMap<u16, [f32; 4]>,
    out: &mut Vec<GlChunkVertex>,
) {
    let x = (quad.word0 & 0x3FF) as f32 / 16.0;
    let y = ((quad.word0 >> 10) & 0x3FF) as f32 / 16.0;
    let z = ((quad.word0 >> 20) & 0x3FF) as f32 / 16.0;

    let w = (((quad.word1 & 0x1FF) + 1) as f32) / 16.0;
    let h = ((((quad.word1 >> 9) & 0x1FF) + 1) as f32) / 16.0;
    let dir = FaceDir::from_u8(((quad.word1 >> 18) & 0x7) as u8).unwrap_or(FaceDir::PosY);
    let mat = (quad.word2 >> 16) as u16;

    let layer = get_gl_texture_layer(mat, dir, custom_layers);
    let tint = get_material_tint(mat, dir, custom_tints);

    let (norm, u_dir, v_dir) = match dir {
        FaceDir::PosX => (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        FaceDir::NegX => (
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        FaceDir::PosY => (
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        ),
        FaceDir::NegY => (
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        FaceDir::PosZ => (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        FaceDir::NegZ => (
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ),
    };

    let p0 = origin + Vec3::new(x, y, z);
    let c0 = p0;
    let c1 = p0 + u_dir * w;
    let c2 = p0 + u_dir * w + v_dir * h;
    let c3 = p0 + v_dir * h;

    let norm_arr = [norm.x, norm.y, norm.z];
    let light = [0.0, 15.0, 0.0];

    let v0 = GlChunkVertex {
        position: [c0.x, c0.y, c0.z],
        normal: norm_arr,
        uv: [0.0, 0.0],
        layer,
        light,
        tint,
    };
    let v1 = GlChunkVertex {
        position: [c1.x, c1.y, c1.z],
        normal: norm_arr,
        uv: [w, 0.0],
        layer,
        light,
        tint,
    };
    let v2 = GlChunkVertex {
        position: [c2.x, c2.y, c2.z],
        normal: norm_arr,
        uv: [w, h],
        layer,
        light,
        tint,
    };
    let v3 = GlChunkVertex {
        position: [c3.x, c3.y, c3.z],
        normal: norm_arr,
        uv: [0.0, h],
        layer,
        light,
        tint,
    };

    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn unpack_t2_quad(
    quad: &T2Quad,
    origin: Vec3,
    custom_layers: &HashMap<u16, f32>,
    custom_tints: &HashMap<u16, [f32; 4]>,
    out: &mut Vec<GlChunkVertex>,
) {
    let verts = [&quad.v0, &quad.v1, &quad.v2, &quad.v3];
    let mut gverts = [GlChunkVertex {
        position: [0.0; 3],
        normal: [0.0; 3],
        uv: [0.0; 2],
        layer: 0.0,
        light: [0.0; 3],
        tint: [1.0, 1.0, 1.0, 1.0],
    }; 4];

    for (i, v) in verts.iter().enumerate() {
        let p = origin + v.pos();
        let norm = v.normal();
        let uv = v.uv();
        let mat = v.material();
        let layer = get_gl_texture_layer(mat, FaceDir::PosY, custom_layers);
        let tint = get_material_tint(mat, FaceDir::PosY, custom_tints);
        let ao = f32::from(v.ao());
        let sky = f32::from(v.sky());
        let block = f32::from(v.block());

        gverts[i] = GlChunkVertex {
            position: [p.x, p.y, p.z],
            normal: [norm.x, norm.y, norm.z],
            uv,
            layer,
            light: [ao, sky, block],
            tint,
        };
    }

    out.extend_from_slice(&[
        gverts[0], gverts[1], gverts[2], gverts[0], gverts[2], gverts[3],
    ]);
}

fn get_gl_texture_layer(mat: u16, dir: FaceDir, custom_layers: &HashMap<u16, f32>) -> f32 {
    if let Some(&layer) = custom_layers.get(&mat) {
        return layer;
    }
    match mat {
        1 => 0.0, // Stone
        2 => 1.0, // Dirt
        3 => match dir {
            FaceDir::PosY => 2.0, // Grass Top
            FaceDir::NegY => 1.0, // Dirt
            _ => 3.0,             // Grass Side
        },
        4 => 4.0,   // Bedrock
        5 => 5.0,   // Sand
        7 => 6.0,   // Oak Planks
        8 => 7.0,   // Oak Leaves
        9 => 8.0,   // Glass
        21 => 17.0, // logic_power_block
        24 => 18.0, // logic_lamp
        25 => 19.0, // logic_lamp_lit
        74 => match dir {
            FaceDir::PosY | FaceDir::NegY => 25.0, // oak_log_top
            _ => 24.0,                             // oak_log
        },
        75 => match dir {
            FaceDir::PosY | FaceDir::NegY => 27.0, // birch_log_top
            _ => 26.0,                             // birch_log
        },
        76 => match dir {
            FaceDir::PosY | FaceDir::NegY => 29.0, // spruce_log_top
            _ => 28.0,                             // spruce_log
        },
        77 => 30.0, // birch_leaves
        78 => 31.0, // spruce_leaves
        _ => f32::from(mat),
    }
}

fn get_material_tint(mat: u16, dir: FaceDir, custom_tints: &HashMap<u16, [f32; 4]>) -> [f32; 4] {
    if let Some(&tint) = custom_tints.get(&mat) {
        return tint;
    }
    match mat {
        3 => {
            // Grass
            if dir == FaceDir::PosY {
                [0.55, 0.78, 0.35, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        }
        8 => [0.298, 0.600, 0.129, 1.0],             // Oak Leaves
        77 => [0.502, 0.655, 0.333, 1.0],            // Birch Leaves
        78 => [0.380, 0.600, 0.380, 1.0],            // Spruce Leaves
        6 | 37..=43 => [0.20, 0.45, 0.85, 0.65],     // Water
        31 | 32 | 44..=50 => [1.0, 0.45, 0.05, 1.0], // Lava
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

/// Converts an array of retained 2D `UiQuad` items into vertex data suitable for OpenGL drawing.
#[allow(clippy::similar_names, clippy::many_single_char_names)]
pub fn ui_quads_to_gl_vertices(quads: &[telos_ui::UiQuad], out: &mut Vec<GlUiVertex>) {
    out.clear();
    out.reserve(quads.len() * 6);

    for q in quads {
        let x0 = q.pos[0] as f32;
        let y0 = q.pos[1] as f32;
        let w = (q.size_kind[0] & 0xFFFF) as f32;
        let h = ((q.size_kind[0] >> 16) & 0xFFFF) as f32;
        let kind = (q.size_kind[1] & 0xFFFF) as u16;

        let u0 = (q.uv[0] & 0xFFFF) as f32 / 65535.0;
        let v0 = ((q.uv[0] >> 16) & 0xFFFF) as f32 / 65535.0;
        let u1 = (q.uv[1] & 0xFFFF) as f32 / 65535.0;
        let v1 = ((q.uv[1] >> 16) & 0xFFFF) as f32 / 65535.0;
        let layer = q.uv[2] as f32;

        let r = (q.color & 0xFF) as f32 / 255.0;
        let g = ((q.color >> 8) & 0xFF) as f32 / 255.0;
        let b = ((q.color >> 16) & 0xFF) as f32 / 255.0;
        let a = ((q.color >> 24) & 0xFF) as f32 / 255.0;
        let color = [r, g, b, a];

        let mode = if kind == 0 { 0.0 } else { 1.0 }; // 0: solid, 1: texture/font

        let v_tl = GlUiVertex {
            position: [x0, y0],
            uv: [u0, v0],
            color,
            mode,
            layer,
        };
        let v_tr = GlUiVertex {
            position: [x0 + w, y0],
            uv: [u1, v0],
            color,
            mode,
            layer,
        };
        let v_br = GlUiVertex {
            position: [x0 + w, y0 + h],
            uv: [u1, v1],
            color,
            mode,
            layer,
        };
        let v_bl = GlUiVertex {
            position: [x0, y0 + h],
            uv: [u0, v1],
            color,
            mode,
            layer,
        };

        out.extend_from_slice(&[v_tl, v_tr, v_br, v_tl, v_br, v_bl]);
    }
}
