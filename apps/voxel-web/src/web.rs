//! WebGL2 browser client wrapper for the voxel engine.

use glam::{Mat4, Vec3};
use hashbrown::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::{
    Document, HtmlCanvasElement, KeyboardEvent, MouseEvent, WebGl2RenderingContext, WebGlBuffer,
    WebGlProgram, WebGlShader, WebGlUniformLocation, Window,
};

use vx_core::coords::{BlockPos, ChunkPos};
use vx_core::raycast::{RaycastHit, raycast_voxels};
use vx_mesh::mesher::mesh_chunk_multilayers;
use vx_mesh::quad::FaceDir;
use vx_voxel::chunk::{Chunk, ChunkSnapshot};
use vx_voxel::coords::LocalIdx;
use vx_voxel::registry::BlockRegistry;
use vx_voxel::shape::BlockShape;
use vx_voxel::state::BlockStateId;
use vx_worldgen::WorldGenerator;

const VERTEX_STRIDE_FLOATS: i32 = 10; // x, y, z, nx, ny, nz, r, g, b, a
const REACH_DISTANCE: f32 = 6.0;

struct ChunkGlMesh {
    opaque_vbo: WebGlBuffer,
    opaque_count: i32,
    cutout_vbo: WebGlBuffer,
    cutout_count: i32,
    translucent_vbo: WebGlBuffer,
    translucent_count: i32,
}

struct WebGpuContext {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    highlight_program: WebGlProgram,
    u_view_proj: WebGlUniformLocation,
    u_sun_dir: WebGlUniformLocation,
    hl_u_view_proj: WebGlUniformLocation,
    hl_u_min: WebGlUniformLocation,
    hl_u_max: WebGlUniformLocation,
    highlight_vbo: WebGlBuffer,
}

struct EngineState {
    camera_pos: Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    keys: HashMap<String, bool>,
    pointer_locked: bool,
    selected_slot: usize,
    hotbar: [BlockStateId; 9],
    registry: BlockRegistry,
    generator: WorldGenerator,
    chunks: HashMap<ChunkPos, Chunk>,
    snapshots: HashMap<ChunkPos, Arc<ChunkSnapshot>>,
    meshes: HashMap<ChunkPos, ChunkGlMesh>,
    last_hit: Option<RaycastHit>,
    fps: u32,
    frame_count: u32,
    last_fps_time: f64,
}

/// Browser entry point invoked on WebAssembly module initialization.
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("voxel-canvas")
        .expect("missing #voxel-canvas")
        .dyn_into()?;

    // Resize canvas to window dimensions
    let width = window.inner_width()?.as_f64().unwrap_or(1280.0) as u32;
    let height = window.inner_height()?.as_f64().unwrap_or(720.0) as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let gl: WebGl2RenderingContext = canvas
        .get_context("webgl2")?
        .expect("webgl2 not supported")
        .dyn_into()?;

    let gpu = init_webgl(&gl)?;

    let registry = BlockRegistry::standard();
    let generator = WorldGenerator::new(1337, &registry);

    let hotbar = [
        BlockStateId(1),  // grass
        BlockStateId(2),  // dirt
        BlockStateId(3),  // stone
        BlockStateId(5),  // oak log
        BlockStateId(7),  // oak planks
        BlockStateId(8),  // oak leaves
        BlockStateId(9),  // glass
        BlockStateId(10), // stone slab
        BlockStateId(11), // oak stairs
    ];

    let state = Rc::new(RefCell::new(EngineState {
        camera_pos: Vec3::new(16.0, 72.0, 16.0),
        camera_yaw: -90.0,
        camera_pitch: -15.0,
        keys: HashMap::new(),
        pointer_locked: false,
        selected_slot: 0,
        hotbar,
        registry,
        generator,
        chunks: HashMap::new(),
        snapshots: HashMap::new(),
        meshes: HashMap::new(),
        last_hit: None,
        fps: 60,
        frame_count: 0,
        last_fps_time: window.performance().unwrap().now(),
    }));

    // Initial terrain generation: 5x3x5 chunks around camera
    generate_initial_world(&gpu, &mut state.borrow_mut());

    setup_event_listeners(&window, &document, &canvas, state.clone())?;

    // Start requestAnimationFrame render loop
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    let state_loop = state.clone();
    let window_loop = window.clone();

    let mut last_frame_time = window.performance().unwrap().now();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let now = window_loop.performance().unwrap().now();
        let dt = ((now - last_frame_time) / 1000.0).clamp(0.0, 0.1) as f32;
        last_frame_time = now;

        update_and_render(
            &gpu,
            &canvas,
            &document,
            &mut state_loop.borrow_mut(),
            dt,
            now,
        );

        request_animation_frame(&window_loop, f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(&window, g.borrow().as_ref().unwrap());

    Ok(())
}

fn request_animation_frame(window: &Window, f: &Closure<dyn FnMut()>) {
    window
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame`");
}

fn init_webgl(gl: &WebGl2RenderingContext) -> Result<WebGpuContext, JsValue> {
    let vert_src = r#"#version 300 es
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec3 a_normal;
    layout(location = 2) in vec4 a_color;

    uniform mat4 u_view_proj;

    out vec3 v_normal;
    out vec4 v_color;
    out vec3 v_world_pos;

    void main() {
        v_normal = a_normal;
        v_color = a_color;
        v_world_pos = a_position;
        gl_Position = u_view_proj * vec4(a_position, 1.0);
    }
    "#;

    let frag_src = r#"#version 300 es
    precision mediump float;

    in vec3 v_normal;
    in vec4 v_color;
    in vec3 v_world_pos;

    uniform vec3 u_sun_dir;

    out vec4 frag_color;

    void main() {
        float face_shade = 0.8;
        if (v_normal.y > 0.5) face_shade = 1.0;
        else if (v_normal.y < -0.5) face_shade = 0.5;
        else if (abs(v_normal.z) > 0.5) face_shade = 0.8;
        else if (abs(v_normal.x) > 0.5) face_shade = 0.6;

        float ndotl = max(dot(v_normal, normalize(u_sun_dir)), 0.0);
        float sun_light = 0.45 + 0.55 * ndotl;

        vec3 rgb = v_color.rgb * face_shade * sun_light;
        frag_color = vec4(rgb, v_color.a);
    }
    "#;

    let program = create_program(gl, vert_src, frag_src)?;
    let u_view_proj = gl
        .get_uniform_location(&program, "u_view_proj")
        .expect("u_view_proj");
    let u_sun_dir = gl
        .get_uniform_location(&program, "u_sun_dir")
        .expect("u_sun_dir");

    // Highlight wireframe shader
    let hl_vert_src = r#"#version 300 es
    layout(location = 0) in vec3 a_unit_pos;

    uniform mat4 u_view_proj;
    uniform vec3 u_min;
    uniform vec3 u_max;

    void main() {
        vec3 world_pos = mix(u_min, u_max, a_unit_pos);
        gl_Position = u_view_proj * vec4(world_pos, 1.0);
    }
    "#;

    let hl_frag_src = r#"#version 300 es
    precision mediump float;
    out vec4 frag_color;
    void main() {
        frag_color = vec4(0.0, 0.0, 0.0, 0.9);
    }
    "#;

    let highlight_program = create_program(gl, hl_vert_src, hl_frag_src)?;
    let hl_u_view_proj = gl
        .get_uniform_location(&highlight_program, "u_view_proj")
        .expect("hl u_view_proj");
    let hl_u_min = gl
        .get_uniform_location(&highlight_program, "u_min")
        .expect("hl u_min");
    let hl_u_max = gl
        .get_uniform_location(&highlight_program, "u_max")
        .expect("hl u_max");

    // 12 lines for a bounding box wireframe
    #[rustfmt::skip]
    let box_lines: [f32; 72] = [
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,  1.0, 0.0, 1.0,
        1.0, 0.0, 1.0,  0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,  0.0, 0.0, 0.0,

        0.0, 1.0, 0.0,  1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,  1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,  0.0, 1.0, 1.0,
        0.0, 1.0, 1.0,  0.0, 1.0, 0.0,

        0.0, 0.0, 0.0,  0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,  1.0, 1.0, 0.0,
        1.0, 0.0, 1.0,  1.0, 1.0, 1.0,
        0.0, 0.0, 1.0,  0.0, 1.0, 1.0,
    ];

    let highlight_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&highlight_vbo));
    unsafe {
        let view = js_sys::Float32Array::view(&box_lines);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    Ok(WebGpuContext {
        gl: gl.clone(),
        program,
        highlight_program,
        u_view_proj,
        u_sun_dir,
        hl_u_view_proj,
        hl_u_min,
        hl_u_max,
        highlight_vbo,
    })
}

fn create_program(
    gl: &WebGl2RenderingContext,
    vert_src: &str,
    frag_src: &str,
) -> Result<WebGlProgram, JsValue> {
    let vert_shader = compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vert_src)?;
    let frag_shader = compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, frag_src)?;

    let program = gl.create_program().ok_or("failed to create program")?;
    gl.attach_shader(&program, &vert_shader);
    gl.attach_shader(&program, &frag_shader);
    gl.link_program(&program);

    if !gl.get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS) {
        let log = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "unknown error".into());
        return Err(JsValue::from_str(&format!("shader link error: {log}")));
    }

    Ok(program)
}

fn compile_shader(
    gl: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, JsValue> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or("failed to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if !gl.get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS) {
        let log = gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "unknown error".into());
        return Err(JsValue::from_str(&format!("shader compile error: {log}")));
    }

    Ok(shader)
}

fn generate_initial_world(gpu: &WebGpuContext, state: &mut EngineState) {
    let radius = 2;
    for cz in -radius..=radius {
        for cy in 0..=3 {
            for cx in -radius..=radius {
                let pos = ChunkPos::new(cx, cy, cz);
                let mut chunk = state.generator.generate_chunk(pos);
                state.snapshots.insert(pos, chunk.publish_snapshot());
                state.chunks.insert(pos, chunk);
            }
        }
    }

    // Mesh all generated chunks
    let positions: Vec<ChunkPos> = state.chunks.keys().copied().collect();
    for pos in positions {
        remesh_chunk(gpu, state, pos);
    }
}

fn remesh_chunk(gpu: &WebGpuContext, state: &mut EngineState, pos: ChunkPos) {
    let Some(snap) = state.snapshots.get(&pos) else {
        return;
    };

    let get_s = |dx: i32, dy: i32, dz: i32| {
        let np = ChunkPos::new(pos.x() + dx, pos.y() + dy, pos.z() + dz);
        state.snapshots.get(&np).map(std::convert::AsRef::as_ref)
    };

    let neighbors = [
        get_s(1, 0, 0),
        get_s(-1, 0, 0),
        get_s(0, 1, 0),
        get_s(0, -1, 0),
        get_s(0, 0, 1),
        get_s(0, 0, -1),
    ];

    let layers = mesh_chunk_multilayers(snap.as_ref(), &neighbors, &state.registry);

    let mut opaque_verts = Vec::new();
    let mut cutout_verts = Vec::new();
    let mut translucent_verts = Vec::new();

    let chunk_origin = Vec3::new(
        (pos.x() * 32) as f32,
        (pos.y() * 32) as f32,
        (pos.z() * 32) as f32,
    );

    // Unpack T0 opaque
    for q in &layers.opaque.quads {
        unpack_t0_quad(q, chunk_origin, &mut opaque_verts);
    }
    // Unpack T1 opaque (slabs/stairs)
    for q in &layers.t1_opaque.quads {
        unpack_t1_quad(q, chunk_origin, &mut opaque_verts);
    }
    // Unpack T0 cutout (leaves)
    for q in &layers.cutout.quads {
        unpack_t0_quad(q, chunk_origin, &mut cutout_verts);
    }
    // Unpack T0 translucent (water/glass)
    for q in &layers.translucent.quads {
        unpack_t0_quad(q, chunk_origin, &mut translucent_verts);
    }

    let gl = &gpu.gl;

    let opaque_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&opaque_vbo));
    unsafe {
        let view = js_sys::Float32Array::view(&opaque_verts);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let cutout_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&cutout_vbo));
    unsafe {
        let view = js_sys::Float32Array::view(&cutout_verts);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let translucent_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&translucent_vbo));
    unsafe {
        let view = js_sys::Float32Array::view(&translucent_verts);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    state.meshes.insert(
        pos,
        ChunkGlMesh {
            opaque_vbo,
            opaque_count: (opaque_verts.len() / VERTEX_STRIDE_FLOATS as usize) as i32,
            cutout_vbo,
            cutout_count: (cutout_verts.len() / VERTEX_STRIDE_FLOATS as usize) as i32,
            translucent_vbo,
            translucent_count: (translucent_verts.len() / VERTEX_STRIDE_FLOATS as usize) as i32,
        },
    );
}

fn get_material_color(mat: u16, dir: FaceDir) -> [f32; 4] {
    match mat {
        1 => {
            // Grass
            match dir {
                FaceDir::PosY => [0.38, 0.65, 0.22, 1.0], // Top
                FaceDir::NegY => [0.55, 0.40, 0.26, 1.0], // Bottom dirt
                _ => [0.46, 0.50, 0.24, 1.0],             // Side
            }
        }
        2 => [0.55, 0.40, 0.26, 1.0], // Dirt
        3 => [0.50, 0.50, 0.50, 1.0], // Stone
        4 => [0.42, 0.42, 0.42, 1.0], // Cobblestone
        5 => {
            // Oak Log
            match dir {
                FaceDir::PosY | FaceDir::NegY => [0.70, 0.58, 0.38, 1.0],
                _ => [0.40, 0.30, 0.18, 1.0],
            }
        }
        6 => [0.20, 0.45, 0.85, 0.65], // Water
        7 => [0.67, 0.54, 0.35, 1.0],  // Oak Planks
        8 => [0.25, 0.52, 0.18, 0.95], // Oak Leaves
        9 => [0.85, 0.95, 1.00, 0.35], // Glass
        10 => [0.50, 0.50, 0.50, 1.0], // Stone Slab
        11 => [0.67, 0.54, 0.35, 1.0], // Oak Stairs
        _ => [0.8, 0.8, 0.8, 1.0],
    }
}

fn unpack_t0_quad(quad: &vx_mesh::quad::T0Quad, origin: Vec3, out: &mut Vec<f32>) {
    let x = quad.x() as f32;
    let y = quad.y() as f32;
    let z = quad.z() as f32;
    let w = quad.w() as f32;
    let h = quad.h() as f32;
    let dir = quad.dir();
    let mat = quad.material();

    let col = get_material_color(mat, dir);

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

    // Two triangles per quad: [c0, c1, c2] and [c0, c2, c3]
    push_vertex(out, c0, norm, col);
    push_vertex(out, c1, norm, col);
    push_vertex(out, c2, norm, col);

    push_vertex(out, c0, norm, col);
    push_vertex(out, c2, norm, col);
    push_vertex(out, c3, norm, col);
}

fn unpack_t1_quad(quad: &vx_mesh::t1::T1Quad, origin: Vec3, out: &mut Vec<f32>) {
    let x = quad.x() as f32 / 16.0;
    let y = quad.y() as f32 / 16.0;
    let z = quad.z() as f32 / 16.0;
    let w = quad.w() as f32 / 16.0;
    let h = quad.h() as f32 / 16.0;
    let dir = quad.dir();
    let mat = quad.material();

    let col = get_material_color(mat, dir);

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

    push_vertex(out, c0, norm, col);
    push_vertex(out, c1, norm, col);
    push_vertex(out, c2, norm, col);

    push_vertex(out, c0, norm, col);
    push_vertex(out, c2, norm, col);
    push_vertex(out, c3, norm, col);
}

#[inline]
fn push_vertex(out: &mut Vec<f32>, p: Vec3, n: Vec3, c: [f32; 4]) {
    out.extend_from_slice(&[p.x, p.y, p.z, n.x, n.y, n.z, c[0], c[1], c[2], c[3]]);
}

fn setup_event_listeners(
    window: &Window,
    document: &Document,
    canvas: &HtmlCanvasElement,
    state: Rc<RefCell<EngineState>>,
) -> Result<(), JsValue> {
    // Keydown
    {
        let state = state.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key().to_uppercase();
            state.borrow_mut().keys.insert(key.clone(), true);

            // Number keys 1..=9
            if let Ok(slot) = event.key().parse::<usize>() {
                if (1..=9).contains(&slot) {
                    state.borrow_mut().selected_slot = slot - 1;
                }
            }
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Keyup
    {
        let state = state.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key().to_uppercase();
            state.borrow_mut().keys.insert(key, false);
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Canvas click -> request pointer lock
    {
        let c = canvas.clone();
        let closure = Closure::wrap(Box::new(move |_: MouseEvent| {
            let _ = c.request_pointer_lock();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse down -> break/place block
    {
        let state = state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            let mut st = state.borrow_mut();
            if !st.pointer_locked {
                return;
            }

            let Some(hit) = st.last_hit else { return };

            if event.button() == 0 {
                // Left click: Break block
                set_world_block(&mut st, hit.pos, BlockStateId(0));
            } else if event.button() == 2 {
                // Right click: Place selected block
                let place_pos = hit.pos.offset(hit.face);
                let block = st.hotbar[st.selected_slot];
                set_world_block(&mut st, place_pos, block);
            }
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Prevent context menu on right click
    {
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            event.prevent_default();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("contextmenu", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse move -> camera look
    {
        let state = state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            let mut st = state.borrow_mut();
            if st.pointer_locked {
                let dx = event.movement_x() as f32;
                let dy = event.movement_y() as f32;
                st.camera_yaw += dx * 0.15;
                st.camera_pitch = (st.camera_pitch - dy * 0.15).clamp(-89.0, 89.0);
            }
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Pointer lock change
    {
        let state = state.clone();
        let doc = document.clone();
        let closure = Closure::wrap(Box::new(move || {
            let locked = doc.pointer_lock_element().is_some();
            state.borrow_mut().pointer_locked = locked;
        }) as Box<dyn FnMut()>);
        document.add_event_listener_with_callback(
            "pointerlockchange",
            closure.as_ref().unchecked_ref(),
        )?;
        closure.forget();
    }

    // Resize
    {
        let c = canvas.clone();
        let win = window.clone();
        let closure = Closure::wrap(Box::new(move || {
            let width = win.inner_width().unwrap().as_f64().unwrap_or(1280.0) as u32;
            let height = win.inner_height().unwrap().as_f64().unwrap_or(720.0) as u32;
            c.set_width(width);
            c.set_height(height);
        }) as Box<dyn FnMut()>);
        window.add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    Ok(())
}

fn set_world_block(st: &mut EngineState, pos: BlockPos, block: BlockStateId) {
    let (chunk_pos, local_pos) = pos.to_chunk_and_local();
    let local_idx = LocalIdx::from_coords_unchecked(
        local_pos.x() as u32,
        local_pos.y() as u32,
        local_pos.z() as u32,
    );
    let new_flags = st.registry.flags(block);
    if let Some(chunk) = st.chunks.get_mut(&chunk_pos) {
        let old_block = chunk.get(local_idx);
        let old_flags = st.registry.flags(old_block);
        chunk.set(local_idx, block, old_flags, new_flags, 0);
        st.snapshots.insert(chunk_pos, chunk.publish_snapshot());
    }
}

fn update_and_render(
    gpu: &WebGpuContext,
    canvas: &HtmlCanvasElement,
    document: &Document,
    st: &mut EngineState,
    dt: f32,
    now: f64,
) {
    // 1. Camera movement
    let speed = 12.0 * dt;
    let yaw_rad = st.camera_yaw.to_radians();
    let pitch_rad = st.camera_pitch.to_radians();

    let forward = Vec3::new(
        yaw_rad.cos() * pitch_rad.cos(),
        pitch_rad.sin(),
        yaw_rad.sin() * pitch_rad.cos(),
    )
    .normalize();

    let horizontal_forward = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin()).normalize_or_zero();
    let right = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos()).normalize();

    if *st.keys.get("W").unwrap_or(&false) {
        st.camera_pos += horizontal_forward * speed;
    }
    if *st.keys.get("S").unwrap_or(&false) {
        st.camera_pos -= horizontal_forward * speed;
    }
    if *st.keys.get("A").unwrap_or(&false) {
        st.camera_pos -= right * speed;
    }
    if *st.keys.get("D").unwrap_or(&false) {
        st.camera_pos += right * speed;
    }
    if *st.keys.get(" ").unwrap_or(&false) {
        st.camera_pos.y += speed;
    }
    if *st.keys.get("SHIFT").unwrap_or(&false) {
        st.camera_pos.y -= speed;
    }

    // 2. Raycast from camera
    let get_block = |pos: BlockPos| {
        let (cp, lp) = pos.to_chunk_and_local();
        let li = LocalIdx::from_coords_unchecked(lp.x() as u32, lp.y() as u32, lp.z() as u32);
        st.chunks
            .get(&cp)
            .map(|c| c.get(li))
            .unwrap_or(BlockStateId(0))
    };
    st.last_hit = raycast_voxels(st.camera_pos, forward, REACH_DISTANCE, |pos| {
        let b = get_block(pos);
        b != BlockStateId(0) && b != BlockStateId(6) // ignore air and water
    });

    // 3. FPS counter update
    st.frame_count += 1;
    if now - st.last_fps_time >= 1000.0 {
        st.fps = st.frame_count;
        st.frame_count = 0;
        st.last_fps_time = now;

        if let Some(el) = document.get_element_by_id("fps-text") {
            el.set_text_content(Some(&format!("FPS: {}", st.fps)));
        }
    }

    // Update position text
    if let Some(el) = document.get_element_by_id("pos-text") {
        el.set_text_content(Some(&format!(
            "Pos: {:.1}, {:.1}, {:.1}",
            st.camera_pos.x, st.camera_pos.y, st.camera_pos.z
        )));
    }

    // Update hotbar indicator
    if let Some(el) = document.get_element_by_id("slot-text") {
        let name = match st.selected_slot {
            0 => "1: Grass Block",
            1 => "2: Dirt",
            2 => "3: Stone",
            3 => "4: Oak Log",
            4 => "5: Oak Planks",
            5 => "6: Oak Leaves",
            6 => "7: Glass",
            7 => "8: Stone Slab",
            8 => "9: Oak Stairs",
            _ => "Block",
        };
        el.set_text_content(Some(name));
    }

    // 4. Render with WebGL2
    let gl = &gpu.gl;
    let width = canvas.width() as i32;
    let height = canvas.height() as i32;

    gl.viewport(0, 0, width, height);

    // Clear color: bright sky blue
    gl.clear_color(0.53, 0.81, 0.92, 1.0);
    gl.clear_depth(1.0);
    gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT);

    gl.enable(WebGl2RenderingContext::DEPTH_TEST);
    gl.depth_func(WebGl2RenderingContext::LEQUAL);
    gl.enable(WebGl2RenderingContext::CULL_FACE);
    gl.cull_face(WebGl2RenderingContext::BACK);

    let aspect = width as f32 / height.max(1) as f32;
    let proj = Mat4::perspective_rh(70.0f32.to_radians(), aspect, 0.1, 1000.0);
    let view = Mat4::look_to_rh(st.camera_pos, forward, Vec3::Y);
    let view_proj = proj * view;

    gl.use_program(Some(&gpu.program));

    let view_proj_slice = view_proj.to_cols_array();
    gl.uniform_matrix4fv_with_f32_array(Some(&gpu.u_view_proj), false, &view_proj_slice);
    gl.uniform3f(Some(&gpu.u_sun_dir), 0.4, 0.8, 0.3);

    // Helper to draw a VBO
    let draw_mesh_layer = |vbo: &WebGlBuffer, count: i32| {
        if count == 0 {
            return;
        }
        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(vbo));
        let stride = VERTEX_STRIDE_FLOATS * 4;

        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, stride, 0);

        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(
            1,
            3,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            3 * 4,
        );

        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(
            2,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            6 * 4,
        );

        gl.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, count);
    };

    // Draw opaque chunks
    for mesh in st.meshes.values() {
        draw_mesh_layer(&mesh.opaque_vbo, mesh.opaque_count);
    }

    // Draw cutout (leaves)
    for mesh in st.meshes.values() {
        draw_mesh_layer(&mesh.cutout_vbo, mesh.cutout_count);
    }

    // Draw translucent (water, glass) with blending
    gl.enable(WebGl2RenderingContext::BLEND);
    gl.blend_func(
        WebGl2RenderingContext::SRC_ALPHA,
        WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
    );
    gl.depth_mask(false);

    for mesh in st.meshes.values() {
        draw_mesh_layer(&mesh.translucent_vbo, mesh.translucent_count);
    }

    gl.depth_mask(true);
    gl.disable(WebGl2RenderingContext::BLEND);

    // Draw wireframe selection highlight box
    if let Some(hit) = st.last_hit {
        gl.use_program(Some(&gpu.highlight_program));
        let view_proj_slice = view_proj.to_cols_array();
        gl.uniform_matrix4fv_with_f32_array(Some(&gpu.hl_u_view_proj), false, &view_proj_slice);

        let hit_block = get_block(hit.pos);
        // Targeted block bounds: match sub-box or unit cube
        let (bmin, bmax) = match st.registry.shape(hit_block) {
            BlockShape::Boxes(boxes) if !boxes.is_empty() => {
                let b = boxes[0];
                (
                    Vec3::new(
                        hit.pos.x() as f32 + b.min[0] as f32 / 16.0,
                        hit.pos.y() as f32 + b.min[1] as f32 / 16.0,
                        hit.pos.z() as f32 + b.min[2] as f32 / 16.0,
                    ),
                    Vec3::new(
                        hit.pos.x() as f32 + b.max[0] as f32 / 16.0,
                        hit.pos.y() as f32 + b.max[1] as f32 / 16.0,
                        hit.pos.z() as f32 + b.max[2] as f32 / 16.0,
                    ),
                )
            }
            _ => (
                Vec3::new(hit.pos.x() as f32, hit.pos.y() as f32, hit.pos.z() as f32),
                Vec3::new(
                    hit.pos.x() as f32 + 1.0,
                    hit.pos.y() as f32 + 1.0,
                    hit.pos.z() as f32 + 1.0,
                ),
            ),
        };

        // Slight expansion to prevent z-fighting with block face
        let eps = 0.002;
        gl.uniform3f(
            Some(&gpu.hl_u_min),
            bmin.x - eps,
            bmin.y - eps,
            bmin.z - eps,
        );
        gl.uniform3f(
            Some(&gpu.hl_u_max),
            bmax.x + eps,
            bmax.y + eps,
            bmax.z + eps,
        );

        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&gpu.highlight_vbo),
        );
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, 3 * 4, 0);

        gl.draw_arrays(WebGl2RenderingContext::LINES, 0, 24);
    }
}
