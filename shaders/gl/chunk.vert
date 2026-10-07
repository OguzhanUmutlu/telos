#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_layer;
layout(location = 4) in vec3 a_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)
layout(location = 5) in vec4 a_tint;

uniform mat4 u_view_proj;
uniform vec3 u_camera_pos;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world_pos;
layout(location = 2) out vec2 v_uv;
layout(location = 3) flat out float v_layer;
layout(location = 4) out vec3 v_light;
layout(location = 5) out vec4 v_tint;

void main() {
    v_normal = a_normal;
    v_world_pos = a_position;
    v_uv = a_uv;
    v_layer = a_layer;
    v_light = a_light;
    v_tint = a_tint;

    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
