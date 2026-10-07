#version 450 core

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in float a_mode;
layout(location = 4) in float a_layer;

uniform vec2 u_viewport_size;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) flat out float v_mode;
layout(location = 3) flat out float v_layer;

void main() {
    v_uv = a_uv;
    v_color = a_color;
    v_mode = a_mode;
    v_layer = a_layer;

    vec2 ndc = vec2(
        (a_position.x / u_viewport_size.x) * 2.0 - 1.0,
        1.0 - (a_position.y / u_viewport_size.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
