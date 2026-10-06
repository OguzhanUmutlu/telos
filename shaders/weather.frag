#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in uint v_layer;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_weather;

void main() {
    vec4 tex = texture(u_weather, vec3(v_uv, float(v_layer)));
    vec4 col = v_color * tex;
    if (col.a < 0.005) {
        discard;
    }
    out_color = col;
}
