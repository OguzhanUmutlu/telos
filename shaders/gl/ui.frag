#version 450 core

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in float v_mode;
layout(location = 3) flat in float v_layer;

uniform sampler2DArray u_ui_textures;
uniform int u_has_textures;

layout(location = 0) out vec4 frag_color;

void main() {
    // Mode 0: Solid color
    if (v_mode < 0.5 || u_has_textures == 0 || v_layer < 0.0) {
        frag_color = v_color;
        return;
    }

    // Mode 1: Font glyph or sprite from texture array
    vec4 tex = texture(u_ui_textures, vec3(v_uv, v_layer));
    frag_color = tex * v_color;
}
