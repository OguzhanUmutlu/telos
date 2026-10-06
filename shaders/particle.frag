#version 460

layout(set = 0, binding = 0) uniform sampler2DArray u_particle_textures;
layout(set = 0, binding = 1) uniform sampler2DArray u_terrain_textures;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in uint v_layer;
layout(location = 3) flat in uint v_tex_source;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex_color;
    if (v_tex_source == 0u) {
        tex_color = texture(u_particle_textures, vec3(v_uv, float(v_layer)));
    } else {
        tex_color = texture(u_terrain_textures, vec3(v_uv, float(v_layer)));
    }

    vec4 final_color = tex_color * v_color;
    if (final_color.a < 0.01) {
        discard;
    }
    out_color = final_color;
}
