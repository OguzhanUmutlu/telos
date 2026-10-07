#version 450 core

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in float v_layer;
layout(location = 4) in vec3 v_light;
layout(location = 5) in vec4 v_tint;

uniform sampler2DArray u_terrain_textures;
uniform sampler2D u_lightmap;
uniform vec3 u_sun_dir;
uniform float u_daylight;
uniform int u_use_textures;

layout(location = 0) out vec4 frag_color;

void main() {
    vec4 base_color = v_tint;

    if (u_use_textures != 0 && v_layer >= 0.0) {
        vec4 tex_color = texture(u_terrain_textures, vec3(v_uv, v_layer));
        base_color = tex_color * v_tint;
    }

    // Alpha test for cutout foliage, glass, etc.
    if (base_color.a < 0.1) {
        discard;
    }

    // Directional cardinal face shading
    float face_shade = 0.8;
    if (v_normal.y > 0.5) {
        face_shade = 1.0;
    } else if (v_normal.y < -0.5) {
        face_shade = 0.5;
    } else if (abs(v_normal.z) > 0.5) {
        face_shade = 0.8;
    } else if (abs(v_normal.x) > 0.5) {
        face_shade = 0.6;
    }

    // Sample dynamic 16x16 lighting lookup table (X: block light, Y: sky light)
    vec2 light_uv = clamp(vec2((v_light.z + 0.5) / 16.0, (v_light.y + 0.5) / 16.0), 0.0, 1.0);
    vec3 light_color = texture(u_lightmap, light_uv).rgb;

    // Ambient occlusion factor (0..3 mapped to 1.0..0.4)
    float ao_factor = 1.0 - clamp(v_light.x, 0.0, 3.0) * 0.20;

    // Directional sunlight modulation
    float ndotl = max(dot(v_normal, normalize(u_sun_dir)), 0.0);
    float sun_boost = 0.85 + 0.15 * ndotl;

    vec3 final_rgb = base_color.rgb * face_shade * light_color * ao_factor * sun_boost;

    frag_color = vec4(final_rgb, base_color.a);
}
