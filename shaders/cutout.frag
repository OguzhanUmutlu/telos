#version 460

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_dir;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_textures;
layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

uint get_texture_layer(uint mat) {
    switch (mat) {
        case 8u: // Oak Leaves
            return 7u;
        case 9u: // Glass
            return 8u;
        default:
            return 7u;
    }
}

void main() {
    uint layer = get_texture_layer(v_material);

    // Sample repeating 2D texture array slice
    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Alpha test: discard transparent texels immediately to preserve early-Z
    if (tex_color.a < 0.5) {
        discard;
    }

    // Biome foliage tint for oak leaves (Classic Voxel plains green #48b518)
    if (v_material == 8u) {
        tex_color.rgb *= vec3(0.298, 0.600, 0.129);
    }

    // Directional face shading factor
    float face_shade = 0.75;
    if (abs(v_normal.y) > 0.5) {
        face_shade = (v_normal.y > 0.0) ? 1.0 : 0.5;
    } else if (abs(v_normal.z) > 0.5) {
        face_shade = 0.85;
    } else {
        face_shade = 0.75;
    }

    // Unpack smooth interpolated light components
    float ao_raw = v_light.x;    // 0.0 .. 3.0
    float sky_raw = v_light.y;   // 0.0 .. 15.0
    float block_raw = v_light.z; // 0.0 .. 15.0

    // Smooth ambient occlusion factor
    float ao_factor = mix(0.40, 1.0, ao_raw / 3.0);

    // Sample 16x16 lightmap LUT with bilinear filtering
    vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
    vec3 light_color = texture(u_lightmap, lightmap_uv).rgb;

    vec3 total_light = clamp(light_color * face_shade * ao_factor, 0.0, 1.0);

    out_color = vec4(tex_color.rgb * total_light, 1.0);
}
