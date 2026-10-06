#version 460

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_dir;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_textures;

uint get_texture_layer(uint mat, uint dir) {
    switch (mat) {
        case 1u: // Stone
            return 0u;
        case 2u: // Dirt
            return 1u;
        case 3u: // Grass Block
            if (dir == 2u) return 2u; // Top face -> grass_block_top
            if (dir == 3u) return 1u; // Bottom face -> dirt
            return 3u;                // Side faces -> grass_block_side
        case 4u: // Bedrock
            return 4u;
        case 5u: // Sand
            return 5u;
        default:
            return 0u;
    }
}

const vec3 TORCH_COLOR = vec3(1.0, 0.82, 0.55);

void main() {
    uint layer = get_texture_layer(v_material, v_dir);

    // Sample repeating 2D texture array slice
    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Biome tint for grass top (Classic Voxel plains green tint #79c05a)
    if (v_material == 3u && v_dir == 2u) {
        tex_color.rgb *= vec3(0.474, 0.753, 0.353);
    }

    // Directional face shading factor
    float face_shade = 0.75;
    if (v_normal.y > 0.5) {
        face_shade = 1.0;
    } else if (v_normal.y < -0.5) {
        face_shade = 0.5;
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
    float ao_factor = mix(0.35, 1.0, ao_raw / 3.0);

    // Sky light with directional sunlight modulation
    float sky_norm = clamp(sky_raw / 15.0, 0.0, 1.0);
    vec3 sun_light = vec3(sky_norm * face_shade);

    // Block light (warm torch color with quadratic falloff)
    float block_norm = clamp(block_raw / 15.0, 0.0, 1.0);
    vec3 torch_light = TORCH_COLOR * (block_norm * block_norm * 0.85 + block_norm * 0.15);

    // Ambient light floor for deep caves
    const float AMBIENT_FLOOR = 0.04;
    vec3 total_light = (max(vec3(AMBIENT_FLOOR), sun_light) + torch_light) * ao_factor;
    total_light = clamp(total_light, 0.0, 1.0);

    out_color = vec4(tex_color.rgb * total_light, tex_color.a);
}
