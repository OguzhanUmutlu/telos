#version 460

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_dir;
layout(location = 4) in vec2 v_uv;

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

void main() {
    uint layer = get_texture_layer(v_material, v_dir);

    // Sample repeating 2D texture array slice
    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Biome tint for grass top (Classic Voxel plains green tint #79c05a)
    if (v_material == 3u && v_dir == 2u) {
        tex_color.rgb *= vec3(0.474, 0.753, 0.353);
    }

    // Directional face shading for voxel readability
    float light = 0.75;
    if (v_normal.y > 0.5) {
        light = 1.0;
    } else if (v_normal.y < -0.5) {
        light = 0.5;
    } else if (abs(v_normal.z) > 0.5) {
        light = 0.85;
    } else {
        light = 0.75;
    }

    out_color = vec4(tex_color.rgb * light, tex_color.a);
}
