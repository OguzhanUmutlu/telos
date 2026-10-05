#version 460

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;

layout(location = 0) out vec4 out_color;

vec3 get_material_color(uint mat) {
    switch (mat) {
        case 1u: // Stone
            return vec3(0.490, 0.490, 0.490);
        case 2u: // Dirt
            return vec3(0.525, 0.376, 0.263);
        case 3u: // Grass
            return vec3(0.357, 0.549, 0.196);
        case 4u: // Bedrock
            return vec3(0.200, 0.200, 0.200);
        default:
            return vec3(0.700, 0.700, 0.700);
    }
}

void main() {
    vec3 base_color = get_material_color(v_material);

    // Directional face shading for voxel depth perception
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

    out_color = vec4(base_color * light, 1.0);
}
