#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_is_emissive;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)
layout(location = 6) flat in uint v_has_shading;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_textures;
layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

layout(push_constant, std430) uniform PushConstants {
    mat4 view_proj;
    uint64_t quad_buffer_address;
    int chunk_x;
    int chunk_y;
    int chunk_z;
    uint frame_tick;
    uint water_base_layer;
    uint water_frame_count;
    uint is_translucent;
    uint _pad;
} pc;

uint get_texture_layer(uint mat) {
    switch (mat) {
        case 6u:  // Water (standard)
        case 8u:  // Water (core pack)
            return pc.water_base_layer;
        case 11u: // Poppy fallback
        case 12u: // Poppy (standard)
        case 16u: // Poppy (core pack)
            return 9u;
        case 13u: // Dandelion (standard)
        case 17u: // Dandelion (core pack)
            return 10u;
        case 14u: // Torch (standard)
        case 15u: // Torch (core pack)
            return 11u;
        case 18u: // Short grass (core pack)
            return 12u;
        case 19u: // Fern (core pack)
            return 13u;
        case 20u: // Dead bush (core pack)
            return 14u;
        default:
            return 0u;
    }
}

void main() {
    uint base_layer = get_texture_layer(v_material);

    // If translucent fluid pass, animate water frame
    uint layer = base_layer;
    if (pc.is_translucent != 0u || v_material == 6u || v_material == 8u) {
        uint frame = (pc.water_frame_count > 1u) ? ((pc.frame_tick / 3u) % pc.water_frame_count) : 0u;
        layer = pc.water_base_layer + frame;
    }

    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Alpha test for cutout pass (plants, torches)
    if (pc.is_translucent == 0u && v_material != 6u && v_material != 8u) {
        if (tex_color.a < 0.5) {
            discard;
        }
    }

    // Directional face shading
    float face_shade = 1.0;
    if (v_has_shading != 0u) {
        if (abs(v_normal.y) > 0.5) {
            face_shade = (v_normal.y > 0.0) ? 1.0 : 0.5;
        } else if (abs(v_normal.z) > 0.5) {
            face_shade = 0.85;
        } else {
            face_shade = 0.75;
        }
    }

    // Light calculation
    vec3 light_color;
    if (v_is_emissive != 0u) {
        light_color = vec3(1.0);
    } else {
        float ao_raw = v_light.x;    // 0.0 .. 3.0
        float sky_raw = v_light.y;   // 0.0 .. 15.0
        float block_raw = v_light.z; // 0.0 .. 15.0

        float ao_factor = mix(0.40, 1.0, ao_raw / 3.0);
        vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
        vec3 sampled_light = texture(u_lightmap, lightmap_uv).rgb;
        light_color = clamp(sampled_light * face_shade * ao_factor, 0.0, 1.0);
    }

    if (pc.is_translucent != 0u || v_material == 6u || v_material == 8u) {
        // Water blue tint
        vec3 water_tint = vec3(0.247, 0.463, 0.894);
        out_color = vec4(tex_color.rgb * water_tint * light_color, 0.72);
    } else {
        out_color = vec4(tex_color.rgb * light_color, 1.0);
    }
}
