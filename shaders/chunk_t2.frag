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
    mat4 view_proj;                   // 64 bytes (0..64)
    uint64_t quad_buffer_address;     // 8 bytes (64..72)
    int chunk_x;                      // 4 bytes (72..76)
    int chunk_y;                      // 4 bytes (76..80)
    vec4 camera_pos;                  // 16 bytes (80..96, xyz = camera pos, w = sim_dist_meters)
    int chunk_z;                      // 4 bytes (96..100)
    uint frame_tick_flags;            // 4 bytes (100..104, bit 31 = is_translucent, bits 0..30 = frame_tick)
    uint water_base_layer;            // 4 bytes (104..108)
    uint water_frame_count;           // 4 bytes (108..112)
    uint lava_base_layer;             // 4 bytes (112..116)
    uint lava_frame_count;            // 4 bytes (116..120)
    uint fire_base_layer;             // 4 bytes (120..124)
    uint fire_frame_count;            // 4 bytes (124..128)
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
        case 31u: // Lava
        case 32u: // Flowing lava
            return pc.lava_base_layer;
        case 33u: // Fire
            return pc.fire_base_layer;
        default:
            return 0u;
    }
}

void main() {
    float dist = length(v_world_pos - pc.camera_pos.xyz);
    bool is_simulated = (dist <= pc.camera_pos.w);
    uint frame_tick = pc.frame_tick_flags & 0x7FFFFFFFu;
    uint is_translucent = (pc.frame_tick_flags >> 31u) & 1u;

    uint base_layer = get_texture_layer(v_material);

    // If fluid or animated surface, animate frame when inside simulation distance
    uint layer = base_layer;
    if (v_material == 31u || v_material == 32u) {
        uint frame = (is_simulated && pc.lava_frame_count > 1u) ? ((frame_tick / 2u) % pc.lava_frame_count) : 0u;
        layer = pc.lava_base_layer + frame;
    } else if (v_material == 33u) {
        uint frame = (is_simulated && pc.fire_frame_count > 1u) ? (frame_tick % pc.fire_frame_count) : 0u;
        layer = pc.fire_base_layer + frame;
    } else if (is_translucent != 0u || v_material == 6u || v_material == 8u) {
        uint frame = (is_simulated && pc.water_frame_count > 1u) ? ((frame_tick / 3u) % pc.water_frame_count) : 0u;
        layer = pc.water_base_layer + frame;
    }

    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Alpha test for cutout pass (plants, torches, fire)
    if (is_translucent == 0u && v_material != 6u && v_material != 8u && v_material != 31u && v_material != 32u) {
        if (tex_color.a < 0.1) {
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
    if (v_is_emissive != 0u || v_material == 31u || v_material == 32u || v_material == 33u) {
        light_color = vec3(1.0);
    } else {
        float ao_raw = v_light.x;    // 0.0 .. 3.0
        float sky_raw = v_light.y;   // 0.0 .. 15.0
        float block_raw = v_light.z; // 0.0 .. 15.0

        float ao_factor = mix(0.40, 1.0, ao_raw / 3.0);
        vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
        vec3 sampled_light = texture(u_lightmap, lightmap_uv).rgb;

        // Daylight ambient floor: prevents outdoor side faces from turning pitch black
        vec3 sky_day_color = texture(u_lightmap, vec2(0.5 / 16.0, 15.5 / 16.0)).rgb;
        vec3 daylight_floor = sky_day_color * 0.20;
        float sky_exposure = clamp(sky_raw / 2.0, 0.0, 1.0);
        vec3 final_light = mix(sampled_light, max(sampled_light, daylight_floor), sky_exposure);

        light_color = clamp(final_light * face_shade * ao_factor, 0.0, 1.0);
    }

    if (v_material == 31u || v_material == 32u) {
        // Lava
        out_color = vec4(tex_color.rgb * light_color, 1.0);
    } else if (is_translucent != 0u || v_material == 6u || v_material == 8u) {
        // Water blue tint
        vec3 water_tint = vec3(0.247, 0.463, 0.894);
        out_color = vec4(tex_color.rgb * water_tint * light_color, 0.72);
    } else {
        out_color = vec4(tex_color.rgb * light_color, tex_color.a);
    }
}
