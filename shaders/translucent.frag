#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_dir;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_textures;
layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

layout(push_constant, std430) uniform PushConstants {
    mat4 view_proj;                   // 64 bytes (0..64)
    uint64_t draw_info_buffer_address; // 8 bytes (64..72)
    uint frame_tick;                  // 4 bytes (72..76)
    uint water_base_layer;             // 4 bytes (76..80)
    vec4 camera_pos;                  // 16 bytes (80..96, xyz = camera pos, w = sim_dist_meters)
    uint water_frame_count;            // 4 bytes (96..100)
    uint water_flow_base_layer;        // 4 bytes (100..104)
    uint water_flow_frame_count;       // 4 bytes (104..108)
    uint lava_base_layer;              // 4 bytes (108..112)
    uint lava_frame_count;             // 4 bytes (112..116)
    uint fire_base_layer;              // 4 bytes (116..120)
    uint fire_frame_count;             // 4 bytes (120..124)
    uint _pad;                         // 4 bytes (124..128)
} pc;

void main() {
    float dist = length(v_world_pos - pc.camera_pos.xyz);
    bool is_simulated = (dist <= pc.camera_pos.w);

    uint frame = 0u;
    uint layer = pc.water_base_layer;
    vec3 fluid_tint = vec3(0.247, 0.463, 0.894); // Plains water tint #3f76e4
    float alpha = 0.72;

    if (v_material == 31u || v_material == 32u) {
        // Lava
        if (is_simulated && pc.lava_frame_count > 1u) {
            frame = (pc.frame_tick / 2u) % pc.lava_frame_count;
        }
        layer = pc.lava_base_layer + frame;
        fluid_tint = vec3(1.0);
        alpha = 1.0;
    } else if (v_material == 15u) {
        // Flowing water
        if (is_simulated && pc.water_flow_frame_count > 1u) {
            frame = (pc.frame_tick / 3u) % pc.water_flow_frame_count;
        }
        layer = pc.water_flow_base_layer + frame;
    } else {
        // Still water
        if (is_simulated && pc.water_frame_count > 1u) {
            frame = (pc.frame_tick / 3u) % pc.water_frame_count;
        }
        layer = pc.water_base_layer + frame;
    }

    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Procedural wave normal perturbation for fluid surfaces
    vec3 normal = v_normal;
    if (v_material != 31u && v_material != 32u) {
        float wave_time = float(pc.frame_tick) * 0.05;
        vec2 p = v_world_pos.xz;
        if (v_normal.y > 0.5) {
            float dx = cos(p.x * 1.5 + wave_time * 1.2) * 1.5 * cos(p.y * 1.2 + wave_time * 0.9) * 0.035
                     + cos((p.x + p.y) * 2.3 + wave_time * 1.8) * 2.3 * 0.02
                     + cos(p.x * 3.8 - wave_time * 2.4) * 3.8 * 0.01;
            float dz = -sin(p.x * 1.5 + wave_time * 1.2) * sin(p.y * 1.2 + wave_time * 0.9) * 1.2 * 0.035
                     + cos((p.x + p.y) * 2.3 + wave_time * 1.8) * 2.3 * 0.02
                     + cos(p.y * 3.6 + wave_time * 2.1) * 3.6 * 0.01;
            normal = normalize(vec3(-dx * 1.6, 1.0, -dz * 1.6));
        } else if (abs(v_normal.y) <= 0.5) {
            float ripple = sin(v_world_pos.y * 4.0 + wave_time * 2.0) * 0.12;
            normal = normalize(v_normal + vec3(0.0, ripple, 0.0));
        }
    }

    // Directional face shading factor
    float face_shade = 0.80;
    if (normal.y > 0.5) {
        face_shade = 1.0;
    } else if (normal.y < -0.5) {
        face_shade = 0.6;
    } else {
        face_shade = 0.80;
    }

    // Unpack smooth interpolated light components
    float ao_raw = v_light.x;    // 0.0 .. 3.0
    float sky_raw = v_light.y;   // 0.0 .. 15.0
    float block_raw = v_light.z; // 0.0 .. 15.0

    // Ambient occlusion factor
    float ao_factor = mix(0.45, 1.0, ao_raw / 3.0);

    // Sample 16x16 lightmap LUT with bilinear filtering
    vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
    vec3 light_color = texture(u_lightmap, lightmap_uv).rgb;

    vec3 total_light = clamp(light_color * face_shade * ao_factor, 0.0, 1.0);

    // Emissive glow for lava
    if (v_material == 31u || v_material == 32u) {
        total_light = max(total_light, vec3(1.0, 0.9, 0.7));
        out_color = vec4(tex_color.rgb * fluid_tint * total_light, alpha);
    } else {
        vec3 V = normalize(pc.camera_pos.xyz - v_world_pos);
        float NdotV = clamp(dot(normal, V), 0.0, 1.0);

        // Schlick Fresnel reflection factor: F = F0 + (1 - F0) * (1 - cos_theta)^5
        float fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);

        // Ambient sky dome & horizon reflection color based on sky light level
        float sky_light_ratio = clamp(sky_raw / 15.0, 0.0, 1.0);
        vec3 sky_reflection = mix(vec3(0.05, 0.08, 0.15), vec3(0.65, 0.82, 0.98), sky_light_ratio);

        // Sun / celestial specular reflection highlight (Blinn-Phong NdotH^96)
        vec3 sun_dir = normalize(vec3(0.5, 0.8, 0.3));
        vec3 H = normalize(V + sun_dir);
        float NdotH = max(0.0, dot(normal, H));
        float specular = pow(NdotH, 96.0) * sky_light_ratio;
        vec3 spec_color = vec3(1.0, 0.96, 0.88) * (specular * 1.2);

        // Beer-Lambert chromatic depth absorption
        float depth_est = mix(1.2, 6.0, 1.0 - NdotV);
        vec3 absorption = exp(-vec3(0.15, 0.05, 0.02) * depth_est);

        vec3 water_color = tex_color.rgb * fluid_tint * absorption * total_light;
        vec3 final_rgb = mix(water_color, sky_reflection * total_light, fresnel * 0.65) + spec_color;
        float final_alpha = clamp(mix(alpha, 0.90, fresnel), 0.0, 1.0);

        out_color = vec4(final_rgb, final_alpha);
    }
}
