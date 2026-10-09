#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_scene_color;
layout(set = 0, binding = 1) uniform sampler2D u_depth;
layout(set = 0, binding = 2) uniform sampler2D u_ssao;
layout(set = 0, binding = 3) uniform sampler2DArray u_shadow_map;

layout(std140, set = 0, binding = 4) uniform CascadeBuffer {
    mat4 u_light_view_proj[4];
    vec4 u_cascade_splits;
    float u_shadow_bias;
    float u_shadow_normal_bias;
    vec2 _pad;
};

layout(push_constant) uniform PostCompositePushConstants {
    mat4 u_inv_view_proj;
    vec3 u_cam_pos;
    float u_time_of_day;
    vec3 u_sun_dir;
    float u_view_distance;
    float u_fog_density;
    uint u_flags; // bit 0: ssao, bit 1: volumetric_fog, bit 2: tonemapping, bit 3: vignette, bit 4: shadows, bit 5: underwater
    vec2 u_screen_size;
};

float sample_cascade_pcf(int c_idx, vec3 w_pos, vec3 norm, vec3 light_d) {
    float cos_theta = clamp(dot(norm, light_d), 0.0, 1.0);
    float n_bias = u_shadow_normal_bias * (1.0 - cos_theta);
    vec3 biased_pos = w_pos + norm * n_bias;

    vec4 light_clip = u_light_view_proj[c_idx] * vec4(biased_pos, 1.0);
    vec3 proj_coords = light_clip.xyz / light_clip.w;

    // Vulkan NDC: x,y in [-1, 1], z in [0, 1]
    vec2 shadow_uv = proj_coords.xy * 0.5 + 0.5;
    float current_depth = proj_coords.z;

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || current_depth > 1.0 || current_depth < 0.0) {
        return 1.0;
    }

    float bias = max(u_shadow_bias * (1.0 - cos_theta), u_shadow_bias * 0.2);

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(u_shadow_map, 0).xy;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(u_shadow_map, vec3(shadow_uv + vec2(x, y) * texel_size, float(c_idx))).r;
            shadow += (current_depth - bias <= pcf_depth) ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

void main() {
    vec2 sample_uv = v_uv;
    // Bit 5: Underwater screen-space optical refraction & wobble
    if ((u_flags & 32u) != 0u) {
        float t = u_time_of_day * 0.05;
        vec2 wobble = vec2(
            sin(v_uv.y * 28.0 + t * 0.3) * 0.003 + cos(v_uv.x * 35.0 + t * 0.22) * 0.002,
            cos(v_uv.x * 28.0 + t * 0.28) * 0.003 + sin(v_uv.y * 35.0 + t * 0.18) * 0.002
        );
        sample_uv = clamp(v_uv + wobble, 0.0, 1.0);
    }

    vec4 scene_sample = texture(u_scene_color, sample_uv);
    vec3 color = scene_sample.rgb;
    float depth = texture(u_depth, sample_uv).r;
    bool is_sky = (depth <= 0.00001);

    // Fast 4-tap box blur on SSAO to soften ambient contact shadows
    vec2 texel = 1.0 / u_screen_size;
    float ssao = 0.0;
    ssao += texture(u_ssao, v_uv + vec2(-texel.x, -texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2( texel.x, -texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2(-texel.x,  texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2( texel.x,  texel.y)).r;
    ssao *= 0.25;

    // Bit 0: SSAO enabled (only for geometry, not sky)
    if ((u_flags & 1u) != 0u && !is_sky) {
        color *= mix(1.0, ssao, 0.65);
    }

    // World position reconstruction from depth
    vec4 clip = vec4(v_uv * 2.0 - 1.0, depth, 1.0);
    vec4 world_h = u_inv_view_proj * clip;
    vec3 world_pos = world_h.xyz / max(world_h.w, 0.00001);
    vec3 ray = world_pos - u_cam_pos;
    float dist = length(ray);
    vec3 ray_dir = ray / max(dist, 0.001);

    // Light direction (sun during day, moon during night)
    vec3 light_dir = normalize(u_sun_dir);
    if (light_dir.y < 0.0) {
        light_dir = -light_dir;
    }
    light_dir = normalize(light_dir);

    // Bit 4: Cascaded Shadow Mapping (CSM)
    if ((u_flags & 16u) != 0u && !is_sky) {
        vec3 normal = normalize(cross(dFdx(world_pos), dFdy(world_pos)));
        if (dot(normal, -ray_dir) < 0.0) {
            normal = -normal;
        }

        // Determine cascade by distance
        int cascade = 3;
        for (int i = 0; i < 3; ++i) {
            if (dist < u_cascade_splits[i]) {
                cascade = i;
                break;
            }
        }

        float shadow_factor = sample_cascade_pcf(cascade, world_pos, normal, light_dir);

        // Smooth cross-fade between cascades at outer boundary to eliminate seam artifacts
        if (cascade < 3) {
            float split_val = u_cascade_splits[cascade];
            float prev_split = (cascade == 0) ? 0.0 : u_cascade_splits[cascade - 1];
            float fade_margin = (split_val - prev_split) * 0.15;
            if (dist > split_val - fade_margin) {
                float next_shadow = sample_cascade_pcf(cascade + 1, world_pos, normal, light_dir);
                float blend = clamp((dist - (split_val - fade_margin)) / max(fade_margin, 0.001), 0.0, 1.0);
                shadow_factor = mix(shadow_factor, next_shadow, blend);
            }
        }

        float cos_theta = clamp(dot(normal, light_dir), 0.0, 1.0);
        float direct_contrib = shadow_factor * cos_theta;
        // Blend between ambient shadow floor (0.38) and fully illuminated surface (1.0)
        float shadow_light = mix(0.38, 1.0, direct_contrib);
        color *= shadow_light;
    }

    // Bit 1: Volumetric horizon fog & atmospheric scattering
    if ((u_flags & 2u) != 0u) {
        float effective_dist = min(dist, u_view_distance * 32.0);
        float height_density = exp(-max(0.0, world_pos.y - 48.0) * 0.015);
        float fog_factor = 1.0 - exp(-effective_dist * (u_fog_density * 0.0016) * (1.0 + height_density * 1.8));
        fog_factor = clamp(fog_factor, 0.0, 1.0);

        // Dual-scattering Mie forward phase function
        float sun_cos = max(0.0, dot(ray_dir, normalize(u_sun_dir)));
        float mie_phase = pow(max(0.0, sun_cos * 0.5 + 0.5), 6.0) * 0.7 + pow(max(0.0, sun_cos), 32.0) * 0.3;

        // Time of day atmospheric colors harmonized with sky.frag
        float sun_elev = sin((u_time_of_day / 24000.0) * 6.2831853 - 1.5707963);
        vec3 day_fog = vec3(0.68, 0.82, 0.98);
        vec3 sunset_fog = vec3(0.98, 0.45, 0.15);
        vec3 night_fog = vec3(0.04, 0.05, 0.09);

        vec3 base_fog_color;
        if (sun_elev > 0.2) {
            base_fog_color = day_fog;
        } else if (sun_elev > -0.1) {
            float t = (sun_elev - (-0.1)) / 0.3;
            base_fog_color = mix(sunset_fog, day_fog, t);
        } else {
            float t = clamp((sun_elev - (-0.4)) / 0.3, 0.0, 1.0);
            base_fog_color = mix(night_fog, sunset_fog, t);
        }

        vec3 sun_glow = vec3(1.0, 0.90, 0.70) * mie_phase * 1.2;
        if (sun_elev <= 0.25 && sun_elev > -0.15) {
            sun_glow = vec3(1.0, 0.48, 0.16) * mie_phase * 2.2;
        }

        if (!is_sky) {
            color = mix(color, base_fog_color, fog_factor) + sun_glow * fog_factor * 0.45;
        } else {
            float horizon_haze = exp(-max(0.0, ray_dir.y) * 4.0) * 0.40;
            color = mix(color, base_fog_color, horizon_haze) + sun_glow * horizon_haze * 0.40;
        }
    }

    // Bit 5: Underwater volumetric absorption, screen-space caustics & pressure vignette
    if ((u_flags & 32u) != 0u) {
        float water_dist = min(dist, 40.0);
        // Beer-Lambert wavelength-dependent absorption across submerged ray
        vec3 water_transmittance = exp(-vec3(0.12, 0.04, 0.015) * water_dist);
        vec3 underwater_ambient = vec3(0.03, 0.16, 0.26);

        if (!is_sky) {
            // Screen-space sunlight caustics projected on submerged geometry
            float caustic_time = u_time_of_day * 0.06;
            float c1 = sin(world_pos.x * 2.2 + world_pos.z * 1.8 + caustic_time * 1.5);
            float c2 = cos(world_pos.x * 1.7 - world_pos.z * 2.4 - caustic_time * 1.3);
            float caustic = max(0.0, c1 + c2 - 0.6) * 0.22;
            caustic *= exp(-water_dist * 0.09);
            color += vec3(0.12, 0.32, 0.42) * caustic;

            color = color * water_transmittance + underwater_ambient * (vec3(1.0) - water_transmittance);
        } else {
            color = mix(color, underwater_ambient, 0.88);
        }

        // Underwater peripheral pressure vignette
        vec2 vig_uv = (v_uv - 0.5) * 2.0;
        float underwater_vig = clamp(1.0 - dot(vig_uv, vig_uv) * 0.35, 0.0, 1.0);
        color *= mix(vec3(0.65, 0.85, 0.98), vec3(1.0), underwater_vig);
    }

    // Bit 2: Filmic ACES Tonemapping
    if ((u_flags & 4u) != 0u) {
        vec3 x = color;
        const float a = 2.51;
        const float b = 0.03;
        const float c = 2.43;
        const float d = 0.59;
        const float e = 0.14;
        color = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    // Bit 3: Subtle Cinematic Vignette
    if ((u_flags & 8u) != 0u) {
        vec2 vig_uv = (v_uv - 0.5) * 2.0;
        float vig = clamp(1.0 - dot(vig_uv, vig_uv) * 0.25, 0.0, 1.0);
        color *= vig;
    }

    out_color = vec4(color, scene_sample.a);
}
