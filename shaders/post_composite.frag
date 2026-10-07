#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_scene_color;
layout(set = 0, binding = 1) uniform sampler2D u_depth;
layout(set = 0, binding = 2) uniform sampler2D u_ssao;

layout(push_constant) uniform PostCompositePushConstants {
    mat4 u_inv_view_proj;
    vec3 u_cam_pos;
    float u_time_of_day;
    vec3 u_sun_dir;
    float u_view_distance;
    float u_fog_density;
    uint u_flags; // bit 0: ssao, bit 1: volumetric_fog, bit 2: tonemapping, bit 3: vignette
    vec2 u_screen_size;
};

void main() {
    vec4 scene_sample = texture(u_scene_color, v_uv);
    vec3 color = scene_sample.rgb;
    float depth = texture(u_depth, v_uv).r;

    // Fast 4-tap box blur on SSAO to soften ambient contact shadows
    vec2 texel = 1.0 / u_screen_size;
    float ssao = 0.0;
    ssao += texture(u_ssao, v_uv + vec2(-texel.x, -texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2( texel.x, -texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2(-texel.x,  texel.y)).r;
    ssao += texture(u_ssao, v_uv + vec2( texel.x,  texel.y)).r;
    ssao *= 0.25;

    // Bit 0: SSAO enabled (only for geometry, not sky)
    if ((u_flags & 1u) != 0u && depth > 0.00001) {
        color *= mix(1.0, ssao, 0.65);
    }

    // Bit 1: Volumetric horizon fog & atmospheric scattering
    if ((u_flags & 2u) != 0u) {
        vec4 clip = vec4(v_uv * 2.0 - 1.0, depth, 1.0);
        vec4 world_h = u_inv_view_proj * clip;
        vec3 world_pos = world_h.xyz / max(world_h.w, 0.00001);
        vec3 ray = world_pos - u_cam_pos;
        float dist = length(ray);
        vec3 ray_dir = ray / max(dist, 0.001);

        bool is_sky = (depth <= 0.00001);
        if (is_sky) {
            dist = u_view_distance * 32.0;
        }

        // Height-modulated exponential fog (denser in valleys, clearer at mountain peaks)
        float height_factor = exp(-max(0.0, world_pos.y - 45.0) * 0.02);
        float effective_dist = min(dist, u_view_distance * 32.0);
        float fog_factor = 1.0 - exp(-effective_dist * (u_fog_density * 0.0015) * (1.0 + height_factor * 1.5));
        fog_factor = clamp(fog_factor, 0.0, 1.0);

        // Sun forward-scattering glare (Mie scattering)
        float sun_cos = max(0.0, dot(ray_dir, normalize(u_sun_dir)));
        float mie = pow(sun_cos, 8.0) * 0.5 + pow(sun_cos, 32.0) * 0.5;

        // Time of day atmospheric colors:
        // u_time_of_day: 0..=24000 (6000=noon, 12000=sunset, 18000=midnight, 24000=sunrise)
        float sun_elev = sin((u_time_of_day / 24000.0) * 6.2831853 - 1.5707963);
        vec3 day_fog = vec3(0.68, 0.78, 0.90);
        vec3 sunset_fog = vec3(0.95, 0.55, 0.25);
        vec3 night_fog = vec3(0.03, 0.04, 0.08);

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

        vec3 sun_glow = vec3(1.0, 0.85, 0.55) * mie;
        if (sun_elev <= 0.2 && sun_elev > -0.2) {
            sun_glow = vec3(1.0, 0.45, 0.15) * mie * 1.5;
        }

        if (!is_sky) {
            color = mix(color, base_fog_color, fog_factor) + sun_glow * fog_factor * 0.4;
        } else {
            float horizon_haze = clamp(1.0 - abs(ray_dir.y) * 4.0, 0.0, 1.0) * 0.35;
            color = mix(color, base_fog_color, horizon_haze) + sun_glow * horizon_haze * 0.35;
        }
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

    // Bit 3: Vignette
    if ((u_flags & 8u) != 0u) {
        vec2 v_coord = (v_uv - 0.5) * vec2(1.0, u_screen_size.y / u_screen_size.x);
        float vig = 1.0 - smoothstep(0.35, 0.75, length(v_coord)) * 0.22;
        color *= vig;
    }

    out_color = vec4(color, 1.0);
}
