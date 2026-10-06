#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_celestial;

layout(push_constant) uniform SkyPushConstants {
    mat4 inv_view_proj;
    vec3 sun_dir;
    float time_of_day;
    vec3 moon_dir;
    uint moon_phase;
    vec3 camera_pos;
    float rain_level;
    float thunder_level;
    float lightning_flash;
    float _pad0;
    float _pad1;
} pc;

const vec3 NOON_ZENITH = vec3(0.35, 0.58, 0.95);
const vec3 NOON_HORIZON = vec3(0.68, 0.82, 0.98);

const vec3 SUNSET_ZENITH = vec3(0.20, 0.16, 0.38);
const vec3 SUNSET_HORIZON = vec3(0.98, 0.45, 0.15);

const vec3 NIGHT_ZENITH = vec3(0.015, 0.02, 0.045);
const vec3 NIGHT_HORIZON = vec3(0.04, 0.05, 0.09);

const vec3 OVERCAST_RAIN = vec3(0.42, 0.45, 0.48);
const vec3 OVERCAST_THUNDER = vec3(0.12, 0.14, 0.18);

void main() {
    // Reconstruct world-space ray direction from NDC at far plane
    vec4 ndc = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 world_h = pc.inv_view_proj * ndc;
    vec3 ray_dir = normalize(world_h.xyz / world_h.w);

    float y = ray_dir.y;
    float y_pos = max(0.0, y);

    // Daylight and sunset transition factors
    float sun_elev = pc.sun_dir.y;
    float daylight = clamp((sun_elev + 0.25) / 0.5, 0.0, 1.0);
    float sunset = clamp(1.0 - abs(sun_elev) * 4.0, 0.0, 1.0);

    // Base atmospheric sky gradients
    float alt_curve = pow(y_pos, 0.75);
    vec3 day_sky = mix(NOON_HORIZON, NOON_ZENITH, alt_curve);
    vec3 sunset_sky = mix(SUNSET_HORIZON, SUNSET_ZENITH, alt_curve);
    vec3 night_sky = mix(NIGHT_HORIZON, NIGHT_ZENITH, alt_curve);

    vec3 sky = mix(night_sky, day_sky, daylight);

    // Directional sunset golden hour glow
    float cos_sun = dot(ray_dir, pc.sun_dir);
    float sun_azimuth_glow = pow(max(0.0, cos_sun * 0.5 + 0.5), 3.0);
    sky = mix(sky, sunset_sky, sunset * (0.35 + 0.65 * sun_azimuth_glow));

    // Overcast weather modulation (rain slate grey and thunder charcoal)
    vec3 overcast_day = mix(OVERCAST_RAIN, OVERCAST_THUNDER, pc.thunder_level);
    vec3 overcast_night = mix(OVERCAST_RAIN * 0.15, OVERCAST_THUNDER * 0.35, pc.thunder_level);
    vec3 overcast_target = mix(overcast_night, overcast_day, daylight);
    float weather_weight = clamp(pc.rain_level * 0.85 + pc.thunder_level * 0.15, 0.0, 1.0);
    sky = mix(sky, overcast_target, weather_weight);

    // Darken below horizon
    float void_factor = clamp(1.0 + y * 2.0, 0.0, 1.0);
    sky *= void_factor;

    // Stars at night (obscured during overcast weather)
    float weather_clear = clamp(1.0 - pc.rain_level * 1.25, 0.0, 1.0);
    if (daylight < 0.85 && y > -0.05 && weather_clear > 0.01) {
        vec3 star_p = ray_dir * 180.0;
        vec3 star_cell = floor(star_p);
        vec3 star_f = fract(star_p) - 0.5;
        float star_h = fract(sin(dot(star_cell, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
        if (star_h > 0.988) {
            float dist = length(star_f);
            float star_intensity = smoothstep(0.12, 0.0, dist);
            float twinkle = 0.7 + 0.3 * sin(star_h * 628.0 + pc.time_of_day * 0.05);
            float star_fade = (1.0 - daylight) * smoothstep(-0.05, 0.1, y) * weather_clear;
            sky += vec3(star_intensity * twinkle * star_fade);
        }
    }

    // Sun disc and corona (attenuated by rain/thunder clouds)
    vec3 up = abs(pc.sun_dir.y) > 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    if (cos_sun > 0.0) {
        float corona = pow(cos_sun, 128.0) * 0.45 * daylight * weather_clear;
        sky += vec3(1.0, 0.95, 0.8) * corona;
    }

    vec3 sun_t = normalize(cross(up, pc.sun_dir));
    vec3 sun_b = cross(pc.sun_dir, sun_t);
    float sun_u = dot(ray_dir, sun_t) / 0.16 + 0.5;
    float sun_v = dot(ray_dir, sun_b) / 0.16 + 0.5;
    if (cos_sun > 0.85 && sun_u >= 0.0 && sun_u <= 1.0 && sun_v >= 0.0 && sun_v <= 1.0) {
        vec4 sun_tex = texture(u_celestial, vec3(sun_u, 1.0 - sun_v, 0.0));
        sky = mix(sky, sun_tex.rgb, sun_tex.a * weather_clear);
    }

    // Moon disc and phases (attenuated by rain/thunder clouds)
    float cos_moon = dot(ray_dir, pc.moon_dir);
    if (cos_moon > 0.0) {
        float moon_glow = pow(cos_moon, 64.0) * 0.2 * (1.0 - daylight) * weather_clear;
        sky += vec3(0.7, 0.8, 1.0) * moon_glow;
    }

    vec3 moon_up = abs(pc.moon_dir.y) > 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 moon_t = normalize(cross(moon_up, pc.moon_dir));
    vec3 moon_b = cross(pc.moon_dir, moon_t);
    float moon_u = dot(ray_dir, moon_t) / 0.14 + 0.5;
    float moon_v = dot(ray_dir, moon_b) / 0.14 + 0.5;
    if (cos_moon > 0.85 && moon_u >= 0.0 && moon_u <= 1.0 && moon_v >= 0.0 && moon_v <= 1.0) {
        uint phase_layer = 1u + (pc.moon_phase & 7u);
        vec4 moon_tex = texture(u_celestial, vec3(moon_u, 1.0 - moon_v, float(phase_layer)));
        sky = mix(sky, moon_tex.rgb, moon_tex.a * weather_clear);
    }

    // Lightning flash wash
    if (pc.lightning_flash > 0.001) {
        vec3 flash_wash = vec3(0.75, 0.82, 1.0) * pc.lightning_flash;
        sky = max(sky, flash_wash);
    }

    out_color = vec4(sky, 1.0);
}
