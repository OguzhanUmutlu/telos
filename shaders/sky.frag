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
    float wind_time;
    float cloud_coverage;
} pc;

const vec3 NOON_ZENITH = vec3(0.35, 0.58, 0.95);
const vec3 NOON_HORIZON = vec3(0.68, 0.82, 0.98);

const vec3 SUNSET_ZENITH = vec3(0.20, 0.16, 0.38);
const vec3 SUNSET_HORIZON = vec3(0.98, 0.45, 0.15);

const vec3 NIGHT_ZENITH = vec3(0.015, 0.02, 0.045);
const vec3 NIGHT_HORIZON = vec3(0.04, 0.05, 0.09);

const vec3 OVERCAST_RAIN = vec3(0.42, 0.45, 0.48);
const vec3 OVERCAST_THUNDER = vec3(0.12, 0.14, 0.18);

// ----------------------------------------------------------------------------
// Procedural 3D Noise for Volumetric Clouds
// ----------------------------------------------------------------------------

float hash3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.1, 0.1));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3d(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3(i + vec3(0.0, 0.0, 0.0)), hash3(i + vec3(1.0, 0.0, 0.0)), f.x),
            mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
        mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), f.x),
            mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), f.x), f.y),
        f.z
    );
}

float cloud_fbm(vec3 p) {
    float f = 0.500 * noise3d(p); p = p * 2.02 + vec3(1.2, 3.4, 5.6);
    f += 0.300 * noise3d(p); p = p * 2.03 + vec3(2.3, 4.5, 6.7);
    f += 0.200 * noise3d(p);
    return f;
}

float sample_cloud_density(vec3 pos, vec3 wind, float coverage) {
    if (pos.y < 192.0 || pos.y > 280.0) {
        return 0.0;
    }
    float h_norm = (pos.y - 192.0) / 88.0;
    float height_profile = smoothstep(0.0, 0.18, h_norm) * smoothstep(1.0, 0.65, h_norm);
    vec3 uvw = (pos + wind) * 0.0035;
    float n = cloud_fbm(uvw);
    float threshold = 1.0 - coverage;
    return clamp((n * height_profile - threshold * 0.45) / max(0.05, 1.0 - threshold * 0.45), 0.0, 1.0);
}

float hg_phase(float cos_theta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(0.001, 1.0 + g2 - 2.0 * g * cos_theta), 1.5));
}

// ----------------------------------------------------------------------------
// Main Shader Entry Point
// ----------------------------------------------------------------------------

void main() {
    // Reconstruct world-space ray direction from NDC at far plane
    vec4 ndc = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 world_h = pc.inv_view_proj * ndc;
    // In reversed-Z infinite perspective, z=0 is at infinity where homogeneous w=0; world_h.xyz is already a direction vector
    vec3 ray_dir = length(world_h.xyz) > 1e-6 ? normalize(world_h.xyz) : vec3(0.0, 1.0, 0.0);

    float y = ray_dir.y;
    float y_pos = max(0.0, y);

    // Daylight and sunset transition factors
    float sun_elev = pc.sun_dir.y;
    float daylight = clamp((sun_elev + 0.25) / 0.5, 0.0, 1.0);
    float sunset = clamp(1.0 - abs(sun_elev) * 4.0, 0.0, 1.0);

    // Base atmospheric sky gradients (Rayleigh scattering)
    float alt_curve = pow(y_pos, 0.75);
    vec3 day_sky = mix(NOON_HORIZON, NOON_ZENITH, alt_curve);
    vec3 sunset_sky = mix(SUNSET_HORIZON, SUNSET_ZENITH, alt_curve);
    vec3 night_sky = mix(NIGHT_HORIZON, NIGHT_ZENITH, alt_curve);

    vec3 sky = mix(night_sky, day_sky, daylight);

    // Directional sunset golden hour glow
    float cos_sun = dot(ray_dir, normalize(pc.sun_dir));
    float sun_azimuth_glow = pow(max(0.0, cos_sun * 0.5 + 0.5), 3.0);
    sky = mix(sky, sunset_sky, sunset * (0.35 + 0.65 * sun_azimuth_glow));

    // Dual-scattering atmospheric horizon: Rayleigh zenith gradient + Mie aerosol horizon glow
    float horizon_optical_depth = exp(-max(0.0, y) * 3.5);
    float mie_horizon_phase = pow(max(0.0, cos_sun * 0.5 + 0.5), 6.0) * 0.7 + pow(max(0.0, cos_sun), 32.0) * 0.3;
    vec3 horizon_mie_color = mix(vec3(0.98, 0.55, 0.20), vec3(1.0, 0.95, 0.85), daylight);
    if (daylight < 0.15) {
        horizon_mie_color = vec3(0.12, 0.16, 0.28); // Silvery moonlight haze
    }
    vec3 atmospheric_haze = mix(sky, horizon_mie_color, mie_horizon_phase * 0.45 * (1.0 - pc.rain_level * 0.8));
    sky = mix(sky, atmospheric_haze, horizon_optical_depth * 0.65);

    // Overcast weather modulation (rain slate grey and thunder charcoal)
    vec3 overcast_day = mix(OVERCAST_RAIN, OVERCAST_THUNDER, pc.thunder_level);
    vec3 overcast_night = mix(OVERCAST_RAIN * 0.15, OVERCAST_THUNDER * 0.35, pc.thunder_level);
    vec3 overcast_target = mix(overcast_night, overcast_day, daylight);
    float weather_weight = clamp(pc.rain_level * 0.85 + pc.thunder_level * 0.15, 0.0, 1.0);
    sky = mix(sky, overcast_target, weather_weight);

    // Smooth atmospheric horizon transition below 0
    float void_factor = clamp(1.0 + min(0.0, y) * 1.5, 0.25, 1.0);
    sky *= void_factor;

    // Stars at night (rendered behind clouds)
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

    // Sun disc and corona (rendered behind clouds)
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

    // Moon disc and phases (rendered behind clouds)
    float cos_moon = dot(ray_dir, normalize(pc.moon_dir));
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

    // ------------------------------------------------------------------------
    // Procedural 3D Ray-Marched Volumetric Clouds Layer
    // ------------------------------------------------------------------------
    if (pc.cloud_coverage > 0.01) {
        float cam_y = pc.camera_pos.y;
        float y_bot = 192.0;
        float y_top = 280.0;

        float t0 = -1.0;
        float t1 = -1.0;

        if (ray_dir.y > 0.001) {
            // Ray pointing upwards towards cloud slab
            if (cam_y < y_bot) {
                t0 = (y_bot - cam_y) / ray_dir.y;
                t1 = (y_top - cam_y) / ray_dir.y;
            } else if (cam_y < y_top) {
                t0 = 0.0;
                t1 = (y_top - cam_y) / ray_dir.y;
            }
        } else if (ray_dir.y < -0.001) {
            // Ray pointing downwards from above cloud slab
            if (cam_y > y_top) {
                t0 = (y_top - cam_y) / ray_dir.y;
                t1 = (y_bot - cam_y) / ray_dir.y;
            } else if (cam_y > y_bot) {
                t0 = 0.0;
                t1 = (y_bot - cam_y) / ray_dir.y;
            }
        }

        if (t1 > t0 && t1 > 0.0) {
            t0 = max(0.0, t0);
            t1 = min(t1, t0 + 2400.0); // Clamp max distance to prevent horizon slowdown

            const int STEPS = 20;
            float dt = (t1 - t0) / float(STEPS);

            // Temporal / screen-space dithering jitter to eliminate banding
            float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
            float t = t0 + dt * jitter;

            vec3 wind = vec3(pc.wind_time * 8.0, 0.0, pc.wind_time * 4.0);
            float coverage = clamp(0.38 + (pc.rain_level * 0.35 + pc.thunder_level * 0.25) * pc.cloud_coverage, 0.0, 0.95);

            // Primary lighting setup
            vec3 light_dir = daylight > 0.05 ? normalize(pc.sun_dir) : normalize(pc.moon_dir);
            vec3 sun_light_col = mix(vec3(1.0, 0.5, 0.15) * 1.8, vec3(1.0, 0.98, 0.92) * 1.2, daylight);
            vec3 moon_light_col = vec3(0.45, 0.55, 0.75) * 0.6;
            vec3 light_col = daylight > 0.05 ? sun_light_col : moon_light_col;

            float cos_light = dot(ray_dir, light_dir);
            // Dual-lobe Henyey-Greenstein phase function for forward silver lining
            float phase = 0.72 * hg_phase(cos_light, 0.75) + 0.28 * hg_phase(cos_light, -0.22);

            vec3 ambient_light = mix(night_sky * 2.0, day_sky * 0.85, daylight);
            ambient_light *= mix(1.0, 0.25, pc.rain_level * 0.8 + pc.thunder_level * 0.2);

            float cloud_trans = 1.0;
            vec3 cloud_rgb = vec3(0.0);

            for (int i = 0; i < STEPS; ++i) {
                vec3 pos = pc.camera_pos + ray_dir * t;
                float d = sample_cloud_density(pos, wind, coverage);

                if (d > 0.005) {
                    // Secondary light ray march (2 sample steps for self-shadowing)
                    float d1 = sample_cloud_density(pos + light_dir * 18.0, wind, coverage);
                    float d2 = sample_cloud_density(pos + light_dir * 45.0, wind, coverage);
                    float light_tau = d1 * 18.0 + d2 * 27.0;
                    float light_trans = exp(-light_tau * 0.06);

                    // Powder effect (forward edge brightening)
                    float powder = 1.0 - exp(-d * 3.0);

                    vec3 step_scatter = (light_col * phase * light_trans * powder * 4.0 + ambient_light) * d;

                    // Internal lightning flash illumination inside storm clouds
                    if (pc.lightning_flash > 0.001) {
                        step_scatter += vec3(0.85, 0.92, 1.0) * pc.lightning_flash * 6.0 * d;
                    }

                    float step_extinction = exp(-d * dt * 0.025);
                    cloud_rgb += cloud_trans * step_scatter * (1.0 - step_extinction);
                    cloud_trans *= step_extinction;

                    if (cloud_trans < 0.01) {
                        break;
                    }
                }

                t += dt;
            }

            // Composite cloud layer over the celestial sky
            sky = sky * cloud_trans + cloud_rgb;
        }
    }

    // Lightning flash wash across entire sky
    if (pc.lightning_flash > 0.001) {
        vec3 flash_wash = vec3(0.75, 0.82, 1.0) * pc.lightning_flash;
        sky = max(sky, flash_wash);
    }

    out_color = vec4(sky, 1.0);
}
