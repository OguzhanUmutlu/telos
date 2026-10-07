#version 450 core

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

uniform mat4 u_inv_view_proj;
uniform vec3 u_sun_dir;
uniform float u_daylight;
uniform float u_sunset;

const vec3 NOON_ZENITH = vec3(0.35, 0.58, 0.95);
const vec3 NOON_HORIZON = vec3(0.68, 0.82, 0.98);
const vec3 SUNSET_ZENITH = vec3(0.20, 0.16, 0.38);
const vec3 SUNSET_HORIZON = vec3(0.98, 0.45, 0.15);
const vec3 NIGHT_ZENITH = vec3(0.015, 0.02, 0.045);
const vec3 NIGHT_HORIZON = vec3(0.04, 0.05, 0.09);

void main() {
    vec4 ndc = vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world_h = u_inv_view_proj * ndc;
    vec3 ray_dir = normalize(world_h.xyz / world_h.w);

    float y_pos = max(0.0, ray_dir.y);
    float alt_curve = pow(y_pos, 0.75);

    vec3 day_sky = mix(NOON_HORIZON, NOON_ZENITH, alt_curve);
    vec3 sunset_sky = mix(SUNSET_HORIZON, SUNSET_ZENITH, alt_curve);
    vec3 night_sky = mix(NIGHT_HORIZON, NIGHT_ZENITH, alt_curve);

    vec3 sky = mix(night_sky, day_sky, u_daylight);
    float cos_sun = dot(ray_dir, normalize(u_sun_dir));
    float sun_glow = pow(max(0.0, cos_sun * 0.5 + 0.5), 3.0);
    sky = mix(sky, sunset_sky, u_sunset * (0.35 + 0.65 * sun_glow));

    // Sun disc
    if (cos_sun > 0.998) {
        sky += vec3(1.0, 0.95, 0.8) * 1.5;
    }

    frag_color = vec4(sky, 1.0);
}
