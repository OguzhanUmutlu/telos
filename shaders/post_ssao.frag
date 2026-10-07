#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_occlusion;

layout(set = 0, binding = 0) uniform sampler2D u_depth;

layout(push_constant) uniform SsaorPushConstants {
    mat4 u_inv_proj;
    mat4 u_proj;
    vec2 u_screen_size;
    float u_radius;
    float u_bias;
};

// 16 Poisson/spiral hemisphere sample points
const vec3 SAMPLES[16] = vec3[16](
    vec3( 0.23,  0.18, 0.42), vec3(-0.31,  0.22, 0.51),
    vec3( 0.15, -0.38, 0.63), vec3(-0.25, -0.19, 0.35),
    vec3( 0.45,  0.42, 0.71), vec3(-0.52,  0.35, 0.62),
    vec3( 0.38, -0.49, 0.58), vec3(-0.41, -0.45, 0.73),
    vec3( 0.12,  0.15, 0.88), vec3(-0.18,  0.11, 0.92),
    vec3( 0.09, -0.14, 0.85), vec3(-0.11, -0.16, 0.95),
    vec3( 0.58,  0.12, 0.31), vec3(-0.62,  0.08, 0.28),
    vec3( 0.11,  0.64, 0.29), vec3(-0.09, -0.59, 0.33)
);

vec3 get_view_pos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view_pos = u_inv_proj * clip;
    return view_pos.xyz / max(view_pos.w, 0.00001);
}

void main() {
    float depth = texture(u_depth, v_uv).r;

    // Reversed-Z: depth <= 0.00001 is sky/infinite background
    if (depth <= 0.00001) {
        out_occlusion = 1.0;
        return;
    }

    vec3 origin = get_view_pos(v_uv, depth);

    // Reconstruct surface normal from screen-space view-position derivatives
    vec3 dx = dFdx(origin);
    vec3 dy = dFdy(origin);
    vec3 normal = cross(dx, dy);
    float normal_len = length(normal);
    if (normal_len < 0.00001) {
        out_occlusion = 1.0;
        return;
    }
    normal /= normal_len;

    // Pseudo-random noise rotation based on pixel coordinates
    float angle = fract(sin(dot(v_uv * u_screen_size, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    float occlusion = 0.0;
    float valid_samples = 0.0;

    for (int i = 0; i < 16; i++) {
        vec3 sample_dir = SAMPLES[i];
        sample_dir.xy = rot * sample_dir.xy;

        // Orient sample into normal-facing hemisphere
        if (dot(sample_dir, normal) < 0.0) {
            sample_dir = -sample_dir;
        }

        vec3 sample_pos = origin + sample_dir * u_radius;

        // Project sample point back to screen UV
        vec4 sample_clip = u_proj * vec4(sample_pos, 1.0);
        vec2 sample_uv = (sample_clip.xy / max(sample_clip.w, 0.00001)) * 0.5 + 0.5;

        // Boundary check
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        float sample_depth = texture(u_depth, sample_uv).r;
        vec3 actual_view_pos = get_view_pos(sample_uv, sample_depth);

        // Distance range check to avoid false occlusion across depth discontinuities
        float dist_diff = origin.z - actual_view_pos.z;
        float range_check = smoothstep(0.0, 1.0, u_radius / (abs(dist_diff) + 0.001));

        // In reversed-Z, a higher depth value means closer to camera
        // So actual geometry occludes sample if actual_view_pos is closer than (sample_pos + bias)
        if (actual_view_pos.z >= sample_pos.z + u_bias) {
            occlusion += range_check;
        }
        valid_samples += 1.0;
    }

    if (valid_samples > 0.0) {
        out_occlusion = clamp(1.0 - (occlusion / valid_samples) * 1.5, 0.0, 1.0);
    } else {
        out_occlusion = 1.0;
    }
}
