#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)
layout(location = 3) in float v_distance;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

layout(push_constant, std430) uniform LodPushConstants {
    mat4 view_proj;
    vec3 camera_pos;
    float max_distance;
    uint64_t draw_info_buffer_address;
} pc;

void main() {
    // Directional face shading factor matching chunk.frag
    float face_shade = 0.75;
    if (v_normal.y > 0.5) {
        face_shade = 1.0;
    } else if (v_normal.y < -0.5) {
        face_shade = 0.5;
    } else if (abs(v_normal.z) > 0.5) {
        face_shade = 0.85;
    } else {
        face_shade = 0.75;
    }

    // Unpack light
    float ao_raw = v_light.x;    // 0..3
    float sky_raw = v_light.y;   // 0..15
    float block_raw = v_light.z; // 0..15

    float ao_factor = mix(0.5, 1.0, ao_raw / 3.0);

    // Sample 16x16 lightmap LUT with bilinear filtering
    vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
    vec3 light_color = texture(u_lightmap, lightmap_uv).rgb;

    vec3 total_light = clamp(light_color * face_shade * ao_factor, 0.0, 1.0);

    vec3 lit_color = v_color.rgb * total_light;

    // Atmospheric distance fog to seamlessly blend far terrain into sky
    float fog_start = pc.max_distance * 0.70;
    float fog_end = pc.max_distance;
    float fog_linear = clamp((v_distance - fog_start) / max(1.0, fog_end - fog_start), 0.0, 1.0);
    float fog_factor = fog_linear * fog_linear; // Smooth extinction curve
    // Sky fog matches ambient sky light color and horizon gradient dynamically
    vec3 sky_day_color = vec3(0.68, 0.82, 0.98);
    vec3 light_intensity = texture(u_lightmap, vec2(0.5 / 16.0, 15.5 / 16.0)).rgb;
    vec3 sky_fog_color = sky_day_color * light_intensity;
    vec3 final_rgb = mix(lit_color, sky_fog_color, fog_factor);

    out_color = vec4(final_rgb, v_color.a);
}
