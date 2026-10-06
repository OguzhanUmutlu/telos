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

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    ivec3 chunk_pos;
    uint pattern_offset;
    uint64_t quad_buffer_address;
    uint frame_tick;
    uint water_base_layer;
    uint water_frame_count;
    uint _pad[3];
} pc;

const vec3 TORCH_COLOR = vec3(1.0, 0.82, 0.55);

void main() {
    uint frame = (pc.water_frame_count > 1u) ? ((pc.frame_tick / 3u) % pc.water_frame_count) : 0u;
    uint layer = pc.water_base_layer + frame;

    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Directional face shading factor
    float face_shade = 0.80;
    if (v_normal.y > 0.5) {
        face_shade = 1.0;
    } else if (v_normal.y < -0.5) {
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

    // Sky light with directional sunlight modulation
    float sky_norm = clamp(sky_raw / 15.0, 0.0, 1.0);
    vec3 sun_light = vec3(sky_norm * face_shade);

    // Block light (warm torch color with quadratic falloff)
    float block_norm = clamp(block_raw / 15.0, 0.0, 1.0);
    vec3 torch_light = TORCH_COLOR * (block_norm * block_norm * 0.85 + block_norm * 0.15);

    const float AMBIENT_FLOOR = 0.10;
    vec3 total_light = (max(vec3(AMBIENT_FLOOR), sun_light) + torch_light) * ao_factor;
    total_light = clamp(total_light, 0.0, 1.0);

    // Water blue tint (Classic Voxel plains water tint #3f76e4)
    vec3 water_tint = vec3(0.247, 0.463, 0.894);

    out_color = vec4(tex_color.rgb * water_tint * total_light, 0.72);
}
