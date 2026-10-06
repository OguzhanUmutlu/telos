#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct ParticleGpu {
    vec3 pos;
    float size;
    vec2 uv_min;
    vec2 uv_max;
    uint color;
    uint layer;
    uint tex_source;
    uint flags;
};

layout(buffer_reference, scalar) readonly buffer ParticleBuffer {
    ParticleGpu particles[];
};

layout(push_constant, std430) uniform ParticlePushConstants {
    mat4 view_proj;
    vec3 camera_right;
    float _pad0;
    vec3 camera_up;
    float _pad1;
    uint64_t particle_buffer_address;
    uint _pad2[2];
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) flat out uint v_layer;
layout(location = 3) flat out uint v_tex_source;

const uint CORNERS[6] = uint[](0, 1, 2, 2, 1, 3);
const vec2 CORNER_UV[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0)
);

void main() {
    uint corner_idx = CORNERS[gl_VertexIndex];
    vec2 corner = CORNER_UV[corner_idx];

    ParticleBuffer buf = ParticleBuffer(pc.particle_buffer_address);
    ParticleGpu p = buf.particles[gl_InstanceIndex];

    // Spherical camera-facing billboard quad offset in view plane
    vec3 right_offset = pc.camera_right * ((corner.x - 0.5) * p.size);
    vec3 up_offset = pc.camera_up * ((0.5 - corner.y) * p.size);

    vec3 world_pos = p.pos + right_offset + up_offset;
    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    // Map unit quad corner into [uv_min, uv_max] sub-rectangle
    v_uv = mix(p.uv_min, p.uv_max, corner);
    v_layer = p.layer;
    v_tex_source = p.tex_source;

    uint r = p.color & 0xFFu;
    uint g = (p.color >> 8u) & 0xFFu;
    uint b = (p.color >> 16u) & 0xFFu;
    uint a = (p.color >> 24u) & 0xFFu;
    v_color = vec4(float(r), float(g), float(b), float(a)) / 255.0;
}
