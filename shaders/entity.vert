#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct EntityVertexGpu {
    vec3 pos;
    uint light; // packed: (sky & 0xF) | ((block & 0xF) << 4) | ((normal_idx & 0x7) << 8)
    vec2 uv;
    uint layer;
    float hurt_tint;
};

layout(buffer_reference, scalar) readonly buffer EntityVertexBuffer {
    EntityVertexGpu vertices[];
};

layout(push_constant, std430) uniform EntityPushConstants {
    mat4 view_proj;
    uint64_t vertex_buffer_address;
    uint _pad[2];
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out uint v_layer;
layout(location = 2) out vec2 v_light; // x: block (0..15), y: sky (0..15)
layout(location = 3) out float v_face_shade;
layout(location = 4) out float v_hurt_tint;

// Directional face shade lookup for normal_idx (0..5: -Y, +Y, -Z, +Z, -X, +X)
const float FACE_SHADES[6] = float[](0.5, 1.0, 0.85, 0.85, 0.75, 0.75);

void main() {
    EntityVertexBuffer buf = EntityVertexBuffer(pc.vertex_buffer_address);
    EntityVertexGpu v = buf.vertices[gl_VertexIndex];

    gl_Position = pc.view_proj * vec4(v.pos, 1.0);

    v_uv = v.uv;
    v_layer = v.layer;

    uint sky = v.light & 0xFu;
    uint block = (v.light >> 4u) & 0xFu;
    uint norm_idx = (v.light >> 8u) & 0x7u;

    v_light = vec2(float(block), float(sky));
    v_face_shade = norm_idx < 6u ? FACE_SHADES[norm_idx] : 0.85;
    v_hurt_tint = v.hurt_tint;
}
