#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct UiQuadGpu {
    ivec2 pos;
    uint size;       // w | (h << 16)
    uint kind_flags; // kind | (flags << 16)
    uint uv0;        // u0 | (v0 << 16)
    uint uv1;        // u1 | (v1 << 16)
    uint layer;
    uint tex_idx;
    uint color;      // RGBA8
    uint clip;
    uint param0;
    uint param1;
};

layout(buffer_reference, scalar) readonly buffer UiQuadBuffer {
    UiQuadGpu quads[];
};

layout(push_constant, std430) uniform UiPushConstants {
    vec2 viewport_size; // width, height in physical pixels
    uint64_t quad_buffer_address;
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) flat out uint v_kind;
layout(location = 3) flat out uint v_layer;
layout(location = 4) out vec2 v_local_pos;
layout(location = 5) flat out vec2 v_quad_size;
layout(location = 6) flat out uint v_param0;
layout(location = 7) flat out uint v_param1;

const uint CORNERS[6] = uint[](0, 1, 2, 2, 1, 3);
const vec2 CORNER_POS[4] = vec2[](
    vec2(0.0, 0.0), // 0: Top-Left
    vec2(1.0, 0.0), // 1: Top-Right
    vec2(0.0, 1.0), // 2: Bottom-Left
    vec2(1.0, 1.0)  // 3: Bottom-Right
);

void main() {
    uint corner_idx = CORNERS[gl_VertexIndex];
    vec2 corner = CORNER_POS[corner_idx];

    UiQuadBuffer quad_buf = UiQuadBuffer(pc.quad_buffer_address);
    UiQuadGpu quad = quad_buf.quads[gl_InstanceIndex];

    float w = float(quad.size & 0xFFFFu);
    float h = float((quad.size >> 16u) & 0xFFFFu);
    vec2 quad_size = vec2(w, h);

    vec2 local_pos = corner * quad_size;
    vec2 physical_pos = vec2(quad.pos) + local_pos;

    // Convert physical pixels [0..W, 0..H] to Vulkan NDC [-1..1, -1..1]
    vec2 ndc = (physical_pos / pc.viewport_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    // Unpack normalized UV coordinates (unorm16)
    float u0 = float(quad.uv0 & 0xFFFFu) / 65535.0;
    float v0 = float((quad.uv0 >> 16u) & 0xFFFFu) / 65535.0;
    float u1 = float(quad.uv1 & 0xFFFFu) / 65535.0;
    float v1 = float((quad.uv1 >> 16u) & 0xFFFFu) / 65535.0;
    v_uv = mix(vec2(u0, v0), vec2(u1, v1), corner);

    // Unpack RGBA8 color
    uint r = quad.color & 0xFFu;
    uint g = (quad.color >> 8u) & 0xFFu;
    uint b = (quad.color >> 16u) & 0xFFu;
    uint a = (quad.color >> 24u) & 0xFFu;
    v_color = vec4(float(r), float(g), float(b), float(a)) / 255.0;

    v_kind = quad.kind_flags & 0xFFFFu;
    v_layer = quad.layer;
    v_local_pos = local_pos;
    v_quad_size = quad_size;
    v_param0 = quad.param0;
    v_param1 = quad.param1;
}
