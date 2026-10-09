#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct PackedQuad {
    uint word0;
    uint word1;
};

layout(buffer_reference, scalar) readonly buffer QuadBuffer {
    PackedQuad quads[];
};

layout(push_constant, std430) uniform ShadowPushConstants {
    mat4 u_light_view_proj;
    uint64_t u_vertex_addr;
    int u_chunk_x;
    int u_chunk_y;
    int u_chunk_z;
    uint _pad;
} pc;

const uint CORNER_INDICES[6] = uint[](0, 1, 2, 2, 3, 0);

const vec3 U_DIRS[6] = vec3[](
    vec3(0.0, 1.0, 0.0),   // +X: u = +Y
    vec3(0.0, 0.0, 1.0),   // -X: u = +Z
    vec3(0.0, 0.0, 1.0),   // +Y: u = +Z
    vec3(1.0, 0.0, 0.0),   // -Y: u = +X
    vec3(1.0, 0.0, 0.0),   // +Z: u = +X
    vec3(0.0, 1.0, 0.0)    // -Z: u = +Y
);

const vec3 V_DIRS[6] = vec3[](
    vec3(0.0, 0.0, 1.0),   // +X: v = +Z
    vec3(0.0, 1.0, 0.0),   // -X: v = +Y
    vec3(1.0, 0.0, 0.0),   // +Y: v = +X
    vec3(0.0, 0.0, 1.0),   // -Y: v = +Z
    vec3(0.0, 1.0, 0.0),   // +Z: v = +Y
    vec3(1.0, 0.0, 0.0)    // -Z: v = +X
);

const vec3 PLANE_OFFSETS[6] = vec3[](
    vec3(1.0, 0.0, 0.0),   // +X: offset +1
    vec3(0.0, 0.0, 0.0),   // -X: offset 0
    vec3(0.0, 1.0, 0.0),   // +Y: offset +1
    vec3(0.0, 0.0, 0.0),   // -Y: offset 0
    vec3(0.0, 0.0, 1.0),   // +Z: offset +1
    vec3(0.0, 0.0, 0.0)    // -Z: offset 0
);

void main() {
    uint quad_idx = gl_VertexIndex / 6;
    uint vert_sub_idx = gl_VertexIndex % 6;

    QuadBuffer quad_buffer = QuadBuffer(pc.u_vertex_addr);
    PackedQuad quad = quad_buffer.quads[quad_idx];

    // Decode Word 0
    uint x = quad.word0 & 0x1Fu;
    uint y = (quad.word0 >> 5) & 0x1Fu;
    uint z = (quad.word0 >> 10) & 0x1Fu;
    uint w = ((quad.word0 >> 15) & 0x1Fu) + 1u;
    uint h = ((quad.word0 >> 20) & 0x1Fu) + 1u;
    uint dir = (quad.word0 >> 25) & 0x7u;

    uint corner = CORNER_INDICES[vert_sub_idx];

    vec2 uv_offsets[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(float(w), 0.0),
        vec2(float(w), float(h)),
        vec2(0.0, float(h))
    );

    vec2 corner_uv = uv_offsets[corner];

    vec3 base_pos = vec3(float(x), float(y), float(z)) + PLANE_OFFSETS[dir];
    vec3 local_pos = base_pos + U_DIRS[dir] * corner_uv.x + V_DIRS[dir] * corner_uv.y;
    vec3 chunk_pos = vec3(float(pc.u_chunk_x), float(pc.u_chunk_y), float(pc.u_chunk_z));
    vec3 world_pos = local_pos + chunk_pos;

    gl_Position = pc.u_light_view_proj * vec4(world_pos, 1.0);
}
