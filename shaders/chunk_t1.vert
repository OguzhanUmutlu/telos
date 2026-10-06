#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct PackedT1Quad {
    uint word0;
    uint word1;
    uint word2;
    uint word3;
};

struct PackedPattern {
    uint word0;
    uint word1;
};

struct TerrainDrawInfo {
    uint64_t quad_buffer_address;
    int chunk_x;
    int chunk_y;
    int chunk_z;
    uint pattern_offset;
    uint _pad;
};

layout(buffer_reference, scalar) readonly buffer T1QuadBuffer {
    PackedT1Quad quads[];
};

layout(buffer_reference, scalar) readonly buffer PatternBuffer {
    PackedPattern patterns[];
};

layout(buffer_reference, std430) readonly buffer DrawInfoBuffer {
    TerrainDrawInfo draws[];
};

layout(push_constant, std430) uniform PushConstants {
    mat4 view_proj;
    uint64_t draw_info_buffer_address;
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world_pos;
layout(location = 2) flat out uint v_material;
layout(location = 3) flat out uint v_dir;
layout(location = 4) out vec2 v_uv;
layout(location = 5) out vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)

const uint CORNER_INDICES[6] = uint[](0, 1, 2, 2, 3, 0);

const vec3 NORMALS[6] = vec3[](
    vec3(1.0, 0.0, 0.0),   // +X
    vec3(-1.0, 0.0, 0.0),  // -X
    vec3(0.0, 1.0, 0.0),   // +Y
    vec3(0.0, -1.0, 0.0),  // -Y
    vec3(0.0, 0.0, 1.0),   // +Z
    vec3(0.0, 0.0, -1.0)   // -Z
);

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

void main() {
    uint quad_idx = gl_VertexIndex / 6;
    uint vert_sub_idx = gl_VertexIndex % 6;
    uint corner = CORNER_INDICES[vert_sub_idx];

    TerrainDrawInfo draw_info = DrawInfoBuffer(pc.draw_info_buffer_address).draws[gl_InstanceIndex];

    T1QuadBuffer quad_buffer = T1QuadBuffer(draw_info.quad_buffer_address);
    PackedT1Quad quad = quad_buffer.quads[quad_idx];

    // Decode Word 0
    uint x = quad.word0 & 0x3FFu;
    uint y = (quad.word0 >> 10) & 0x3FFu;
    uint z = (quad.word0 >> 20) & 0x3FFu;

    // Decode Word 1
    uint w = (quad.word1 & 0x1FFu) + 1u;
    uint h = ((quad.word1 >> 9) & 0x1FFu) + 1u;
    uint dir = (quad.word1 >> 18) & 0x7u;

    // Decode Word 2
    uint material = (quad.word2 >> 16) & 0xFFFFu;

    // Decode Word 3
    uint pattern_idx = quad.word3 & 0x3FFFu;

    PatternBuffer pattern_buffer = PatternBuffer(draw_info.quad_buffer_address + uint64_t(draw_info.pattern_offset * 16));
    PackedPattern pattern = pattern_buffer.patterns[pattern_idx];
    uint sky = (pattern.word0 >> (corner * 4u)) & 0xFu;
    uint block = (pattern.word0 >> (16u + corner * 4u)) & 0xFu;
    uint ao = (pattern.word1 >> (corner * 2u)) & 0x3u;

    vec2 uv_offsets[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(float(w), 0.0),
        vec2(float(w), float(h)),
        vec2(0.0, float(h))
    );

    vec2 corner_uv = uv_offsets[corner];

    vec3 local_pos = vec3(float(x), float(y), float(z))
        + U_DIRS[dir] * corner_uv.x
        + V_DIRS[dir] * corner_uv.y;

    vec3 chunk_pos = vec3(draw_info.chunk_x, draw_info.chunk_y, draw_info.chunk_z);
    vec3 world_pos = chunk_pos + local_pos / 16.0;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    v_normal = NORMALS[dir];
    v_world_pos = world_pos;
    v_material = material;
    v_dir = dir;
    v_uv = corner_uv / 16.0;
    v_light = vec3(float(ao), float(sky), float(block));
}
