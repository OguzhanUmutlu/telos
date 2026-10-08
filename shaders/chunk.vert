#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct PackedQuad {
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

layout(buffer_reference, scalar) readonly buffer QuadBuffer {
    PackedQuad quads[];
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
const uint CORNER_INDICES_FLIPPED[6] = uint[](1, 2, 3, 3, 0, 1);

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

    TerrainDrawInfo draw_info = DrawInfoBuffer(pc.draw_info_buffer_address).draws[gl_InstanceIndex];

    QuadBuffer quad_buffer = QuadBuffer(draw_info.quad_buffer_address);
    PackedQuad quad = quad_buffer.quads[quad_idx];

    // Decode Word 0
    uint x = quad.word0 & 0x1Fu;
    uint y = (quad.word0 >> 5) & 0x1Fu;
    uint z = (quad.word0 >> 10) & 0x1Fu;
    uint w = ((quad.word0 >> 15) & 0x1Fu) + 1u;
    uint h = ((quad.word0 >> 20) & 0x1Fu) + 1u;
    uint dir = (quad.word0 >> 25) & 0x7u;

    // Decode Word 1
    uint material = quad.word1 & 0xFFFFu;
    uint pattern_idx = (quad.word1 >> 16) & 0x3FFFu;

    // Unpack LightPattern from table
    PackedQuad pattern = quad_buffer.quads[draw_info.pattern_offset + pattern_idx];
    uint ao0 = pattern.word1 & 0x3u;
    uint ao1 = (pattern.word1 >> 2u) & 0x3u;
    uint ao2 = (pattern.word1 >> 4u) & 0x3u;
    uint ao3 = (pattern.word1 >> 6u) & 0x3u;

    uint sky0 = pattern.word0 & 0xFu;
    uint sky1 = (pattern.word0 >> 4u) & 0xFu;
    uint sky2 = (pattern.word0 >> 8u) & 0xFu;
    uint sky3 = (pattern.word0 >> 12u) & 0xFu;

    uint blk0 = (pattern.word0 >> 16u) & 0xFu;
    uint blk1 = (pattern.word0 >> 20u) & 0xFu;
    uint blk2 = (pattern.word0 >> 24u) & 0xFu;
    uint blk3 = (pattern.word0 >> 28u) & 0xFu;

    // Split quad along brighter diagonal to eliminate corner half-shadow pinch
    uint l02 = (ao0 + ao2) * 32u + (sky0 + sky2) + (blk0 + blk2);
    uint l13 = (ao1 + ao3) * 32u + (sky1 + sky3) + (blk1 + blk3);
    bool flip = l02 < l13;
    uint corner = flip ? CORNER_INDICES_FLIPPED[vert_sub_idx] : CORNER_INDICES[vert_sub_idx];

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

    vec3 base_pos = vec3(float(x), float(y), float(z)) + PLANE_OFFSETS[dir];
    vec3 local_pos = base_pos + U_DIRS[dir] * corner_uv.x + V_DIRS[dir] * corner_uv.y;
    vec3 chunk_pos = vec3(draw_info.chunk_x, draw_info.chunk_y, draw_info.chunk_z);
    vec3 world_pos = local_pos + chunk_pos;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    vec2 out_uv;
    if (dir == 0u || dir == 5u) {
        out_uv = vec2(corner_uv.y, float(w) - corner_uv.x);
    } else if (dir == 1u || dir == 4u) {
        out_uv = vec2(corner_uv.x, float(h) - corner_uv.y);
    } else {
        out_uv = corner_uv;
    }

    v_normal = NORMALS[dir];
    v_world_pos = world_pos;
    v_material = material;
    v_dir = dir;
    v_uv = out_uv;
    v_light = vec3(float(ao), float(sky), float(block));
}
