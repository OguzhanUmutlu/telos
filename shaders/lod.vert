#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct LodDrawInfo {
    uint64_t buffer_address;
    int node_x;
    int node_y;
    int node_z;
    uint level;
    uint quad_count;
    uint _pad0;
    uint _pad1;
};

layout(buffer_reference, scalar) readonly buffer LodBuffer {
    uint words[];
};

layout(buffer_reference, std430) readonly buffer LodDrawInfoBuffer {
    LodDrawInfo draws[];
};

layout(push_constant, std430) uniform LodPushConstants {
    mat4 view_proj;
    vec3 camera_pos;
    float max_distance;
    uint64_t draw_info_buffer_address;
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)
layout(location = 3) out float v_distance;

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
    uint corner = CORNER_INDICES[vert_sub_idx];

    LodDrawInfo draw_info = LodDrawInfoBuffer(pc.draw_info_buffer_address).draws[gl_InstanceIndex];

    LodBuffer lod_buffer = LodBuffer(draw_info.buffer_address);
    uint word0 = lod_buffer.words[quad_idx * 2u];
    uint word1 = lod_buffer.words[quad_idx * 2u + 1u];

    // Decode Word 0
    uint x = word0 & 0x1Fu;
    uint y = (word0 >> 5u) & 0x1Fu;
    uint z = (word0 >> 10u) & 0x1Fu;
    uint w = ((word0 >> 15u) & 0x1Fu) + 1u;
    uint h = ((word0 >> 20u) & 0x1Fu) + 1u;
    uint dir = (word0 >> 25u) & 0x7u;
    uint sky = (word0 >> 28u) & 0xFu;

    // Decode Word 1
    uint color_idx = word1 & 0xFFFFu;
    uint ao = (word1 >> 16u) & 0x3u;
    uint block = (word1 >> 18u) & 0xFu;

    // Lookup color in trailing palette
    uint palette_offset = draw_info.quad_count * 2u;
    uint rgba = lod_buffer.words[palette_offset + color_idx];
    vec4 base_color = unpackUnorm4x8(rgba);

    vec2 uv_offsets[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(float(w), 0.0),
        vec2(float(w), float(h)),
        vec2(0.0, float(h))
    );

    vec2 corner_uv = uv_offsets[corner];

    float voxel_size = float(1u << draw_info.level);
    vec3 base_pos = (vec3(float(x), float(y), float(z)) + PLANE_OFFSETS[dir]) * voxel_size;
    vec3 local_pos = base_pos + (U_DIRS[dir] * corner_uv.x + V_DIRS[dir] * corner_uv.y) * voxel_size;

    // Node origin in world coordinates: node_pos << (5 + level)
    int shift = 5 + int(draw_info.level);
    ivec3 node_pos = ivec3(draw_info.node_x, draw_info.node_y, draw_info.node_z);
    vec3 node_origin = vec3(node_pos << shift);
    vec3 world_pos = local_pos + node_origin;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    v_normal = NORMALS[dir];
    v_color = base_color;
    v_light = vec3(float(ao), float(sky), float(block));
    v_distance = distance(world_pos, pc.camera_pos);
}
