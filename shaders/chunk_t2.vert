#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

struct PackedT2Vertex {
    uint word0; // pos.x (16b) | pos.y (16b)
    uint word1; // pos.z (16b) | uv.u (16b)
    uint word2; // uv.v (16b)  | material (16b)
    uint word3; // light (16b) | normal (16b)
};

layout(buffer_reference, scalar) readonly buffer T2VertexBuffer {
    PackedT2Vertex vertices[];
};

layout(push_constant, std430) uniform PushConstants {
    mat4 view_proj;                   // 64 bytes (0..64)
    uint64_t quad_buffer_address;     // 8 bytes (64..72)
    int chunk_x;                      // 4 bytes (72..76)
    int chunk_y;                      // 4 bytes (76..80)
    vec4 camera_pos;                  // 16 bytes (80..96, xyz = camera pos, w = sim_dist_meters)
    int chunk_z;                      // 4 bytes (96..100)
    uint frame_tick_flags;            // 4 bytes (100..104, bit 31 = is_translucent, bits 0..30 = frame_tick)
    uint water_base_layer;            // 4 bytes (104..108)
    uint water_frame_count;           // 4 bytes (108..112)
    uint lava_base_layer;             // 4 bytes (112..116)
    uint lava_frame_count;            // 4 bytes (116..120)
    uint fire_base_layer;             // 4 bytes (120..124)
    uint fire_frame_count;            // 4 bytes (124..128)
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world_pos;
layout(location = 2) flat out uint v_material;
layout(location = 3) flat out uint v_is_emissive;
layout(location = 4) out vec2 v_uv;
layout(location = 5) out vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)
layout(location = 6) flat out uint v_has_shading;

const uint CORNER_INDICES[6] = uint[](0, 1, 2, 2, 3, 0);

vec3 decode_octahedral_normal(int nx, int nz) {
    float x = clamp(float(nx) / 127.0, -1.0, 1.0);
    float z = clamp(float(nz) / 127.0, -1.0, 1.0);
    float y = 1.0 - abs(x) - abs(z);
    vec3 n = vec3(x, y, z);
    if (n.y < 0.0) {
        float ox = (1.0 - abs(n.z)) * ((n.x >= 0.0) ? 1.0 : -1.0);
        float oz = (1.0 - abs(n.x)) * ((n.z >= 0.0) ? 1.0 : -1.0);
        n.x = ox;
        n.z = oz;
    }
    return normalize(n);
}

void main() {
    uint quad_idx = gl_VertexIndex / 6;
    uint vert_sub_idx = gl_VertexIndex % 6;
    uint corner = CORNER_INDICES[vert_sub_idx];
    uint vert_idx = quad_idx * 4u + corner;

    T2VertexBuffer vertex_buffer = T2VertexBuffer(pc.quad_buffer_address);
    PackedT2Vertex v = vertex_buffer.vertices[vert_idx];

    // Decode position (in 1/1024th units within chunk)
    float px = float(v.word0 & 0xFFFFu) / 1024.0;
    float py = float((v.word0 >> 16) & 0xFFFFu) / 1024.0;
    float pz = float(v.word1 & 0xFFFFu) / 1024.0;

    // Decode UVs
    float u = float((v.word1 >> 16) & 0xFFFFu) / 65535.0;
    float v_coord = float(v.word2 & 0xFFFFu) / 65535.0;

    // Material
    uint material = (v.word2 >> 16) & 0xFFFFu;

    // Light and flags
    uint light_word = v.word3 & 0xFFFFu;
    uint ao = light_word & 0x03u;
    uint sky = (light_word >> 2) & 0x0Fu;
    uint block = (light_word >> 6) & 0x0Fu;
    uint emissive = (light_word >> 10) & 0x01u;
    uint shade = (light_word >> 11) & 0x01u;

    // Normal
    int nx = int(int8_t((v.word3 >> 16) & 0xFFu));
    int nz = int(int8_t((v.word3 >> 24) & 0xFFu));
    vec3 normal = decode_octahedral_normal(nx, nz);

    vec3 chunk_pos = vec3(pc.chunk_x, pc.chunk_y, pc.chunk_z);
    vec3 world_pos = chunk_pos + vec3(px, py, pz);

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    v_normal = normal;
    v_world_pos = world_pos;
    v_material = material;
    v_is_emissive = emissive;
    v_uv = vec2(u, v_coord);
    v_light = vec3(float(ao), float(sky), float(block));
    v_has_shading = shade;
}
