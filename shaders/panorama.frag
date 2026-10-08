#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_panorama;

layout(push_constant) uniform PanoramaPushConstants {
    mat4 inv_view_proj;
} pc;

void main() {
    // Reconstruct world ray direction from NDC coordinates
    vec4 ndc = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 world_h = pc.inv_view_proj * ndc;
    vec3 dir = length(world_h.xyz) > 1e-6 ? normalize(world_h.xyz) : vec3(0.0, 0.0, -1.0);

    // Minecraft Panorama Cubemap mapping:
    // Face 0: Back (+Z)
    // Face 1: Left (-X)
    // Face 2: Front (-Z)
    // Face 3: Right (+X)
    // Face 4: Up (+Y)
    // Face 5: Down (-Y)

    vec3 abs_dir = abs(dir);
    float max_coord = max(abs_dir.x, max(abs_dir.y, abs_dir.z));

    uint layer = 0u;
    vec2 st = vec2(0.0);

    if (max_coord == abs_dir.z) {
        if (dir.z > 0.0) {
            // Face 0: Back (+Z)
            layer = 0u;
            st = vec2(-dir.x / dir.z, dir.y / dir.z);
        } else {
            // Face 2: Front (-Z)
            layer = 2u;
            st = vec2(dir.x / -dir.z, dir.y / -dir.z);
        }
    } else if (max_coord == abs_dir.x) {
        if (dir.x > 0.0) {
            // Face 3: Right (+X)
            layer = 3u;
            st = vec2(dir.z / dir.x, dir.y / dir.x);
        } else {
            // Face 1: Left (-X)
            layer = 1u;
            st = vec2(dir.z / dir.x, dir.y / -dir.x);
        }
    } else {
        if (dir.y > 0.0) {
            // Face 4: Up (+Y)
            layer = 4u;
            st = vec2(dir.x / dir.y, -dir.z / dir.y);
        } else {
            // Face 5: Down (-Y)
            layer = 5u;
            st = vec2(dir.x / -dir.y, dir.z / dir.y);
        }
    }

    vec2 uv = vec2(0.5 + 0.5 * st.x, 0.5 - 0.5 * st.y);
    out_color = texture(u_panorama, vec3(uv, float(layer)));
}
