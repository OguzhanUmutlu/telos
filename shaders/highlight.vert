#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 min_bound;
    vec4 max_bound;
    vec4 color;
} pc;

const vec3 box_verts[24] = vec3[24](
    // Bottom square
    vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0),

    // Top square
    vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0),
    vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0),

    // Vertical pillars
    vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0)
);

void main() {
    vec3 corner = box_verts[gl_VertexIndex];
    vec3 pos = mix(pc.min_bound.xyz, pc.max_bound.xyz, corner);
    gl_Position = pc.view_proj * vec4(pos, 1.0);
}
