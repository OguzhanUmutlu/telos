#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 min_bound;
    vec4 max_bound;
    vec4 color;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = pc.color;
}
