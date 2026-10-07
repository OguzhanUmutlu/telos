#version 450 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_view_proj;
uniform vec3 u_min;
uniform vec3 u_max;

void main() {
    vec3 world_pos = mix(u_min, u_max, a_position);
    gl_Position = u_view_proj * vec4(world_pos, 1.0);
}
