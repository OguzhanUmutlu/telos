#version 450

// Hardcoded clip-space positions and vertex colors for debug triangle
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.2, 0.2), // Red
    vec3(0.2, 1.0, 0.2), // Green
    vec3(0.2, 0.4, 1.0)  // Blue
);

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
