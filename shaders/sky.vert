#version 460

layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    // NDC: x in [-1, 1], y in [-1, 1], reversed-Z far plane depth at z = 0.0
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
