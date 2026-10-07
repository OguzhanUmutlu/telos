#version 450 core

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 uvs[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 uv = uvs[gl_VertexID];
    v_uv = uv * 0.5 + 0.5;
    gl_Position = vec4(uv, 0.9999, 1.0);
}
