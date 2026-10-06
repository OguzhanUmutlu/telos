#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in uint v_layer;
layout(location = 2) in vec2 v_light; // x: block, y: sky
layout(location = 3) in float v_face_shade;
layout(location = 4) in float v_hurt_tint;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_entity_textures;
layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

void main() {
    vec4 tex_color = texture(u_entity_textures, vec3(v_uv, float(v_layer)));
    if (tex_color.a < 0.1) {
        discard;
    }

    vec2 lightmap_uv = clamp(vec2(v_light.x + 0.5, v_light.y + 0.5) / 16.0, 0.0, 1.0);
    vec3 light_color = texture(u_lightmap, lightmap_uv).rgb;

    vec3 lit_color = tex_color.rgb * light_color * v_face_shade;
    vec3 final_color = mix(lit_color, vec3(1.0, 0.2, 0.2), v_hurt_tint);

    out_color = vec4(final_color, tex_color.a);
}
