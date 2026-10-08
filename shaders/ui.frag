#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in uint v_kind;
layout(location = 3) flat in uint v_layer;
layout(location = 4) in vec2 v_local_pos;
layout(location = 5) flat in vec2 v_quad_size;
layout(location = 6) flat in uint v_param0;
layout(location = 7) flat in uint v_param1;
layout(location = 8) flat in vec4 v_uv_bounds;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_ui_textures;

vec2 compute_nine_slice_uv(vec2 local_pos, vec2 quad_size, uint borders, uint src_size, vec4 uv_bounds) {
    float b_left = float(borders & 0xFFu);
    float b_top = float((borders >> 8u) & 0xFFu);
    float b_right = float((borders >> 16u) & 0xFFu);
    float b_bottom = float((borders >> 24u) & 0xFFu);

    float src_w = float(src_size & 0xFFFFu);
    float src_h = float((src_size >> 16u) & 0xFFFFu);

    if (src_w <= 0.0 || src_h <= 0.0) {
        return mix(uv_bounds.xy, uv_bounds.zw, local_pos / max(quad_size, vec2(1.0)));
    }

    // Piecewise mapping for X
    float x = local_pos.x;
    float norm_x = 0.0;
    if (x < b_left) {
        norm_x = x / src_w;
    } else if (x > quad_size.x - b_right) {
        norm_x = (src_w - b_right + (x - (quad_size.x - b_right))) / src_w;
    } else {
        float mid_w = max(quad_size.x - b_left - b_right, 1.0);
        float src_mid_w = max(src_w - b_left - b_right, 1.0);
        norm_x = (b_left + (x - b_left) * (src_mid_w / mid_w)) / src_w;
    }

    // Piecewise mapping for Y
    float y = local_pos.y;
    float norm_y = 0.0;
    if (y < b_top) {
        norm_y = y / src_h;
    } else if (y > quad_size.y - b_bottom) {
        norm_y = (src_h - b_bottom + (y - (quad_size.y - b_bottom))) / src_h;
    } else {
        float mid_h = max(quad_size.y - b_top - b_bottom, 1.0);
        float src_mid_h = max(src_h - b_top - b_bottom, 1.0);
        norm_y = (b_top + (y - b_top) * (src_mid_h / mid_h)) / src_h;
    }

    return mix(uv_bounds.xy, uv_bounds.zw, vec2(clamp(norm_x, 0.0, 1.0), clamp(norm_y, 0.0, 1.0)));
}

void main() {
    switch (v_kind) {
        case 0u: { // Solid
            out_color = v_color;
            break;
        }
        case 1u: { // Sprite
            vec4 tex = texture(u_ui_textures, vec3(v_uv, float(v_layer)));
            out_color = tex * v_color;
            break;
        }
        case 2u: { // GlyphBitmap
            vec4 tex = texture(u_ui_textures, vec3(v_uv, float(v_layer)));
            if (tex.a < 0.1) {
                discard;
            }
            out_color = vec4(v_color.rgb, tex.a * v_color.a);
            break;
        }
        case 3u: { // NineSlice
            vec2 slice_uv = compute_nine_slice_uv(v_local_pos, v_quad_size, v_param0, v_param1, v_uv_bounds);
            vec4 tex = texture(u_ui_textures, vec3(slice_uv, float(v_layer)));
            out_color = tex * v_color;
            break;
        }
        case 4u: { // Crosshair
            vec4 tex = texture(u_ui_textures, vec3(v_uv, float(v_layer)));
            if (tex.a < 0.05) {
                discard;
            }
            out_color = tex * v_color;
            break;
        }
        default: {
            out_color = v_color;
            break;
        }
    }

    if (out_color.a <= 0.001) {
        discard;
    }
}
