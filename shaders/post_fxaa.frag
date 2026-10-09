#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_color_texture;

layout(push_constant, std430) uniform FxaaPushConstants {
    vec2 u_texel_size;          // (1.0 / width, 1.0 / height)
    uint u_flags;               // bit 0 = FXAA active (1 = on, 0 = passthrough)
    float u_subpix;             // Subpixel quality (default: 0.75)
    float u_edge_threshold;     // Edge detection threshold (default: 0.125)
    float u_edge_threshold_min; // Minimum edge detection threshold (default: 0.0312)
    float _pad0;
    float _pad1;
} pc;

// Perceptual luminance calculation (green-dominant luma)
float rgb_to_luma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

// Exploration step distances along the edge tangent
const float QUALITY_STEPS[10] = float[](1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 3.0, 4.0);

void main() {
    vec4 center_color = texture(u_color_texture, v_uv);

    // If FXAA is disabled or toggled off via flags, pass through immediately
    if ((pc.u_flags & 1u) == 0u) {
        out_color = center_color;
        return;
    }

    vec2 texel = pc.u_texel_size;

    // 1. Center and 4 direct orthogonal neighbors
    float luma_m = rgb_to_luma(center_color.rgb);
    float luma_n = rgb_to_luma(texture(u_color_texture, v_uv + vec2(0.0, -texel.y)).rgb);
    float luma_s = rgb_to_luma(texture(u_color_texture, v_uv + vec2(0.0,  texel.y)).rgb);
    float luma_w = rgb_to_luma(texture(u_color_texture, v_uv + vec2(-texel.x, 0.0)).rgb);
    float luma_e = rgb_to_luma(texture(u_color_texture, v_uv + vec2( texel.x, 0.0)).rgb);

    float luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)));
    float luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)));
    float luma_range = luma_max - luma_min;

    // Early exit if contrast is below edge detection threshold (flat surface or faint noise)
    float edge_threshold = max(pc.u_edge_threshold_min, luma_max * pc.u_edge_threshold);
    if (luma_range < edge_threshold) {
        out_color = center_color;
        return;
    }

    // 2. Sample 4 diagonal neighbors for 3x3 low-pass subpixel evaluation
    float luma_nw = rgb_to_luma(texture(u_color_texture, v_uv + vec2(-texel.x, -texel.y)).rgb);
    float luma_ne = rgb_to_luma(texture(u_color_texture, v_uv + vec2( texel.x, -texel.y)).rgb);
    float luma_sw = rgb_to_luma(texture(u_color_texture, v_uv + vec2(-texel.x,  texel.y)).rgb);
    float luma_se = rgb_to_luma(texture(u_color_texture, v_uv + vec2( texel.x,  texel.y)).rgb);

    // Compute subpixel blend factor using 3x3 box/tent filter
    float luma_l = (2.0 * (luma_n + luma_s + luma_w + luma_e) + (luma_nw + luma_ne + luma_sw + luma_se)) / 12.0;
    float subpix_a = clamp(abs(luma_l - luma_m) / max(luma_range, 0.0001), 0.0, 1.0);
    float subpix_b = (3.0 - 2.0 * subpix_a) * (subpix_a * subpix_a);
    float subpix_blend = subpix_b * subpix_b * pc.u_subpix;

    // 3. Edge direction detection (Horizontal vs Vertical gradient)
    float edge_h = abs((luma_nw - luma_w) + 2.0 * (luma_n - luma_m) + (luma_ne - luma_e)) +
                   abs((luma_sw - luma_w) + 2.0 * (luma_s - luma_m) + (luma_se - luma_e));
    float edge_v = abs((luma_nw - luma_n) + 2.0 * (luma_w - luma_m) + (luma_sw - luma_s)) +
                   abs((luma_ne - luma_n) + 2.0 * (luma_e - luma_m) + (luma_se - luma_s));
    bool is_horizontal = (edge_h >= edge_v);

    // 4. Edge orientation and step direction
    float luma_pos = is_horizontal ? luma_s : luma_e;
    float luma_neg = is_horizontal ? luma_n : luma_w;
    float grad_pos = abs(luma_pos - luma_m);
    float grad_neg = abs(luma_neg - luma_m);

    float step_length = is_horizontal ? texel.y : texel.x;
    float luma_local_avg;

    if (grad_pos >= grad_neg) {
        luma_local_avg = 0.5 * (luma_pos + luma_m);
    } else {
        step_length = -step_length;
        luma_local_avg = 0.5 * (luma_neg + luma_m);
    }

    // Offset sampling point by half a texel in the normal direction toward the edge
    vec2 current_uv = v_uv;
    if (is_horizontal) {
        current_uv.y += step_length * 0.5;
    } else {
        current_uv.x += step_length * 0.5;
    }

    // 5. Edge span exploration along tangent in positive and negative directions
    vec2 edge_tangent = is_horizontal ? vec2(texel.x, 0.0) : vec2(0.0, texel.y);
    vec2 uv_pos = current_uv + edge_tangent;
    vec2 uv_neg = current_uv - edge_tangent;

    float delta_luma_pos = rgb_to_luma(texture(u_color_texture, uv_pos).rgb) - luma_local_avg;
    float delta_luma_neg = rgb_to_luma(texture(u_color_texture, uv_neg).rgb) - luma_local_avg;

    float gradient_threshold = luma_range * 0.25;
    bool reached_pos = abs(delta_luma_pos) >= gradient_threshold;
    bool reached_neg = abs(delta_luma_neg) >= gradient_threshold;

    for (int i = 1; i < 10; ++i) {
        if (!reached_pos) {
            uv_pos += edge_tangent * QUALITY_STEPS[i];
            delta_luma_pos = rgb_to_luma(texture(u_color_texture, uv_pos).rgb) - luma_local_avg;
            reached_pos = abs(delta_luma_pos) >= gradient_threshold;
        }
        if (!reached_neg) {
            uv_neg -= edge_tangent * QUALITY_STEPS[i];
            delta_luma_neg = rgb_to_luma(texture(u_color_texture, uv_neg).rgb) - luma_local_avg;
            reached_neg = abs(delta_luma_neg) >= gradient_threshold;
        }
        if (reached_pos && reached_neg) {
            break;
        }
    }

    // 6. Compute distance to endpoints and calculate sub-pixel offset
    float dist_pos = is_horizontal ? (uv_pos.x - v_uv.x) : (uv_pos.y - v_uv.y);
    float dist_neg = is_horizontal ? (v_uv.x - uv_neg.x) : (v_uv.y - uv_neg.y);

    bool is_pos_closer = (dist_pos < dist_neg);
    float dist_min = min(dist_pos, dist_neg);
    float total_span = dist_pos + dist_neg;

    // Check if the gradient at the closer endpoint matches the expected polarity
    bool delta_closer = is_pos_closer ? (delta_luma_pos < 0.0) : (delta_luma_neg < 0.0);
    bool center_smaller = (luma_m < luma_local_avg);
    bool correct_polarity = (delta_closer != center_smaller);

    float pixel_offset = 0.0;
    if (correct_polarity && total_span > 0.0001) {
        pixel_offset = 0.5 - (dist_min / total_span);
    }

    float final_blend = max(pixel_offset, subpix_blend);

    // 7. Final bilinear sampling at computed offset
    vec2 final_uv = v_uv;
    if (is_horizontal) {
        final_uv.y += step_length * final_blend;
    } else {
        final_uv.x += step_length * final_blend;
    }

    out_color = texture(u_color_texture, final_uv);
}
