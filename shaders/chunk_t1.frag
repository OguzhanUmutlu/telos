#version 460

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) flat in uint v_material;
layout(location = 3) flat in uint v_dir;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_light; // x: ao (0..3), y: sky (0..15), z: block (0..15)

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray u_textures;
layout(set = 0, binding = 1) uniform sampler2D u_lightmap;

uint get_texture_layer(uint mat) {
    switch (mat) {
        case 10u: // Stone Slab
            return 0u; // stone
        case 11u: // Oak Stairs
            return 6u; // oak_planks
        case 19u: // logic_wire (unpowered)
        case 20u: // logic_wire_powered
            return 20u; // redstone_dust_line0
        case 22u: // logic_lever
        case 23u: // logic_lever_on
            return 23u; // lever
        case 26u: // logic_repeater
        case 30u: // logic_diode
            return 21u; // repeater
        case 27u: // logic_repeater_powered
            return 22u; // repeater_on
        case 28u: // logic_inverter
            return 15u; // redstone_torch
        case 29u: // logic_inverter_off
            return 16u; // redstone_torch_off
        default:
            return 0u;
    }
}

void main() {
    uint layer = get_texture_layer(v_material);

    // Sample repeating 2D texture array slice
    vec4 tex_color = texture(u_textures, vec3(v_uv, float(layer)));

    // Discard transparent texels for cutout sub-cube geometry (wires, levers, torches)
    if (tex_color.a < 0.1) {
        discard;
    }

    // Directional face shading factor
    float face_shade = 0.75;
    if (abs(v_normal.y) > 0.5) {
        face_shade = (v_normal.y > 0.0) ? 1.0 : 0.5;
    } else if (abs(v_normal.z) > 0.5) {
        face_shade = 0.85;
    } else {
        face_shade = 0.75;
    }

    // Unpack smooth interpolated light components
    float ao_raw = v_light.x;    // 0.0 .. 3.0
    float sky_raw = v_light.y;   // 0.0 .. 15.0
    float block_raw = v_light.z; // 0.0 .. 15.0

    // Smooth ambient occlusion factor
    float ao_factor = mix(0.35, 1.0, ao_raw / 3.0);

    // Sample 16x16 lightmap LUT with bilinear filtering
    vec2 lightmap_uv = clamp(vec2(block_raw + 0.5, sky_raw + 0.5) / 16.0, 0.0, 1.0);
    vec3 sampled_light = texture(u_lightmap, lightmap_uv).rgb;

    // Daylight ambient floor: prevents outdoor side faces from turning pitch black
    vec3 sky_day_color = texture(u_lightmap, vec2(0.5 / 16.0, 15.5 / 16.0)).rgb;
    vec3 daylight_floor = sky_day_color * 0.20;
    float sky_exposure = clamp(sky_raw / 2.0, 0.0, 1.0);
    vec3 light_color = mix(sampled_light, max(sampled_light, daylight_floor), sky_exposure);

    vec3 total_light = clamp(light_color * face_shade * ao_factor, 0.0, 1.0);

    // Tinting logic components (Phase 36)
    if (v_material == 19u) {
        // Unpowered wire: dark crimson red
        tex_color.rgb *= vec3(0.35, 0.05, 0.05);
    } else if (v_material == 20u) {
        // Powered wire: bright glowing red with emissive pop
        tex_color.rgb *= vec3(1.0, 0.15, 0.15);
        total_light = max(total_light, vec3(0.9, 0.2, 0.2));
    } else if (v_material == 27u || v_material == 28u) {
        // Powered repeater / active inverter torch glow
        total_light = max(total_light, vec3(0.85, 0.45, 0.2));
    }

    out_color = vec4(tex_color.rgb * total_light, 1.0);
}
