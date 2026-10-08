//! Integration tests for `telos-ui`.

use telos_ui::{
    BitmapFont, HudState, QuadKind, UiFrame, UiLayers, UiQuad, UiTree, WidgetKind,
    compute_gui_scale, render_hud, snap_to_physical, to_physical_pixels,
};

#[test]
fn test_ui_quad_layout_and_size() {
    assert_eq!(std::mem::size_of::<UiQuad>(), 48);
    assert_eq!(std::mem::align_of::<UiQuad>(), 4);

    let quad = UiQuad::solid([10, 20], [100, 200], UiQuad::rgba(255, 128, 64, 255));
    assert_eq!(quad.pos, [10, 20]);
    assert_eq!(quad.size_kind[0], UiQuad::pack_size(100, 200));
    assert_eq!(quad.size_kind[1], UiQuad::pack_kind(QuadKind::Solid, 0));
    assert_eq!(quad.color, UiQuad::rgba(255, 128, 64, 255));
}

#[test]
fn test_integer_gui_scale_calculation() {
    assert_eq!(compute_gui_scale(1920, 1080), 4);
    assert_eq!(compute_gui_scale(1280, 720), 3);
    assert_eq!(compute_gui_scale(854, 480), 2);
    assert_eq!(compute_gui_scale(640, 480), 2);
    assert_eq!(compute_gui_scale(320, 240), 1);
    assert_eq!(compute_gui_scale(100, 100), 1);
    assert_eq!(compute_gui_scale(3840, 2160), 9);

    assert_eq!(to_physical_pixels(10, 4), 40);
    assert_eq!(snap_to_physical(10.25, 4), 41);
}

#[test]
fn test_bitmap_font_glyph_advances() {
    let font = BitmapFont::new_fallback(0);

    let adv_i = font.glyph(b'i').advance;
    let adv_w = font.glyph(b'W').advance;
    let adv_a = font.glyph(b'a').advance;
    let adv_space = font.glyph(b' ').advance;

    assert!(adv_i < adv_a);
    assert!(adv_a < adv_w);
    assert!((adv_space - 4.0).abs() < f32::EPSILON);

    let (measured_w, measured_h) = font.measure_text("Hello");
    assert!(measured_w > 0.0);
    assert!((measured_h - 9.0).abs() < f32::EPSILON);
}

#[test]
fn test_retained_tree_layout_and_tessellation() {
    let mut tree = UiTree::new(4);
    let root = tree.root();

    let node = tree.add_child(
        root,
        WidgetKind::Solid {
            color: UiQuad::rgba(255, 0, 0, 255),
        },
        taffy::Style {
            size: taffy::Size {
                width: taffy::Dimension::length(50.0),
                height: taffy::Dimension::length(20.0),
            },
            ..Default::default()
        },
    );

    let mut frame = UiFrame::new([1920, 1080], 4);
    let font = BitmapFont::new_fallback(0);

    tree.update(1920, 1080, &font, &mut frame.quads);
    assert_eq!(frame.quad_count(), 1);
    assert_eq!(frame.quads[0].size_kind[0], UiQuad::pack_size(200, 80)); // 50*4, 20*4

    // Modifying visibility marks dirty
    tree.set_visible(node, false);
    assert!(tree.is_dirty());
    frame.clear();
    tree.update(1920, 1080, &font, &mut frame.quads);
    assert_eq!(frame.quad_count(), 0);
}

#[test]
fn test_render_hud_quad_generation() {
    let font = BitmapFont::new_fallback(3);
    let layers = UiLayers::default();
    let mut state = HudState {
        selected_slot: 3,
        f3_open: false,
        ..Default::default()
    };

    let mut quads = Vec::new();
    render_hud(&state, &font, &layers, 1920, 1080, 4, &mut quads);

    // 1 crosshair + 1 hotbar + 1 selection indicator + 1 xp bar bg + 20 hearts + 20 food drumsticks = 44 quads
    assert_eq!(quads.len(), 44);

    // Turn on F3 overlay
    state.f3_open = true;
    quads.clear();
    render_hud(&state, &font, &layers, 1920, 1080, 4, &mut quads);
    // Should have crosshair, hotbar, selection indicator, plus multiple lines with backing boxes
    assert!(quads.len() > 10);

    // Add active status effect badge
    state.f3_open = false;
    state.active_effects.push(telos_ui::HudEffectDisplay {
        effect_id: 1,
        name: "Speed II".to_string(),
        amplifier: 1,
        duration_ticks: 200,
        color: [124, 175, 198],
        ambient: false,
        is_beneficial: true,
    });
    quads.clear();
    render_hud(&state, &font, &layers, 1920, 1080, 4, &mut quads);
    // Base 44 quads + badge background + accent bar + text glyphs
    assert!(quads.len() > 44, "Active effects must generate HUD quads");

    // Add harmful + ambient effect
    state.active_effects.push(telos_ui::HudEffectDisplay {
        effect_id: 7,
        name: "Wither".to_string(),
        amplifier: 0,
        duration_ticks: 100,
        color: [53, 42, 39],
        ambient: true,
        is_beneficial: false,
    });
    let prev_count = quads.len();
    quads.clear();
    render_hud(&state, &font, &layers, 1920, 1080, 4, &mut quads);
    assert!(
        quads.len() > prev_count,
        "Multiple active effects including ambient line generate extra quads"
    );
}
