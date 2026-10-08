//! Tests for `AdvancementsScreen` GUI, node layout, tab switching, and toast rendering.

use telos_ui::font::BitmapFont;
use telos_ui::hud::{HudState, ToastState, UiLayers, render_hud};
use telos_ui::menu::advancements_screen::{
    AdvancementsScreen, UiAdvancementCategory, UiAdvancementFrame, UiAdvancementNode,
};

#[test]
fn test_advancements_screen_tab_switch_and_hover() {
    let mut screen = AdvancementsScreen::new();
    let nodes = vec![
        UiAdvancementNode {
            id: "telos:story/root".to_string(),
            parent_id: None,
            category: UiAdvancementCategory::Story,
            title: "Telos".to_string(),
            description: "The heart and story of the world".to_string(),
            icon_item: 7, // Planks
            frame: UiAdvancementFrame::Task,
            x: 0.0,
            y: 0.0,
            completed: true,
        },
        UiAdvancementNode {
            id: "telos:adventure/root".to_string(),
            parent_id: None,
            category: UiAdvancementCategory::Adventure,
            title: "Adventure".to_string(),
            description: "Adventure, exploration and combat".to_string(),
            icon_item: 16, // Map / Compass / Item
            frame: UiAdvancementFrame::Task,
            x: 0.0,
            y: 0.0,
            completed: false,
        },
    ];
    screen.set_nodes(nodes);

    assert_eq!(screen.active_tab, UiAdvancementCategory::Story);

    // Switch tab to Adventure via mouse click at top of window
    // Window width = 252, height = 140, screen width_gui = 800, height_gui = 600
    // origin_x = (800 - 252) / 2 = 274, origin_y = (600 - 140) / 2 = 230
    // Adventure tab is at x in [origin_x + 69, origin_x + 69 + 72] = [343, 415]
    // y in [origin_y - 26, origin_y] = [204, 230]
    screen.handle_mouse_down(350.0, 215.0, 800.0, 600.0);
    assert_eq!(screen.active_tab, UiAdvancementCategory::Adventure);

    // Node is at canvas base + node.x * 32.
    // canvas_x = 274 + 9 = 283, canvas_y = 230 + 18 = 248, canvas_h = 113.
    // base_x = 283 + 24 = 307. base_y = 248 + 113 * 0.5 = 304.5.
    screen.handle_mouse_move(310.0, 310.0, 800.0, 600.0);
    assert!(screen.hovered_node.is_some());
    assert_eq!(
        screen.hovered_node.as_ref().unwrap().id,
        "telos:adventure/root"
    );

    // Render verification
    let font_bytes = vec![255u8; 128 * 128 * 4];
    let font = BitmapFont::from_rgba(&font_bytes, 128, 128, 3);
    let layers = UiLayers::default();
    let mut out = Vec::new();

    screen.render(&layers, &font, 800.0, 600.0, 2, &mut out);
    assert!(
        !out.is_empty(),
        "Expected quads generated for advancements screen"
    );
}

#[test]
fn test_hud_toast_banner_rendering() {
    let state = HudState {
        active_toast: Some(ToastState {
            id: "telos:story/mine_wood".to_string(),
            title: "Getting Wood".to_string(),
            icon_item: 5,
            frame: 0,
            elapsed_secs: 1.0,
            duration_secs: 5.0,
        }),
        ..Default::default()
    };

    let layers = UiLayers::default();
    let font_bytes = vec![255u8; 128 * 128 * 4];
    let font = BitmapFont::from_rgba(&font_bytes, 128, 128, 3);
    let mut out = Vec::new();

    render_hud(&state, &font, &layers, 1920, 1080, 2, &mut out);

    // Verify toast background quad is generated with toast_bg layer (10)
    let found_toast_bg = out.iter().any(|q| q.uv[2] == layers.toast_bg);
    assert!(
        found_toast_bg,
        "Expected toast background quad in HUD output"
    );

    // Verify icon item quad is generated with item_icons layer (6)
    let found_item_icon = out.iter().any(|q| q.uv[2] == layers.item_icons);
    assert!(
        found_item_icon,
        "Expected toast item icon quad in HUD output"
    );
}
