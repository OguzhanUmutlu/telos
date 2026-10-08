//! Integration tests for dynamic GUI style sheets, custom slot layouts, and 9-slice container skinning.

use telos_ui::font::BitmapFont;
use telos_ui::hud::{HudState, UiLayers, render_hud, render_hud_styled};
use telos_ui::inventory::{INVENTORY_SLOT_COUNT, UiSlotItem, render_inventory_screen_styled};
use telos_ui::quad::QuadKind;
use telos_ui::style::{
    ContainerLayoutDef, HudThemeDef, NineSliceBorderDef, SlotLayoutDef, container_bounds,
    slot_at_pos_styled,
};
use telos_ui::{
    chest_slot_at_pos_styled, crafting_table_slot_at_pos_styled, furnace_slot_at_pos_styled,
    render_chest_container_styled, render_crafting_table_container_styled,
    render_furnace_container_styled,
};

#[test]
fn test_custom_container_bounds_and_hit_testing() {
    let mut layout = ContainerLayoutDef {
        width: 200,
        height: 180,
        title_pos: [8, 6],
        inventory_title_pos: Some([8, 72]),
        background_texture: None,
        nine_slice: None,
        slots: Vec::new(),
        flame_pos: None,
        arrow_pos: None,
    };
    layout
        .slots
        .push(SlotLayoutDef::with_size(0, 20, 30, 24, 24));
    layout.slots.push(SlotLayoutDef::new(1, 50, 30));

    let screen_w = 800;
    let screen_h = 600;
    let gui_scale = 1;

    let ([ox, oy], [w, h]) = container_bounds(&layout, screen_w, screen_h, gui_scale);
    assert_eq!(w, 200);
    assert_eq!(h, 180);
    assert_eq!(ox, (800 - 200) / 2);
    assert_eq!(oy, (600 - 180) / 2);

    // Hit slot 0 at (ox + 20 + 5, oy + 30 + 5)
    let hit0 = slot_at_pos_styled(
        &layout,
        [ox as f32 + 25.0, oy as f32 + 35.0],
        screen_w,
        screen_h,
        gui_scale,
    );
    assert_eq!(hit0, Some(0));

    // Hit slot 1 at (ox + 50 + 5, oy + 30 + 5)
    let hit1 = slot_at_pos_styled(
        &layout,
        [ox as f32 + 55.0, oy as f32 + 35.0],
        screen_w,
        screen_h,
        gui_scale,
    );
    assert_eq!(hit1, Some(1));

    // Miss between slots
    let miss = slot_at_pos_styled(
        &layout,
        [ox as f32 + 46.0, oy as f32 + 35.0],
        screen_w,
        screen_h,
        gui_scale,
    );
    assert_eq!(miss, None);

    // Hit testing using container specific helpers
    assert_eq!(
        chest_slot_at_pos_styled(
            [ox as f32 + 25.0, oy as f32 + 35.0],
            screen_w,
            screen_h,
            gui_scale,
            &layout,
        ),
        Some(0)
    );
    assert_eq!(
        furnace_slot_at_pos_styled(
            [ox as f32 + 55.0, oy as f32 + 35.0],
            screen_w,
            screen_h,
            gui_scale,
            &layout,
        ),
        Some(1)
    );
    assert_eq!(
        crafting_table_slot_at_pos_styled(
            [ox as f32 + 25.0, oy as f32 + 35.0],
            screen_w,
            screen_h,
            gui_scale,
            &layout,
        ),
        Some(0)
    );
}

#[test]
fn test_nine_slice_quad_generation_in_inventory() {
    let font = BitmapFont::new_fallback(3);
    let layers = UiLayers::default();
    let slots = vec![UiSlotItem::EMPTY; INVENTORY_SLOT_COUNT];

    let mut layout = ContainerLayoutDef::default_inventory();
    layout.nine_slice = Some(NineSliceBorderDef::uniform(4));

    let mut quads = Vec::new();
    render_inventory_screen_styled(
        &slots,
        UiSlotItem::EMPTY,
        None,
        800,
        600,
        2,
        &font,
        &layers,
        &layout,
        |_| "Air",
        [0.0, 0.0],
        &mut quads,
    );

    // Quad 0 is fullscreen dim
    assert_eq!(quads[0].kind(), QuadKind::Solid);

    // Quad 1 is the container background. Since nine_slice is set, it MUST be NineSlice!
    assert_eq!(quads[1].kind(), QuadKind::NineSlice);
    // Borders scaled by gui_scale 2: 4 * 2 = 8
    assert_eq!(quads[1].nine_slice_borders(), [8, 8, 8, 8]);
    // Source size: 176 * 2 = 352, 166 * 2 = 332
    assert_eq!(quads[1].nine_slice_src_size(), [352, 332]);
}

#[test]
fn test_nine_slice_quad_generation_in_containers() {
    let font = BitmapFont::new_fallback(3);
    let layers = UiLayers::default();

    // Chest with detailed nine-slice
    let mut chest_layout = ContainerLayoutDef::default_chest();
    chest_layout.nine_slice = Some(NineSliceBorderDef::new(4, 8, 6, 10));

    let mut quads = Vec::new();
    render_chest_container_styled(
        &vec![UiSlotItem::EMPTY; 63],
        UiSlotItem::EMPTY,
        None,
        800,
        600,
        1,
        &font,
        &layers,
        "Chest",
        |_| "Air",
        [0.0, 0.0],
        &chest_layout,
        &mut quads,
    );

    // Container background is quad 1
    assert_eq!(quads[1].kind(), QuadKind::NineSlice);
    assert_eq!(quads[1].nine_slice_borders(), [4, 8, 6, 10]);

    // Furnace with uniform nine-slice
    let mut furnace_layout = ContainerLayoutDef::default_furnace();
    furnace_layout.nine_slice = Some(NineSliceBorderDef::uniform(5));

    let mut f_quads = Vec::new();
    render_furnace_container_styled(
        &vec![UiSlotItem::EMPTY; 39],
        UiSlotItem::EMPTY,
        None,
        0,
        0,
        0,
        0,
        800,
        600,
        1,
        &font,
        &layers,
        "Furnace",
        |_| "Air",
        [0.0, 0.0],
        &furnace_layout,
        &mut f_quads,
    );
    assert_eq!(f_quads[1].kind(), QuadKind::NineSlice);
    assert_eq!(f_quads[1].nine_slice_borders(), [5, 5, 5, 5]);

    // Crafting table with uniform nine-slice
    let mut craft_layout = ContainerLayoutDef::default_crafting_table();
    craft_layout.nine_slice = Some(NineSliceBorderDef::uniform(6));

    let mut c_quads = Vec::new();
    render_crafting_table_container_styled(
        &vec![UiSlotItem::EMPTY; 46],
        UiSlotItem::EMPTY,
        None,
        800,
        600,
        1,
        &font,
        &layers,
        "Crafting",
        |_| "Air",
        [0.0, 0.0],
        &craft_layout,
        &mut c_quads,
    );
    assert_eq!(c_quads[1].kind(), QuadKind::NineSlice);
    assert_eq!(c_quads[1].nine_slice_borders(), [6, 6, 6, 6]);
}

#[test]
fn test_hud_theme_styling() {
    let font = BitmapFont::new_fallback(3);
    let layers = UiLayers::default();
    let hud_state = HudState::default();

    let theme = HudThemeDef {
        hotbar_y_offset: -8,
        ..Default::default()
    };

    let mut default_quads = Vec::new();
    render_hud(&hud_state, &font, &layers, 800, 600, 1, &mut default_quads);

    let mut styled_quads = Vec::new();
    render_hud_styled(
        &hud_state,
        &font,
        &layers,
        800,
        600,
        1,
        &theme,
        &mut styled_quads,
    );

    // Crosshair is quad 0, hotbar frame is quad 1
    let default_hotbar_pos = default_quads[1].pos;
    let styled_hotbar_pos = styled_quads[1].pos;

    // Styled hotbar should have y shifted by +8 when offset is -8 (moved down towards bottom)
    assert_eq!(styled_hotbar_pos[0], default_hotbar_pos[0]);
    assert_eq!(styled_hotbar_pos[1] - default_hotbar_pos[1], 8);
}
