//! Retained dual-inventory container UI layout, slot hit-testing, and quad rendering.

use crate::font::BitmapFont;
use crate::hud::UiLayers;
use crate::inventory::{CONTAINER_HEIGHT, CONTAINER_WIDTH, UiSlotItem, item_icon_uv};
use crate::quad::UiQuad;
use crate::scale::to_physical_pixels;

/// Total number of item slots in a dual chest container window (63).
/// Total number of item slots in a dual chest container window (63).
/// - 0..27: Chest container storage (3 rows of 9)
/// - 27..54: Player main storage (3 rows of 9)
/// - 54..63: Player hotbar (1 row of 9)
pub const DUAL_CONTAINER_SLOT_COUNT: usize = 63;

/// Number of item slots in the chest portion of the container.
pub const CHEST_CONTAINER_SLOTS: usize = 27;

/// Total number of item slots in a dual furnace container window (39).
/// - 0..3: Furnace slots (0: Input, 1: Fuel, 2: Output)
/// - 3..30: Player main storage (3 rows of 9)
/// - 30..39: Player hotbar (1 row of 9)
pub const DUAL_FURNACE_SLOT_COUNT: usize = 39;

/// Number of item slots in the furnace portion of the container.
pub const FURNACE_CONTAINER_SLOTS: usize = 3;

/// Returns the container-relative `[x, y]` coordinates of the 16x16 interior of a slot in GUI pixels.
///
/// Returns None if `slot >= DUAL_CONTAINER_SLOT_COUNT`.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn chest_slot_pos(slot: usize) -> Option<[i32; 2]> {
    if slot < CHEST_CONTAINER_SLOTS {
        // Chest storage (slots 0..=26): 3 rows of 9 starting at y = 18, x = 8
        let col = (slot % 9) as i32;
        let row = (slot / 9) as i32;
        Some([8 + col * 18, 18 + row * 18])
    } else if (27..54).contains(&slot) {
        // Player main storage (slots 27..=53): 3 rows of 9 starting at y = 84, x = 8
        let idx = (slot - 27) as i32;
        let col = idx % 9;
        let row = idx / 9;
        Some([8 + col * 18, 84 + row * 18])
    } else if (54..63).contains(&slot) {
        // Player hotbar (slots 54..=62): 1 row of 9 at y = 142, x = 8
        let col = (slot - 54) as i32;
        Some([8 + col * 18, 142])
    } else {
        None
    }
}

/// Hit-tests a screen mouse position against the 63 dual container slots.
///
/// Returns `Some(slot_index)` if the mouse is inside any slot's 16x16 bounds.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
pub fn chest_slot_at_pos(
    mouse_pos: [f32; 2],
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
) -> Option<usize> {
    let container_w = to_physical_pixels(CONTAINER_WIDTH as i32, gui_scale);
    let container_h = to_physical_pixels(CONTAINER_HEIGHT as i32, gui_scale);
    let origin_x = (screen_w as i32 - container_w) / 2;
    let origin_y = (screen_h as i32 - container_h) / 2;

    let mx = mouse_pos[0] as i32;
    let my = mouse_pos[1] as i32;

    let slot_sz = to_physical_pixels(16, gui_scale);

    for i in 0..DUAL_CONTAINER_SLOT_COUNT {
        if let Some([sx, sy]) = chest_slot_pos(i) {
            let px = origin_x + to_physical_pixels(sx, gui_scale);
            let py = origin_y + to_physical_pixels(sy, gui_scale);
            if mx >= px && mx < px + slot_sz && my >= py && my < py + slot_sz {
                return Some(i);
            }
        }
    }

    None
}

/// Returns the container-relative `[x, y]` coordinates of the interior of a furnace slot in GUI pixels.
///
/// Returns None if `slot >= DUAL_FURNACE_SLOT_COUNT`.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn furnace_slot_pos(slot: usize) -> Option<[i32; 2]> {
    match slot {
        0 => Some([56, 17]),  // Input
        1 => Some([56, 53]),  // Fuel
        2 => Some([116, 35]), // Output
        3..=29 => {
            let idx = (slot - 3) as i32;
            let col = idx % 9;
            let row = idx / 9;
            Some([8 + col * 18, 84 + row * 18])
        }
        30..=38 => {
            let col = (slot - 30) as i32;
            Some([8 + col * 18, 142])
        }
        _ => None,
    }
}

/// Hit-tests a screen mouse position against the 39 dual furnace container slots.
///
/// Returns `Some(slot_index)` if the mouse is inside any slot's bounds.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
pub fn furnace_slot_at_pos(
    mouse_pos: [f32; 2],
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
) -> Option<usize> {
    let container_w = to_physical_pixels(CONTAINER_WIDTH as i32, gui_scale);
    let container_h = to_physical_pixels(CONTAINER_HEIGHT as i32, gui_scale);
    let origin_x = (screen_w as i32 - container_w) / 2;
    let origin_y = (screen_h as i32 - container_h) / 2;

    let mx = mouse_pos[0] as i32;
    let my = mouse_pos[1] as i32;

    let slot_sz = to_physical_pixels(16, gui_scale);

    for i in 0..DUAL_FURNACE_SLOT_COUNT {
        if let Some([sx, sy]) = furnace_slot_pos(i) {
            let px = origin_x + to_physical_pixels(sx, gui_scale);
            let py = origin_y + to_physical_pixels(sy, gui_scale);
            if mx >= px && mx < px + slot_sz && my >= py && my < py + slot_sz {
                return Some(i);
            }
        }
    }

    None
}

/// Renders the complete interactive dual-inventory container screen:
/// - Dimmed fullscreen background
/// - Centered 176x166 container background
/// - 27 recessed chest slot frames (3 rows of 9)
/// - Container title (e.g. "Chest") and "Inventory" labels
/// - 63 interactive slots with 16x16 item icons and count labels
/// - Slot hover highlight
/// - Carried cursor item stack
/// - Hover tooltip with formatted item name and border
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn render_chest_container<'a>(
    slots: &[UiSlotItem],
    carried: UiSlotItem,
    hovered_slot: Option<usize>,
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
    font: &BitmapFont,
    layers: &UiLayers,
    title: &str,
    item_names: impl Fn(u32) -> &'a str,
    mouse_pos: [f32; 2],
    out: &mut Vec<UiQuad>,
) {
    let sw = screen_w as i32;
    let sh = screen_h as i32;

    // 1. Fullscreen dark dimming overlay (50% black)
    out.push(UiQuad::solid(
        [0, 0],
        [screen_w as u16, screen_h as u16],
        UiQuad::rgba(0, 0, 0, 128),
    ));

    // 2. Centered container background (176x166 source px)
    let container_w = to_physical_pixels(CONTAINER_WIDTH as i32, gui_scale) as u16;
    let container_h = to_physical_pixels(CONTAINER_HEIGHT as i32, gui_scale) as u16;
    let origin_x = (sw - i32::from(container_w)) / 2;
    let origin_y = (sh - i32::from(container_h)) / 2;

    out.push(UiQuad::sprite(
        [origin_x, origin_y],
        [container_w, container_h],
        [layers.inventory_bg_uv[0], layers.inventory_bg_uv[1]],
        [layers.inventory_bg_uv[2], layers.inventory_bg_uv[3]],
        layers.inventory_bg,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    // 2b. Upper panel backdrop covering crafting area with solid GUI gray
    let upper_panel_x = origin_x + to_physical_pixels(7, gui_scale);
    let upper_panel_y = origin_y + to_physical_pixels(17, gui_scale);
    let upper_panel_w = to_physical_pixels(162, gui_scale) as u16;
    let upper_panel_h = to_physical_pixels(54, gui_scale) as u16;
    out.push(UiQuad::solid(
        [upper_panel_x, upper_panel_y],
        [upper_panel_w, upper_panel_h],
        UiQuad::rgba(198, 198, 198, 255),
    ));

    // 2c. Recessed slot frames for the 27 chest slots (3 rows of 9)
    let slot_outer_sz = to_physical_pixels(18, gui_scale) as u16;
    let border_px = gui_scale.max(1) as u16;
    let shadow_col = UiQuad::rgba(55, 55, 55, 255);
    let light_col = UiQuad::rgba(255, 255, 255, 255);
    let slot_inner_col = UiQuad::rgba(139, 139, 139, 255);

    for row in 0..3 {
        for col in 0..9 {
            let sx = origin_x + to_physical_pixels(7 + col * 18, gui_scale);
            let sy = origin_y + to_physical_pixels(17 + row * 18, gui_scale);

            // Interior fill
            out.push(UiQuad::solid(
                [sx, sy],
                [slot_outer_sz, slot_outer_sz],
                slot_inner_col,
            ));
            // Dark shadow border (top & left)
            out.push(UiQuad::solid(
                [sx, sy],
                [slot_outer_sz, border_px],
                shadow_col,
            ));
            out.push(UiQuad::solid(
                [sx, sy],
                [border_px, slot_outer_sz],
                shadow_col,
            ));
            // Light highlight border (bottom & right)
            out.push(UiQuad::solid(
                [sx, sy + i32::from(slot_outer_sz - border_px)],
                [slot_outer_sz, border_px],
                light_col,
            ));
            out.push(UiQuad::solid(
                [sx + i32::from(slot_outer_sz - border_px), sy],
                [border_px, slot_outer_sz],
                light_col,
            ));
        }
    }

    // 3. Titles: Container title (e.g. "Chest") and "Inventory"
    let title_color = UiQuad::rgba(64, 64, 64, 255);
    let scale_f = gui_scale as f32;
    font.layout_text(
        title,
        (origin_x as f32 / scale_f) + 8.0,
        (origin_y as f32 / scale_f) + 6.0,
        title_color,
        false,
        gui_scale,
        out,
    );
    font.layout_text(
        "Inventory",
        (origin_x as f32 / scale_f) + 8.0,
        (origin_y as f32 / scale_f) + 72.0,
        title_color,
        false,
        gui_scale,
        out,
    );

    let slot_sz = to_physical_pixels(16, gui_scale) as u16;

    // 4. Slots quads
    for (i, item) in slots
        .iter()
        .copied()
        .enumerate()
        .take(DUAL_CONTAINER_SLOT_COUNT)
    {
        if let Some([sx, sy]) = chest_slot_pos(i) {
            let slot_x = origin_x + to_physical_pixels(sx, gui_scale);
            let slot_y = origin_y + to_physical_pixels(sy, gui_scale);

            // Slot hover highlight (semi-transparent white)
            if hovered_slot == Some(i) {
                out.push(UiQuad::solid(
                    [slot_x, slot_y],
                    [slot_sz, slot_sz],
                    UiQuad::rgba(255, 255, 255, 80),
                ));
            }

            if !item.is_empty() {
                let uv = item_icon_uv(item.item);
                // Item icon
                out.push(UiQuad::sprite(
                    [slot_x, slot_y],
                    [slot_sz, slot_sz],
                    [uv[0], uv[1]],
                    [uv[2], uv[3]],
                    layers.item_icons,
                    UiQuad::rgba(255, 255, 255, 255),
                ));

                // Stack count if > 1
                if item.count > 1 {
                    let count_str = format!("{}", item.count);
                    let (text_w, _) = font.measure_text(&count_str);
                    let text_x = (slot_x as f32 / scale_f) + 17.0 - text_w;
                    let text_y = (slot_y as f32 / scale_f) + 9.0;
                    font.layout_text(
                        &count_str,
                        text_x,
                        text_y,
                        UiQuad::rgba(255, 255, 255, 255),
                        true,
                        gui_scale,
                        out,
                    );
                }
            }
        }
    }

    // 5. Carried stack on cursor
    if !carried.is_empty() {
        let half_sz = i32::from(slot_sz / 2);
        let carried_x = mouse_pos[0] as i32 - half_sz;
        let carried_y = mouse_pos[1] as i32 - half_sz;
        let uv = item_icon_uv(carried.item);

        out.push(UiQuad::sprite(
            [carried_x, carried_y],
            [slot_sz, slot_sz],
            [uv[0], uv[1]],
            [uv[2], uv[3]],
            layers.item_icons,
            UiQuad::rgba(255, 255, 255, 255),
        ));

        if carried.count > 1 {
            let count_str = format!("{}", carried.count);
            let (text_w, _) = font.measure_text(&count_str);
            let text_x = (carried_x as f32 / scale_f) + 17.0 - text_w;
            let text_y = (carried_y as f32 / scale_f) + 9.0;
            font.layout_text(
                &count_str,
                text_x,
                text_y,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                gui_scale,
                out,
            );
        }
    }

    // 6. Tooltip when hovering over a slot
    if let Some(hovered_idx) = hovered_slot
        && carried.is_empty()
        && hovered_idx < slots.len()
        && !slots[hovered_idx].is_empty()
    {
        let item = slots[hovered_idx];
        let name = item_names(item.item);
        let (text_w, text_h) = font.measure_text(name);

        let pad = to_physical_pixels(3, gui_scale);
        let box_w = (text_w * scale_f) as i32 + pad * 2;
        let box_h = (text_h * scale_f) as i32 + pad * 2;

        let tip_x =
            ((mouse_pos[0] as i32 + to_physical_pixels(10, gui_scale)).min(sw - box_w - 4)).max(4);
        let tip_y =
            ((mouse_pos[1] as i32 - to_physical_pixels(12, gui_scale)).min(sh - box_h - 4)).max(4);

        // Dark box background (#100010 at 94% opacity)
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [box_w as u16, box_h as u16],
            UiQuad::rgba(16, 0, 16, 240),
        ));

        // Purple borders (#5000ff)
        let border_w = gui_scale.max(1) as u16;
        let border_col = UiQuad::rgba(80, 0, 255, 255);
        // Top border
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [box_w as u16, border_w],
            border_col,
        ));
        // Bottom border
        out.push(UiQuad::solid(
            [tip_x, tip_y + box_h - i32::from(border_w)],
            [box_w as u16, border_w],
            border_col,
        ));
        // Left border
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [border_w, box_h as u16],
            border_col,
        ));
        // Right border
        out.push(UiQuad::solid(
            [tip_x + box_w - i32::from(border_w), tip_y],
            [border_w, box_h as u16],
            border_col,
        ));

        // Tooltip text in pure white
        font.layout_text(
            name,
            (tip_x + pad) as f32 / scale_f,
            (tip_y + pad) as f32 / scale_f,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            gui_scale,
            out,
        );
    }
}

/// Renders the complete interactive dual-inventory furnace container screen:
/// - Dimmed fullscreen background
/// - Centered 176x166 furnace background (from layer `furnace_bg`)
/// - Lit flame meter at (56, 36) burning down as `burn_time_remaining / total_burn_time`
/// - Cook progress arrow at (79, 34) filling left-to-right as `cook_progress / cook_duration`
/// - Container title (e.g. "Furnace") and "Inventory" labels
/// - 39 interactive slots (3 furnace slots + 27 main inventory + 9 hotbar) with 16x16 item icons and count labels
/// - Slot hover highlight
/// - Carried cursor item stack
/// - Hover tooltip with formatted item name and border
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn render_furnace_container<'a>(
    slots: &[UiSlotItem],
    carried: UiSlotItem,
    hovered_slot: Option<usize>,
    burn_time_remaining: i16,
    total_burn_time: i16,
    cook_progress: i16,
    cook_duration: i16,
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
    font: &BitmapFont,
    layers: &UiLayers,
    title: &str,
    item_names: impl Fn(u32) -> &'a str,
    mouse_pos: [f32; 2],
    out: &mut Vec<UiQuad>,
) {
    let sw = screen_w as i32;
    let sh = screen_h as i32;

    // 1. Fullscreen dark dimming overlay (50% black)
    out.push(UiQuad::solid(
        [0, 0],
        [screen_w as u16, screen_h as u16],
        UiQuad::rgba(0, 0, 0, 128),
    ));

    // 2. Centered container background (176x166 source px)
    let container_w = to_physical_pixels(CONTAINER_WIDTH as i32, gui_scale) as u16;
    let container_h = to_physical_pixels(CONTAINER_HEIGHT as i32, gui_scale) as u16;
    let origin_x = (sw - i32::from(container_w)) / 2;
    let origin_y = (sh - i32::from(container_h)) / 2;

    out.push(UiQuad::sprite(
        [origin_x, origin_y],
        [container_w, container_h],
        [layers.furnace_bg_uv[0], layers.furnace_bg_uv[1]],
        [layers.furnace_bg_uv[2], layers.furnace_bg_uv[3]],
        layers.furnace_bg,
        UiQuad::rgba(255, 255, 255, 255),
    ));

    // 2a. Lit flame meter: burns down from bottom-up
    if total_burn_time > 0 && burn_time_remaining > 0 {
        let frac = (f32::from(burn_time_remaining) / f32::from(total_burn_time)).clamp(0.0, 1.0);
        let flame_h = ((14.0 * frac).ceil() as i32).clamp(1, 14);
        let top_offset = 14 - flame_h;
        let fx = origin_x + to_physical_pixels(56, gui_scale);
        let fy = origin_y + to_physical_pixels(36 + top_offset, gui_scale);
        let fw = to_physical_pixels(14, gui_scale) as u16;
        let fh = to_physical_pixels(flame_h, gui_scale) as u16;

        let v_top = layers.furnace_flame_uv[1] + (top_offset as f32 / 256.0);
        let v_bottom = layers.furnace_flame_uv[3];

        out.push(UiQuad::sprite(
            [fx, fy],
            [fw, fh],
            [layers.furnace_flame_uv[0], v_top],
            [layers.furnace_flame_uv[2], v_bottom],
            layers.furnace_bg,
            UiQuad::rgba(255, 255, 255, 255),
        ));
    }

    // 2b. Cook progress arrow: fills left-to-right
    if cook_duration > 0 && cook_progress > 0 {
        let frac = (f32::from(cook_progress) / f32::from(cook_duration)).clamp(0.0, 1.0);
        let arrow_w = ((24.0 * frac).floor() as i32).clamp(1, 24);
        let ax = origin_x + to_physical_pixels(79, gui_scale);
        let ay = origin_y + to_physical_pixels(34, gui_scale);
        let aw = to_physical_pixels(arrow_w, gui_scale) as u16;
        let ah = to_physical_pixels(16, gui_scale) as u16;

        let u_left = layers.furnace_arrow_uv[0];
        let u_right = layers.furnace_arrow_uv[0] + (arrow_w as f32 / 256.0);

        out.push(UiQuad::sprite(
            [ax, ay],
            [aw, ah],
            [u_left, layers.furnace_arrow_uv[1]],
            [u_right, layers.furnace_arrow_uv[3]],
            layers.furnace_bg,
            UiQuad::rgba(255, 255, 255, 255),
        ));
    }

    // 3. Titles: Container title (e.g. "Furnace") and "Inventory"
    let title_color = UiQuad::rgba(64, 64, 64, 255);
    let scale_f = gui_scale as f32;
    font.layout_text(
        title,
        (origin_x as f32 / scale_f) + 8.0,
        (origin_y as f32 / scale_f) + 6.0,
        title_color,
        false,
        gui_scale,
        out,
    );
    font.layout_text(
        "Inventory",
        (origin_x as f32 / scale_f) + 8.0,
        (origin_y as f32 / scale_f) + 72.0,
        title_color,
        false,
        gui_scale,
        out,
    );

    let slot_sz = to_physical_pixels(16, gui_scale) as u16;

    // 4. Slots quads
    for (i, item) in slots
        .iter()
        .copied()
        .enumerate()
        .take(DUAL_FURNACE_SLOT_COUNT)
    {
        if let Some([sx, sy]) = furnace_slot_pos(i) {
            let slot_x = origin_x + to_physical_pixels(sx, gui_scale);
            let slot_y = origin_y + to_physical_pixels(sy, gui_scale);

            // Slot hover highlight (semi-transparent white)
            if hovered_slot == Some(i) {
                out.push(UiQuad::solid(
                    [slot_x, slot_y],
                    [slot_sz, slot_sz],
                    UiQuad::rgba(255, 255, 255, 80),
                ));
            }

            if !item.is_empty() {
                let uv = item_icon_uv(item.item);
                // Item icon
                out.push(UiQuad::sprite(
                    [slot_x, slot_y],
                    [slot_sz, slot_sz],
                    [uv[0], uv[1]],
                    [uv[2], uv[3]],
                    layers.item_icons,
                    UiQuad::rgba(255, 255, 255, 255),
                ));

                // Stack count if > 1
                if item.count > 1 {
                    let count_str = format!("{}", item.count);
                    let (text_w, _) = font.measure_text(&count_str);
                    let text_x = (slot_x as f32 / scale_f) + 17.0 - text_w;
                    let text_y = (slot_y as f32 / scale_f) + 9.0;
                    font.layout_text(
                        &count_str,
                        text_x,
                        text_y,
                        UiQuad::rgba(255, 255, 255, 255),
                        true,
                        gui_scale,
                        out,
                    );
                }
            }
        }
    }

    // 5. Carried stack on cursor
    if !carried.is_empty() {
        let half_sz = i32::from(slot_sz / 2);
        let carried_x = mouse_pos[0] as i32 - half_sz;
        let carried_y = mouse_pos[1] as i32 - half_sz;
        let uv = item_icon_uv(carried.item);

        out.push(UiQuad::sprite(
            [carried_x, carried_y],
            [slot_sz, slot_sz],
            [uv[0], uv[1]],
            [uv[2], uv[3]],
            layers.item_icons,
            UiQuad::rgba(255, 255, 255, 255),
        ));

        if carried.count > 1 {
            let count_str = format!("{}", carried.count);
            let (text_w, _) = font.measure_text(&count_str);
            let text_x = (carried_x as f32 / scale_f) + 17.0 - text_w;
            let text_y = (carried_y as f32 / scale_f) + 9.0;
            font.layout_text(
                &count_str,
                text_x,
                text_y,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                gui_scale,
                out,
            );
        }
    }

    // 6. Tooltip when hovering over a slot
    if let Some(hovered_idx) = hovered_slot
        && carried.is_empty()
        && hovered_idx < slots.len()
        && !slots[hovered_idx].is_empty()
    {
        let item = slots[hovered_idx];
        let name = item_names(item.item);
        let (text_w, text_h) = font.measure_text(name);

        let pad = to_physical_pixels(3, gui_scale);
        let box_w = (text_w * scale_f) as i32 + pad * 2;
        let box_h = (text_h * scale_f) as i32 + pad * 2;

        let tip_x =
            ((mouse_pos[0] as i32 + to_physical_pixels(10, gui_scale)).min(sw - box_w - 4)).max(4);
        let tip_y =
            ((mouse_pos[1] as i32 - to_physical_pixels(12, gui_scale)).min(sh - box_h - 4)).max(4);

        // Dark box background (#100010 at 94% opacity)
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [box_w as u16, box_h as u16],
            UiQuad::rgba(16, 0, 16, 240),
        ));

        // Purple borders (#5000ff)
        let border_w = gui_scale.max(1) as u16;
        let border_col = UiQuad::rgba(80, 0, 255, 255);
        // Top border
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [box_w as u16, border_w],
            border_col,
        ));
        // Bottom border
        out.push(UiQuad::solid(
            [tip_x, tip_y + box_h - i32::from(border_w)],
            [box_w as u16, border_w],
            border_col,
        ));
        // Left border
        out.push(UiQuad::solid(
            [tip_x, tip_y],
            [border_w, box_h as u16],
            border_col,
        ));
        // Right border
        out.push(UiQuad::solid(
            [tip_x + box_w - i32::from(border_w), tip_y],
            [border_w, box_h as u16],
            border_col,
        ));

        // Tooltip text in pure white
        font.layout_text(
            name,
            (tip_x + pad) as f32 / scale_f,
            (tip_y + pad) as f32 / scale_f,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            gui_scale,
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chest_slot_coordinates_bounds() {
        assert_eq!(chest_slot_pos(0), Some([8, 18]));
        assert_eq!(chest_slot_pos(8), Some([8 + 8 * 18, 18]));
        assert_eq!(chest_slot_pos(26), Some([8 + 8 * 18, 18 + 2 * 18]));

        // Storage
        assert_eq!(chest_slot_pos(27), Some([8, 84]));
        assert_eq!(chest_slot_pos(53), Some([8 + 8 * 18, 84 + 2 * 18]));

        // Hotbar
        assert_eq!(chest_slot_pos(54), Some([8, 142]));
        assert_eq!(chest_slot_pos(62), Some([8 + 8 * 18, 142]));

        // Out of bounds
        assert_eq!(chest_slot_pos(63), None);
        assert_eq!(chest_slot_pos(100), None);
    }

    #[test]
    fn test_furnace_slot_coordinates_bounds() {
        assert_eq!(furnace_slot_pos(0), Some([56, 17]));
        assert_eq!(furnace_slot_pos(1), Some([56, 53]));
        assert_eq!(furnace_slot_pos(2), Some([116, 35]));

        // Storage
        assert_eq!(furnace_slot_pos(3), Some([8, 84]));
        assert_eq!(furnace_slot_pos(29), Some([8 + 8 * 18, 84 + 2 * 18]));

        // Hotbar
        assert_eq!(furnace_slot_pos(30), Some([8, 142]));
        assert_eq!(furnace_slot_pos(38), Some([8 + 8 * 18, 142]));

        // Out of bounds
        assert_eq!(furnace_slot_pos(39), None);
        assert_eq!(furnace_slot_pos(100), None);
    }

    #[test]
    fn test_furnace_slot_hit_test() {
        // Center of container in 800x600 with scale 1
        let container_w = 176;
        let container_h = 166;
        let ox = (800 - container_w) / 2;
        let oy = (600 - container_h) / 2;

        // Inside furnace input slot (slot 0): (ox + 56 + 2, oy + 17 + 2)
        let hit_in = furnace_slot_at_pos([ox as f32 + 58.0, oy as f32 + 19.0], 800, 600, 1);
        assert_eq!(hit_in, Some(0));

        // Inside furnace fuel slot (slot 1): (ox + 56 + 2, oy + 53 + 2)
        let hit_fuel = furnace_slot_at_pos([ox as f32 + 58.0, oy as f32 + 55.0], 800, 600, 1);
        assert_eq!(hit_fuel, Some(1));

        // Inside furnace output slot (slot 2): (ox + 116 + 2, oy + 35 + 2)
        let hit_out = furnace_slot_at_pos([ox as f32 + 118.0, oy as f32 + 37.0], 800, 600, 1);
        assert_eq!(hit_out, Some(2));

        // Inside player hotbar slot 0 (slot 30): (ox + 8 + 2, oy + 142 + 2)
        let hit_hotbar = furnace_slot_at_pos([ox as f32 + 10.0, oy as f32 + 144.0], 800, 600, 1);
        assert_eq!(hit_hotbar, Some(30));

        // Outside container
        let hit_outside = furnace_slot_at_pos([10.0, 10.0], 800, 600, 1);
        assert_eq!(hit_outside, None);
    }

    #[test]
    fn test_chest_slot_hit_test() {
        // Center of container in 800x600 with scale 1
        let container_w = 176;
        let container_h = 166;
        let ox = (800 - container_w) / 2;
        let oy = (600 - container_h) / 2;

        // Inside chest slot 0 interior: (ox + 8 + 2, oy + 18 + 2)
        let hit = chest_slot_at_pos([ox as f32 + 10.0, oy as f32 + 20.0], 800, 600, 1);
        assert_eq!(hit, Some(0));

        // Inside player hotbar slot 0: (ox + 8 + 2, oy + 142 + 2)
        let hit_hotbar = chest_slot_at_pos([ox as f32 + 10.0, oy as f32 + 144.0], 800, 600, 1);
        assert_eq!(hit_hotbar, Some(54));

        // Outside container
        let hit_outside = chest_slot_at_pos([10.0, 10.0], 800, 600, 1);
        assert_eq!(hit_outside, None);
    }
}
