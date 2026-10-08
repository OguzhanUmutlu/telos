//! GUI style sheet, data-driven container metrics, and hit-testing helpers.

pub use telos_assets::gui::{
    ContainerLayoutDef, GuiStyleSheet, HudThemeDef, NineSliceBorderDef, SlotLayoutDef,
};

use crate::scale::to_physical_pixels;

/// Computes the centered physical pixel origin `[x, y]` and dimensions `[w, h]` for a container layout.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn container_bounds(
    layout: &ContainerLayoutDef,
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
) -> ([i32; 2], [u16; 2]) {
    let cw = to_physical_pixels(layout.width as i32, gui_scale);
    let ch = to_physical_pixels(layout.height as i32, gui_scale);
    let origin_x = (screen_w as i32 - cw) / 2;
    let origin_y = (screen_h as i32 - ch) / 2;
    ([origin_x, origin_y], [cw as u16, ch as u16])
}

/// Evaluates a screen mouse position against the slots of a container layout.
///
/// Returns `Some(slot_index)` if the mouse is inside any slot's bounds.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
pub fn slot_at_pos_styled(
    layout: &ContainerLayoutDef,
    mouse_pos: [f32; 2],
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
) -> Option<usize> {
    let ([origin_x, origin_y], _) = container_bounds(layout, screen_w, screen_h, gui_scale);
    let mx = mouse_pos[0] as i32;
    let my = mouse_pos[1] as i32;

    for slot in &layout.slots {
        let px = origin_x + to_physical_pixels(slot.x, gui_scale);
        let py = origin_y + to_physical_pixels(slot.y, gui_scale);
        let sw = to_physical_pixels(i32::from(slot.size[0]), gui_scale);
        let sh = to_physical_pixels(i32::from(slot.size[1]), gui_scale);

        if mx >= px && mx < px + sw && my >= py && my < py + sh {
            return Some(slot.index);
        }
    }

    None
}
