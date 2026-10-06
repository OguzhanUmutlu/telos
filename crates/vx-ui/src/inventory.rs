//! Interactive survival inventory screen layout, slot hit-testing, and quad rendering.

use crate::font::BitmapFont;
use crate::hud::UiLayers;
use crate::quad::UiQuad;
use crate::scale::to_physical_pixels;

/// Width of the standard player survival inventory container in GUI pixels.
pub const CONTAINER_WIDTH: u32 = 176;
/// Height of the standard player survival inventory container in GUI pixels.
pub const CONTAINER_HEIGHT: u32 = 166;

/// Total number of slots in the player survival inventory screen (46).
pub const INVENTORY_SLOT_COUNT: usize = 46;

/// A lightweight slot item view for UI rendering and hit-testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UiSlotItem {
    /// Item identifier (0 = Air / Empty).
    pub item: u32,
    /// Number of items in this stack.
    pub count: u16,
}

impl UiSlotItem {
    /// Empty item stack constant.
    pub const EMPTY: Self = Self { item: 0, count: 0 };

    /// Creates a new slot item.
    #[must_use]
    pub const fn new(item: u32, count: u16) -> Self {
        if item == 0 || count == 0 {
            Self::EMPTY
        } else {
            Self { item, count }
        }
    }

    /// Whether this item stack is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.item == 0 || self.count == 0
    }
}

/// Returns the container-relative `[x, y]` coordinates of the 16x16 interior of a slot in GUI pixels.
///
/// Returns None if `slot >= INVENTORY_SLOT_COUNT`.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn slot_pos(slot: usize) -> Option<[i32; 2]> {
    if slot < 9 {
        // Hotbar (slots 0..=8): y = 142, x = 8 + col * 18
        Some([8 + (slot as i32) * 18, 142])
    } else if (9..36).contains(&slot) {
        // Main storage (slots 9..=35): 3 rows of 9 starting at y = 84
        let idx = (slot - 9) as i32;
        let col = idx % 9;
        let row = idx / 9;
        Some([8 + col * 18, 84 + row * 18])
    } else if (36..40).contains(&slot) {
        // Armor (slots 36..=39): Helmet (36), Chestplate (37), Leggings (38), Boots (39)
        let row = (slot - 36) as i32;
        Some([8, 8 + row * 18])
    } else if (40..44).contains(&slot) {
        // 2x2 Crafting inputs (slots 40..=43): 2 rows of 2 at x = 98, y = 18
        let idx = (slot - 40) as i32;
        let col = idx % 2;
        let row = idx / 2;
        Some([98 + col * 18, 18 + row * 18])
    } else if slot == 44 {
        // Crafting result output (slot 44)
        Some([154, 28])
    } else if slot == 45 {
        // Offhand slot (slot 45)
        Some([77, 62])
    } else {
        None
    }
}

/// Returns the normalized UV bounds `[u0, v0, u1, v1]` of a 16x16 item icon in the 256x256 atlas.
#[must_use]
pub fn item_icon_uv(item_id: u32) -> [f32; 4] {
    let idx = item_id.saturating_sub(1) % 256;
    let col = (idx % 16) as f32;
    let row = (idx / 16) as f32;
    [
        col / 16.0,
        row / 16.0,
        (col + 1.0) / 16.0,
        (row + 1.0) / 16.0,
    ]
}

/// Hit-tests a screen mouse position against the 46 inventory slots.
///
/// Returns `Some(slot_index)` if the mouse is inside any slot's 16x16 bounds.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
pub fn slot_at_pos(
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

    for i in 0..INVENTORY_SLOT_COUNT {
        if let Some([sx, sy]) = slot_pos(i) {
            let px = origin_x + to_physical_pixels(sx, gui_scale);
            let py = origin_y + to_physical_pixels(sy, gui_scale);
            if mx >= px && mx < px + slot_sz && my >= py && my < py + slot_sz {
                return Some(i);
            }
        }
    }

    None
}

/// Renders the complete interactive inventory screen:
/// - Dimmed fullscreen background
/// - Centered 176x166 container background sprite
/// - "Crafting" and "Inventory" titles
/// - 46 slots with 16x16 item icons and count numbers
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
pub fn render_inventory_screen(
    slots: &[UiSlotItem],
    carried: UiSlotItem,
    hovered_slot: Option<usize>,
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
    font: &BitmapFont,
    layers: &UiLayers,
    item_names: impl Fn(u32) -> &'static str,
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

    // 3. Titles: "Crafting" and "Inventory"
    let title_color = UiQuad::rgba(64, 64, 64, 255);
    let scale_f = gui_scale as f32;
    font.layout_text(
        "Crafting",
        (origin_x as f32 / scale_f) + 97.0,
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
    for (i, item) in slots.iter().copied().enumerate().take(INVENTORY_SLOT_COUNT) {
        if let Some([sx, sy]) = slot_pos(i) {
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
    fn test_slot_positions_count() {
        for i in 0..INVENTORY_SLOT_COUNT {
            assert!(slot_pos(i).is_some(), "Slot {i} must have valid position");
        }
        assert!(slot_pos(INVENTORY_SLOT_COUNT).is_none());
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn test_slot_hit_testing() {
        let sw = 1920;
        let sh = 1080;
        let gui_scale = 2;

        let container_w = to_physical_pixels(CONTAINER_WIDTH as i32, gui_scale);
        let container_h = to_physical_pixels(CONTAINER_HEIGHT as i32, gui_scale);
        let origin_x = (sw as i32 - container_w) / 2;
        let origin_y = (sh as i32 - container_h) / 2;

        // Mouse right in the middle of Hotbar slot 0: (origin_x + 8*2 + 8, origin_y + 142*2 + 8)
        let mouse_pos = [(origin_x + 16 + 8) as f32, (origin_y + 284 + 8) as f32];
        assert_eq!(slot_at_pos(mouse_pos, sw, sh, gui_scale), Some(0));

        // Mouse way off in the corner
        assert_eq!(slot_at_pos([0.0, 0.0], sw, sh, gui_scale), None);
    }

    #[test]
    fn test_render_inventory_screen_quad_generation() {
        let font = BitmapFont::new_fallback(3);
        let layers = UiLayers::default();
        let mut quads = Vec::new();

        let mut slots = [UiSlotItem::EMPTY; 46];
        slots[0] = UiSlotItem::new(1, 64); // Stone
        slots[44] = UiSlotItem::new(7, 4); // Crafting result 4 planks

        render_inventory_screen(
            &slots,
            UiSlotItem::EMPTY,
            Some(0), // Hover slot 0
            1920,
            1080,
            2,
            &font,
            &layers,
            |_| "Stone",
            [960.0, 540.0],
            &mut quads,
        );

        assert!(!quads.is_empty(), "Must produce inventory screen quads");
    }
}
