//! Advancements & achievement tree interactive GUI screen.
//!
//! Provides a 252x140 window with category tabs (Story, Adventure), pannable tree canvas,
//! connecting branch lines, status node frames (Task, Goal, Challenge), and mouse hover tooltips.

use crate::font::BitmapFont;
use crate::hud::UiLayers;
use crate::inventory::item_icon_uv;
use crate::quad::UiQuad;

/// Visual border frame presentation for an advancement node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UiAdvancementFrame {
    /// Standard square/rounded frame for normal milestone advancements.
    Task,
    /// Rounded bracket/circle frame for key gameplay goals.
    Goal,
    /// Spiky / pointed bracket frame for difficult challenges.
    Challenge,
}

/// Category / tab grouping in the advancements menu tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UiAdvancementCategory {
    /// Core gameplay progression.
    Story,
    /// Combat, archery, and exploration progression.
    Adventure,
}

impl UiAdvancementCategory {
    /// Returns the tab title.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Story => "Story",
            Self::Adventure => "Adventure",
        }
    }
}

/// A node in the advancement tree shown in the GUI.
#[derive(Debug, Clone)]
pub struct UiAdvancementNode {
    /// Namespaced identifier (e.g. "telos:story/root").
    pub id: String,
    /// Parent advancement identifier if not a root node.
    pub parent_id: Option<String>,
    /// Category / tab grouping.
    pub category: UiAdvancementCategory,
    /// Display title.
    pub title: String,
    /// Description text.
    pub description: String,
    /// Icon item ID (rendered in center of frame).
    pub icon_item: u32,
    /// Frame visual tier.
    pub frame: UiAdvancementFrame,
    /// Grid column coordinate.
    pub x: f32,
    /// Grid row coordinate.
    pub y: f32,
    /// Whether the local player has unlocked this advancement.
    pub completed: bool,
}

/// Interactive Advancements Screen modal state.
#[derive(Debug, Clone)]
pub struct AdvancementsScreen {
    /// All advancement tree nodes.
    pub nodes: Vec<UiAdvancementNode>,
    /// Currently selected category tab.
    pub active_tab: UiAdvancementCategory,
    /// Canvas panning offset X in GUI pixels.
    pub pan_x: f32,
    /// Canvas panning offset Y in GUI pixels.
    pub pan_y: f32,
    /// Whether the user is currently mouse-dragging the canvas.
    pub is_dragging: bool,
    /// Mouse position at drag start.
    pub drag_start_mouse: [f32; 2],
    /// Panning offset at drag start.
    pub drag_start_pan: [f32; 2],
    /// Currently hovered node under the mouse cursor.
    pub hovered_node: Option<UiAdvancementNode>,
}

impl Default for AdvancementsScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancementsScreen {
    /// Creates a new empty `AdvancementsScreen`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            active_tab: UiAdvancementCategory::Story,
            pan_x: 0.0,
            pan_y: 0.0,
            is_dragging: false,
            drag_start_mouse: [0.0, 0.0],
            drag_start_pan: [0.0, 0.0],
            hovered_node: None,
        }
    }

    /// Sets the tree nodes to display.
    pub fn set_nodes(&mut self, nodes: Vec<UiAdvancementNode>) {
        self.nodes = nodes;
    }

    /// Handles mouse motion in GUI pixels.
    pub fn handle_mouse_move(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
    ) {
        if self.is_dragging {
            let dx = mouse_x - self.drag_start_mouse[0];
            let dy = mouse_y - self.drag_start_mouse[1];
            self.pan_x = self.drag_start_pan[0] + dx;
            self.pan_y = self.drag_start_pan[1] + dy;
        }

        let origin_x = ((width_gui - 252.0) * 0.5).round();
        let origin_y = ((height_gui - 140.0) * 0.5).round();
        let canvas_x = origin_x + 9.0;
        let canvas_y = origin_y + 18.0;
        let canvas_w = 234.0;
        let canvas_h = 113.0;

        self.hovered_node = None;
        if mouse_x >= canvas_x
            && mouse_x <= canvas_x + canvas_w
            && mouse_y >= canvas_y
            && mouse_y <= canvas_y + canvas_h
        {
            let base_x = canvas_x + 24.0 + self.pan_x;
            let base_y = canvas_y + canvas_h * 0.5 + self.pan_y;

            for node in &self.nodes {
                if node.category != self.active_tab {
                    continue;
                }
                let nx = base_x + node.x * 32.0;
                let ny = base_y + node.y * 32.0;
                let frame_sz = 26.0;

                if mouse_x >= nx
                    && mouse_x <= nx + frame_sz
                    && mouse_y >= ny
                    && mouse_y <= ny + frame_sz
                {
                    self.hovered_node = Some(node.clone());
                    break;
                }
            }
        }
    }

    /// Handles mouse button press.
    pub fn handle_mouse_down(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width_gui: f32,
        height_gui: f32,
    ) {
        let origin_x = ((width_gui - 252.0) * 0.5).round();
        let origin_y = ((height_gui - 140.0) * 0.5).round();

        // Check tabs at top
        // Tab 0: Story (origin_x + 9, origin_y - 28, 56, 28)
        let tab0_x = origin_x + 9.0;
        let tab0_y = origin_y - 26.0;
        if mouse_x >= tab0_x && mouse_x <= tab0_x + 56.0 && mouse_y >= tab0_y && mouse_y <= origin_y
        {
            self.active_tab = UiAdvancementCategory::Story;
            self.pan_x = 0.0;
            self.pan_y = 0.0;
            return;
        }

        // Tab 1: Adventure (origin_x + 69, origin_y - 28, 72, 28)
        let tab1_x = origin_x + 69.0;
        let tab1_y = origin_y - 26.0;
        if mouse_x >= tab1_x && mouse_x <= tab1_x + 72.0 && mouse_y >= tab1_y && mouse_y <= origin_y
        {
            self.active_tab = UiAdvancementCategory::Adventure;
            self.pan_x = 0.0;
            self.pan_y = 0.0;
            return;
        }

        // Check canvas area for drag start
        let canvas_x = origin_x + 9.0;
        let canvas_y = origin_y + 18.0;
        let canvas_w = 234.0;
        let canvas_h = 113.0;

        if mouse_x >= canvas_x
            && mouse_x <= canvas_x + canvas_w
            && mouse_y >= canvas_y
            && mouse_y <= canvas_y + canvas_h
        {
            self.is_dragging = true;
            self.drag_start_mouse = [mouse_x, mouse_y];
            self.drag_start_pan = [self.pan_x, self.pan_y];
        }
    }

    /// Handles mouse button release.
    pub fn handle_mouse_up(&mut self) {
        self.is_dragging = false;
    }

    /// Renders the complete advancements screen to `out`.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    #[allow(
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_lossless,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    pub fn render(
        &self,
        layers: &UiLayers,
        font: &BitmapFont,
        width_gui: f32,
        height_gui: f32,
        scale: u32,
        out: &mut Vec<UiQuad>,
    ) {
        let scale_f = scale as f32;

        // 1. Dark semi-transparent full-screen backdrop
        let px_sw = (width_gui * scale_f).round() as u16;
        let px_sh = (height_gui * scale_f).round() as u16;
        out.push(UiQuad::solid(
            [0, 0],
            [px_sw, px_sh],
            UiQuad::rgba(0, 0, 0, 160),
        ));

        // 2. Centered window position (252x140 logical units)
        let origin_x = ((width_gui - 252.0) * 0.5).round();
        let origin_y = ((height_gui - 140.0) * 0.5).round();

        // 3. Category tabs above the window
        let render_tab =
            |cat: UiAdvancementCategory, tx: f32, tw: f32, active: bool, out: &mut Vec<UiQuad>| {
                let ty = origin_y - 24.0;
                let th = 26.0;
                let px_tx = (tx * scale_f).round() as i32;
                let px_ty = (ty * scale_f).round() as i32;
                let px_tw = (tw * scale_f).round() as u16;
                let px_th = (th * scale_f).round() as u16;

                let bg_color = if active {
                    UiQuad::rgba(45, 45, 52, 255)
                } else {
                    UiQuad::rgba(28, 28, 34, 255)
                };
                out.push(UiQuad::solid([px_tx, px_ty], [px_tw, px_th], bg_color));

                // Top accent border
                let border_color = if active {
                    UiQuad::rgba(230, 200, 50, 255)
                } else {
                    UiQuad::rgba(70, 70, 80, 255)
                };
                let bar_h = (2.0 * scale_f).round() as u16;
                out.push(UiQuad::solid([px_tx, px_ty], [px_tw, bar_h], border_color));

                let (text_w, _) = font.measure_text(cat.title());
                let text_x = tx + (tw - text_w) * 0.5;
                let text_y = ty + 8.0;
                font.layout_text(
                    cat.title(),
                    text_x,
                    text_y,
                    if active {
                        UiQuad::rgba(255, 255, 255, 255)
                    } else {
                        UiQuad::rgba(160, 160, 170, 255)
                    },
                    true,
                    scale,
                    out,
                );
            };

        render_tab(
            UiAdvancementCategory::Story,
            origin_x + 9.0,
            56.0,
            self.active_tab == UiAdvancementCategory::Story,
            out,
        );
        render_tab(
            UiAdvancementCategory::Adventure,
            origin_x + 69.0,
            72.0,
            self.active_tab == UiAdvancementCategory::Adventure,
            out,
        );

        // 4. Main 252x140 Window Frame
        let px_ox = (origin_x * scale_f).round() as i32;
        let px_oy = (origin_y * scale_f).round() as i32;
        let px_ow = (252.0 * scale_f).round() as u16;
        let px_oh = (140.0 * scale_f).round() as u16;

        // Base window texture
        out.push(UiQuad::sprite(
            [px_ox, px_oy],
            [px_ow, px_oh],
            [
                layers.advancement_window_uv[0],
                layers.advancement_window_uv[1],
            ],
            [
                layers.advancement_window_uv[2],
                layers.advancement_window_uv[3],
            ],
            layers.advancement_window,
            UiQuad::rgba(255, 255, 255, 255),
        ));

        // 5. Canvas Viewport (Inner canvas: x = 9, y = 18, w = 234, h = 113)
        let canvas_x = origin_x + 9.0;
        let canvas_y = origin_y + 18.0;
        let canvas_w = 234.0;
        let canvas_h = 113.0;

        let base_x = canvas_x + 24.0 + self.pan_x;
        let base_y = canvas_y + canvas_h * 0.5 + self.pan_y;

        // 6. Draw Connecting Branch Lines between parent and child nodes
        for node in &self.nodes {
            if node.category != self.active_tab {
                continue;
            }
            let Some(parent_id) = &node.parent_id else {
                continue;
            };
            let Some(parent) = self.nodes.iter().find(|n| n.id == *parent_id) else {
                continue;
            };

            let p_cx = base_x + parent.x * 32.0 + 13.0;
            let p_cy = base_y + parent.y * 32.0 + 13.0;
            let n_cx = base_x + node.x * 32.0 + 13.0;
            let n_cy = base_y + node.y * 32.0 + 13.0;

            let line_color = if node.completed {
                UiQuad::rgba(220, 190, 40, 255)
            } else {
                UiQuad::rgba(70, 70, 75, 255)
            };

            // Horizontal segment from (p_cx, p_cy) to (n_cx, p_cy)
            let x0 = p_cx.min(n_cx).max(canvas_x);
            let x1 = p_cx.max(n_cx).min(canvas_x + canvas_w);
            if x1 > x0 && p_cy >= canvas_y && p_cy <= canvas_y + canvas_h {
                let px_lx = (x0 * scale_f).round() as i32;
                let px_ly = ((p_cy - 1.0) * scale_f).round() as i32;
                let px_lw = ((x1 - x0) * scale_f).round() as u16;
                let px_lh = (2.0 * scale_f).round() as u16;
                out.push(UiQuad::solid([px_lx, px_ly], [px_lw, px_lh], line_color));
            }

            // Vertical segment from (n_cx, p_cy) to (n_cx, n_cy)
            let y0 = p_cy.min(n_cy).max(canvas_y);
            let y1 = p_cy.max(n_cy).min(canvas_y + canvas_h);
            if y1 > y0 && n_cx >= canvas_x && n_cx <= canvas_x + canvas_w {
                let px_lx = ((n_cx - 1.0) * scale_f).round() as i32;
                let px_ly = (y0 * scale_f).round() as i32;
                let px_lw = (2.0 * scale_f).round() as u16;
                let px_lh = ((y1 - y0) * scale_f).round() as u16;
                out.push(UiQuad::solid([px_lx, px_ly], [px_lw, px_lh], line_color));
            }
        }

        // 7. Draw Nodes (Frames + Centered Icons)
        for node in &self.nodes {
            if node.category != self.active_tab {
                continue;
            }
            let nx = base_x + node.x * 32.0;
            let ny = base_y + node.y * 32.0;
            let frame_sz = 26.0;

            // Cull nodes completely outside canvas bounds
            if nx + frame_sz < canvas_x
                || nx > canvas_x + canvas_w
                || ny + frame_sz < canvas_y
                || ny > canvas_y + canvas_h
            {
                continue;
            }

            let px_nx = (nx * scale_f).round() as i32;
            let px_ny = (ny * scale_f).round() as i32;
            let px_nsz = (frame_sz * scale_f).round() as u16;

            // Frame background & border
            let (bg_color, border_color) = if node.completed {
                (
                    UiQuad::rgba(35, 30, 20, 240),
                    match node.frame {
                        UiAdvancementFrame::Challenge => UiQuad::rgba(255, 120, 240, 255),
                        UiAdvancementFrame::Goal => UiQuad::rgba(100, 230, 255, 255),
                        UiAdvancementFrame::Task => UiQuad::rgba(235, 205, 50, 255),
                    },
                )
            } else {
                (UiQuad::rgba(20, 20, 25, 240), UiQuad::rgba(75, 75, 82, 255))
            };

            // Background
            out.push(UiQuad::solid([px_nx, px_ny], [px_nsz, px_nsz], bg_color));

            // Frame border (2 physical pixels)
            let b_thick = (2.0 * scale_f).round() as u16;
            out.push(UiQuad::solid(
                [px_nx, px_ny],
                [px_nsz, b_thick],
                border_color,
            ));
            out.push(UiQuad::solid(
                [px_nx, px_ny + i32::from(px_nsz - b_thick)],
                [px_nsz, b_thick],
                border_color,
            ));
            out.push(UiQuad::solid(
                [px_nx, px_ny],
                [b_thick, px_nsz],
                border_color,
            ));
            out.push(UiQuad::solid(
                [px_nx + i32::from(px_nsz - b_thick), px_ny],
                [b_thick, px_nsz],
                border_color,
            ));

            // Centered 16x16 Item Icon
            if node.icon_item > 0 {
                let icon_x = ((nx + 5.0) * scale_f).round() as i32;
                let icon_y = ((ny + 5.0) * scale_f).round() as i32;
                let icon_sz = (16.0 * scale_f).round() as u16;
                let uv = item_icon_uv(node.icon_item);
                let icon_tint = if node.completed {
                    UiQuad::rgba(255, 255, 255, 255)
                } else {
                    UiQuad::rgba(120, 120, 130, 180)
                };
                out.push(UiQuad::sprite(
                    [icon_x, icon_y],
                    [icon_sz, icon_sz],
                    [uv[0], uv[1]],
                    [uv[2], uv[3]],
                    layers.item_icons,
                    icon_tint,
                ));
            }
        }

        // 8. Hover Tooltip
        if let Some(hovered) = &self.hovered_node {
            let (tw_title, _) = font.measure_text(&hovered.title);
            let (tw_desc, _) = font.measure_text(&hovered.description);
            let tip_w = tw_title.max(tw_desc).max(120.0) + 16.0;
            let tip_h = 36.0;

            let tip_x = ((origin_x + 12.0) * scale_f).round() as i32;
            let tip_y = ((origin_y + 140.0 - 42.0) * scale_f).round() as i32;
            let px_tw = (tip_w * scale_f).round() as u16;
            let px_th = (tip_h * scale_f).round() as u16;

            // Dark translucent card
            out.push(UiQuad::solid(
                [tip_x, tip_y],
                [px_tw, px_th],
                UiQuad::rgba(15, 12, 22, 245),
            ));
            let tb_thick = (1.5 * scale_f).round() as u16;
            out.push(UiQuad::solid(
                [tip_x, tip_y],
                [px_tw, tb_thick],
                UiQuad::rgba(70, 50, 110, 255),
            ));

            // Title
            let title_color = match hovered.frame {
                UiAdvancementFrame::Challenge => UiQuad::rgba(255, 110, 240, 255),
                UiAdvancementFrame::Goal => UiQuad::rgba(100, 240, 255, 255),
                UiAdvancementFrame::Task => UiQuad::rgba(255, 255, 85, 255),
            };
            font.layout_text(
                &hovered.title,
                origin_x + 18.0,
                origin_y + 140.0 - 38.0,
                title_color,
                true,
                scale,
                out,
            );

            // Description
            font.layout_text(
                &hovered.description,
                origin_x + 18.0,
                origin_y + 140.0 - 26.0,
                UiQuad::rgba(200, 200, 205, 255),
                true,
                scale,
                out,
            );

            // Status tag
            let (status_text, status_color) = if hovered.completed {
                ("[Completed]", UiQuad::rgba(85, 255, 85, 255))
            } else {
                ("[Incomplete]", UiQuad::rgba(180, 180, 180, 200))
            };
            let (tag_w, _) = font.measure_text(status_text);
            font.layout_text(
                status_text,
                origin_x + 12.0 + tip_w - tag_w - 6.0,
                origin_y + 140.0 - 38.0,
                status_color,
                true,
                scale,
                out,
            );
        }
    }
}
