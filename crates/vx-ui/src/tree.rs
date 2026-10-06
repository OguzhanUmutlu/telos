//! Retained UI widget tree, generational handle store, and Taffy layout integration.

use bitflags::bitflags;
use slotmap::{SlotMap, new_key_type};
use taffy::{AvailableSpace, NodeId as TaffyNodeId, Size, Style, TaffyTree};

use crate::font::BitmapFont;
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;

new_key_type! {
    /// Generational slotmap handle identifying a widget node in `UiTree`.
    pub struct NodeId;
}

bitflags! {
    /// Invalidation and dirty flags for retained nodes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct DirtyFlags: u8 {
        /// Layout subtree is dirty and needs Taffy recomputation.
        const LAYOUT = 0x01;
        /// Visual appearance is dirty and needs re-tessellation.
        const PAINT  = 0x02;
    }
}

/// Category of UI widget.
#[derive(Debug, Clone, PartialEq)]
pub enum WidgetKind {
    /// Empty container for flex/grid layout and grouping.
    Container,
    /// Solid background rectangle.
    Solid {
        /// RGBA8 packed tint color.
        color: u32,
    },
    /// Static sprite with fixed UV coordinates in atlas.
    Sprite {
        /// UV min coordinates.
        uv_min: [f32; 2],
        /// UV max coordinates.
        uv_max: [f32; 2],
        /// Texture array layer.
        layer: u32,
        /// RGBA8 tint color.
        color: u32,
    },
    /// Formatted text string.
    Text {
        /// Display text.
        text: String,
        /// Default text color.
        color: u32,
        /// Render drop shadow behind text.
        shadow: bool,
    },
}

/// Retained node structure in the `UiTree`.
pub struct WidgetNode {
    /// Category and parameters of widget.
    pub kind: WidgetKind,
    /// Taffy layout node handle.
    pub taffy_node: TaffyNodeId,
    /// Parent widget node.
    pub parent: Option<NodeId>,
    /// Child widget nodes in submission order.
    pub children: Vec<NodeId>,
    /// Reactive dirty tracking flags.
    pub flags: DirtyFlags,
    /// Visible flag (skip layout/render when false).
    pub visible: bool,
}

/// Retained widget tree managing layout computation and quad emission.
pub struct UiTree {
    /// Generational node store.
    nodes: SlotMap<NodeId, WidgetNode>,
    /// Taffy flexbox and block layout engine.
    taffy: TaffyTree<()>,
    /// Root node of the tree.
    root: NodeId,
    /// Last computed physical dimensions.
    dimensions: [u32; 2],
    /// Active GUI scale.
    pub gui_scale: u32,
}

impl UiTree {
    /// Creates a new `UiTree` with root container.
    #[must_use]
    pub fn new(gui_scale: u32) -> Self {
        let mut taffy = TaffyTree::new();
        let root_style = Style {
            size: Size {
                width: taffy::Dimension::percent(1.0),
                height: taffy::Dimension::percent(1.0),
            },
            ..Default::default()
        };
        let root_taffy = taffy
            .new_leaf(root_style)
            .expect("Failed to initialize Taffy root");

        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(WidgetNode {
            kind: WidgetKind::Container,
            taffy_node: root_taffy,
            parent: None,
            children: Vec::new(),
            flags: DirtyFlags::all(),
            visible: true,
        });

        Self {
            nodes,
            taffy,
            root,
            dimensions: [0, 0],
            gui_scale,
        }
    }

    /// Accesses the root node ID.
    #[must_use]
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// Inserts a new child widget under `parent`.
    pub fn add_child(&mut self, parent: NodeId, kind: WidgetKind, style: Style) -> NodeId {
        let taffy_node = self
            .taffy
            .new_leaf(style)
            .expect("Failed to allocate Taffy node");

        let parent_taffy = self.nodes[parent].taffy_node;
        self.taffy
            .add_child(parent_taffy, taffy_node)
            .expect("Failed to add Taffy child");

        let child_id = self.nodes.insert(WidgetNode {
            kind,
            taffy_node,
            parent: Some(parent),
            children: Vec::new(),
            flags: DirtyFlags::all(),
            visible: true,
        });

        self.nodes[parent].children.push(child_id);
        self.nodes[parent].flags |= DirtyFlags::LAYOUT;

        child_id
    }

    /// Sets node visibility.
    pub fn set_visible(&mut self, node: NodeId, visible: bool) {
        if let Some(n) = self.nodes.get_mut(node)
            && n.visible != visible
        {
            n.visible = visible;
            n.flags |= DirtyFlags::all();
        }
    }

    /// Computes layout for the given physical window size.
    pub fn compute_layout(&mut self, width: u32, height: u32) {
        self.dimensions = [width, height];
        let root_taffy = self.nodes[self.root].taffy_node;

        let available_w = (width as f32) / (self.gui_scale as f32);
        let available_h = (height as f32) / (self.gui_scale as f32);

        let available_space = Size {
            width: AvailableSpace::Definite(available_w),
            height: AvailableSpace::Definite(available_h),
        };

        self.taffy
            .compute_layout(root_taffy, available_space)
            .expect("Failed to compute UI layout");
    }

    /// Tessellates the retained tree into GPU quad instances.
    pub fn tessellate(&mut self, font: &BitmapFont, out: &mut Vec<UiQuad>) {
        let root = self.root;
        self.tessellate_node(root, 0.0, 0.0, font, out);
    }

    /// Computes layout and tessellates all visible nodes into the output quad buffer.
    pub fn update(&mut self, width: u32, height: u32, font: &BitmapFont, out: &mut Vec<UiQuad>) {
        self.compute_layout(width, height);
        self.tessellate(font, out);
    }

    /// Checks if any node in the tree has dirty flags set.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.nodes.values().any(|n| !n.flags.is_empty())
    }

    fn tessellate_node(
        &self,
        node_id: NodeId,
        parent_x: f32,
        parent_y: f32,
        font: &BitmapFont,
        out: &mut Vec<UiQuad>,
    ) {
        let node = match self.nodes.get(node_id) {
            Some(n) if n.visible => n,
            _ => return,
        };

        let Ok(layout) = self.taffy.layout(node.taffy_node) else {
            return;
        };

        let cur_x = parent_x + layout.location.x;
        let cur_y = parent_y + layout.location.y;

        let px_x = snap_to_physical(cur_x, self.gui_scale);
        let px_y = snap_to_physical(cur_y, self.gui_scale);
        let px_w = snap_to_physical(layout.size.width, self.gui_scale).max(0) as u16;
        let px_h = snap_to_physical(layout.size.height, self.gui_scale).max(0) as u16;

        match &node.kind {
            WidgetKind::Container => {}
            WidgetKind::Solid { color } => {
                out.push(UiQuad::solid([px_x, px_y], [px_w, px_h], *color));
            }
            WidgetKind::Sprite {
                uv_min,
                uv_max,
                layer,
                color,
            } => {
                out.push(UiQuad::sprite(
                    [px_x, px_y],
                    [px_w, px_h],
                    *uv_min,
                    *uv_max,
                    *layer,
                    *color,
                ));
            }
            WidgetKind::Text {
                text,
                color,
                shadow,
            } => {
                font.layout_text(text, cur_x, cur_y, *color, *shadow, self.gui_scale, out);
            }
        }

        for &child in &node.children {
            self.tessellate_node(child, cur_x, cur_y, font, out);
        }
    }
}
