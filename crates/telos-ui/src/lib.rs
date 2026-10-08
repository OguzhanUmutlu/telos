//! Retained UI framework, taffy layout, bitmap font engine, and in-game HUD.
//!
//! This crate contains pure CPU UI logic with zero Vulkan or windowing dependencies,
//! ensuring complete headless testability. It produces GPU-ready 48-byte `UiQuad`
//! instances for single-draw-call instanced rendering.

pub mod chat;
pub mod container;
pub mod font;
pub mod frame;
pub mod hud;
pub mod inventory;
pub mod menu;
pub mod quad;
pub mod scale;
pub mod server_list;
pub mod settings;
pub mod tree;

pub use chat::{ChatEntry, ChatHudState, render_chat_hud};
pub use container::{
    CHEST_CONTAINER_SLOTS, DUAL_CONTAINER_SLOT_COUNT, chest_slot_at_pos, chest_slot_pos,
    render_chest_container,
};
pub use font::{BitmapFont, GlyphMetrics};
pub use frame::UiFrame;
pub use hud::{HudEffectDisplay, HudState, UiLayers, render_hud};
pub use inventory::{
    CONTAINER_HEIGHT, CONTAINER_WIDTH, INVENTORY_SLOT_COUNT, UiSlotItem, item_icon_uv,
    render_inventory_screen, slot_at_pos, slot_pos,
};
pub use menu::{
    MainMenuAction, MainMenuScreen, MenuButton, MenuSlider, MenuTextInput, PauseMenuAction,
    PauseMenuScreen, SettingsScreen, SettingsTab, WorldCreateAction, WorldCreateWizard, WorldEntry,
    WorldSelectAction, WorldSelectScreen,
};
pub use quad::{QuadKind, UiQuad};
pub use scale::{compute_gui_scale, snap_to_physical, to_physical_pixels};
pub use server_list::{ServerEntry, ServerListScreen, render_server_list};
pub use settings::{AudioSettings, ControlSettings, GameSettings, GameplaySettings, VideoSettings};
pub use tree::{DirtyFlags, NodeId, UiTree, WidgetKind, WidgetNode};
