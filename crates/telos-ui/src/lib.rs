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
pub mod keybinds;
pub mod menu;
pub mod quad;
pub mod scale;
pub mod server_list;
pub mod settings;
pub mod style;
pub mod tree;

pub use chat::{ChatEntry, ChatHudState, render_chat_hud};
pub use container::{
    ANVIL_CONTAINER_SLOTS, CHEST_CONTAINER_SLOTS, CRAFTING_TABLE_CONTAINER_SLOTS,
    DUAL_ANVIL_SLOT_COUNT, DUAL_CONTAINER_SLOT_COUNT, DUAL_CRAFTING_TABLE_SLOT_COUNT,
    DUAL_FURNACE_SLOT_COUNT, FURNACE_CONTAINER_SLOTS, anvil_slot_at_pos, anvil_slot_at_pos_styled,
    anvil_slot_pos, chest_slot_at_pos, chest_slot_at_pos_styled, chest_slot_pos,
    crafting_table_slot_at_pos, crafting_table_slot_at_pos_styled, crafting_table_slot_pos,
    furnace_slot_at_pos, furnace_slot_at_pos_styled, furnace_slot_pos, render_anvil_container,
    render_anvil_container_styled, render_chest_container, render_chest_container_styled,
    render_crafting_table_container, render_crafting_table_container_styled,
    render_furnace_container, render_furnace_container_styled,
};
pub use font::{BitmapFont, GlyphMetrics};
pub use frame::UiFrame;
pub use hud::{HudEffectDisplay, HudState, ToastState, UiLayers, render_hud, render_hud_styled};
pub use inventory::{
    CONTAINER_HEIGHT, CONTAINER_WIDTH, INVENTORY_SLOT_COUNT, UiSlotItem,
    enchantment_name_and_level, item_icon_uv, render_inventory_screen,
    render_inventory_screen_styled, render_item_tooltip, slot_at_pos,
    slot_at_pos_styled as inventory_slot_at_pos_styled, slot_pos,
};
pub use keybinds::{InputKey, KeyAction, KeyCategory, KeybindSettings};
pub use menu::{
    AdvancementsScreen, ButtonStyle, CustomWidgetState, MainMenuAction, MainMenuScreen, MenuButton,
    MenuSlider, MenuTextInput, ModalFormAction, ModalFormScreen, PauseMenuAction, PauseMenuScreen,
    SettingsScreen, SettingsTab, UiAdvancementCategory, UiAdvancementFrame, UiAdvancementNode,
    WorldCreateAction, WorldCreateWizard, WorldDeletePrompt, WorldEntry, WorldSelectAction,
    WorldSelectScreen,
};

pub use quad::{QuadKind, UiQuad};
pub use scale::{compute_gui_scale, snap_to_physical, to_physical_pixels};
pub use server_list::{ServerEntry, ServerListScreen, render_server_list};
pub use settings::{AudioSettings, ControlSettings, GameSettings, GameplaySettings, VideoSettings};
pub use style::{
    ContainerLayoutDef, GuiStyleSheet, HudThemeDef, NineSliceBorderDef, SlotLayoutDef,
    container_bounds, slot_at_pos_styled,
};
pub use tree::{DirtyFlags, NodeId, UiTree, WidgetKind, WidgetNode};
