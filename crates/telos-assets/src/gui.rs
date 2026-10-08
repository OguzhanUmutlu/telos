//! Data-driven GUI style sheets, container layouts, and 9-slice border definitions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

const fn default_slot_size() -> [u16; 2] {
    [16, 16]
}

const fn default_hotbar_size() -> [u16; 2] {
    [182, 22]
}

const fn default_hotbar_slot_origin() -> [i32; 2] {
    [3, 3]
}

const fn default_hotbar_slot_stride() -> i32 {
    20
}

const fn default_selection_size() -> [u16; 2] {
    [24, 23]
}

const fn default_selection_offset() -> [i32; 2] {
    [-1, -1]
}

const fn default_xp_bar_size() -> [u16; 2] {
    [182, 5]
}

const fn default_xp_bar_y_offset() -> i32 {
    29
}

/// Border specification for 9-slice scalable GUI frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NineSliceBorderDef {
    /// Left border in source pixels.
    pub left: u8,
    /// Top border in source pixels.
    pub top: u8,
    /// Right border in source pixels.
    pub right: u8,
    /// Bottom border in source pixels.
    pub bottom: u8,
    /// Whether to stretch the inner region instead of tiling.
    #[serde(default)]
    pub stretch_inner: bool,
}

impl NineSliceBorderDef {
    /// Creates a new 9-slice border with explicit per-edge dimensions.
    #[must_use]
    pub const fn new(left: u8, top: u8, right: u8, bottom: u8) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            stretch_inner: false,
        }
    }

    /// Creates a uniform 9-slice border where all 4 edges share the same width.
    #[must_use]
    pub const fn uniform(border: u8) -> Self {
        Self::new(border, border, border, border)
    }

    /// Sets whether the inner region should stretch rather than tile.
    #[must_use]
    pub const fn with_stretch_inner(mut self, stretch: bool) -> Self {
        self.stretch_inner = stretch;
        self
    }

    /// Returns the border edges as `[left, top, right, bottom]`.
    #[must_use]
    pub const fn as_array(&self) -> [u8; 4] {
        [self.left, self.top, self.right, self.bottom]
    }
}

/// Data-driven slot layout position and bounds inside a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotLayoutDef {
    /// Slot index inside the container session.
    pub index: usize,
    /// X coordinate relative to container top-left in GUI pixels.
    pub x: i32,
    /// Y coordinate relative to container top-left in GUI pixels.
    pub y: i32,
    /// Width and height of the slot in GUI pixels (default: [16, 16]).
    #[serde(default = "default_slot_size")]
    pub size: [u16; 2],
}

impl SlotLayoutDef {
    /// Creates a standard 16x16 slot at the specified relative position.
    #[must_use]
    pub const fn new(index: usize, x: i32, y: i32) -> Self {
        Self {
            index,
            x,
            y,
            size: default_slot_size(),
        }
    }

    /// Creates a custom-sized slot at the specified relative position.
    #[must_use]
    pub const fn with_size(index: usize, x: i32, y: i32, w: u16, h: u16) -> Self {
        Self {
            index,
            x,
            y,
            size: [w, h],
        }
    }
}

/// Complete data-driven container window layout definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerLayoutDef {
    /// Width of the container window in GUI pixels.
    pub width: u32,
    /// Height of the container window in GUI pixels.
    pub height: u32,
    /// Top-left relative position of the container primary title label.
    pub title_pos: [i32; 2],
    /// Optional relative position of the secondary "Inventory" label.
    #[serde(default)]
    pub inventory_title_pos: Option<[i32; 2]>,
    /// Optional custom background texture path or sprite identifier.
    #[serde(default)]
    pub background_texture: Option<String>,
    /// Optional 9-slice border configuration for scalable window frames.
    #[serde(default)]
    pub nine_slice: Option<NineSliceBorderDef>,
    /// Slot positions and layout.
    #[serde(default)]
    pub slots: Vec<SlotLayoutDef>,
    /// Optional furnace lit flame sprite position `[x, y]` in GUI pixels.
    #[serde(default)]
    pub flame_pos: Option<[i32; 2]>,
    /// Optional furnace / crafting progress arrow sprite position `[x, y]` in GUI pixels.
    #[serde(default)]
    pub arrow_pos: Option<[i32; 2]>,
}

impl ContainerLayoutDef {
    /// Returns the container-relative `[x, y]` coordinates of a slot by its index.
    #[must_use]
    pub fn slot_pos(&self, slot: usize) -> Option<[i32; 2]> {
        self.slots
            .iter()
            .find(|s| s.index == slot)
            .map(|s| [s.x, s.y])
    }

    /// Returns the slot layout definition by slot index.
    #[must_use]
    pub fn slot_def(&self, slot: usize) -> Option<&SlotLayoutDef> {
        self.slots.iter().find(|s| s.index == slot)
    }

    /// Default Minecraft standard survival player inventory layout (176x166, 46 slots).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn default_inventory() -> Self {
        let mut slots = Vec::with_capacity(46);

        // Hotbar (slots 0..=8): y = 142, x = 8 + col * 18
        for col in 0..9 {
            slots.push(SlotLayoutDef::new(col, 8 + (col as i32) * 18, 142));
        }

        // Main storage (slots 9..=35): 3 rows of 9 starting at y = 84
        for idx in 0..27 {
            let col = (idx % 9) as i32;
            let row = (idx / 9) as i32;
            slots.push(SlotLayoutDef::new(9 + idx, 8 + col * 18, 84 + row * 18));
        }

        // Armor (slots 36..=39): Helmet (36), Chestplate (37), Leggings (38), Boots (39)
        for row in 0..4 {
            slots.push(SlotLayoutDef::new(36 + row, 8, 8 + (row as i32) * 18));
        }

        // 2x2 Crafting inputs (slots 40..=43): 2 rows of 2 at x = 98, y = 18
        for idx in 0..4 {
            let col = (idx % 2) as i32;
            let row = (idx / 2) as i32;
            slots.push(SlotLayoutDef::new(40 + idx, 98 + col * 18, 18 + row * 18));
        }

        // Crafting result output (slot 44)
        slots.push(SlotLayoutDef::new(44, 154, 28));

        // Offhand slot (slot 45)
        slots.push(SlotLayoutDef::new(45, 77, 62));

        Self {
            width: 176,
            height: 166,
            title_pos: [8, 6],
            inventory_title_pos: None,
            background_texture: Some("textures/gui/container/inventory.png".to_string()),
            nine_slice: None,
            slots,
            flame_pos: None,
            arrow_pos: None,
        }
    }

    /// Default Minecraft standard dual chest container layout (176x166, 63 slots).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn default_chest() -> Self {
        let mut slots = Vec::with_capacity(63);

        // Chest storage (slots 0..=26): 3 rows of 9 starting at y = 18, x = 8
        for slot in 0..27 {
            let col = (slot % 9) as i32;
            let row = (slot / 9) as i32;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 18 + row * 18));
        }

        // Player main storage (slots 27..=53): 3 rows of 9 starting at y = 84, x = 8
        for slot in 27..54 {
            let idx = (slot - 27) as i32;
            let col = idx % 9;
            let row = idx / 9;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 84 + row * 18));
        }

        // Player hotbar (slots 54..=62): 1 row of 9 at y = 142, x = 8
        for slot in 54..63 {
            let col = (slot - 54) as i32;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 142));
        }

        Self {
            width: 176,
            height: 166,
            title_pos: [8, 6],
            inventory_title_pos: Some([8, 72]),
            background_texture: Some("textures/gui/container/generic_54.png".to_string()),
            nine_slice: None,
            slots,
            flame_pos: None,
            arrow_pos: None,
        }
    }

    /// Default Minecraft standard dual furnace container layout (176x166, 39 slots).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn default_furnace() -> Self {
        let mut slots = Vec::with_capacity(39);

        // Furnace slots (0: Input, 1: Fuel, 2: Output)
        slots.push(SlotLayoutDef::new(0, 56, 17));
        slots.push(SlotLayoutDef::new(1, 56, 53));
        slots.push(SlotLayoutDef::new(2, 116, 35));

        // Player main storage (slots 3..=29): 3 rows of 9 starting at y = 84, x = 8
        for slot in 3..=29 {
            let idx = (slot - 3) as i32;
            let col = idx % 9;
            let row = idx / 9;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 84 + row * 18));
        }

        // Player hotbar (slots 30..=38): 1 row of 9 at y = 142, x = 8
        for slot in 30..=38 {
            let col = (slot - 30) as i32;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 142));
        }

        Self {
            width: 176,
            height: 166,
            title_pos: [8, 6],
            inventory_title_pos: Some([8, 74]),
            background_texture: Some("textures/gui/container/furnace.png".to_string()),
            nine_slice: None,
            slots,
            flame_pos: Some([56, 36]),
            arrow_pos: Some([79, 34]),
        }
    }

    /// Default Minecraft standard dual crafting table container layout (176x166, 46 slots).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn default_crafting_table() -> Self {
        let mut slots = Vec::with_capacity(46);

        // Slot 0: Output preview slot
        slots.push(SlotLayoutDef::new(0, 124, 35));

        // Slots 1..=9: 3x3 crafting grid
        for slot in 1..=9 {
            let idx = (slot - 1) as i32;
            let col = idx % 3;
            let row = idx / 3;
            slots.push(SlotLayoutDef::new(slot, 30 + col * 18, 17 + row * 18));
        }

        // Slots 10..=36: Player main storage (3 rows of 9)
        for slot in 10..=36 {
            let idx = (slot - 10) as i32;
            let col = idx % 9;
            let row = idx / 9;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 84 + row * 18));
        }

        // Slots 37..=45: Player hotbar (1 row of 9)
        for slot in 37..=45 {
            let col = (slot - 37) as i32;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 142));
        }

        Self {
            width: 176,
            height: 166,
            title_pos: [28, 6],
            inventory_title_pos: Some([8, 74]),
            background_texture: Some("textures/gui/container/crafting_table.png".to_string()),
            nine_slice: None,
            slots,
            flame_pos: None,
            arrow_pos: Some([90, 35]),
        }
    }

    /// Default Minecraft standard dual anvil container layout (176x166, 39 slots).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn default_anvil() -> Self {
        let mut slots = Vec::with_capacity(39);

        // Slot 0: Left input slot
        slots.push(SlotLayoutDef::new(0, 27, 47));

        // Slot 1: Right input slot (sacrifice/book)
        slots.push(SlotLayoutDef::new(1, 76, 47));

        // Slot 2: Output result slot
        slots.push(SlotLayoutDef::new(2, 134, 47));

        // Slots 3..=29: Player main storage (3 rows of 9)
        for slot in 3..=29 {
            let idx = (slot - 3) as i32;
            let col = idx % 9;
            let row = idx / 9;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 84 + row * 18));
        }

        // Slots 30..=38: Player hotbar (1 row of 9)
        for slot in 30..=38 {
            let col = (slot - 30) as i32;
            slots.push(SlotLayoutDef::new(slot, 8 + col * 18, 142));
        }

        Self {
            width: 176,
            height: 166,
            title_pos: [60, 6],
            inventory_title_pos: Some([8, 74]),
            background_texture: Some("textures/gui/container/anvil.png".to_string()),
            nine_slice: None,
            slots,
            flame_pos: None,
            arrow_pos: Some([99, 45]),
        }
    }

    /// Parses a `ContainerLayoutDef` from a JSON string.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str::<Self>(json)
    }

    /// Parses a `ContainerLayoutDef` from a RON string.
    ///
    /// # Errors
    /// Returns a `ron::error::SpannedError` if parsing fails.
    pub fn from_ron_str(ron_str: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str::<Self>(ron_str)
    }
}

/// HUD theme and positioning overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HudThemeDef {
    /// Hotbar vertical offset from viewport bottom in GUI pixels.
    #[serde(default)]
    pub hotbar_y_offset: i32,
    /// Hotbar dimensions in GUI pixels.
    #[serde(default = "default_hotbar_size")]
    pub hotbar_size: [u16; 2],
    /// Origin of slot 0 relative to hotbar top-left in GUI pixels.
    #[serde(default = "default_hotbar_slot_origin")]
    pub hotbar_slot_origin: [i32; 2],
    /// Stride between consecutive hotbar slots in GUI pixels.
    #[serde(default = "default_hotbar_slot_stride")]
    pub hotbar_slot_stride: i32,
    /// Hotbar selection indicator size in GUI pixels.
    #[serde(default = "default_selection_size")]
    pub selection_size: [u16; 2],
    /// Hotbar selection indicator offset relative to active slot.
    #[serde(default = "default_selection_offset")]
    pub selection_offset: [i32; 2],
    /// Experience bar size in GUI pixels.
    #[serde(default = "default_xp_bar_size")]
    pub xp_bar_size: [u16; 2],
    /// Experience bar vertical offset from viewport bottom in GUI pixels.
    #[serde(default = "default_xp_bar_y_offset")]
    pub xp_bar_y_offset: i32,
}

impl Default for HudThemeDef {
    fn default() -> Self {
        Self {
            hotbar_y_offset: 0,
            hotbar_size: default_hotbar_size(),
            hotbar_slot_origin: default_hotbar_slot_origin(),
            hotbar_slot_stride: default_hotbar_slot_stride(),
            selection_size: default_selection_size(),
            selection_offset: default_selection_offset(),
            xp_bar_size: default_xp_bar_size(),
            xp_bar_y_offset: default_xp_bar_y_offset(),
        }
    }
}

/// Complete GUI style sheet containing container and HUD layouts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuiStyleSheet {
    /// Survival player inventory container layout.
    #[serde(default = "ContainerLayoutDef::default_inventory")]
    pub inventory: ContainerLayoutDef,
    /// Chest container layout.
    #[serde(default = "ContainerLayoutDef::default_chest")]
    pub chest: ContainerLayoutDef,
    /// Furnace container layout.
    #[serde(default = "ContainerLayoutDef::default_furnace")]
    pub furnace: ContainerLayoutDef,
    /// Crafting table container layout.
    #[serde(default = "ContainerLayoutDef::default_crafting_table")]
    pub crafting_table: ContainerLayoutDef,
    /// Anvil container layout.
    #[serde(default = "ContainerLayoutDef::default_anvil")]
    pub anvil: ContainerLayoutDef,
    /// Additional custom mod or pack container layouts indexed by identifier.
    #[serde(default)]
    pub custom_containers: HashMap<String, ContainerLayoutDef>,
    /// In-game HUD styling and layout parameters.
    #[serde(default)]
    pub hud: HudThemeDef,
    /// 9-slice borders indexed by sprite name (e.g. `"widget/button"`, `"container/slot"`).
    #[serde(default)]
    pub nine_slices: HashMap<String, NineSliceBorderDef>,
}

impl Default for GuiStyleSheet {
    fn default() -> Self {
        Self {
            inventory: ContainerLayoutDef::default_inventory(),
            chest: ContainerLayoutDef::default_chest(),
            furnace: ContainerLayoutDef::default_furnace(),
            crafting_table: ContainerLayoutDef::default_crafting_table(),
            anvil: ContainerLayoutDef::default_anvil(),
            custom_containers: HashMap::new(),
            hud: HudThemeDef::default(),
            nine_slices: HashMap::new(),
        }
    }
}

impl GuiStyleSheet {
    /// Parses a `GuiStyleSheet` from a JSON string.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str::<Self>(json)
    }

    /// Parses a `GuiStyleSheet` from a RON string.
    ///
    /// # Errors
    /// Returns a `ron::error::SpannedError` if parsing fails.
    pub fn from_ron_str(ron_str: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str::<Self>(ron_str)
    }

    /// Overrides or adds a container layout by identifier.
    pub fn override_container(&mut self, name: &str, layout: ContainerLayoutDef) {
        match name {
            "inventory" => self.inventory = layout,
            "chest" => self.chest = layout,
            "furnace" => self.furnace = layout,
            "crafting_table" => self.crafting_table = layout,
            other => {
                self.custom_containers.insert(other.to_string(), layout);
            }
        }
    }

    /// Retrieves a container layout by identifier.
    #[must_use]
    pub fn get_container(&self, name: &str) -> Option<&ContainerLayoutDef> {
        match name {
            "inventory" => Some(&self.inventory),
            "chest" => Some(&self.chest),
            "furnace" => Some(&self.furnace),
            "crafting_table" => Some(&self.crafting_table),
            other => self.custom_containers.get(other),
        }
    }

    /// Merges another style sheet into `self`, taking overrides from `other`.
    pub fn merge(&mut self, other: Self) {
        if other.inventory != ContainerLayoutDef::default_inventory() {
            self.inventory = other.inventory;
        }
        if other.chest != ContainerLayoutDef::default_chest() {
            self.chest = other.chest;
        }
        if other.furnace != ContainerLayoutDef::default_furnace() {
            self.furnace = other.furnace;
        }
        if other.crafting_table != ContainerLayoutDef::default_crafting_table() {
            self.crafting_table = other.crafting_table;
        }
        if other.hud != HudThemeDef::default() {
            self.hud = other.hud;
        }
        self.custom_containers.extend(other.custom_containers);
        self.nine_slices.extend(other.nine_slices);
    }
}
