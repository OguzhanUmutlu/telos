//! Central keybind registry and input key mapping.
//!
//! Provides customizable bindings for movement, gameplay, UI, and camera controls,
//! with conflict detection and persistence to `settings.toml`.

use serde::{Deserialize, Serialize};

/// High-level game actions that can be bound to physical keyboard keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyAction {
    /// Move player forward.
    Forward,
    /// Move player backward.
    Backward,
    /// Strafe player left.
    Left,
    /// Strafe player right.
    Right,
    /// Jump or ascend in flight.
    Jump,
    /// Sneak / edge-guard or descend in flight.
    Sneak,
    /// Sprint boost.
    Sprint,
    /// Open/close survival inventory.
    Inventory,
    /// Open/close advancements tree screen (L).
    Advancements,
    /// Drop active hotbar item stack.
    Drop,
    /// Open multiplayer / local chat box.
    Chat,
    /// Open command prompt with initial slash.
    Command,
    /// Capture in-game screenshot (F2).
    Screenshot,
    /// Toggle F3 performance and debug overlay.
    ToggleF3,
    /// Toggle or cycle game mode (Survival/Creative/Adventure/Spectator).
    ToggleGameMode,
    /// Hot-reload SPIR-V / GLSL shaders (F5).
    ReloadShaders,
    /// Select hotbar slot 1.
    Hotbar1,
    /// Select hotbar slot 2.
    Hotbar2,
    /// Select hotbar slot 3.
    Hotbar3,
    /// Select hotbar slot 4.
    Hotbar4,
    /// Select hotbar slot 5.
    Hotbar5,
    /// Select hotbar slot 6.
    Hotbar6,
    /// Select hotbar slot 7.
    Hotbar7,
    /// Select hotbar slot 8.
    Hotbar8,
    /// Select hotbar slot 9.
    Hotbar9,
    /// Push-to-talk voice transmission.
    PushToTalk,
}

/// Category grouping for display in settings panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCategory {
    /// Walking, running, jumping, and sneaking.
    Movement,
    /// Inventory, dropping, chat, and commands.
    Gameplay,
    /// Direct slot selection 1 through 9.
    Hotbar,
    /// Multiplayer, voice chat, and communication.
    Multiplayer,
    /// Debug overlays, screenshots, game modes, and tools.
    System,
}

impl KeyAction {
    /// Returns the user-facing display label for this action.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Forward => "Walk Forward",
            Self::Backward => "Walk Backward",
            Self::Left => "Strafe Left",
            Self::Right => "Strafe Right",
            Self::Jump => "Jump / Fly Up",
            Self::Sneak => "Sneak / Fly Down",
            Self::Sprint => "Sprint",
            Self::Inventory => "Open Inventory",
            Self::Advancements => "Advancements (L)",
            Self::Drop => "Drop Item",
            Self::Chat => "Open Chat",
            Self::Command => "Open Command",
            Self::Screenshot => "Take Screenshot",
            Self::ToggleF3 => "Debug Screen (F3)",
            Self::ToggleGameMode => "Cycle Game Mode (F4)",
            Self::ReloadShaders => "Reload Shaders (F5)",
            Self::Hotbar1 => "Hotbar Slot 1",
            Self::Hotbar2 => "Hotbar Slot 2",
            Self::Hotbar3 => "Hotbar Slot 3",
            Self::Hotbar4 => "Hotbar Slot 4",
            Self::Hotbar5 => "Hotbar Slot 5",
            Self::Hotbar6 => "Hotbar Slot 6",
            Self::Hotbar7 => "Hotbar Slot 7",
            Self::Hotbar8 => "Hotbar Slot 8",
            Self::Hotbar9 => "Hotbar Slot 9",
            Self::PushToTalk => "Push to Talk (V)",
        }
    }

    /// Returns the category for this action.
    #[must_use]
    pub const fn category(&self) -> KeyCategory {
        match self {
            Self::Forward
            | Self::Backward
            | Self::Left
            | Self::Right
            | Self::Jump
            | Self::Sneak
            | Self::Sprint => KeyCategory::Movement,
            Self::Inventory | Self::Advancements | Self::Drop | Self::Chat | Self::Command => {
                KeyCategory::Gameplay
            }
            Self::Hotbar1
            | Self::Hotbar2
            | Self::Hotbar3
            | Self::Hotbar4
            | Self::Hotbar5
            | Self::Hotbar6
            | Self::Hotbar7
            | Self::Hotbar8
            | Self::Hotbar9 => KeyCategory::Hotbar,
            Self::PushToTalk => KeyCategory::Multiplayer,
            Self::Screenshot | Self::ToggleF3 | Self::ToggleGameMode | Self::ReloadShaders => {
                KeyCategory::System
            }
        }
    }

    /// Returns an ordered list of all standard actions.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Forward,
            Self::Backward,
            Self::Left,
            Self::Right,
            Self::Jump,
            Self::Sneak,
            Self::Sprint,
            Self::Inventory,
            Self::Advancements,
            Self::Drop,
            Self::Chat,
            Self::Command,
            Self::PushToTalk,
            Self::Hotbar1,
            Self::Hotbar2,
            Self::Hotbar3,
            Self::Hotbar4,
            Self::Hotbar5,
            Self::Hotbar6,
            Self::Hotbar7,
            Self::Hotbar8,
            Self::Hotbar9,
            Self::Screenshot,
            Self::ToggleF3,
            Self::ToggleGameMode,
            Self::ReloadShaders,
        ]
    }
}

/// Lightweight, serializable representation of physical keyboard keys.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InputKey {
    #[default]
    None,
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    Space,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    Escape,
    Enter,
    Tab,
    Backspace,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Slash,
    Backslash,
    Minus,
    Equal,
    Semicolon,
    Quote,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
}

impl InputKey {
    /// Returns the concise user-facing name for this key.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::KeyA => "A",
            Self::KeyB => "B",
            Self::KeyC => "C",
            Self::KeyD => "D",
            Self::KeyE => "E",
            Self::KeyF => "F",
            Self::KeyG => "G",
            Self::KeyH => "H",
            Self::KeyI => "I",
            Self::KeyJ => "J",
            Self::KeyK => "K",
            Self::KeyL => "L",
            Self::KeyM => "M",
            Self::KeyN => "N",
            Self::KeyO => "O",
            Self::KeyP => "P",
            Self::KeyQ => "Q",
            Self::KeyR => "R",
            Self::KeyS => "S",
            Self::KeyT => "T",
            Self::KeyU => "U",
            Self::KeyV => "V",
            Self::KeyW => "W",
            Self::KeyX => "X",
            Self::KeyY => "Y",
            Self::KeyZ => "Z",
            Self::Digit0 => "0",
            Self::Digit1 => "1",
            Self::Digit2 => "2",
            Self::Digit3 => "3",
            Self::Digit4 => "4",
            Self::Digit5 => "5",
            Self::Digit6 => "6",
            Self::Digit7 => "7",
            Self::Digit8 => "8",
            Self::Digit9 => "9",
            Self::Space => "SPACE",
            Self::ShiftLeft => "L-SHIFT",
            Self::ShiftRight => "R-SHIFT",
            Self::ControlLeft => "L-CTRL",
            Self::ControlRight => "R-CTRL",
            Self::AltLeft => "L-ALT",
            Self::AltRight => "R-ALT",
            Self::Escape => "ESC",
            Self::Enter => "ENTER",
            Self::Tab => "TAB",
            Self::Backspace => "BACKSPACE",
            Self::ArrowUp => "UP",
            Self::ArrowDown => "DOWN",
            Self::ArrowLeft => "LEFT",
            Self::ArrowRight => "RIGHT",
            Self::Slash => "/",
            Self::Backslash => "\\",
            Self::Minus => "-",
            Self::Equal => "=",
            Self::Semicolon => ";",
            Self::Quote => "'",
            Self::F1 => "F1",
            Self::F2 => "F2",
            Self::F3 => "F3",
            Self::F4 => "F4",
            Self::F5 => "F5",
            Self::F6 => "F6",
            Self::F7 => "F7",
            Self::F8 => "F8",
            Self::F9 => "F9",
            Self::F10 => "F10",
            Self::F11 => "F11",
            Self::F12 => "F12",
        }
    }
}

/// Keybind configuration mapping all `KeyAction` variants to physical `InputKey`s.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeybindSettings {
    pub forward: InputKey,
    pub backward: InputKey,
    pub left: InputKey,
    pub right: InputKey,
    pub jump: InputKey,
    pub sneak: InputKey,
    pub sprint: InputKey,
    pub inventory: InputKey,
    pub advancements: InputKey,
    pub drop: InputKey,
    pub chat: InputKey,
    pub command: InputKey,
    pub screenshot: InputKey,
    pub toggle_f3: InputKey,
    pub toggle_gamemode: InputKey,
    pub reload_shaders: InputKey,
    pub hotbar_1: InputKey,
    pub hotbar_2: InputKey,
    pub hotbar_3: InputKey,
    pub hotbar_4: InputKey,
    pub hotbar_5: InputKey,
    pub hotbar_6: InputKey,
    pub hotbar_7: InputKey,
    pub hotbar_8: InputKey,
    pub hotbar_9: InputKey,
    pub push_to_talk: InputKey,
}

impl Default for KeybindSettings {
    fn default() -> Self {
        Self {
            forward: InputKey::KeyW,
            backward: InputKey::KeyS,
            left: InputKey::KeyA,
            right: InputKey::KeyD,
            jump: InputKey::Space,
            sneak: InputKey::ShiftLeft,
            sprint: InputKey::ControlLeft,
            inventory: InputKey::KeyE,
            advancements: InputKey::KeyL,
            drop: InputKey::KeyQ,
            chat: InputKey::KeyT,
            command: InputKey::Slash,
            screenshot: InputKey::F2,
            toggle_f3: InputKey::F3,
            toggle_gamemode: InputKey::F4,
            reload_shaders: InputKey::F5,
            hotbar_1: InputKey::Digit1,
            hotbar_2: InputKey::Digit2,
            hotbar_3: InputKey::Digit3,
            hotbar_4: InputKey::Digit4,
            hotbar_5: InputKey::Digit5,
            hotbar_6: InputKey::Digit6,
            hotbar_7: InputKey::Digit7,
            hotbar_8: InputKey::Digit8,
            hotbar_9: InputKey::Digit9,
            push_to_talk: InputKey::KeyV,
        }
    }
}

impl KeybindSettings {
    /// Retrieves the bound key for an action.
    #[must_use]
    pub const fn get(&self, action: KeyAction) -> InputKey {
        match action {
            KeyAction::Forward => self.forward,
            KeyAction::Backward => self.backward,
            KeyAction::Left => self.left,
            KeyAction::Right => self.right,
            KeyAction::Jump => self.jump,
            KeyAction::Sneak => self.sneak,
            KeyAction::Sprint => self.sprint,
            KeyAction::Inventory => self.inventory,
            KeyAction::Advancements => self.advancements,
            KeyAction::Drop => self.drop,
            KeyAction::Chat => self.chat,
            KeyAction::Command => self.command,
            KeyAction::Screenshot => self.screenshot,
            KeyAction::ToggleF3 => self.toggle_f3,
            KeyAction::ToggleGameMode => self.toggle_gamemode,
            KeyAction::ReloadShaders => self.reload_shaders,
            KeyAction::Hotbar1 => self.hotbar_1,
            KeyAction::Hotbar2 => self.hotbar_2,
            KeyAction::Hotbar3 => self.hotbar_3,
            KeyAction::Hotbar4 => self.hotbar_4,
            KeyAction::Hotbar5 => self.hotbar_5,
            KeyAction::Hotbar6 => self.hotbar_6,
            KeyAction::Hotbar7 => self.hotbar_7,
            KeyAction::Hotbar8 => self.hotbar_8,
            KeyAction::Hotbar9 => self.hotbar_9,
            KeyAction::PushToTalk => self.push_to_talk,
        }
    }

    /// Sets the bound key for an action.
    pub fn set(&mut self, action: KeyAction, key: InputKey) {
        match action {
            KeyAction::Forward => self.forward = key,
            KeyAction::Backward => self.backward = key,
            KeyAction::Left => self.left = key,
            KeyAction::Right => self.right = key,
            KeyAction::Jump => self.jump = key,
            KeyAction::Sneak => self.sneak = key,
            KeyAction::Sprint => self.sprint = key,
            KeyAction::Inventory => self.inventory = key,
            KeyAction::Advancements => self.advancements = key,
            KeyAction::Drop => self.drop = key,
            KeyAction::Chat => self.chat = key,
            KeyAction::Command => self.command = key,
            KeyAction::Screenshot => self.screenshot = key,
            KeyAction::ToggleF3 => self.toggle_f3 = key,
            KeyAction::ToggleGameMode => self.toggle_gamemode = key,
            KeyAction::ReloadShaders => self.reload_shaders = key,
            KeyAction::Hotbar1 => self.hotbar_1 = key,
            KeyAction::Hotbar2 => self.hotbar_2 = key,
            KeyAction::Hotbar3 => self.hotbar_3 = key,
            KeyAction::Hotbar4 => self.hotbar_4 = key,
            KeyAction::Hotbar5 => self.hotbar_5 = key,
            KeyAction::Hotbar6 => self.hotbar_6 = key,
            KeyAction::Hotbar7 => self.hotbar_7 = key,
            KeyAction::Hotbar8 => self.hotbar_8 = key,
            KeyAction::Hotbar9 => self.hotbar_9 = key,
            KeyAction::PushToTalk => self.push_to_talk = key,
        }
    }

    /// Checks if a physical input key matches the bound key for an action.
    #[must_use]
    pub fn matches(&self, action: KeyAction, key: InputKey) -> bool {
        key != InputKey::None && self.get(action) == key
    }

    /// Checks for conflicting key assignments within the same action domain.
    #[must_use]
    pub fn find_conflicts(&self) -> Vec<(KeyAction, KeyAction, InputKey)> {
        let mut conflicts = Vec::new();
        let all = KeyAction::all();
        for i in 0..all.len() {
            let a1 = all[i];
            let k1 = self.get(a1);
            if k1 == InputKey::None {
                continue;
            }
            for &a2 in &all[i + 1..] {
                let k2 = self.get(a2);
                if k1 == k2 {
                    conflicts.push((a1, a2, k1));
                }
            }
        }
        conflicts
    }

    /// Resets all keybinds to the canonical defaults.
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_keybinds_no_conflicts() {
        let kb = KeybindSettings::default();
        let conflicts = kb.find_conflicts();
        assert!(
            conflicts.is_empty(),
            "Defaults should have zero key conflicts: {conflicts:?}"
        );
        assert_eq!(kb.get(KeyAction::Forward), InputKey::KeyW);
        assert_eq!(kb.get(KeyAction::Jump), InputKey::Space);
        assert_eq!(kb.get(KeyAction::Inventory), InputKey::KeyE);
    }

    #[test]
    fn test_conflict_detection() {
        let mut kb = KeybindSettings::default();
        kb.set(KeyAction::Backward, InputKey::KeyW); // Conflict with Forward
        let conflicts = kb.find_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, KeyAction::Forward);
        assert_eq!(conflicts[0].1, KeyAction::Backward);
        assert_eq!(conflicts[0].2, InputKey::KeyW);
    }

    #[test]
    fn test_reset_defaults() {
        let mut kb = KeybindSettings::default();
        kb.set(KeyAction::Forward, InputKey::ArrowUp);
        assert_eq!(kb.get(KeyAction::Forward), InputKey::ArrowUp);
        kb.reset_defaults();
        assert_eq!(kb.get(KeyAction::Forward), InputKey::KeyW);
    }
}
