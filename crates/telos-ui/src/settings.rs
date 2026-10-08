//! Persistent game configuration and settings engine.
//!
//! Manages video, audio, controls, and gameplay preferences serialized to `settings.toml`.

use crate::keybinds::KeybindSettings;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Comprehensive game settings serialized to `settings.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GameSettings {
    /// Video and graphics settings.
    pub video: VideoSettings,
    /// Audio volume settings.
    pub audio: AudioSettings,
    /// Mouse and keyboard control settings.
    pub controls: ControlSettings,
    /// Gameplay preferences.
    pub gameplay: GameplaySettings,
}

const fn default_sim_distance() -> u32 {
    8
}

const fn default_true() -> bool {
    true
}

/// Video and display settings.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoSettings {
    /// Horizontal render distance in chunks (2..=32).
    pub view_distance: u32,
    /// Simulation distance in chunks (2..=32).
    #[serde(default = "default_sim_distance")]
    pub simulation_distance: u32,
    /// Vertical render distance in chunks (2..=24).
    pub vertical_view_distance: u32,
    /// Field of view in degrees (30.0..=110.0).
    pub fov: f32,
    /// GUI scale mode (0 = auto, 1..=4 = manual).
    pub gui_scale: u32,
    /// Bypass Hi-Z GPU occlusion culling.
    pub no_cull: bool,
    /// Enable `VSync` (caps to display refresh rate).
    pub vsync: bool,
    /// Target FPS limit (0 = unlimited, 60, 120, 144, 240).
    pub fps_limit: u32,
    /// Enable post-processing pipeline.
    #[serde(default = "default_true")]
    pub post_processing: bool,
    /// Enable screen-space ambient occlusion (SSAO).
    #[serde(default = "default_true")]
    pub ssao: bool,
    /// Enable volumetric horizon distance fog & scattering.
    #[serde(default = "default_true")]
    pub volumetric_fog: bool,
    /// Enable ACES filmic tonemapping.
    #[serde(default = "default_true")]
    pub tonemapping: bool,
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            view_distance: 8,
            simulation_distance: 8,
            vertical_view_distance: 12,
            fov: 70.0,
            gui_scale: 0,
            no_cull: false,
            vsync: false,
            fps_limit: 144,
            post_processing: true,
            ssao: true,
            volumetric_fog: true,
            tonemapping: true,
        }
    }
}

/// Audio volume settings (0.0..=1.0).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioSettings {
    /// Master volume bus (0.0..=1.0).
    pub master_volume: f32,
    /// Background music volume (0.0..=1.0).
    pub music_volume: f32,
    /// Weather effects volume (0.0..=1.0).
    pub weather_volume: f32,
    /// Block breaking and placing sounds (0.0..=1.0).
    pub blocks_volume: f32,
    /// Entity and mob footsteps/cries (0.0..=1.0).
    pub entities_volume: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            music_volume: 0.7,
            weather_volume: 0.85,
            blocks_volume: 0.95,
            entities_volume: 1.0,
        }
    }
}

/// Control settings and input sensitivities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlSettings {
    /// Mouse look sensitivity multiplier (0.1..=3.0).
    pub mouse_sensitivity: f32,
    /// Invert pitch look axis.
    pub invert_mouse_y: bool,
    /// Player keyboard keybind mappings.
    #[serde(default)]
    pub keybinds: KeybindSettings,
}

impl Default for ControlSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 1.0,
            invert_mouse_y: false,
            keybinds: KeybindSettings::default(),
        }
    }
}

/// Gameplay preferences and locale.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameplaySettings {
    /// Player display name.
    pub player_name: String,
    /// Default game mode: "Survival" or "Creative".
    pub game_mode: String,
    /// Active UI language code (e.g. `"en_us"`, `"es_es"`, `"de_de"`, `"fr_fr"`, `"tr_tr"`).
    pub language: String,
    /// Whether to allow dynamic JavaScript scripts inside resource packs (default: false).
    #[serde(default)]
    pub allow_pack_scripts: bool,
}

impl Default for GameplaySettings {
    fn default() -> Self {
        Self {
            player_name: "Player".to_string(),
            game_mode: "Survival".to_string(),
            language: "en_us".to_string(),
            allow_pack_scripts: false,
        }
    }
}

impl GameSettings {
    /// Loads settings from a TOML file, or creates default settings if missing/invalid.
    #[must_use]
    pub fn load_or_create(path: &Path) -> Self {
        if path.exists()
            && let Ok(content) = fs::read_to_string(path)
            && let Ok(mut settings) = toml::from_str::<Self>(&content)
        {
            settings.clamp();
            return settings;
        }

        let settings = Self::default();
        let _ = settings.save(path);
        settings
    }

    /// Serializes and writes settings to the given path.
    ///
    /// # Errors
    /// Returns `std::io::Error` if directory creation or file writing fails.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let serialized = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, serialized)
    }

    /// Clamps numeric values to valid engine ranges.
    pub fn clamp(&mut self) {
        self.video.view_distance = self.video.view_distance.clamp(2, 32);
        self.video.simulation_distance = self.video.simulation_distance.clamp(2, 32);
        self.video.vertical_view_distance = self.video.vertical_view_distance.clamp(2, 24);
        self.video.fov = self.video.fov.clamp(30.0, 110.0);
        self.video.gui_scale = self.video.gui_scale.min(4);

        self.audio.master_volume = self.audio.master_volume.clamp(0.0, 1.0);
        self.audio.music_volume = self.audio.music_volume.clamp(0.0, 1.0);
        self.audio.weather_volume = self.audio.weather_volume.clamp(0.0, 1.0);
        self.audio.blocks_volume = self.audio.blocks_volume.clamp(0.0, 1.0);
        self.audio.entities_volume = self.audio.entities_volume.clamp(0.0, 1.0);

        self.controls.mouse_sensitivity = self.controls.mouse_sensitivity.clamp(0.1, 3.0);

        if self.gameplay.language.trim().is_empty() {
            self.gameplay.language = "en_us".to_string();
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_roundtrip() {
        let original = GameSettings::default();
        let toml_str = toml::to_string_pretty(&original).expect("Serialize failed");
        let deserialized: GameSettings = toml::from_str(&toml_str).expect("Deserialize failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_settings_clamping() {
        let mut settings = GameSettings {
            video: VideoSettings {
                view_distance: 100,
                simulation_distance: 100,
                vertical_view_distance: 0,
                fov: 150.0,
                gui_scale: 10,
                no_cull: true,
                vsync: false,
                fps_limit: 144,
                post_processing: true,
                ssao: true,
                volumetric_fog: true,
                tonemapping: true,
            },
            audio: AudioSettings {
                master_volume: 5.0,
                music_volume: -2.0,
                weather_volume: 1.5,
                blocks_volume: -0.1,
                entities_volume: 0.5,
            },
            controls: ControlSettings {
                mouse_sensitivity: 10.0,
                invert_mouse_y: true,
                keybinds: KeybindSettings::default(),
            },
            gameplay: GameplaySettings::default(),
        };

        settings.clamp();
        assert_eq!(settings.video.view_distance, 32);
        assert_eq!(settings.video.simulation_distance, 32);
        assert_eq!(settings.video.vertical_view_distance, 2);
        assert_eq!(settings.video.fov, 110.0);
        assert_eq!(settings.video.gui_scale, 4);
        assert_eq!(settings.audio.master_volume, 1.0);
        assert_eq!(settings.audio.music_volume, 0.0);
        assert_eq!(settings.controls.mouse_sensitivity, 3.0);
    }
}
