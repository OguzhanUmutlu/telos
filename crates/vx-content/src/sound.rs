//! Sound events, categories, and block acoustic group definitions.

use serde::{Deserialize, Serialize};

/// Audio channels and volume grouping categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SoundCategory {
    /// Global master audio bus.
    #[default]
    Master,
    /// Background music.
    Music,
    /// Jukebox and record audio.
    Records,
    /// Ambient environmental weather (rain, thunder).
    Weather,
    /// Block interaction sounds (step, break, place, hit).
    Blocks,
    /// Hostile mob sounds.
    Hostile,
    /// Passive and neutral mob sounds.
    Neutral,
    /// Player actions and footsteps.
    Players,
    /// Subterranean and atmospheric ambient loops.
    Ambient,
    /// Voice and narration.
    Voice,
}

/// Strongly typed sound event identifier (e.g. `voxel:block.grass.step`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SoundEvent(pub String);

impl SoundEvent {
    /// Creates a sound event from namespace and path.
    #[must_use]
    pub fn new(namespace: &str, path: &str) -> Self {
        Self(format!("{namespace}:{path}"))
    }

    /// Creates a sound event under the default engine namespace (`voxel`).
    #[must_use]
    pub fn engine(path: &str) -> Self {
        Self(format!("voxel:{path}"))
    }

    /// Returns the sound event path as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SoundEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Acoustic profile for a block material (step, break, place, hit, fall sounds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockSoundGroup {
    /// Sound played when block is destroyed.
    pub break_sound: SoundEvent,
    /// Sound played when an entity walks over the block.
    pub step_sound: SoundEvent,
    /// Sound played when block is placed in the world.
    pub place_sound: SoundEvent,
    /// Sound played when hitting/mining the block.
    pub hit_sound: SoundEvent,
    /// Sound played when falling onto the block.
    pub fall_sound: SoundEvent,
    /// Base volume multiplier (typically 1.0).
    pub volume: f32,
    /// Base pitch multiplier (typically 1.0).
    pub pitch: f32,
}

impl BlockSoundGroup {
    /// Creates a new block sound group.
    #[must_use]
    pub fn new(
        break_sound: SoundEvent,
        step_sound: SoundEvent,
        place_sound: SoundEvent,
        hit_sound: SoundEvent,
        fall_sound: SoundEvent,
        volume: f32,
        pitch: f32,
    ) -> Self {
        Self {
            break_sound,
            step_sound,
            place_sound,
            hit_sound,
            fall_sound,
            volume,
            pitch,
        }
    }

    /// Standard grass sound group.
    #[must_use]
    pub fn grass() -> Self {
        Self::new(
            SoundEvent::engine("block.grass.break"),
            SoundEvent::engine("block.grass.step"),
            SoundEvent::engine("block.grass.place"),
            SoundEvent::engine("block.grass.hit"),
            SoundEvent::engine("block.grass.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard stone sound group.
    #[must_use]
    pub fn stone() -> Self {
        Self::new(
            SoundEvent::engine("block.stone.break"),
            SoundEvent::engine("block.stone.step"),
            SoundEvent::engine("block.stone.place"),
            SoundEvent::engine("block.stone.hit"),
            SoundEvent::engine("block.stone.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard wood sound group.
    #[must_use]
    pub fn wood() -> Self {
        Self::new(
            SoundEvent::engine("block.wood.break"),
            SoundEvent::engine("block.wood.step"),
            SoundEvent::engine("block.wood.place"),
            SoundEvent::engine("block.wood.hit"),
            SoundEvent::engine("block.wood.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard dirt / gravel sound group.
    #[must_use]
    pub fn dirt() -> Self {
        Self::new(
            SoundEvent::engine("block.gravel.break"),
            SoundEvent::engine("block.gravel.step"),
            SoundEvent::engine("block.gravel.place"),
            SoundEvent::engine("block.gravel.hit"),
            SoundEvent::engine("block.gravel.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard sand sound group.
    #[must_use]
    pub fn sand() -> Self {
        Self::new(
            SoundEvent::engine("block.sand.break"),
            SoundEvent::engine("block.sand.step"),
            SoundEvent::engine("block.sand.place"),
            SoundEvent::engine("block.sand.hit"),
            SoundEvent::engine("block.sand.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard glass sound group.
    #[must_use]
    pub fn glass() -> Self {
        Self::new(
            SoundEvent::engine("block.glass.break"),
            SoundEvent::engine("block.glass.step"),
            SoundEvent::engine("block.glass.place"),
            SoundEvent::engine("block.glass.hit"),
            SoundEvent::engine("block.glass.fall"),
            1.0,
            1.0,
        )
    }

    /// Standard water sound group.
    #[must_use]
    pub fn water() -> Self {
        Self::new(
            SoundEvent::engine("block.water.ambient"),
            SoundEvent::engine("block.water.step"),
            SoundEvent::engine("block.water.place"),
            SoundEvent::engine("block.water.hit"),
            SoundEvent::engine("block.water.fall"),
            1.0,
            1.0,
        )
    }

    /// Resolves a sound group from a sound key name (e.g. `"voxel:stone"`, `"voxel:grass"`).
    #[must_use]
    pub fn from_sound_key(key: &str) -> Self {
        let normalized = key.strip_prefix("voxel:").unwrap_or(key);
        match normalized {
            "grass" | "plant" | "foliage" => Self::grass(),
            "wood" | "log" | "planks" => Self::wood(),
            "dirt" | "gravel" | "ground" => Self::dirt(),
            "sand" => Self::sand(),
            "glass" => Self::glass(),
            "water" => Self::water(),
            _ => Self::stone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sound_event_format() {
        let ev = SoundEvent::engine("block.stone.step");
        assert_eq!(ev.as_str(), "voxel:block.stone.step");
        assert_eq!(format!("{ev}"), "voxel:block.stone.step");

        let custom = SoundEvent::new("custom_mod", "ambient.wind");
        assert_eq!(custom.as_str(), "custom_mod:ambient.wind");
    }

    #[test]
    fn test_standard_block_sound_groups() {
        let groups = [
            BlockSoundGroup::grass(),
            BlockSoundGroup::stone(),
            BlockSoundGroup::wood(),
            BlockSoundGroup::dirt(),
            BlockSoundGroup::sand(),
            BlockSoundGroup::glass(),
            BlockSoundGroup::water(),
        ];

        for g in groups {
            assert!(g.volume > 0.0);
            assert!(g.pitch > 0.0);
            assert!(!g.break_sound.as_str().is_empty());
            assert!(!g.step_sound.as_str().is_empty());
            assert!(!g.place_sound.as_str().is_empty());
        }
    }

    #[test]
    fn test_from_sound_key_resolution() {
        assert_eq!(
            BlockSoundGroup::from_sound_key("voxel:grass"),
            BlockSoundGroup::grass()
        );
        assert_eq!(
            BlockSoundGroup::from_sound_key("planks"),
            BlockSoundGroup::wood()
        );
        assert_eq!(
            BlockSoundGroup::from_sound_key("gravel"),
            BlockSoundGroup::dirt()
        );
        assert_eq!(
            BlockSoundGroup::from_sound_key("unknown"),
            BlockSoundGroup::stone()
        );
    }
}
