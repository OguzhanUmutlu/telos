//! Authoritative player game modes and capability attributes.
//!
//! Controls flight permissions, damage invulnerability, instant block breaking,
//! build restrictions, and noclip states across singleplayer and multiplayer.

use serde::{Deserialize, Serialize};

/// Capability bitflag: player takes no damage from hits, falls, drowning, or hunger.
pub const CAP_FLAG_INVINCIBLE: u8 = 1 << 0;
/// Capability bitflag: player is currently actively flying through the air.
pub const CAP_FLAG_FLYING: u8 = 1 << 1;
/// Capability bitflag: player is permitted to toggle flight mode (e.g. double-tap jump).
pub const CAP_FLAG_ALLOW_FLIGHT: u8 = 1 << 2;
/// Capability bitflag: player breaks voxels instantaneously in one click.
pub const CAP_FLAG_INSTABREAK: u8 = 1 << 3;
/// Capability bitflag: player is permitted to modify the world (break & place blocks).
pub const CAP_FLAG_CAN_BUILD: u8 = 1 << 4;
/// Capability bitflag: player passes through solid voxels without physical collision.
pub const CAP_FLAG_NOCLIP: u8 = 1 << 5;

/// Player game mode controlling gameplay capabilities, physics constraints, and build rights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum GameMode {
    /// Strict swept AABB collision, gravity, survival hunger/health, and normal reach.
    #[default]
    Survival = 0,
    /// Flight toggle, instant break, god-mode invulnerability, and expanded reach.
    Creative = 1,
    /// Survival movement mechanics with restricted world modification rights.
    Adventure = 2,
    /// Free noclip flight through solid geometry, invisible to mobs, cannot modify world.
    Spectator = 3,
}

impl GameMode {
    /// Returns the canonical numeric wire ID of this game mode.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Resolves a game mode from its numeric ID (0..=3).
    #[must_use]
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Survival),
            1 => Some(Self::Creative),
            2 => Some(Self::Adventure),
            3 => Some(Self::Spectator),
            _ => None,
        }
    }

    /// Resolves a game mode from a case-insensitive string name, shorthand, or numeric ID.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let trimmed = name.trim();
        if trimmed.eq_ignore_ascii_case("survival")
            || trimmed.eq_ignore_ascii_case("s")
            || trimmed == "0"
        {
            Some(Self::Survival)
        } else if trimmed.eq_ignore_ascii_case("creative")
            || trimmed.eq_ignore_ascii_case("c")
            || trimmed == "1"
        {
            Some(Self::Creative)
        } else if trimmed.eq_ignore_ascii_case("adventure")
            || trimmed.eq_ignore_ascii_case("a")
            || trimmed == "2"
        {
            Some(Self::Adventure)
        } else if trimmed.eq_ignore_ascii_case("spectator")
            || trimmed.eq_ignore_ascii_case("sp")
            || trimmed == "3"
        {
            Some(Self::Spectator)
        } else {
            None
        }
    }

    /// Returns the user-facing display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Survival => "Survival",
            Self::Creative => "Creative",
            Self::Adventure => "Adventure",
            Self::Spectator => "Spectator",
        }
    }

    /// Returns the localization catalog translation key.
    #[must_use]
    pub const fn translation_key(self) -> &'static str {
        match self {
            Self::Survival => "gameMode.survival",
            Self::Creative => "gameMode.creative",
            Self::Adventure => "gameMode.adventure",
            Self::Spectator => "gameMode.spectator",
        }
    }

    /// Returns true if this game mode allows creative flight.
    #[must_use]
    pub const fn allows_flight(self) -> bool {
        matches!(self, Self::Creative | Self::Spectator)
    }

    /// Returns true if this game mode is god-mode invincible.
    #[must_use]
    pub const fn is_invincible(self) -> bool {
        matches!(self, Self::Creative | Self::Spectator)
    }

    /// Returns true if this game mode permits world voxel modifications.
    #[must_use]
    pub const fn can_build(self) -> bool {
        matches!(self, Self::Survival | Self::Creative)
    }

    /// Returns true if this game mode bypasses solid block collisions.
    #[must_use]
    pub const fn is_noclip(self) -> bool {
        matches!(self, Self::Spectator)
    }

    /// Cycles to the next game mode in sequence: Survival -> Creative -> Adventure -> Spectator -> Survival.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Survival => Self::Creative,
            Self::Creative => Self::Adventure,
            Self::Adventure => Self::Spectator,
            Self::Spectator => Self::Survival,
        }
    }

    /// Returns default capabilities associated with this game mode.
    #[must_use]
    pub fn default_capabilities(self) -> PlayerCapabilities {
        PlayerCapabilities::from_game_mode(self)
    }
}

/// Comprehensive player capabilities synchronized authoritatively between server and client.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlayerCapabilities {
    /// The associated game mode.
    pub game_mode: GameMode,
    /// Whether the player is permitted to toggle flight mode.
    pub allow_flight: bool,
    /// Whether the player is currently actively flying.
    pub flying: bool,
    /// Whether the player is immune to all damage and hunger depletion.
    pub invincible: bool,
    /// Whether voxels break instantaneously in 1 tick.
    pub instabreak: bool,
    /// Whether the player can place and destroy blocks.
    pub can_build: bool,
    /// Whether the player translates through solid geometry without collision.
    pub noclip: bool,
    /// Interaction and combat reach distance in blocks.
    pub reach_distance: f32,
    /// Base horizontal flying speed in blocks/second.
    pub fly_speed: f32,
    /// Base horizontal walking speed in blocks/second.
    pub walk_speed: f32,
}

impl Default for PlayerCapabilities {
    fn default() -> Self {
        Self::from_game_mode(GameMode::Survival)
    }
}

impl PlayerCapabilities {
    /// Constructs default capabilities corresponding to the given game mode.
    #[must_use]
    pub fn from_game_mode(mode: GameMode) -> Self {
        match mode {
            GameMode::Survival => Self {
                game_mode: GameMode::Survival,
                allow_flight: false,
                flying: false,
                invincible: false,
                instabreak: false,
                can_build: true,
                noclip: false,
                reach_distance: 5.0,
                fly_speed: 14.0,
                walk_speed: 4.317,
            },
            GameMode::Creative => Self {
                game_mode: GameMode::Creative,
                allow_flight: true,
                flying: false,
                invincible: true,
                instabreak: true,
                can_build: true,
                noclip: false,
                reach_distance: 6.0,
                fly_speed: 14.0,
                walk_speed: 4.317,
            },
            GameMode::Adventure => Self {
                game_mode: GameMode::Adventure,
                allow_flight: false,
                flying: false,
                invincible: false,
                instabreak: false,
                can_build: false,
                noclip: false,
                reach_distance: 5.0,
                fly_speed: 14.0,
                walk_speed: 4.317,
            },
            GameMode::Spectator => Self {
                game_mode: GameMode::Spectator,
                allow_flight: true,
                flying: true,
                invincible: true,
                instabreak: false,
                can_build: false,
                noclip: true,
                reach_distance: 0.0,
                fly_speed: 20.0,
                walk_speed: 4.317,
            },
        }
    }

    /// Updates this capability set to match a new game mode while preserving current flight status if valid.
    pub fn apply_game_mode(&mut self, mode: GameMode) {
        let prev_flying = self.flying;
        *self = Self::from_game_mode(mode);
        if self.allow_flight && prev_flying {
            self.flying = true;
        }
    }

    /// Creates default survival capabilities.
    #[must_use]
    pub fn survival() -> Self {
        Self::from_game_mode(GameMode::Survival)
    }

    /// Creates default creative capabilities.
    #[must_use]
    pub fn creative() -> Self {
        Self::from_game_mode(GameMode::Creative)
    }

    /// Packs boolean capability attributes into a compact 1-byte bitflag.
    #[must_use]
    pub fn to_flags(&self) -> u8 {
        self.flags()
    }

    /// Packs boolean capability attributes into a compact 1-byte bitflag.
    #[must_use]
    pub fn flags(&self) -> u8 {
        let mut flags = 0u8;
        if self.invincible {
            flags |= CAP_FLAG_INVINCIBLE;
        }
        if self.flying {
            flags |= CAP_FLAG_FLYING;
        }
        if self.allow_flight {
            flags |= CAP_FLAG_ALLOW_FLIGHT;
        }
        if self.instabreak {
            flags |= CAP_FLAG_INSTABREAK;
        }
        if self.can_build {
            flags |= CAP_FLAG_CAN_BUILD;
        }
        if self.noclip {
            flags |= CAP_FLAG_NOCLIP;
        }
        flags
    }

    /// Reconstructs capabilities from wire attributes and bitflags.
    #[must_use]
    pub fn from_wire(
        game_mode_id: u8,
        flags: u8,
        fly_speed: f32,
        walk_speed: f32,
        reach_distance: f32,
    ) -> Self {
        let game_mode = GameMode::from_id(game_mode_id).unwrap_or(GameMode::Survival);
        Self {
            game_mode,
            allow_flight: (flags & CAP_FLAG_ALLOW_FLIGHT) != 0,
            flying: (flags & CAP_FLAG_FLYING) != 0,
            invincible: (flags & CAP_FLAG_INVINCIBLE) != 0,
            instabreak: (flags & CAP_FLAG_INSTABREAK) != 0,
            can_build: (flags & CAP_FLAG_CAN_BUILD) != 0,
            noclip: (flags & CAP_FLAG_NOCLIP) != 0,
            reach_distance,
            fly_speed,
            walk_speed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_mode_resolution() {
        assert_eq!(GameMode::from_name("survival"), Some(GameMode::Survival));
        assert_eq!(GameMode::from_name("s"), Some(GameMode::Survival));
        assert_eq!(GameMode::from_name("0"), Some(GameMode::Survival));

        assert_eq!(GameMode::from_name("creative"), Some(GameMode::Creative));
        assert_eq!(GameMode::from_name("c"), Some(GameMode::Creative));
        assert_eq!(GameMode::from_name("1"), Some(GameMode::Creative));

        assert_eq!(GameMode::from_name("adventure"), Some(GameMode::Adventure));
        assert_eq!(GameMode::from_name("a"), Some(GameMode::Adventure));
        assert_eq!(GameMode::from_name("2"), Some(GameMode::Adventure));

        assert_eq!(GameMode::from_name("spectator"), Some(GameMode::Spectator));
        assert_eq!(GameMode::from_name("sp"), Some(GameMode::Spectator));
        assert_eq!(GameMode::from_name("3"), Some(GameMode::Spectator));

        assert_eq!(GameMode::from_name("invalid"), None);
    }

    #[test]
    fn test_game_mode_cycling() {
        let mut m = GameMode::Survival;
        m = m.next();
        assert_eq!(m, GameMode::Creative);
        m = m.next();
        assert_eq!(m, GameMode::Adventure);
        m = m.next();
        assert_eq!(m, GameMode::Spectator);
        m = m.next();
        assert_eq!(m, GameMode::Survival);
    }

    #[test]
    fn test_capabilities_bitflags_roundtrip() {
        for mode in [
            GameMode::Survival,
            GameMode::Creative,
            GameMode::Adventure,
            GameMode::Spectator,
        ] {
            let caps = PlayerCapabilities::from_game_mode(mode);
            let flags = caps.flags();
            let decoded = PlayerCapabilities::from_wire(
                caps.game_mode.id(),
                flags,
                caps.fly_speed,
                caps.walk_speed,
                caps.reach_distance,
            );
            assert_eq!(caps, decoded);
        }
    }
}
