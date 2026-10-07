//! Navigation capabilities and physical dimensions for pathfinding entities.

/// Navigation profile specifying size and traversal capabilities for an entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathProfile {
    /// Required vertical headroom clearance in blocks (e.g. 2 for Humanoid, 1 for small animals).
    pub height: u32,
    /// Bounding box collision width in blocks (e.g. 0.6).
    pub width: f32,
    /// Maximum vertical step-up jump capability in blocks (typically 1).
    pub max_step_up: i32,
    /// Maximum safe vertical drop-down height in blocks (typically 3).
    pub max_drop_down: i32,
    /// Whether the entity can traverse through water.
    pub can_swim: bool,
    /// Whether the entity actively avoids hazardous blocks (lava, fire).
    pub avoid_hazards: bool,
}

impl Default for PathProfile {
    fn default() -> Self {
        Self::humanoid()
    }
}

impl PathProfile {
    /// Standard humanoid navigation profile (Zombie, Player, Skeleton).
    #[must_use]
    pub const fn humanoid() -> Self {
        Self {
            height: 2,
            width: 0.6,
            max_step_up: 1,
            max_drop_down: 3,
            can_swim: true,
            avoid_hazards: true,
        }
    }

    /// Profile for small animals (Pig, Chicken, Baby Zombie).
    #[must_use]
    pub const fn small_animal() -> Self {
        Self {
            height: 1,
            width: 0.6,
            max_step_up: 1,
            max_drop_down: 3,
            can_swim: true,
            avoid_hazards: true,
        }
    }

    /// Profile for large animals (Cow, Horse).
    #[must_use]
    pub const fn large_animal() -> Self {
        Self {
            height: 2,
            width: 0.9,
            max_step_up: 1,
            max_drop_down: 3,
            can_swim: true,
            avoid_hazards: true,
        }
    }
}
