//! Experience points, leveling, and progress bar calculations.

use bevy_ecs::component::Component;

/// Calculates the total experience points required to reach level `l`.
#[must_use]
pub const fn total_points_for_level(l: u32) -> u32 {
    if l <= 16 {
        l * l + 6 * l
    } else if l <= 31 {
        // (5*L^2 + 720 - 81*L) / 2
        (5 * l * l + 720 - 81 * l) / 2
    } else {
        // (9*L^2 + 4440 - 325*L) / 2
        (9 * l * l + 4440 - 325 * l) / 2
    }
}

/// Calculates the experience points required to advance from level `l` to `l + 1`.
#[must_use]
pub const fn points_for_next_level(l: u32) -> u32 {
    if l <= 15 {
        2 * l + 7
    } else if l <= 30 {
        5 * l - 38
    } else {
        9 * l - 158
    }
}

/// Finds the player's level from total accumulated experience points.
#[must_use]
pub fn level_from_total_points(total_xp: u32) -> u32 {
    // Binary search over reasonable levels (0..=5000)
    let mut low = 0u32;
    let mut high = 5000u32;

    while low < high {
        let mid = (low + high).div_ceil(2);
        if total_points_for_level(mid) <= total_xp {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

/// Player experience component.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Component)]
pub struct Experience {
    /// Total experience points collected over the lifetime of the player.
    pub total_xp: u32,
}

impl Experience {
    /// Creates an experience component with given total points.
    #[must_use]
    pub const fn new(total_xp: u32) -> Self {
        Self { total_xp }
    }

    /// Current experience level.
    #[must_use]
    pub fn level(&self) -> u32 {
        level_from_total_points(self.total_xp)
    }

    /// Fraction of progress toward the next level, in range `[0.0, 1.0)`.
    #[must_use]
    pub fn progress(&self) -> f32 {
        let current_lvl = self.level();
        let pts_at_current = total_points_for_level(current_lvl);
        let pts_needed = points_for_next_level(current_lvl);
        if pts_needed == 0 {
            return 0.0;
        }
        let excess = self.total_xp.saturating_sub(pts_at_current);
        #[allow(clippy::cast_precision_loss)]
        let progress = (excess as f32) / (pts_needed as f32);
        progress.clamp(0.0, 0.9999)
    }

    /// Experience points remaining until reaching the next level.
    #[must_use]
    pub fn points_to_next_level(&self) -> u32 {
        let current_lvl = self.level();
        let pts_next_level = total_points_for_level(current_lvl + 1);
        pts_next_level.saturating_sub(self.total_xp)
    }

    /// Adds experience points. Returns `true` if this caused a level up.
    pub fn add_xp(&mut self, points: u32) -> bool {
        let old_level = self.level();
        self.total_xp = self.total_xp.saturating_add(points);
        self.level() > old_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_progression_closed_forms() {
        // Verify level total points continuity:
        // Level 0: 0
        assert_eq!(total_points_for_level(0), 0);
        // Level 1: 7
        assert_eq!(total_points_for_level(1), 7);
        // Level 15: 15^2 + 6*15 = 225 + 90 = 315
        assert_eq!(total_points_for_level(15), 315);
        // Level 16: 352
        assert_eq!(total_points_for_level(16), 352);
        assert_eq!(total_points_for_level(15) + points_for_next_level(15), 352);

        // Level 30: 1395
        assert_eq!(total_points_for_level(30), 1395);
        // Level 31: 1507
        assert_eq!(total_points_for_level(31), 1507);
        assert_eq!(total_points_for_level(30) + points_for_next_level(30), 1507);

        // Test level recovery roundtrip for levels 0..100
        for lvl in 0..=100 {
            let pts = total_points_for_level(lvl);
            assert_eq!(level_from_total_points(pts), lvl);
            if lvl > 0 {
                assert_eq!(level_from_total_points(pts - 1), lvl - 1);
            }
        }
    }

    #[test]
    fn test_experience_progress_fraction() {
        let mut exp = Experience::new(0);
        assert_eq!(exp.level(), 0);
        assert!(exp.progress().abs() < f32::EPSILON);

        // Add 7 XP -> exactly level 1
        assert!(exp.add_xp(7));
        assert_eq!(exp.level(), 1);
        assert!(exp.progress().abs() < f32::EPSILON);

        // Level 1 needs 2(1) + 7 = 9 XP. Add 4 XP -> progress ~4/9
        exp.add_xp(4);
        assert_eq!(exp.level(), 1);
        let prog = exp.progress();
        assert!((prog - 4.0 / 9.0).abs() < 0.001);
    }
}
