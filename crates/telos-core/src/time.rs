//! Fixed-timestep loop driver for simulation and physics (20 TPS).

use std::time::{Duration, Instant};

/// Drives a fixed-rate simulation loop (e.g. 20 ticks per second = 50ms per tick).
///
/// Uses an accumulator pattern to ensure simulation determinism regardless of frame rate,
/// with a clamp to prevent spiral-of-death stalls under heavy lag.
#[derive(Debug, Clone)]
pub struct FixedTimestep {
    /// The fixed duration of one simulation tick (e.g. 50ms for 20 TPS).
    tick_duration: Duration,
    /// Accumulator holding unsimulated real time.
    accumulator: Duration,
    /// Maximum real time that can accumulate before ticks are dropped (spiral-of-death clamp).
    max_accumulated: Duration,
    /// Total ticks executed so far.
    total_ticks: u64,
    /// Timestamp of the last step call.
    last_time: Instant,
}

impl FixedTimestep {
    /// Creates a new `FixedTimestep` with the given target ticks per second (TPS).
    ///
    /// # Panics
    /// Panics if `tps == 0`.
    #[must_use]
    pub fn new(tps: u32) -> Self {
        assert!(tps > 0, "TPS must be greater than zero");
        let tick_duration = Duration::from_secs_f64(1.0 / f64::from(tps));
        // Clamp accumulator to at most 10 ticks
        let max_accumulated = tick_duration * 10;

        Self {
            tick_duration,
            accumulator: Duration::ZERO,
            max_accumulated,
            total_ticks: 0,
            last_time: Instant::now(),
        }
    }

    /// Creates a standard 20 TPS fixed timestep (50ms per tick).
    #[must_use]
    pub fn standard_20_tps() -> Self {
        Self::new(20)
    }

    /// Steps the clock by the actual elapsed real time and returns an iterator or count
    /// of ticks that need to be simulated.
    ///
    /// Invokes `tick_fn` once for each required simulation step.
    pub fn advance<F>(&mut self, mut tick_fn: F) -> u32
    where
        F: FnMut(u64),
    {
        let now = Instant::now();
        let delta = now.duration_since(self.last_time);
        self.last_time = now;

        self.accumulator += delta;
        if self.accumulator > self.max_accumulated {
            self.accumulator = self.max_accumulated;
        }

        let mut ticks_run = 0;
        while self.accumulator >= self.tick_duration {
            self.accumulator -= self.tick_duration;
            self.total_ticks += 1;
            ticks_run += 1;
            tick_fn(self.total_ticks);
        }

        ticks_run
    }

    /// Returns the interpolation factor `alpha` in `[0.0, 1.0)` between the current
    /// simulation tick and the next, for smooth visual rendering.
    #[must_use]
    pub fn sub_tick_alpha(&self) -> f32 {
        let alpha = self.accumulator.as_secs_f32() / self.tick_duration.as_secs_f32();
        alpha.clamp(0.0, 1.0)
    }

    /// Returns the target tick duration.
    #[must_use]
    pub fn tick_duration(&self) -> Duration {
        self.tick_duration
    }

    /// Returns the total ticks completed so far.
    #[must_use]
    pub fn total_ticks(&self) -> u64 {
        self.total_ticks
    }
}

/// Standard duration of a full in-game day (20 real-time minutes at 20 TPS).
pub const DAY_TICKS: u64 = 24_000;
/// Ticks at midday / noon (sun directly overhead at zenith).
pub const NOON_TICKS: u64 = 6_000;
/// Ticks at sunset / dusk (sun at western horizon).
pub const SUNSET_TICKS: u64 = 12_000;
/// Ticks at midnight (moon directly overhead at zenith).
pub const MIDNIGHT_TICKS: u64 = 18_000;
/// Ticks at sunrise / dawn (sun at eastern horizon).
pub const SUNRISE_TICKS: u64 = 0;
/// Number of days in a full lunar cycle (8 phases: full moon to new moon to waning).
pub const LUNAR_CYCLE_DAYS: u64 = 8;

/// Computes the normalized celestial angle `[0.0, 2pi)` for a given time of day and sub-tick fraction.
///
/// Angle is zero at noon (`t = 6000`), `pi/2` at sunset (`t = 12000`), `pi` at midnight (`t = 18000`),
/// and `3pi/2` at sunrise (`t = 0`).
#[must_use]
pub fn sun_angle(time_of_day: u64, sub_tick: f32) -> f32 {
    let t = ((time_of_day % DAY_TICKS) as f64 + f64::from(sub_tick)) % (DAY_TICKS as f64);
    let offset_t = (t - (NOON_TICKS as f64)).rem_euclid(DAY_TICKS as f64);
    let fraction = offset_t / (DAY_TICKS as f64);
    (fraction * std::f64::consts::TAU) as f32
}

/// Computes the normalized 3D unit direction vector pointing towards the sun on the celestial sphere.
///
/// Sun rises in the East (`+X`), reaches zenith at noon (`+Y`), sets in the West (`-X`), and nadir at midnight (`-Y`).
#[must_use]
pub fn sun_direction(sun_angle: f32) -> glam::Vec3 {
    let x = -sun_angle.sin();
    let y = sun_angle.cos();
    let z = 0.0;
    glam::Vec3::new(x, y, z).normalize()
}

/// Computes the normalized 3D unit direction vector pointing towards the moon on the celestial sphere.
///
/// The moon is always directly opposite the sun (`moon_dir = -sun_dir`).
#[must_use]
pub fn moon_direction(sun_angle: f32) -> glam::Vec3 {
    -sun_direction(sun_angle)
}

/// Computes the active moon phase index `[0..=7]` for the given day number (`world_age / DAY_TICKS`).
///
/// - `0`: Full Moon
/// - `1`: Waning Gibbous
/// - `2`: Third Quarter
/// - `3`: Waning Crescent
/// - `4`: New Moon
/// - `5`: Waxing Crescent
/// - `6`: First Quarter
/// - `7`: Waxing Gibbous
#[must_use]
pub fn moon_phase(day_number: u64) -> u32 {
    (day_number % LUNAR_CYCLE_DAYS) as u32
}

/// Returns the daylight illumination factor in `[0.0, 1.0]` based on sun elevation (`sun_dir.y`).
///
/// `1.0` during full daylight, smoothly transitioning to `0.0` across dusk and dawn.
#[must_use]
pub fn daylight_factor(sun_dir_y: f32) -> f32 {
    ((sun_dir_y + 0.25) / 0.5).clamp(0.0, 1.0)
}

/// Returns the sunset/sunrise golden hour intensity in `[0.0, 1.0]` based on sun elevation (`sun_dir.y`).
///
/// Peaks at `1.0` when the sun is near the horizon (`sun_dir.y == 0.0`) and drops to `0.0` at midday and night.
#[must_use]
pub fn sunset_factor(sun_dir_y: f32) -> f32 {
    let abs_y = sun_dir_y.abs();
    if abs_y < 0.25 {
        1.0 - (abs_y / 0.25)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sun_angle_cardinal_points() {
        // Noon: angle == 0
        let noon_angle = sun_angle(NOON_TICKS, 0.0);
        assert!(noon_angle.abs() < 1e-4);
        let noon_dir = sun_direction(noon_angle);
        assert!((noon_dir.y - 1.0).abs() < 1e-4);
        assert!(noon_dir.x.abs() < 1e-4);

        // Sunset: angle == pi/2, dir == (-1, 0, 0) (West)
        let sunset_angle = sun_angle(SUNSET_TICKS, 0.0);
        let sunset_dir = sun_direction(sunset_angle);
        assert!((sunset_dir.x - (-1.0)).abs() < 1e-4);
        assert!(sunset_dir.y.abs() < 1e-4);

        // Midnight: angle == pi, dir == (0, -1, 0) (Nadir)
        let midnight_angle = sun_angle(MIDNIGHT_TICKS, 0.0);
        let midnight_dir = sun_direction(midnight_angle);
        assert!((midnight_dir.y - (-1.0)).abs() < 1e-4);
        assert!(midnight_dir.x.abs() < 1e-4);

        // Sunrise: angle == 3pi/2, dir == (1, 0, 0) (East)
        let sunrise_angle = sun_angle(SUNRISE_TICKS, 0.0);
        let sunrise_dir = sun_direction(sunrise_angle);
        assert!((sunrise_dir.x - 1.0).abs() < 1e-4);
        assert!(sunrise_dir.y.abs() < 1e-4);
    }

    #[test]
    fn test_moon_opposite_sun() {
        for tick in (0..DAY_TICKS).step_by(1000) {
            let angle = sun_angle(tick, 0.0);
            let s_dir = sun_direction(angle);
            let m_dir = moon_direction(angle);
            let dot = s_dir.dot(m_dir);
            assert!((dot - (-1.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn test_moon_phases_cycle() {
        for day in 0..16 {
            assert_eq!(moon_phase(day), (day % 8) as u32);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_daylight_and_sunset_factors() {
        assert_eq!(daylight_factor(1.0), 1.0);
        assert_eq!(daylight_factor(-1.0), 0.0);
        assert_eq!(daylight_factor(0.0), 0.5);

        assert_eq!(sunset_factor(0.0), 1.0);
        assert_eq!(sunset_factor(0.5), 0.0);
        assert_eq!(sunset_factor(-0.5), 0.0);
    }
}
