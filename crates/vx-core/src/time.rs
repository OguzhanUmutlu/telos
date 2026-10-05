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
