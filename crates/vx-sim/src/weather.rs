//! Server-authoritative weather state machine and precipitation simulation.
//!
//! Provides deterministic cycles across Clear, Rain, and Thunder, smooth level fading (0.01/tick),
//! thunderstorm lightning strike event generation, and altitude-lapse precipitation classification.

use bevy_ecs::prelude::Resource;

/// Active atmospheric weather condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WeatherKind {
    /// Clear sunny skies with no precipitation.
    #[default]
    Clear = 0,
    /// Gentle or heavy rainfall.
    Rain = 1,
    /// Severe thunderstorm with dark overcast skies and lightning strikes.
    Thunder = 2,
}

impl WeatherKind {
    /// Converts numeric index to weather kind.
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            1 => Self::Rain,
            2 => Self::Thunder,
            _ => Self::Clear,
        }
    }
}

/// The physical form of atmospheric precipitation at a given world position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecipitationKind {
    /// No precipitation (clear weather, or dry biomes such as deserts).
    None,
    /// Liquid rainfall streaks.
    Rain,
    /// Solid fluttering snowflakes.
    Snow,
}

/// Rate at which rain and thunder levels transition per simulation tick (0.01 / tick = 5.0 seconds total fade).
pub const LEVEL_FADE_PER_TICK: f32 = 0.01;

/// Standard sea-level reference altitude for temperature lapse calculation (blocks).
pub const SEA_LEVEL_REF: f32 = 64.0;

/// Temperature reduction per block of elevation above sea level ($0.00125\ ^\circ\text{C} / \text{block}$).
pub const ALTITUDE_LAPSE_RATE: f32 = 0.00125;

/// Critical temperature threshold below which rain turns to snow ($t < 0.15$).
pub const SNOW_TEMP_THRESHOLD: f32 = 0.15;

/// Duration in ticks of a lightning flash event (4 ticks = 200ms).
pub const LIGHTNING_FLASH_DURATION_TICKS: u8 = 4;

/// Server-authoritative weather state resource.
#[derive(Debug, Clone, Resource, PartialEq)]
pub struct WeatherState {
    /// Current weather condition.
    pub kind: WeatherKind,
    /// Ticks remaining before transitioning to the next weather state.
    pub timer: u32,
    /// Rain intensity factor in `[0.0, 1.0]`.
    pub rain_level: f32,
    /// Thunderstorm intensity factor in `[0.0, 1.0]`.
    pub thunder_level: f32,
    /// Remaining ticks of active lightning flash illumination (0 if dark).
    pub lightning_flash_ticks: u8,
    /// World seed used for deterministic weather transition pseudo-randomness.
    pub seed: u64,
    /// Total ticks elapsed in simulation.
    pub ticks_elapsed: u64,
}

impl WeatherState {
    /// Creates a new `WeatherState` initialized to Clear skies with a standard timer.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            kind: WeatherKind::Clear,
            timer: 24_000, // 20 minutes default clear duration
            rain_level: 0.0,
            thunder_level: 0.0,
            lightning_flash_ticks: 0,
            seed,
            ticks_elapsed: 0,
        }
    }

    /// Advances the weather simulation by one tick (20 TPS).
    pub fn tick(&mut self) {
        self.ticks_elapsed = self.ticks_elapsed.wrapping_add(1);

        // 1. Advance state timer
        if self.timer > 0 {
            self.timer -= 1;
        }

        if self.timer == 0 {
            self.transition_next_state();
        }

        // 2. Smoothly interpolate levels toward active target
        match self.kind {
            WeatherKind::Clear => {
                self.rain_level = (self.rain_level - LEVEL_FADE_PER_TICK).max(0.0);
                self.thunder_level = (self.thunder_level - LEVEL_FADE_PER_TICK).max(0.0);
            }
            WeatherKind::Rain => {
                self.rain_level = (self.rain_level + LEVEL_FADE_PER_TICK).min(1.0);
                self.thunder_level = (self.thunder_level - LEVEL_FADE_PER_TICK).max(0.0);
            }
            WeatherKind::Thunder => {
                self.rain_level = (self.rain_level + LEVEL_FADE_PER_TICK).min(1.0);
                self.thunder_level = (self.thunder_level + LEVEL_FADE_PER_TICK).min(1.0);
            }
        }

        // 3. Lightning flash countdown
        if self.lightning_flash_ticks > 0 {
            self.lightning_flash_ticks -= 1;
        }

        // 4. Random lightning strikes during thunderstorm
        if self.kind == WeatherKind::Thunder && self.thunder_level > 0.5 {
            let roll = pseudo_rand_u64(self.seed.wrapping_add(self.ticks_elapsed));
            // Average chance of strike: ~1 per 250 ticks (~12.5 seconds)
            if roll.is_multiple_of(250) {
                self.lightning_flash_ticks = LIGHTNING_FLASH_DURATION_TICKS;
            }
        }
    }

    /// Transitions to the next deterministic weather state upon timer expiration.
    fn transition_next_state(&mut self) {
        let roll = pseudo_rand_u64(self.seed.wrapping_add(self.ticks_elapsed));
        match self.kind {
            WeatherKind::Clear => {
                // Clear transitions to Rain (12,000..=24,000 ticks = 10..20 min)
                self.kind = WeatherKind::Rain;
                let offset = (roll % 12_001) as u32;
                self.timer = 12_000 + offset;
            }
            WeatherKind::Rain => {
                // Rain transitions either to Thunder (25% chance) or back to Clear (75% chance)
                if roll.is_multiple_of(4) {
                    self.kind = WeatherKind::Thunder;
                    let offset = (roll % 12_001) as u32;
                    self.timer = 3_600 + offset; // 3 to 13 minutes
                } else {
                    self.kind = WeatherKind::Clear;
                    let offset = (roll % 24_001) as u32;
                    self.timer = 12_000 + offset; // 10 to 30 minutes
                }
            }
            WeatherKind::Thunder => {
                // Thunder transitions back to Rain
                self.kind = WeatherKind::Rain;
                let offset = (roll % 12_001) as u32;
                self.timer = 12_000 + offset;
            }
        }
    }

    /// Sets the weather condition manually (e.g. for commands or debug testing).
    pub fn set_weather(&mut self, kind: WeatherKind, duration_ticks: u32) {
        self.kind = kind;
        self.timer = duration_ticks;
    }

    /// Triggers an immediate lightning strike flash event.
    pub fn trigger_lightning(&mut self) {
        self.lightning_flash_ticks = LIGHTNING_FLASH_DURATION_TICKS;
    }

    /// Returns the current lightning flash illumination fraction in `[0.0, 1.0]`.
    #[must_use]
    pub fn lightning_flash_factor(&self) -> f32 {
        if self.lightning_flash_ticks > 0 {
            f32::from(self.lightning_flash_ticks) / f32::from(LIGHTNING_FLASH_DURATION_TICKS)
        } else {
            0.0
        }
    }
}

/// Computes the type of precipitation (None, Rain, or Snow) for a given surface temperature and elevation.
#[must_use]
pub fn precipitation_at(
    temperature: f32,
    altitude_y: f32,
    is_dry_biome: bool,
) -> PrecipitationKind {
    if is_dry_biome {
        return PrecipitationKind::None;
    }

    // Altitude lapse rate: temperature decreases with altitude above sea level
    let lapse = (altitude_y - SEA_LEVEL_REF).max(0.0) * ALTITUDE_LAPSE_RATE;
    let effective_temp = temperature - lapse;

    if effective_temp < SNOW_TEMP_THRESHOLD {
        PrecipitationKind::Snow
    } else {
        PrecipitationKind::Rain
    }
}

/// Simple 64-bit `SplitMix` pseudo-random generator step.
const fn pseudo_rand_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_state_transitions() {
        let mut weather = WeatherState::new(42);
        assert_eq!(weather.kind, WeatherKind::Clear);
        assert_eq!(weather.rain_level, 0.0);
        assert_eq!(weather.thunder_level, 0.0);

        // Force timer to 1 tick
        weather.timer = 1;
        weather.tick();
        assert_eq!(weather.kind, WeatherKind::Rain);
        assert!(weather.timer >= 12_000);

        // Advance 100 ticks: rain_level should smoothly reach 1.0
        for _ in 0..100 {
            weather.tick();
        }
        assert!((weather.rain_level - 1.0).abs() < 1e-4);
        assert_eq!(weather.thunder_level, 0.0);
    }

    #[test]
    fn test_set_weather_and_thunder() {
        let mut weather = WeatherState::new(12345);
        weather.set_weather(WeatherKind::Thunder, 6_000);
        assert_eq!(weather.kind, WeatherKind::Thunder);
        assert_eq!(weather.timer, 6_000);

        for _ in 0..100 {
            weather.tick();
        }
        assert!((weather.rain_level - 1.0).abs() < 1e-4);
        assert!((weather.thunder_level - 1.0).abs() < 1e-4);

        weather.trigger_lightning();
        assert_eq!(
            weather.lightning_flash_ticks,
            LIGHTNING_FLASH_DURATION_TICKS
        );
        assert!((weather.lightning_flash_factor() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_precipitation_at_thresholds() {
        // Temperate plain at sea level: Rain
        assert_eq!(precipitation_at(0.5, 64.0, false), PrecipitationKind::Rain);

        // Desert at sea level: None
        assert_eq!(precipitation_at(0.8, 64.0, true), PrecipitationKind::None);

        // Cold region at sea level: Snow
        assert_eq!(precipitation_at(0.1, 64.0, false), PrecipitationKind::Snow);

        // Mountain at high elevation (lapse causes temp to drop below 0.15):
        // temp 0.3 at y = 200 -> lapse = (200 - 64) * 0.00125 = 136 * 0.00125 = 0.17
        // effective_temp = 0.3 - 0.17 = 0.13 < 0.15 -> Snow
        assert_eq!(precipitation_at(0.3, 200.0, false), PrecipitationKind::Snow);

        // Mountain at lower elevation:
        // temp 0.3 at y = 80 -> lapse = 16 * 0.00125 = 0.02 -> effective_temp = 0.28 -> Rain
        assert_eq!(precipitation_at(0.3, 80.0, false), PrecipitationKind::Rain);
    }
}
