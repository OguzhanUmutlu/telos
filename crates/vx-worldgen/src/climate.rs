//! Continuous multi-noise climate parameters.

use crate::noise::fbm2d;

/// Climate parameters sampled at a 2D world position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimatePoint {
    /// Continentalness: ocean vs land (-1.0 = deep ocean, 1.0 = far inland).
    pub continentalness: f32,
    /// Erosion: flat terrain vs mountainous (-1.0 = mountainous, 1.0 = flat).
    pub erosion: f32,
    /// Weirdness: base variation input.
    pub weirdness: f32,
    /// Peaks & Valleys (derived from Weirdness): -1.0 = valleys/rivers, 1.0 = high peaks.
    pub peaks_and_valleys: f32,
    /// Temperature: cold vs hot (-1.0 = frozen, 1.0 = arid/tropical).
    pub temperature: f32,
    /// Humidity: dry vs wet (-1.0 = desert, 1.0 = rainforest/swamp).
    pub humidity: f32,
}

impl ClimatePoint {
    /// Evaluates continuous climate parameters at world coordinates `(x, z)`.
    #[must_use]
    pub fn sample(seed: u64, x: f32, z: f32) -> Self {
        let c_seed = seed.wrapping_add(101);
        let e_seed = seed.wrapping_add(202);
        let w_seed = seed.wrapping_add(303);
        let t_seed = seed.wrapping_add(404);
        let h_seed = seed.wrapping_add(505);

        // Continentalness: large scale (~1/2048 blocks)
        let continentalness = fbm2d(c_seed, x, z, 4, 1.0 / 2048.0, 2.0, 0.5).clamp(-1.0, 1.0);

        // Erosion: mid scale (~1/1024 blocks)
        let erosion = fbm2d(e_seed, x, z, 4, 1.0 / 1024.0, 2.0, 0.5).clamp(-1.0, 1.0);

        // Weirdness: (~1/1024 blocks)
        let weirdness = fbm2d(w_seed, x, z, 4, 1.0 / 1024.0, 2.0, 0.5).clamp(-1.0, 1.0);

        // Peaks & Valleys: PV = 1 - |3|W| - 2|
        let peaks_and_valleys = (1.0 - (3.0 * weirdness.abs() - 2.0).abs()).clamp(-1.0, 1.0);

        // Temperature: large scale (~1/3072 blocks)
        let temperature = fbm2d(t_seed, x, z, 4, 1.0 / 3072.0, 2.0, 0.5).clamp(-1.0, 1.0);

        // Humidity: (~1/2048 blocks)
        let humidity = fbm2d(h_seed, x, z, 4, 1.0 / 2048.0, 2.0, 0.5).clamp(-1.0, 1.0);

        Self {
            continentalness,
            erosion,
            weirdness,
            peaks_and_valleys,
            temperature,
            humidity,
        }
    }
}
