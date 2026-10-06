//! Biome classification and fast climate-space nearest lookup.

use crate::climate::ClimatePoint;

/// Distinct biome identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum BiomeId {
    /// Ocean biome (deep water, gravel/sand beds).
    Ocean = 0,
    /// Plains biome (flat or gentle rolling grass fields).
    Plains = 1,
    /// Forest biome (lush grass, temperate trees).
    Forest = 2,
    /// Desert biome (arid, sandy expanse).
    Desert = 3,
    /// Mountains biome (high peaks, exposed stone, jagged crests).
    Mountains = 4,
}

impl BiomeId {
    /// Friendly name of this biome.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ocean => "ocean",
            Self::Plains => "plains",
            Self::Forest => "forest",
            Self::Desert => "desert",
            Self::Mountains => "mountains",
        }
    }
}

/// Target parameters for a registered biome in climate space.
#[derive(Debug, Clone, Copy)]
pub struct BiomeTarget {
    /// Biome identifier.
    pub id: BiomeId,
    /// Target continentalness value in [-1.0, 1.0].
    pub continentalness: f32,
    /// Target erosion value in [-1.0, 1.0].
    pub erosion: f32,
    /// Target peaks and valleys value in [-1.0, 1.0].
    pub peaks_and_valleys: f32,
    /// Target temperature value in [-1.0, 1.0].
    pub temperature: f32,
    /// Target humidity value in [-1.0, 1.0].
    pub humidity: f32,
}

/// Pre-configured standard biomes table.
pub const STANDARD_BIOMES: [BiomeTarget; 5] = [
    // Ocean: negative continentalness
    BiomeTarget {
        id: BiomeId::Ocean,
        continentalness: -0.6,
        erosion: 0.0,
        peaks_and_valleys: -0.5,
        temperature: 0.0,
        humidity: 0.0,
    },
    // Plains: low peaks, moderate erosion, temperate
    BiomeTarget {
        id: BiomeId::Plains,
        continentalness: 0.2,
        erosion: 0.3,
        peaks_and_valleys: -0.2,
        temperature: 0.1,
        humidity: 0.1,
    },
    // Forest: moderate erosion, higher humidity
    BiomeTarget {
        id: BiomeId::Forest,
        continentalness: 0.3,
        erosion: 0.0,
        peaks_and_valleys: 0.1,
        temperature: 0.1,
        humidity: 0.6,
    },
    // Desert: high temperature, very low humidity
    BiomeTarget {
        id: BiomeId::Desert,
        continentalness: 0.2,
        erosion: 0.2,
        peaks_and_valleys: -0.1,
        temperature: 0.8,
        humidity: -0.7,
    },
    // Mountains: low erosion, high peaks & valleys
    BiomeTarget {
        id: BiomeId::Mountains,
        continentalness: 0.4,
        erosion: -0.6,
        peaks_and_valleys: 0.8,
        temperature: -0.3,
        humidity: 0.0,
    },
];

/// Finds the nearest biome for the given continuous climate point.
#[must_use]
pub fn lookup_biome(climate: &ClimatePoint) -> BiomeId {
    // If continentalness is strongly negative, it's always ocean
    if climate.continentalness < -0.2 {
        return BiomeId::Ocean;
    }

    let mut best_id = BiomeId::Plains;
    let mut best_dist = f32::MAX;

    for target in &STANDARD_BIOMES {
        if target.id == BiomeId::Ocean && climate.continentalness >= -0.2 {
            continue;
        }

        let dc = climate.continentalness - target.continentalness;
        let de = climate.erosion - target.erosion;
        let dpv = climate.peaks_and_valleys - target.peaks_and_valleys;
        let dt = climate.temperature - target.temperature;
        let dh = climate.humidity - target.humidity;

        // Weighting factors
        let dist = dc * dc * 1.5 + de * de * 2.0 + dpv * dpv * 2.5 + dt * dt * 1.5 + dh * dh * 1.5;

        if dist < best_dist {
            best_dist = dist;
            best_id = target.id;
        }
    }

    best_id
}
