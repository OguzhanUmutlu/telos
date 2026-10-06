//! Noise primitives and fractal evaluations.

pub mod simplex;

pub use simplex::{noise2, noise3};

/// Evaluates 2D Fractal Brownian Motion (fBm).
#[must_use]
pub fn fbm2d(
    seed: u64,
    mut x: f32,
    mut y: f32,
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut max_amp = 0.0;
    x *= frequency;
    y *= frequency;

    for i in 0..octaves {
        let octave_seed = seed.wrapping_add(i as u64 * 10007);
        total += noise2(octave_seed, x, y) * amp;
        max_amp += amp;
        amp *= gain;
        x *= lacunarity;
        y *= lacunarity;
    }

    if max_amp > 0.0 {
        total / max_amp
    } else {
        total
    }
}

/// Evaluates 3D Fractal Brownian Motion (fBm).
#[must_use]
pub fn fbm3d(
    seed: u64,
    mut x: f32,
    mut y: f32,
    mut z: f32,
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut max_amp = 0.0;
    x *= frequency;
    y *= frequency;
    z *= frequency;

    for i in 0..octaves {
        let octave_seed = seed.wrapping_add(i as u64 * 10007);
        total += noise3(octave_seed, x, y, z) * amp;
        max_amp += amp;
        amp *= gain;
        x *= lacunarity;
        y *= lacunarity;
        z *= lacunarity;
    }

    if max_amp > 0.0 {
        total / max_amp
    } else {
        total
    }
}

/// Evaluates 2D Ridged multifractal noise (used for mountain crests and ridges).
#[must_use]
pub fn ridged2d(
    seed: u64,
    mut x: f32,
    mut y: f32,
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut max_amp = 0.0;
    x *= frequency;
    y *= frequency;

    for i in 0..octaves {
        let octave_seed = seed.wrapping_add(i as u64 * 10007);
        let n = noise2(octave_seed, x, y);
        let ridged = 1.0 - n.abs();
        total += ridged * ridged * amp;
        max_amp += amp;
        amp *= gain;
        x *= lacunarity;
        y *= lacunarity;
    }

    if max_amp > 0.0 {
        total / max_amp
    } else {
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_noise_bounds() {
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = noise2(42, i as f32 * 0.1, i as f32 * 0.2);
            assert!(
                (-1.5..=1.5).contains(&v),
                "noise2 out of expected bound: {v}"
            );
        }
    }

    #[test]
    fn test_fbm2d_smoothness() {
        let v1 = fbm2d(1337, 10.0, 20.0, 4, 0.01, 2.0, 0.5);
        let v2 = fbm2d(1337, 10.05, 20.05, 4, 0.01, 2.0, 0.5);
        assert!((v1 - v2).abs() < 0.1);
    }
}
