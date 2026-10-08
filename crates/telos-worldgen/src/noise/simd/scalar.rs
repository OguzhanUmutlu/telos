//! Reference scalar batch implementations of 2D/3D noise and fractals.
//!
//! Serves as the portable fallback on non-AVX systems and as the ground-truth
//! reference for property and golden-hash tests.

#![allow(clippy::too_many_arguments)]

use crate::noise::simplex::{noise2, noise3};

/// Evaluates 8 samples of 2D Simplex noise sequentially.
#[inline]
pub fn noise2_batch_8_scalar(seed: u64, xs: &[f32; 8], ys: &[f32; 8], out: &mut [f32; 8]) {
    for i in 0..8 {
        out[i] = noise2(seed, xs[i], ys[i]);
    }
}

/// Evaluates 8 samples of 3D Simplex noise sequentially.
#[inline]
pub fn noise3_batch_8_scalar(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    zs: &[f32; 8],
    out: &mut [f32; 8],
) {
    for i in 0..8 {
        out[i] = noise3(seed, xs[i], ys[i], zs[i]);
    }
}

/// Evaluates 8 samples of 2D fBm noise sequentially.
#[inline]
pub fn fbm2d_batch_8_scalar(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    for i in 0..8 {
        out[i] = crate::noise::fbm2d(seed, xs[i], ys[i], octaves, frequency, lacunarity, gain);
    }
}

/// Evaluates 8 samples of 3D fBm noise sequentially.
#[inline]
pub fn fbm3d_batch_8_scalar(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    zs: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    for i in 0..8 {
        out[i] = crate::noise::fbm3d(
            seed, xs[i], ys[i], zs[i], octaves, frequency, lacunarity, gain,
        );
    }
}

/// Evaluates 8 samples of 2D ridged multifractal noise sequentially.
#[inline]
pub fn ridged2d_batch_8_scalar(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    for i in 0..8 {
        out[i] = crate::noise::ridged2d(seed, xs[i], ys[i], octaves, frequency, lacunarity, gain);
    }
}

/// Evaluates an arbitrary slice of 2D Simplex noise samples sequentially.
pub fn noise2_slice_scalar(seed: u64, xs: &[f32], ys: &[f32], out: &mut [f32]) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    for i in 0..xs.len() {
        out[i] = noise2(seed, xs[i], ys[i]);
    }
}

/// Evaluates an arbitrary slice of 3D Simplex noise samples sequentially.
pub fn noise3_slice_scalar(seed: u64, xs: &[f32], ys: &[f32], zs: &[f32], out: &mut [f32]) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), zs.len());
    assert_eq!(xs.len(), out.len());
    for i in 0..xs.len() {
        out[i] = noise3(seed, xs[i], ys[i], zs[i]);
    }
}

/// Evaluates an arbitrary slice of 2D fBm noise samples sequentially.
pub fn fbm2d_slice_scalar(
    seed: u64,
    xs: &[f32],
    ys: &[f32],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32],
) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    for i in 0..xs.len() {
        out[i] = crate::noise::fbm2d(seed, xs[i], ys[i], octaves, frequency, lacunarity, gain);
    }
}

/// Evaluates an arbitrary slice of 3D fBm noise samples sequentially.
pub fn fbm3d_slice_scalar(
    seed: u64,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32],
) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), zs.len());
    assert_eq!(xs.len(), out.len());
    for i in 0..xs.len() {
        out[i] = crate::noise::fbm3d(
            seed, xs[i], ys[i], zs[i], octaves, frequency, lacunarity, gain,
        );
    }
}
