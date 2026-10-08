//! Dynamic runtime SIMD dispatch for 2D/3D Simplex noise and fractals.
//!
//! Automatically selects the fastest available vectorized kernel on the running host
//! (AVX-512, AVX2, or portable scalar fallback) based on `telos_core::cpu::CpuFeatures`.

#![allow(clippy::too_many_arguments)]

pub mod avx2;
pub mod avx512;
pub mod scalar;

use telos_core::cpu::CpuFeatures;

/// Evaluates 8 samples of 2D Simplex noise using the best available SIMD backend.
#[inline]
pub fn noise2_batch_8(seed: u64, xs: &[f32; 8], ys: &[f32; 8], out: &mut [f32; 8]) {
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability via CpuFeatures.
        unsafe {
            avx2::noise2_batch_8_avx2(seed, xs, ys, out);
            return;
        }
    }

    scalar::noise2_batch_8_scalar(seed, xs, ys, out);
}

/// Evaluates 8 samples of 3D Simplex noise using the best available SIMD backend.
#[inline]
pub fn noise3_batch_8(seed: u64, xs: &[f32; 8], ys: &[f32; 8], zs: &[f32; 8], out: &mut [f32; 8]) {
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability via CpuFeatures.
        unsafe {
            avx2::noise3_batch_8_avx2(seed, xs, ys, zs, out);
            return;
        }
    }

    scalar::noise3_batch_8_scalar(seed, xs, ys, zs, out);
}

/// Evaluates 8 samples of 2D Fractal Brownian Motion (fBm) using the best available SIMD backend.
#[inline]
pub fn fbm2d_batch_8(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability via CpuFeatures.
        unsafe {
            avx2::fbm2d_batch_8_avx2(seed, xs, ys, octaves, frequency, lacunarity, gain, out);
            return;
        }
    }

    scalar::fbm2d_batch_8_scalar(seed, xs, ys, octaves, frequency, lacunarity, gain, out);
}

/// Evaluates 8 samples of 3D Fractal Brownian Motion (fBm) using the best available SIMD backend.
#[inline]
pub fn fbm3d_batch_8(
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
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability via CpuFeatures.
        unsafe {
            avx2::fbm3d_batch_8_avx2(seed, xs, ys, zs, octaves, frequency, lacunarity, gain, out);
            return;
        }
    }

    scalar::fbm3d_batch_8_scalar(seed, xs, ys, zs, octaves, frequency, lacunarity, gain, out);
}

/// Evaluates 8 samples of 2D ridged multifractal noise using the best available SIMD backend.
#[inline]
pub fn ridged2d_batch_8(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability via CpuFeatures.
        unsafe {
            avx2::ridged2d_batch_8_avx2(seed, xs, ys, octaves, frequency, lacunarity, gain, out);
            return;
        }
    }

    scalar::ridged2d_batch_8_scalar(seed, xs, ys, octaves, frequency, lacunarity, gain, out);
}

/// Evaluates an arbitrary slice of 2D fBm noise samples in SIMD 8-wide batches with scalar tail.
pub fn fbm2d_slice(
    seed: u64,
    xs: &[f32],
    ys: &[f32],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32],
) {
    let len = xs.len();
    assert_eq!(len, ys.len());
    assert_eq!(len, out.len());

    let chunks = len / 8;
    let mut batch_x = [0.0f32; 8];
    let mut batch_y = [0.0f32; 8];
    let mut batch_out = [0.0f32; 8];

    for c in 0..chunks {
        let base = c * 8;
        batch_x.copy_from_slice(&xs[base..base + 8]);
        batch_y.copy_from_slice(&ys[base..base + 8]);

        fbm2d_batch_8(
            seed,
            &batch_x,
            &batch_y,
            octaves,
            frequency,
            lacunarity,
            gain,
            &mut batch_out,
        );
        out[base..base + 8].copy_from_slice(&batch_out);
    }

    let tail_start = chunks * 8;
    for i in tail_start..len {
        out[i] = crate::noise::fbm2d(seed, xs[i], ys[i], octaves, frequency, lacunarity, gain);
    }
}

/// Evaluates an arbitrary slice of 3D fBm noise samples in SIMD 8-wide batches with scalar tail.
pub fn fbm3d_slice(
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
    let len = xs.len();
    assert_eq!(len, ys.len());
    assert_eq!(len, zs.len());
    assert_eq!(len, out.len());

    let chunks = len / 8;
    let mut batch_x = [0.0f32; 8];
    let mut batch_y = [0.0f32; 8];
    let mut batch_z = [0.0f32; 8];
    let mut batch_out = [0.0f32; 8];

    for c in 0..chunks {
        let base = c * 8;
        batch_x.copy_from_slice(&xs[base..base + 8]);
        batch_y.copy_from_slice(&ys[base..base + 8]);
        batch_z.copy_from_slice(&zs[base..base + 8]);

        fbm3d_batch_8(
            seed,
            &batch_x,
            &batch_y,
            &batch_z,
            octaves,
            frequency,
            lacunarity,
            gain,
            &mut batch_out,
        );
        out[base..base + 8].copy_from_slice(&batch_out);
    }

    let tail_start = chunks * 8;
    for i in tail_start..len {
        out[i] = crate::noise::fbm3d(
            seed, xs[i], ys[i], zs[i], octaves, frequency, lacunarity, gain,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise2_simd_vs_scalar_agreement() {
        let seed = 0x1234_5678_9ABC_DEF0;
        let mut xs = [0.0f32; 8];
        let mut ys = [0.0f32; 8];

        for i in 0..8 {
            xs[i] = (i as f32) * 1.37 + 0.12;
            ys[i] = (i as f32) * -2.41 + 0.58;
        }

        let mut out_simd = [0.0f32; 8];
        let mut out_scalar = [0.0f32; 8];

        noise2_batch_8(seed, &xs, &ys, &mut out_simd);
        scalar::noise2_batch_8_scalar(seed, &xs, &ys, &mut out_scalar);

        for i in 0..8 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                diff < 1e-5,
                "Lanes at index {i} mismatch: simd={}, scalar={}, diff={diff}",
                out_simd[i],
                out_scalar[i]
            );
        }
    }

    #[test]
    fn test_noise3_simd_vs_scalar_agreement() {
        let seed = 0x9ABC_DEF0_1234_5678;
        let mut xs = [0.0f32; 8];
        let mut ys = [0.0f32; 8];
        let mut zs = [0.0f32; 8];

        for i in 0..8 {
            xs[i] = (i as f32) * 0.77 + 0.31;
            ys[i] = (i as f32) * 1.55 - 0.22;
            zs[i] = (i as f32) * -0.93 + 1.14;
        }

        let mut out_simd = [0.0f32; 8];
        let mut out_scalar = [0.0f32; 8];

        noise3_batch_8(seed, &xs, &ys, &zs, &mut out_simd);
        scalar::noise3_batch_8_scalar(seed, &xs, &ys, &zs, &mut out_scalar);

        for i in 0..8 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                diff < 1e-5,
                "Lanes at index {i} mismatch: simd={}, scalar={}, diff={diff}",
                out_simd[i],
                out_scalar[i]
            );
        }
    }

    #[test]
    fn test_fbm3d_slice_agreement() {
        let seed = 42;
        let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let ys: Vec<f32> = (0..20).map(|i| i as f32 * -0.2).collect();
        let zs: Vec<f32> = (0..20).map(|i| i as f32 * 0.3).collect();

        let mut out_simd = vec![0.0f32; 20];
        let mut out_scalar = vec![0.0f32; 20];

        fbm3d_slice(seed, &xs, &ys, &zs, 2, 0.05, 2.0, 0.5, &mut out_simd);
        scalar::fbm3d_slice_scalar(seed, &xs, &ys, &zs, 2, 0.05, 2.0, 0.5, &mut out_scalar);

        for i in 0..20 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at slice index {i}: simd={}, scalar={}, diff={diff}",
                out_simd[i],
                out_scalar[i]
            );
        }
    }
}
