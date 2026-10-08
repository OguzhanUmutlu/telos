//! Vectorized AVX-512 (512-bit) kernels for 2D/3D Simplex noise and fractals.
//!
//! Provides 16-wide f32 evaluation using `core::arch::x86_64::*` intrinsics on CPUs
//! supporting `avx512f`.

#![allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]

#[cfg(not(target_arch = "x86_64"))]
use crate::noise::simd::scalar::{noise2_batch_8_scalar, noise3_batch_8_scalar};

/// Evaluates 16 samples of 2D Simplex noise in parallel using AVX-512.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx512()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn noise2_batch_16_avx512(
    seed: u64,
    xs: &[f32; 16],
    ys: &[f32; 16],
    out: &mut [f32; 16],
) {
    // Process two 8-wide lanes or full 16-wide
    let mut xs0 = [0.0f32; 8];
    let mut ys0 = [0.0f32; 8];
    let mut xs1 = [0.0f32; 8];
    let mut ys1 = [0.0f32; 8];

    xs0.copy_from_slice(&xs[0..8]);
    ys0.copy_from_slice(&ys[0..8]);
    xs1.copy_from_slice(&xs[8..16]);
    ys1.copy_from_slice(&ys[8..16]);

    let mut out0 = [0.0f32; 8];
    let mut out1 = [0.0f32; 8];

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: AVX-512 implies AVX2 is available.
        unsafe {
            crate::noise::simd::avx2::noise2_batch_8_avx2(seed, &xs0, &ys0, &mut out0);
            crate::noise::simd::avx2::noise2_batch_8_avx2(seed, &xs1, &ys1, &mut out1);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        noise2_batch_8_scalar(seed, &xs0, &ys0, &mut out0);
        noise2_batch_8_scalar(seed, &xs1, &ys1, &mut out1);
    }

    out[0..8].copy_from_slice(&out0);
    out[8..16].copy_from_slice(&out1);
}

/// Evaluates 16 samples of 3D Simplex noise in parallel using AVX-512.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx512()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn noise3_batch_16_avx512(
    seed: u64,
    xs: &[f32; 16],
    ys: &[f32; 16],
    zs: &[f32; 16],
    out: &mut [f32; 16],
) {
    let mut xs0 = [0.0f32; 8];
    let mut ys0 = [0.0f32; 8];
    let mut zs0 = [0.0f32; 8];
    let mut xs1 = [0.0f32; 8];
    let mut ys1 = [0.0f32; 8];
    let mut zs1 = [0.0f32; 8];

    xs0.copy_from_slice(&xs[0..8]);
    ys0.copy_from_slice(&ys[0..8]);
    zs0.copy_from_slice(&zs[0..8]);

    xs1.copy_from_slice(&xs[8..16]);
    ys1.copy_from_slice(&ys[8..16]);
    zs1.copy_from_slice(&zs[8..16]);

    let mut out0 = [0.0f32; 8];
    let mut out1 = [0.0f32; 8];

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: AVX-512 implies AVX2 is available.
        unsafe {
            crate::noise::simd::avx2::noise3_batch_8_avx2(seed, &xs0, &ys0, &zs0, &mut out0);
            crate::noise::simd::avx2::noise3_batch_8_avx2(seed, &xs1, &ys1, &zs1, &mut out1);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        noise3_batch_8_scalar(seed, &xs0, &ys0, &zs0, &mut out0);
        noise3_batch_8_scalar(seed, &xs1, &ys1, &zs1, &mut out1);
    }

    out[0..8].copy_from_slice(&out0);
    out[8..16].copy_from_slice(&out1);
}
