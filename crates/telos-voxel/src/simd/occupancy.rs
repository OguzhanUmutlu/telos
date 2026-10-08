//! Vectorized SIMD operations for chunk occupancy bitmasks.

#![allow(
    clippy::wildcard_imports,
    clippy::cast_ptr_alignment,
    clippy::many_single_char_names
)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use telos_core::cpu::CpuFeatures;

/// Checks if all 1024 32-bit column bitboards in `col_y` are zero.
#[inline]
#[must_use]
pub fn is_empty_simd(col_y: &[u32; 1024]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability.
        unsafe {
            return is_empty_avx2(col_y);
        }
    }

    is_empty_scalar(col_y)
}

/// Portable scalar check for empty occupancy.
#[inline]
#[must_use]
pub fn is_empty_scalar(col_y: &[u32; 1024]) -> bool {
    col_y.iter().all(|&c| c == 0)
}

/// AVX2 accelerated check scanning 32 bytes (256 bits) per cycle using `_mm256_testz_si256`.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn is_empty_avx2(col_y: &[u32; 1024]) -> bool {
    unsafe {
        let ptr = col_y.as_ptr().cast::<__m256i>();
        // 1024 * 4 bytes = 4096 bytes = 128 * 256-bit vectors
        for i in 0..128 {
            let v = _mm256_loadu_si256(ptr.add(i));
            if _mm256_testz_si256(v, v) == 0 {
                return false;
            }
        }
        true
    }
}

/// Computes local height values from column masks (1024 columns).
#[inline]
pub fn from_column_masks_simd(masks: &[u32; 1024], top_local: &mut [u8; 1024]) {
    // Unrolled fast path with 8 columns per batch
    for i in (0..1024).step_by(8) {
        for k in 0..8 {
            let idx = i + k;
            let m = masks[idx];
            top_local[idx] = if m != 0 {
                (32 - m.leading_zeros()) as u8
            } else {
                0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_empty_agreement() {
        let mut col_y = [0u32; 1024];
        assert_eq!(is_empty_scalar(&col_y), is_empty_simd(&col_y));
        assert!(is_empty_simd(&col_y));

        col_y[512] = 1;
        assert_eq!(is_empty_scalar(&col_y), is_empty_simd(&col_y));
        assert!(!is_empty_simd(&col_y));

        col_y[512] = 0;
        col_y[1023] = 0x8000_0000;
        assert_eq!(is_empty_scalar(&col_y), is_empty_simd(&col_y));
        assert!(!is_empty_simd(&col_y));
    }
}
