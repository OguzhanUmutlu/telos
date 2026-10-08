//! Vectorized SIMD operations for chunk lighting and heightmaps.

#![allow(
    clippy::wildcard_imports,
    clippy::cast_ptr_alignment,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use telos_core::cpu::CpuFeatures;

/// Scans the 16 KiB nibble buffer and returns `Some(level)` if all 32,768 voxels share that level.
#[must_use]
pub fn try_collapse_simd(bytes: &[u8; 16384]) -> Option<u8> {
    let first_byte = bytes[0];
    let low = first_byte & 0x0F;
    let high = (first_byte >> 4) & 0x0F;
    if low != high {
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // SAFETY: Verified AVX2 capability.
        unsafe {
            if try_collapse_avx2(bytes, first_byte) {
                return Some(low);
            }
            return None;
        }
    }

    if try_collapse_scalar(bytes, first_byte) {
        Some(low)
    } else {
        None
    }
}

/// Portable scalar check for uniform nibble buffer using 64-bit word comparisons.
#[must_use]
pub fn try_collapse_scalar(bytes: &[u8; 16384], first_byte: u8) -> bool {
    let pattern_u64 = u64::from_ne_bytes([first_byte; 8]);
    let (prefix, u64_chunks, suffix) = unsafe { bytes.align_to::<u64>() };

    for &b in prefix {
        if b != first_byte {
            return false;
        }
    }
    for &chunk in u64_chunks {
        if chunk != pattern_u64 {
            return false;
        }
    }
    for &b in suffix {
        if b != first_byte {
            return false;
        }
    }

    true
}

/// AVX2 accelerated uniform scan checking 128 bytes per unrolled loop iteration.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn try_collapse_avx2(bytes: &[u8; 16384], first_byte: u8) -> bool {
    unsafe {
        let target = _mm256_set1_epi8(first_byte.cast_signed());
        let ptr = bytes.as_ptr().cast::<__m256i>();

        // 16384 bytes / 32 bytes = 512 vectors.
        // Unroll 4x: 128 iterations of 128 bytes each.
        for i in 0..128 {
            let base = i * 4;
            let v0 = _mm256_loadu_si256(ptr.add(base));
            let v1 = _mm256_loadu_si256(ptr.add(base + 1));
            let v2 = _mm256_loadu_si256(ptr.add(base + 2));
            let v3 = _mm256_loadu_si256(ptr.add(base + 3));

            let eq0 = _mm256_cmpeq_epi8(v0, target);
            let eq1 = _mm256_cmpeq_epi8(v1, target);
            let eq2 = _mm256_cmpeq_epi8(v2, target);
            let eq3 = _mm256_cmpeq_epi8(v3, target);

            let all_eq = _mm256_and_si256(_mm256_and_si256(eq0, eq1), _mm256_and_si256(eq2, eq3));
            if _mm256_movemask_epi8(all_eq) != -1 {
                return false;
            }
        }
        true
    }
}

/// Updates 1024 column heightmap values with new chunk local heights using SIMD when available.
pub fn update_column_heights_simd(top: &mut [i16; 1024], chunk_cy: i32, top_local: &[u8; 1024]) {
    let chunk_base_y = chunk_cy << 5;

    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().can_avx2() {
        // Prepare 1024 i16 candidates
        let mut candidates = [i16::MIN; 1024];
        for i in 0..1024 {
            let val = top_local[i];
            if val > 0 {
                candidates[i] = (chunk_base_y + i32::from(val - 1)) as i16;
            }
        }

        // SAFETY: Verified AVX2 capability.
        unsafe {
            update_column_heights_avx2(top, &candidates);
            return;
        }
    }

    // Scalar fallback
    for i in 0..1024 {
        let val = top_local[i];
        if val > 0 {
            let world_y = (chunk_base_y + i32::from(val - 1)) as i16;
            if world_y > top[i] {
                top[i] = world_y;
            }
        }
    }
}

/// AVX2 accelerated elementwise signed max over 1024 `i16` values (16 elements per instruction).
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn update_column_heights_avx2(top: &mut [i16; 1024], candidates: &[i16; 1024]) {
    unsafe {
        let top_ptr = top.as_mut_ptr().cast::<__m256i>();
        let cand_ptr = candidates.as_ptr().cast::<__m256i>();

        // 1024 * 2 bytes = 2048 bytes = 64 * 256-bit vectors
        for i in 0..64 {
            let v_top = _mm256_loadu_si256(top_ptr.add(i));
            let v_cand = _mm256_loadu_si256(cand_ptr.add(i));
            let v_max = _mm256_max_epi16(v_top, v_cand);
            _mm256_storeu_si256(top_ptr.add(i), v_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_collapse_agreement() {
        let mut bytes = Box::new([0xAA_u8; 16384]); // 10 and 10
        assert_eq!(try_collapse_simd(&bytes), Some(10));
        assert!(try_collapse_scalar(&bytes, 0xAA));

        // One mismatch
        bytes[8192] = 0xAB;
        assert_eq!(try_collapse_simd(&bytes), None);
        assert!(!try_collapse_scalar(&bytes, 0xAA));
    }

    #[test]
    fn test_update_column_heights_agreement() {
        let mut top_simd = [i16::MIN; 1024];
        let mut top_scalar = [i16::MIN; 1024];

        let mut top_local = [0u8; 1024];
        top_local[0] = 16;
        top_local[500] = 32;
        top_local[1023] = 1;

        update_column_heights_simd(&mut top_simd, 2, &top_local);

        // Compute scalar manually
        let chunk_base_y = 2 << 5;
        for i in 0..1024 {
            let val = top_local[i];
            if val > 0 {
                let world_y = (chunk_base_y + i32::from(val - 1)) as i16;
                if world_y > top_scalar[i] {
                    top_scalar[i] = world_y;
                }
            }
        }

        assert_eq!(top_simd, top_scalar);
    }
}
