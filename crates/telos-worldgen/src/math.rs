//! Deterministic math primitives and stateless positional hashing.

/// Finalizer of `SplitMix64` (Steele et al.).
#[inline]
#[must_use]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Stateless positional hash mapping (key, x, y, z) into a pseudo-random u64.
///
/// Ensures deterministic, order-independent randomness at any 3D coordinate.
#[inline]
#[must_use]
pub const fn hash3(key: u64, x: i32, y: i32, z: i32) -> u64 {
    let h = key
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    mix64(h)
}

/// Maps a 64-bit integer hash to an exact float in the half-open range `[0.0, 1.0)`.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn unit_f32(h: u64) -> f32 {
    (h >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// Standard linear interpolation between `a` and `b` by fraction `t`.
#[inline]
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite smoothstep interpolation clamped to `[0.0, 1.0]`.
#[inline]
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash3_determinism() {
        let h1 = hash3(42, 10, -5, 20);
        let h2 = hash3(42, 10, -5, 20);
        let h3 = hash3(42, 10, -5, 21);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_unit_f32_range() {
        for i in 0..1000 {
            let u = unit_f32(mix64(i));
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_smoothstep() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
    }
}
