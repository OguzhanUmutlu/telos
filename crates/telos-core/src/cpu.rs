//! Dynamic CPU feature detection and runtime SIMD backend dispatch.
//!
//! Provides zero-cost cached query of host CPU capabilities (AVX2, AVX-512, FMA, NEON)
//! without forcing compile-time target-cpu flags.

use std::sync::OnceLock;

/// Active SIMD execution tier selected for vectorized kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdBackend {
    /// Portable scalar baseline (universal fallback).
    Scalar,
    /// 256-bit AVX2 vectorization (`x86_64`).
    Avx2,
    /// 512-bit AVX-512 vectorization (`x86_64`).
    Avx512,
    /// 128-bit NEON vectorization (`aarch64`).
    Neon,
}

impl std::fmt::Display for SimdBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "Scalar"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
            Self::Neon => write!(f, "NEON"),
        }
    }
}

/// Discovered CPU capabilities on the running host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct CpuFeatures {
    /// Host CPU supports 256-bit AVX2 vector instructions.
    pub has_avx2: bool,
    /// Host CPU supports 512-bit AVX-512 foundation instructions (`avx512f`).
    pub has_avx512f: bool,
    /// Host CPU supports Fused Multiply-Add (`fma`).
    pub has_fma: bool,
    /// Host CPU supports ARM NEON vector instructions.
    pub has_neon: bool,
    /// Best available SIMD backend for this host (accounting for env overrides).
    pub backend: SimdBackend,
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

impl CpuFeatures {
    /// Detects host CPU features at runtime.
    ///
    /// Respects the `TELOS_SIMD` environment variable:
    /// - `"scalar"`: Forces scalar fallback even if AVX2/AVX-512 are present.
    /// - `"avx2"`: Limits dispatch to AVX2 even if AVX-512 is present.
    /// - `"avx512"`: Enables AVX-512 if hardware permits.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        let (raw_avx2, raw_avx512, raw_fma, raw_neon) = (
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f"),
            std::is_x86_feature_detected!("fma"),
            false,
        );

        #[cfg(target_arch = "aarch64")]
        let (raw_avx2, raw_avx512, raw_fma, raw_neon) = (false, false, false, true);

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (raw_avx2, raw_avx512, raw_fma, raw_neon) = (false, false, false, false);

        // Check for manual environment override (useful for CI, profiling, determinism verification)
        let override_val = std::env::var("TELOS_SIMD").ok();
        let override_str = override_val.as_deref().unwrap_or("").trim().to_lowercase();

        let (has_avx2, has_avx512f, has_fma, has_neon, backend) = match override_str.as_str() {
            "scalar" | "off" | "none" | "0" => (false, false, false, false, SimdBackend::Scalar),
            "avx2" => {
                let avx2 = raw_avx2;
                let b = if avx2 {
                    SimdBackend::Avx2
                } else {
                    SimdBackend::Scalar
                };
                (avx2, false, raw_fma && avx2, false, b)
            }
            "avx512" | "avx-512" => {
                let avx512 = raw_avx512;
                let avx2 = raw_avx2;
                let b = if avx512 {
                    SimdBackend::Avx512
                } else if avx2 {
                    SimdBackend::Avx2
                } else {
                    SimdBackend::Scalar
                };
                (avx2, avx512, raw_fma, false, b)
            }
            _ => {
                let b = if raw_avx512 {
                    SimdBackend::Avx512
                } else if raw_avx2 {
                    SimdBackend::Avx2
                } else if raw_neon {
                    SimdBackend::Neon
                } else {
                    SimdBackend::Scalar
                };
                (raw_avx2, raw_avx512, raw_fma, raw_neon, b)
            }
        };

        Self {
            has_avx2,
            has_avx512f,
            has_fma,
            has_neon,
            backend,
        }
    }

    /// Returns a cached reference to the host CPU features.
    #[inline]
    #[must_use]
    pub fn get() -> &'static Self {
        CPU_FEATURES.get_or_init(Self::detect)
    }

    /// Returns `true` if AVX2 instructions can be safely executed on this host.
    #[inline]
    #[must_use]
    pub fn can_avx2(&self) -> bool {
        self.has_avx2
    }

    /// Returns `true` if AVX-512 foundation instructions can be safely executed on this host.
    #[inline]
    #[must_use]
    pub fn can_avx512(&self) -> bool {
        self.has_avx512f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detect() {
        let features = CpuFeatures::detect();
        // Backend must match detected flags
        match features.backend {
            SimdBackend::Avx512 => {
                assert!(features.has_avx512f);
                assert!(features.has_avx2);
            }
            SimdBackend::Avx2 => {
                assert!(features.has_avx2);
            }
            SimdBackend::Neon => {
                assert!(features.has_neon);
            }
            SimdBackend::Scalar => {}
        }
    }

    #[test]
    fn test_cpu_features_singleton() {
        let f1 = CpuFeatures::get();
        let f2 = CpuFeatures::get();
        assert_eq!(f1, f2);
    }
}
