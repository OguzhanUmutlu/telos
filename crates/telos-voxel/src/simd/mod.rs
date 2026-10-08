//! Vectorized SIMD operations for chunk occupancy and lighting.
//!
//! Dispatches to AVX2 when supported by host CPU, otherwise falling back
//! transparently to portable scalar algorithms.

pub mod light;
pub mod occupancy;
