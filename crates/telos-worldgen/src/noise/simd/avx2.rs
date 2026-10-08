//! Vectorized AVX2 (256-bit) kernels for 2D/3D Simplex noise and fractals.
//!
//! Provides 8-wide f32 evaluation using `core::arch::x86_64::*` intrinsics.
//! Guaranteed to maintain mathematical determinism with scalar reference code.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::wildcard_imports,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::math::hash3;

const F2: f32 = 0.366_025_4; // 0.5 * (sqrt(3.0) - 1.0)
const G2: f32 = 0.211_324_87; // (3.0 - sqrt(3.0)) / 6.0

const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

#[inline]
fn grad2(hash: u64, dx: f32, dy: f32) -> f32 {
    let h = (hash & 7) as u8;
    let u = if h < 4 { dx } else { dy };
    let v = if h < 4 { dy } else { dx };
    let g1 = if (h & 1) == 0 { u } else { -u };
    let g2 = if (h & 2) == 0 { v } else { -v };
    g1 + g2
}

#[inline]
fn grad3(hash: u64, dx: f32, dy: f32, dz: f32) -> f32 {
    let h = (hash & 15) as u8;
    let u = if h < 8 { dx } else { dy };
    let v = if h < 4 {
        dy
    } else if h == 12 || h == 14 {
        dx
    } else {
        dz
    };
    let g1 = if (h & 1) == 0 { u } else { -u };
    let g2 = if (h & 2) == 0 { v } else { -v };
    g1 + g2
}

/// Evaluates 8 samples of 2D Simplex noise in parallel using AVX2.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn noise2_batch_8_avx2(seed: u64, xs: &[f32; 8], ys: &[f32; 8], out: &mut [f32; 8]) {
    unsafe {
        let vx = _mm256_loadu_ps(xs.as_ptr());
        let vy = _mm256_loadu_ps(ys.as_ptr());

        let s = _mm256_mul_ps(_mm256_add_ps(vx, vy), _mm256_set1_ps(F2));
        let xs_s = _mm256_add_ps(vx, s);
        let ys_s = _mm256_add_ps(vy, s);

        let i_f = _mm256_floor_ps(xs_s);
        let j_f = _mm256_floor_ps(ys_s);

        let t = _mm256_mul_ps(_mm256_add_ps(i_f, j_f), _mm256_set1_ps(G2));
        let x0 = _mm256_sub_ps(vx, _mm256_sub_ps(i_f, t));
        let y0 = _mm256_sub_ps(vy, _mm256_sub_ps(j_f, t));

        let cmp_gt = _mm256_cmp_ps(x0, y0, _CMP_GT_OQ);
        let i1_f = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0), cmp_gt);
        let j1_f = _mm256_blendv_ps(_mm256_set1_ps(1.0), _mm256_setzero_ps(), cmp_gt);

        let x1 = _mm256_add_ps(_mm256_sub_ps(x0, i1_f), _mm256_set1_ps(G2));
        let y1 = _mm256_add_ps(_mm256_sub_ps(y0, j1_f), _mm256_set1_ps(G2));

        let x2 = _mm256_add_ps(
            _mm256_sub_ps(x0, _mm256_set1_ps(1.0)),
            _mm256_set1_ps(2.0 * G2),
        );
        let y2 = _mm256_add_ps(
            _mm256_sub_ps(y0, _mm256_set1_ps(1.0)),
            _mm256_set1_ps(2.0 * G2),
        );

        let t0 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(_mm256_set1_ps(0.5), _mm256_mul_ps(x0, x0)),
                _mm256_mul_ps(y0, y0),
            ),
            _mm256_setzero_ps(),
        );
        let t1 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(_mm256_set1_ps(0.5), _mm256_mul_ps(x1, x1)),
                _mm256_mul_ps(y1, y1),
            ),
            _mm256_setzero_ps(),
        );
        let t2 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(_mm256_set1_ps(0.5), _mm256_mul_ps(x2, x2)),
                _mm256_mul_ps(y2, y2),
            ),
            _mm256_setzero_ps(),
        );

        let t0_2 = _mm256_mul_ps(t0, t0);
        let t0_4 = _mm256_mul_ps(t0_2, t0_2);

        let t1_2 = _mm256_mul_ps(t1, t1);
        let t1_4 = _mm256_mul_ps(t1_2, t1_2);

        let t2_2 = _mm256_mul_ps(t2, t2);
        let t2_4 = _mm256_mul_ps(t2_2, t2_2);

        let mut i_arr = [0.0f32; 8];
        let mut j_arr = [0.0f32; 8];
        let mut i1_arr = [0.0f32; 8];
        let mut j1_arr = [0.0f32; 8];
        let mut x0_arr = [0.0f32; 8];
        let mut y0_arr = [0.0f32; 8];
        let mut x1_arr = [0.0f32; 8];
        let mut y1_arr = [0.0f32; 8];
        let mut x2_arr = [0.0f32; 8];
        let mut y2_arr = [0.0f32; 8];

        _mm256_storeu_ps(i_arr.as_mut_ptr(), i_f);
        _mm256_storeu_ps(j_arr.as_mut_ptr(), j_f);
        _mm256_storeu_ps(i1_arr.as_mut_ptr(), i1_f);
        _mm256_storeu_ps(j1_arr.as_mut_ptr(), j1_f);
        _mm256_storeu_ps(x0_arr.as_mut_ptr(), x0);
        _mm256_storeu_ps(y0_arr.as_mut_ptr(), y0);
        _mm256_storeu_ps(x1_arr.as_mut_ptr(), x1);
        _mm256_storeu_ps(y1_arr.as_mut_ptr(), y1);
        _mm256_storeu_ps(x2_arr.as_mut_ptr(), x2);
        _mm256_storeu_ps(y2_arr.as_mut_ptr(), y2);

        let mut g0 = [0.0f32; 8];
        let mut g1 = [0.0f32; 8];
        let mut g2 = [0.0f32; 8];

        for k in 0..8 {
            let i = i_arr[k] as i32;
            let j = j_arr[k] as i32;
            let i1 = i1_arr[k] as i32;
            let j1 = j1_arr[k] as i32;

            g0[k] = grad2(hash3(seed, i, j, 0), x0_arr[k], y0_arr[k]);
            g1[k] = grad2(hash3(seed, i + i1, j + j1, 0), x1_arr[k], y1_arr[k]);
            g2[k] = grad2(hash3(seed, i + 1, j + 1, 0), x2_arr[k], y2_arr[k]);
        }

        let g0_v = _mm256_loadu_ps(g0.as_ptr());
        let g1_v = _mm256_loadu_ps(g1.as_ptr());
        let g2_v = _mm256_loadu_ps(g2.as_ptr());

        let n0 = _mm256_mul_ps(t0_4, g0_v);
        let n1 = _mm256_mul_ps(t1_4, g1_v);
        let n2 = _mm256_mul_ps(t2_4, g2_v);

        let sum = _mm256_add_ps(_mm256_add_ps(n0, n1), n2);
        let res = _mm256_mul_ps(sum, _mm256_set1_ps(70.148));

        _mm256_storeu_ps(out.as_mut_ptr(), res);
    }
}

/// Evaluates 8 samples of 3D Simplex noise in parallel using AVX2.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn noise3_batch_8_avx2(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    zs: &[f32; 8],
    out: &mut [f32; 8],
) {
    unsafe {
        let vx = _mm256_loadu_ps(xs.as_ptr());
        let vy = _mm256_loadu_ps(ys.as_ptr());
        let vz = _mm256_loadu_ps(zs.as_ptr());

        let s = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(vx, vy), vz), _mm256_set1_ps(F3));
        let i_f = _mm256_floor_ps(_mm256_add_ps(vx, s));
        let j_f = _mm256_floor_ps(_mm256_add_ps(vy, s));
        let k_f = _mm256_floor_ps(_mm256_add_ps(vz, s));

        let t = _mm256_mul_ps(
            _mm256_add_ps(_mm256_add_ps(i_f, j_f), k_f),
            _mm256_set1_ps(G3),
        );
        let x0 = _mm256_sub_ps(vx, _mm256_sub_ps(i_f, t));
        let y0 = _mm256_sub_ps(vy, _mm256_sub_ps(j_f, t));
        let z0 = _mm256_sub_ps(vz, _mm256_sub_ps(k_f, t));

        let mut x0_arr = [0.0f32; 8];
        let mut y0_arr = [0.0f32; 8];
        let mut z0_arr = [0.0f32; 8];
        let mut i_arr = [0.0f32; 8];
        let mut j_arr = [0.0f32; 8];
        let mut k_arr = [0.0f32; 8];

        _mm256_storeu_ps(x0_arr.as_mut_ptr(), x0);
        _mm256_storeu_ps(y0_arr.as_mut_ptr(), y0);
        _mm256_storeu_ps(z0_arr.as_mut_ptr(), z0);
        _mm256_storeu_ps(i_arr.as_mut_ptr(), i_f);
        _mm256_storeu_ps(j_arr.as_mut_ptr(), j_f);
        _mm256_storeu_ps(k_arr.as_mut_ptr(), k_f);

        let mut x1_arr = [0.0f32; 8];
        let mut y1_arr = [0.0f32; 8];
        let mut z1_arr = [0.0f32; 8];

        let mut x2_arr = [0.0f32; 8];
        let mut y2_arr = [0.0f32; 8];
        let mut z2_arr = [0.0f32; 8];

        let mut g0 = [0.0f32; 8];
        let mut g1 = [0.0f32; 8];
        let mut g2 = [0.0f32; 8];
        let mut g3 = [0.0f32; 8];

        for idx in 0..8 {
            let x0_val = x0_arr[idx];
            let y0_val = y0_arr[idx];
            let z0_val = z0_arr[idx];

            let (i1, j1, k1, i2, j2, k2) = if x0_val >= y0_val {
                if y0_val >= z0_val {
                    (1, 0, 0, 1, 1, 0)
                } else if x0_val >= z0_val {
                    (1, 0, 0, 1, 0, 1)
                } else {
                    (0, 0, 1, 1, 0, 1)
                }
            } else if y0_val < z0_val {
                (0, 0, 1, 0, 1, 1)
            } else if x0_val < z0_val {
                (0, 1, 0, 0, 1, 1)
            } else {
                (0, 1, 0, 1, 1, 0)
            };

            let x1_val = x0_val - i1 as f32 + G3;
            let y1_val = y0_val - j1 as f32 + G3;
            let z1_val = z0_val - k1 as f32 + G3;

            let x2_val = x0_val - i2 as f32 + 2.0 * G3;
            let y2_val = y0_val - j2 as f32 + 2.0 * G3;
            let z2_val = z0_val - k2 as f32 + 2.0 * G3;

            x1_arr[idx] = x1_val;
            y1_arr[idx] = y1_val;
            z1_arr[idx] = z1_val;

            x2_arr[idx] = x2_val;
            y2_arr[idx] = y2_val;
            z2_arr[idx] = z2_val;

            let i = i_arr[idx] as i32;
            let j = j_arr[idx] as i32;
            let k = k_arr[idx] as i32;

            let x3_val = x0_val - 1.0 + 3.0 * G3;
            let y3_val = y0_val - 1.0 + 3.0 * G3;
            let z3_val = z0_val - 1.0 + 3.0 * G3;

            g0[idx] = grad3(hash3(seed, i, j, k), x0_val, y0_val, z0_val);
            g1[idx] = grad3(hash3(seed, i + i1, j + j1, k + k1), x1_val, y1_val, z1_val);
            g2[idx] = grad3(hash3(seed, i + i2, j + j2, k + k2), x2_val, y2_val, z2_val);
            g3[idx] = grad3(hash3(seed, i + 1, j + 1, k + 1), x3_val, y3_val, z3_val);
        }

        let x1 = _mm256_loadu_ps(x1_arr.as_ptr());
        let y1 = _mm256_loadu_ps(y1_arr.as_ptr());
        let z1 = _mm256_loadu_ps(z1_arr.as_ptr());

        let x2 = _mm256_loadu_ps(x2_arr.as_ptr());
        let y2 = _mm256_loadu_ps(y2_arr.as_ptr());
        let z2 = _mm256_loadu_ps(z2_arr.as_ptr());

        let x3 = _mm256_add_ps(
            _mm256_sub_ps(x0, _mm256_set1_ps(1.0)),
            _mm256_set1_ps(3.0 * G3),
        );
        let y3 = _mm256_add_ps(
            _mm256_sub_ps(y0, _mm256_set1_ps(1.0)),
            _mm256_set1_ps(3.0 * G3),
        );
        let z3 = _mm256_add_ps(
            _mm256_sub_ps(z0, _mm256_set1_ps(1.0)),
            _mm256_set1_ps(3.0 * G3),
        );

        let t0 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_set1_ps(0.6), _mm256_mul_ps(x0, x0)),
                    _mm256_mul_ps(y0, y0),
                ),
                _mm256_mul_ps(z0, z0),
            ),
            _mm256_setzero_ps(),
        );
        let t1 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_set1_ps(0.6), _mm256_mul_ps(x1, x1)),
                    _mm256_mul_ps(y1, y1),
                ),
                _mm256_mul_ps(z1, z1),
            ),
            _mm256_setzero_ps(),
        );
        let t2 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_set1_ps(0.6), _mm256_mul_ps(x2, x2)),
                    _mm256_mul_ps(y2, y2),
                ),
                _mm256_mul_ps(z2, z2),
            ),
            _mm256_setzero_ps(),
        );
        let t3 = _mm256_max_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_set1_ps(0.6), _mm256_mul_ps(x3, x3)),
                    _mm256_mul_ps(y3, y3),
                ),
                _mm256_mul_ps(z3, z3),
            ),
            _mm256_setzero_ps(),
        );

        let t0_2 = _mm256_mul_ps(t0, t0);
        let t0_4 = _mm256_mul_ps(t0_2, t0_2);

        let t1_2 = _mm256_mul_ps(t1, t1);
        let t1_4 = _mm256_mul_ps(t1_2, t1_2);

        let t2_2 = _mm256_mul_ps(t2, t2);
        let t2_4 = _mm256_mul_ps(t2_2, t2_2);

        let t3_2 = _mm256_mul_ps(t3, t3);
        let t3_4 = _mm256_mul_ps(t3_2, t3_2);

        let g0_v = _mm256_loadu_ps(g0.as_ptr());
        let g1_v = _mm256_loadu_ps(g1.as_ptr());
        let g2_v = _mm256_loadu_ps(g2.as_ptr());
        let g3_v = _mm256_loadu_ps(g3.as_ptr());

        let n0 = _mm256_mul_ps(t0_4, g0_v);
        let n1 = _mm256_mul_ps(t1_4, g1_v);
        let n2 = _mm256_mul_ps(t2_4, g2_v);
        let n3 = _mm256_mul_ps(t3_4, g3_v);

        let sum = _mm256_add_ps(_mm256_add_ps(n0, n1), _mm256_add_ps(n2, n3));
        let res = _mm256_mul_ps(sum, _mm256_set1_ps(32.0));

        _mm256_storeu_ps(out.as_mut_ptr(), res);
    }
}

/// Evaluates 8 samples of 2D Fractal Brownian Motion (fBm) using AVX2.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fbm2d_batch_8_avx2(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    unsafe {
        let mut total = _mm256_setzero_ps();
        let mut amp = 1.0f32;
        let mut max_amp = 0.0f32;

        let mut cur_xs = *xs;
        let mut cur_ys = *ys;

        for i in 0..8 {
            cur_xs[i] *= frequency;
            cur_ys[i] *= frequency;
        }

        let mut octave_out = [0.0f32; 8];

        for i in 0..octaves {
            let octave_seed = seed.wrapping_add(i as u64 * 10007);
            noise2_batch_8_avx2(octave_seed, &cur_xs, &cur_ys, &mut octave_out);
            let oct_vec = _mm256_loadu_ps(octave_out.as_ptr());
            total = _mm256_add_ps(total, _mm256_mul_ps(oct_vec, _mm256_set1_ps(amp)));
            max_amp += amp;
            amp *= gain;

            for k in 0..8 {
                cur_xs[k] *= lacunarity;
                cur_ys[k] *= lacunarity;
            }
        }

        if max_amp > 0.0 {
            total = _mm256_mul_ps(total, _mm256_set1_ps(1.0 / max_amp));
        }

        _mm256_storeu_ps(out.as_mut_ptr(), total);
    }
}

/// Evaluates 8 samples of 3D Fractal Brownian Motion (fBm) using AVX2.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fbm3d_batch_8_avx2(
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
    unsafe {
        let mut total = _mm256_setzero_ps();
        let mut amp = 1.0f32;
        let mut max_amp = 0.0f32;

        let mut cur_xs = *xs;
        let mut cur_ys = *ys;
        let mut cur_zs = *zs;

        for i in 0..8 {
            cur_xs[i] *= frequency;
            cur_ys[i] *= frequency;
            cur_zs[i] *= frequency;
        }

        let mut octave_out = [0.0f32; 8];

        for i in 0..octaves {
            let octave_seed = seed.wrapping_add(i as u64 * 10007);
            noise3_batch_8_avx2(octave_seed, &cur_xs, &cur_ys, &cur_zs, &mut octave_out);
            let oct_vec = _mm256_loadu_ps(octave_out.as_ptr());
            total = _mm256_add_ps(total, _mm256_mul_ps(oct_vec, _mm256_set1_ps(amp)));
            max_amp += amp;
            amp *= gain;

            for k in 0..8 {
                cur_xs[k] *= lacunarity;
                cur_ys[k] *= lacunarity;
                cur_zs[k] *= lacunarity;
            }
        }

        if max_amp > 0.0 {
            total = _mm256_mul_ps(total, _mm256_set1_ps(1.0 / max_amp));
        }

        _mm256_storeu_ps(out.as_mut_ptr(), total);
    }
}

/// Evaluates 8 samples of 2D ridged multifractal noise using AVX2.
///
/// # Safety
/// Caller must verify `CpuFeatures::get().can_avx2()` before invoking this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ridged2d_batch_8_avx2(
    seed: u64,
    xs: &[f32; 8],
    ys: &[f32; 8],
    octaves: usize,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    out: &mut [f32; 8],
) {
    unsafe {
        let mut total = _mm256_setzero_ps();
        let mut amp = 1.0f32;
        let mut max_amp = 0.0f32;

        let mut cur_xs = *xs;
        let mut cur_ys = *ys;

        for i in 0..8 {
            cur_xs[i] *= frequency;
            cur_ys[i] *= frequency;
        }

        let mut octave_out = [0.0f32; 8];
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

        for i in 0..octaves {
            let octave_seed = seed.wrapping_add(i as u64 * 10007);
            noise2_batch_8_avx2(octave_seed, &cur_xs, &cur_ys, &mut octave_out);
            let oct_vec = _mm256_loadu_ps(octave_out.as_ptr());
            let abs_vec = _mm256_and_ps(oct_vec, abs_mask);
            let ridged = _mm256_sub_ps(_mm256_set1_ps(1.0), abs_vec);
            let ridged_sq = _mm256_mul_ps(ridged, ridged);
            total = _mm256_add_ps(total, _mm256_mul_ps(ridged_sq, _mm256_set1_ps(amp)));
            max_amp += amp;
            amp *= gain;

            for k in 0..8 {
                cur_xs[k] *= lacunarity;
                cur_ys[k] *= lacunarity;
            }
        }

        if max_amp > 0.0 {
            total = _mm256_mul_ps(total, _mm256_set1_ps(1.0 / max_amp));
        }

        _mm256_storeu_ps(out.as_mut_ptr(), total);
    }
}
