//! Deterministic simplex gradient noise in 2D and 3D.

use crate::math::hash3;

const F2: f32 = 0.366_025_4; // 0.5 * (sqrt(3.0) - 1.0)
const G2: f32 = 0.211_324_87; // (3.0 - sqrt(3.0)) / 6.0

const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

/// 2D gradient lookup derived directly from hash bits without memory lookups.
#[inline]
fn grad2(hash: u64, dx: f32, dy: f32) -> f32 {
    let h = (hash & 7) as u8;
    let u = if h < 4 { dx } else { dy };
    let v = if h < 4 { dy } else { dx };
    let g1 = if (h & 1) == 0 { u } else { -u };
    let g2 = if (h & 2) == 0 { v } else { -v };
    g1 + g2
}

/// 3D gradient lookup derived directly from hash bits without memory lookups.
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

/// Evaluates 2D Simplex noise in range `[-1.0, 1.0]`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
pub fn noise2(seed: u64, x: f32, y: f32) -> f32 {
    let s = (x + y) * F2;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;

    #[allow(clippy::cast_precision_loss)]
    let t = (i + j) as f32 * G2;
    #[allow(clippy::cast_precision_loss)]
    let x0 = x - (i as f32 - t);
    #[allow(clippy::cast_precision_loss)]
    let y0 = y - (j as f32 - t);

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    #[allow(clippy::cast_precision_loss)]
    let x1 = x0 - i1 as f32 + G2;
    #[allow(clippy::cast_precision_loss)]
    let y1 = y0 - j1 as f32 + G2;

    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let mut n0 = 0.0;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let t0_2 = t0 * t0;
        n0 = t0_2 * t0_2 * grad2(hash3(seed, i, j, 0), x0, y0);
    }

    let mut n1 = 0.0;
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let t1_2 = t1 * t1;
        n1 = t1_2 * t1_2 * grad2(hash3(seed, i + i1, j + j1, 0), x1, y1);
    }

    let mut n2 = 0.0;
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let t2_2 = t2 * t2;
        n2 = t2_2 * t2_2 * grad2(hash3(seed, i + 1, j + 1, 0), x2, y2);
    }

    // Scaling factor to normalize to roughly [-1.0, 1.0]
    (n0 + n1 + n2) * 70.148
}

/// Evaluates 3D Simplex noise in range `[-1.0, 1.0]`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
pub fn noise3(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;
    let k = (z + s).floor() as i32;

    #[allow(clippy::cast_precision_loss)]
    let t = (i + j + k) as f32 * G3;
    #[allow(clippy::cast_precision_loss)]
    let x0 = x - (i as f32 - t);
    #[allow(clippy::cast_precision_loss)]
    let y0 = y - (j as f32 - t);
    #[allow(clippy::cast_precision_loss)]
    let z0 = z - (k as f32 - t);

    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0)
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1)
        } else {
            (0, 0, 1, 1, 0, 1)
        }
    } else if y0 < z0 {
        (0, 0, 1, 0, 1, 1)
    } else if x0 < z0 {
        (0, 1, 0, 0, 1, 1)
    } else {
        (0, 1, 0, 1, 1, 0)
    };

    #[allow(clippy::cast_precision_loss)]
    let x1 = x0 - i1 as f32 + G3;
    #[allow(clippy::cast_precision_loss)]
    let y1 = y0 - j1 as f32 + G3;
    #[allow(clippy::cast_precision_loss)]
    let z1 = z0 - k1 as f32 + G3;

    #[allow(clippy::cast_precision_loss)]
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    #[allow(clippy::cast_precision_loss)]
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    #[allow(clippy::cast_precision_loss)]
    let z2 = z0 - k2 as f32 + 2.0 * G3;

    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let mut n0 = 0.0;
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let t0_2 = t0 * t0;
        n0 = t0_2 * t0_2 * grad3(hash3(seed, i, j, k), x0, y0, z0);
    }

    let mut n1 = 0.0;
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let t1_2 = t1 * t1;
        n1 = t1_2 * t1_2 * grad3(hash3(seed, i + i1, j + j1, k + k1), x1, y1, z1);
    }

    let mut n2 = 0.0;
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let t2_2 = t2 * t2;
        n2 = t2_2 * t2_2 * grad3(hash3(seed, i + i2, j + j2, k + k2), x2, y2, z2);
    }

    let mut n3 = 0.0;
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let t3_2 = t3 * t3;
        n3 = t3_2 * t3_2 * grad3(hash3(seed, i + 1, j + 1, k + 1), x3, y3, z3);
    }

    (n0 + n1 + n2 + n3) * 32.0
}
