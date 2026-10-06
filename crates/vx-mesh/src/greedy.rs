//! 2D binary greedy meshing kernel for 32x32 slice bitboards.

/// Greedily merges contiguous 1x1 face bits into maximal rectangular quads,
/// respecting both material identity and light pattern merging constraints.
///
/// # Arguments
/// - `rows`: 32x32 bitboard where row `v ∈ 0..31` contains bits `u ∈ 0..31`.
/// - `get_cell`: Closure returning `(material: u16, pattern_idx: u16, can_merge_u: bool, can_merge_v: bool)`.
/// - `emit_quad`: Closure called for each greedily merged quad with `(u, v, w, h, material, pattern_idx)`.
#[inline]
pub fn greedy_merge_slice_with_light(
    mut rows: [u32; 32],
    mut get_cell: impl FnMut(u32, u32) -> (u16, u16, bool, bool),
    mut emit_quad: impl FnMut(u32, u32, u32, u32, u16, u16),
) {
    for v in 0..32 {
        while rows[v] != 0 {
            let u = rows[v].trailing_zeros();
            let (material, pattern_idx, can_merge_u, can_merge_v) = get_cell(u, v as u32);

            // Step 1: Expand width along u on row v
            let remaining = rows[v] >> u;
            let max_w = if can_merge_u {
                remaining.trailing_ones().min(32 - u)
            } else {
                1
            };

            let mut w = 1u32;
            while w < max_w {
                let (w_mat, w_pattern, w_can_u, _) = get_cell(u + w, v as u32);
                if w_mat != material || w_pattern != pattern_idx || !w_can_u {
                    break;
                }
                w += 1;
            }

            let mask = if w == 32 {
                u32::MAX
            } else {
                ((1u32 << w) - 1) << u
            };

            // Step 2: Expand height along v
            let mut h = 1u32;
            if can_merge_v {
                while ((v as u32) + h) < 32 {
                    let next_v = (v as u32) + h;
                    // Quick bitwise check: all bits in mask must be set in next row
                    if (rows[next_v as usize] & mask) != mask {
                        break;
                    }
                    // Material and pattern check across width
                    let mut match_row = true;
                    for du in 0..w {
                        let (row_mat, row_pattern, _, row_can_v) = get_cell(u + du, next_v);
                        if row_mat != material || row_pattern != pattern_idx || !row_can_v {
                            match_row = false;
                            break;
                        }
                    }
                    if !match_row {
                        break;
                    }
                    h += 1;
                }
            }

            // Step 3: Clear the merged rectangle from all spanned rows
            for dh in 0..h {
                rows[v + (dh as usize)] &= !mask;
            }

            // Step 4: Emit the merged quad
            emit_quad(u, v as u32, w, h, material, pattern_idx);
        }
    }
}

/// Greedily merges contiguous 1x1 face bits into maximal rectangular quads.
///
/// # Arguments
/// - `rows`: 32x32 bitboard where row `v ∈ 0..31` contains bits `u ∈ 0..31`.
/// - `get_material`: Closure returning the material ID for cell `(u, v)`.
/// - `emit_quad`: Closure called for each greedily merged quad with `(u, v, w, h, material)`.
#[inline]
pub fn greedy_merge_slice(
    rows: [u32; 32],
    mut get_material: impl FnMut(u32, u32) -> u16,
    mut emit_quad: impl FnMut(u32, u32, u32, u32, u16),
) {
    greedy_merge_slice_with_light(
        rows,
        |u, v| (get_material(u, v), 0, true, true),
        |u, v, w, h, mat, _pat| emit_quad(u, v, w, h, mat),
    );
}

/// Fast-path 2D greedy merger for slices where all set bits share an identical `material` and `pattern_idx`.
#[inline]
pub fn greedy_merge_slice_uniform_with_light(
    mut rows: [u32; 32],
    material: u16,
    pattern_idx: u16,
    can_merge_u: bool,
    can_merge_v: bool,
    mut emit_quad: impl FnMut(u32, u32, u32, u32, u16, u16),
) {
    for v in 0..32 {
        while rows[v] != 0 {
            let u = rows[v].trailing_zeros();
            let remaining = rows[v] >> u;
            let max_w = if can_merge_u {
                remaining.trailing_ones().min(32 - u)
            } else {
                1
            };

            let w = max_w;
            let mask = if w == 32 {
                u32::MAX
            } else {
                ((1u32 << w) - 1) << u
            };

            let mut h = 1u32;
            if can_merge_v {
                while ((v as u32) + h) < 32 {
                    let next_v = (v as u32) + h;
                    if (rows[next_v as usize] & mask) != mask {
                        break;
                    }
                    h += 1;
                }
            }

            for dh in 0..h {
                rows[v + (dh as usize)] &= !mask;
            }

            emit_quad(u, v as u32, w, h, material, pattern_idx);
        }
    }
}

/// Fast-path 2D greedy merger for slices where all set bits share an identical `material`.
///
/// Bypasses all material comparison closures, merging purely via bitwise scans.
#[inline]
pub fn greedy_merge_slice_uniform(
    rows: [u32; 32],
    material: u16,
    mut emit_quad: impl FnMut(u32, u32, u32, u32, u16),
) {
    greedy_merge_slice_uniform_with_light(
        rows,
        material,
        0,
        true,
        true,
        |u, v, w, h, mat, _pat| emit_quad(u, v, w, h, mat),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_merge_full_slice() {
        let rows = [u32::MAX; 32];
        let mut count = 0;
        let mut emitted = None;

        greedy_merge_slice_uniform(rows, 10, |u, v, w, h, mat| {
            count += 1;
            emitted = Some((u, v, w, h, mat));
        });

        assert_eq!(count, 1, "Full 32x32 slice must merge into exactly 1 quad");
        assert_eq!(emitted, Some((0, 0, 32, 32, 10)));
    }

    #[test]
    fn test_greedy_merge_material_split() {
        let mut rows = [0u32; 32];
        // 2x2 square: left column material 1, right column material 2
        rows[0] = 0b11;
        rows[1] = 0b11;

        let mut quads = Vec::new();
        greedy_merge_slice(
            rows,
            |u, _v| if u == 0 { 1 } else { 2 },
            |u, v, w, h, mat| {
                quads.push((u, v, w, h, mat));
            },
        );

        assert_eq!(
            quads.len(),
            2,
            "Must produce 2 separate 1x2 quads due to material difference"
        );
        assert_eq!(quads[0], (0, 0, 1, 2, 1));
        assert_eq!(quads[1], (1, 0, 1, 2, 2));
    }
}
