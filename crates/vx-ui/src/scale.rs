//! GUI scale calculation and physical pixel edge-snapping.

/// Computes the Classic Voxel-parity auto integer GUI scale factor based on viewport resolution.
///
/// Rule: Largest integer `S >= 1` such that `width / S >= 320` and `height / S >= 240`.
#[must_use]
pub fn compute_gui_scale(width: u32, height: u32) -> u32 {
    let scale_w = width / 320;
    let scale_h = height / 240;
    let scale = scale_w.min(scale_h);
    scale.max(1)
}

/// Converts a GUI-space coordinate/dimension to physical pixels using the active scale factor.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub const fn to_physical_pixels(gui_units: i32, gui_scale: u32) -> i32 {
    gui_units * (gui_scale as i32)
}

/// Converts a GUI-space float coordinate to snapped physical pixels.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn snap_to_physical(gui_coord: f32, gui_scale: u32) -> i32 {
    (gui_coord * gui_scale as f32).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gui_scale_resolutions() {
        // Standard resolutions
        assert_eq!(compute_gui_scale(1920, 1080), 4);
        assert_eq!(compute_gui_scale(1280, 720), 3);
        assert_eq!(compute_gui_scale(854, 480), 2);
        assert_eq!(compute_gui_scale(640, 480), 2);
        assert_eq!(compute_gui_scale(320, 240), 1);
        assert_eq!(compute_gui_scale(200, 150), 1);
        assert_eq!(compute_gui_scale(3840, 2160), 9);
    }

    #[test]
    fn test_pixel_snapping() {
        assert_eq!(to_physical_pixels(10, 4), 40);
        assert_eq!(snap_to_physical(10.25, 4), 41);
        assert_eq!(snap_to_physical(10.75, 4), 43);
    }
}
