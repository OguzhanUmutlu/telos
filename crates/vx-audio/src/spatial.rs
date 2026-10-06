//! 3D spatial acoustic calculations: distance attenuation and stereo ear panning.

use glam::Vec3;

/// 3D listener coordinate frame representing the player's head and ears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Listener {
    /// World-space position of the listener (camera eye).
    pub position: Vec3,
    /// Forward view vector (must be non-zero).
    pub forward: Vec3,
    /// Up vector (must be non-zero).
    pub up: Vec3,
    /// Right vector derived from forward and up (points towards the listener's right ear).
    pub right: Vec3,
}

impl Listener {
    /// Creates a listener from eye position, forward vector, and up vector.
    #[must_use]
    pub fn new(position: Vec3, forward: Vec3, up: Vec3) -> Self {
        let f = if forward.length_squared() > 1e-6 {
            forward.normalize()
        } else {
            Vec3::new(0.0, 0.0, -1.0)
        };
        let u = if up.length_squared() > 1e-6 {
            up.normalize()
        } else {
            Vec3::Y
        };
        let r = f.cross(u);
        let right = if r.length_squared() > 1e-6 {
            r.normalize()
        } else {
            Vec3::X
        };

        Self {
            position,
            forward: f,
            up: u,
            right,
        }
    }
}

impl Default for Listener {
    fn default() -> Self {
        Self::new(Vec3::ZERO, -Vec3::Z, Vec3::Y)
    }
}

/// Computes distance attenuation and stereo channel gains `(left_gain, right_gain)` for a 3D emitter.
///
/// # Arguments
/// * `listener` - The listener's position and orientation.
/// * `emitter_pos` - 3D world-space coordinates of the sound emitter.
/// * `min_distance` - Distance below which the sound plays at 100% volume (e.g. 1.0 block).
/// * `max_distance` - Distance beyond which the sound is completely inaudible (e.g. 32.0 blocks).
/// * `volume` - Master volume scale of the sound instance (0.0 to 1.0+).
#[must_use]
pub fn calculate_spatial_gains(
    listener: &Listener,
    emitter_pos: Vec3,
    min_distance: f32,
    max_distance: f32,
    volume: f32,
) -> (f32, f32) {
    let delta = emitter_pos - listener.position;
    let dist = delta.length();

    if dist >= max_distance || max_distance <= min_distance || volume <= 0.0 {
        return (0.0, 0.0);
    }

    // Distance attenuation: linear roll-off clamped between min and max distance
    let attenuation = if dist <= min_distance {
        1.0
    } else {
        (1.0 - (dist - min_distance) / (max_distance - min_distance)).clamp(0.0, 1.0)
    };

    // Horizontal panning based on the emitter's projection onto the listener's right axis
    let pan = if dist < 1e-4 {
        0.0
    } else {
        let dir = delta / dist;
        dir.dot(listener.right).clamp(-1.0, 1.0)
    };

    // Constant-power panning law: preserves acoustic power regardless of azimuth angle
    // theta in [0, pi/2] where 0 is hard-left and pi/2 is hard-right
    let theta = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    let left_gain = theta.cos() * std::f32::consts::SQRT_2 * attenuation * volume;
    let right_gain = theta.sin() * std::f32::consts::SQRT_2 * attenuation * volume;

    (left_gain.max(0.0), right_gain.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listener_orientation() {
        let l = Listener::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::Y);
        // Forward is +Z, up is +Y. Right = +Z cross +Y = -X
        assert!(l.right.x < -0.99);
    }

    #[test]
    fn test_distance_attenuation() {
        let listener = Listener::default();
        // Emitter at listener position
        let (left, right) = calculate_spatial_gains(&listener, Vec3::ZERO, 2.0, 10.0, 1.0);
        assert!((left - 1.0).abs() < 1e-3);
        assert!((right - 1.0).abs() < 1e-3);

        // Emitter at distance >= max_distance
        let (left, right) =
            calculate_spatial_gains(&listener, Vec3::new(0.0, 0.0, -10.0), 2.0, 10.0, 1.0);
        assert!(left.abs() < 1e-6);
        assert!(right.abs() < 1e-6);

        // Emitter halfway: dist = 6.0, min = 2.0, max = 10.0 -> atten = 0.5
        let (left, right) =
            calculate_spatial_gains(&listener, Vec3::new(0.0, 0.0, -6.0), 2.0, 10.0, 1.0);
        assert!((left - 0.5).abs() < 1e-3);
        assert!((right - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_stereo_panning() {
        let listener = Listener::new(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        // Emitter straight right (+X)
        let (left, right) =
            calculate_spatial_gains(&listener, Vec3::new(1.0, 0.0, 0.0), 2.0, 10.0, 1.0);
        assert!(right > left);
        assert!(left < 0.1);

        // Emitter straight left (-X)
        let (left, right) =
            calculate_spatial_gains(&listener, Vec3::new(-1.0, 0.0, 0.0), 2.0, 10.0, 1.0);
        assert!(left > right);
        assert!(right < 0.1);
    }
}
