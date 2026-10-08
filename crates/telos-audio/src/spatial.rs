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

/// Ring buffer delay line for modeling fractional Interaural Time Differences (ITD).
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f32>,
    write_cursor: usize,
}

impl DelayLine {
    /// Creates a new delay line with the given sample capacity (at least 64 samples).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity.max(64)],
            write_cursor: 0,
        }
    }

    /// Pushes a new sample into the delay line.
    pub fn push(&mut self, sample: f32) {
        self.buffer[self.write_cursor] = sample;
        self.write_cursor = (self.write_cursor + 1) % self.buffer.len();
    }

    /// Reads a sample delayed by `delay_samples` from the write head.
    #[must_use]
    pub fn read_delayed(&self, delay_samples: usize) -> f32 {
        let len = self.buffer.len();
        let idx = (self.write_cursor + len - 1 - delay_samples.min(len - 1)) % len;
        self.buffer[idx]
    }

    /// Clears the delay line buffer.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_cursor = 0;
    }
}

/// Woodworth-based Binaural HRTF spatializer providing ITD and frequency-dependent head shadowing.
#[derive(Debug, Clone)]
pub struct BinauralSpatializer {
    sample_rate: f32,
    head_radius_m: f32,
    speed_of_sound_m_s: f32,
    left_delay: DelayLine,
    right_delay: DelayLine,
    left_filter: crate::occlusion::LowPassFilter,
    right_filter: crate::occlusion::LowPassFilter,
}

impl BinauralSpatializer {
    /// Creates a new binaural spatializer configured for the specified audio sample rate (e.g. 48,000 Hz).
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            head_radius_m: 0.0875,
            speed_of_sound_m_s: 343.0,
            left_delay: DelayLine::new(64),
            right_delay: DelayLine::new(64),
            left_filter: crate::occlusion::LowPassFilter::new(20_000.0, sample_rate),
            right_filter: crate::occlusion::LowPassFilter::new(20_000.0, sample_rate),
        }
    }

    /// Resets all internal delay buffers and filter states.
    pub fn reset(&mut self) {
        self.left_delay.reset();
        self.right_delay.reset();
        self.left_filter.reset();
        self.right_filter.reset();
    }

    /// Processes a single mono sample into a spatialized binaural `(left, right)` sample pair.
    ///
    /// Applies Woodworth ITD ear arrival delay, frequency-dependent contralateral head shadow filtering,
    /// distance attenuation, and voxel terrain occlusion.
    pub fn spatialize_sample(
        &mut self,
        sample: f32,
        listener: &Listener,
        emitter_pos: Vec3,
        min_distance: f32,
        max_distance: f32,
        volume: f32,
        occlusion: &crate::occlusion::VoiceOcclusion,
    ) -> (f32, f32) {
        let delta = emitter_pos - listener.position;
        let dist = delta.length();

        if dist >= max_distance || max_distance <= min_distance || volume <= 0.0 {
            return (0.0, 0.0);
        }

        // Distance attenuation: smooth power roll-off
        let dist_factor = if dist <= min_distance {
            1.0
        } else {
            (1.0 - (dist - min_distance) / (max_distance - min_distance)).clamp(0.0, 1.0)
        };
        let attenuation = dist_factor.powf(1.2) * volume * occlusion.transmission_gain;

        // Azimuth angle in [-pi, pi]: 0 = straight ahead, +pi/2 = right, -pi/2 = left
        let dir = if dist < 1e-4 {
            listener.forward
        } else {
            delta / dist
        };
        let forward_proj = dir.dot(listener.forward);
        let right_proj = dir.dot(listener.right);
        let azimuth = right_proj.atan2(forward_proj);

        // Woodworth spherical head model for Interaural Time Difference (ITD)
        let abs_azimuth = azimuth.abs().min(std::f32::consts::PI * 0.5);
        let itd_seconds =
            (self.head_radius_m / self.speed_of_sound_m_s) * (abs_azimuth.sin() + abs_azimuth);
        let delay_samples = (itd_seconds * self.sample_rate).round() as usize;

        // Push input sample to both ear delay lines
        self.left_delay.push(sample);
        self.right_delay.push(sample);

        // Left/Right ear delays and head shadow cutoffs
        let (left_delayed, right_delayed, left_cutoff, right_cutoff) = if azimuth >= 0.0 {
            // Sound is to the right: right ear leads (delay=0), left ear is delayed
            let l_samp = self.left_delay.read_delayed(delay_samples);
            let r_samp = self.right_delay.read_delayed(0);

            // Left ear is shadowed by head: cutoff sweeps down to 1500 Hz
            let shadow_cutoff = 1500.0 + (20_000.0 - 1500.0) * f32::midpoint(1.0, (-azimuth).cos());
            let l_cut = occlusion.cutoff_hz.min(shadow_cutoff);
            let r_cut = occlusion.cutoff_hz;
            (l_samp, r_samp, l_cut, r_cut)
        } else {
            // Sound is to the left: left ear leads (delay=0), right ear is delayed
            let l_samp = self.left_delay.read_delayed(0);
            let r_samp = self.right_delay.read_delayed(delay_samples);

            // Right ear is shadowed by head
            let shadow_cutoff = 1500.0 + (20_000.0 - 1500.0) * f32::midpoint(1.0, azimuth.cos());
            let l_cut = occlusion.cutoff_hz;
            let r_cut = occlusion.cutoff_hz.min(shadow_cutoff);
            (l_samp, r_samp, l_cut, r_cut)
        };

        // Update ear low-pass filters
        self.left_filter.set_cutoff(left_cutoff, self.sample_rate);
        self.right_filter.set_cutoff(right_cutoff, self.sample_rate);

        let left_filtered = self.left_filter.process_sample(left_delayed);
        let right_filtered = self.right_filter.process_sample(right_delayed);

        // Constant-power level panning
        let theta = (right_proj.clamp(-1.0, 1.0) + 1.0) * 0.25 * std::f32::consts::PI;
        let left_gain = theta.cos() * std::f32::consts::SQRT_2 * attenuation;
        let right_gain = theta.sin() * std::f32::consts::SQRT_2 * attenuation;

        (left_filtered * left_gain, right_filtered * right_gain)
    }

    /// Spatialize an entire frame of mono samples into interleaved stereo `[L0, R0, L1, R1, ...]`.
    #[allow(clippy::too_many_arguments)]
    pub fn spatialize_frame_interleaved(
        &mut self,
        mono_in: &[f32],
        stereo_out: &mut [f32],
        listener: &Listener,
        emitter_pos: Vec3,
        min_distance: f32,
        max_distance: f32,
        volume: f32,
        occlusion: &crate::occlusion::VoiceOcclusion,
    ) {
        let frame_len = mono_in.len().min(stereo_out.len() / 2);
        for i in 0..frame_len {
            let (l, r) = self.spatialize_sample(
                mono_in[i],
                listener,
                emitter_pos,
                min_distance,
                max_distance,
                volume,
                occlusion,
            );
            stereo_out[i * 2] = l;
            stereo_out[i * 2 + 1] = r;
        }
    }
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

    #[test]
    fn test_delay_line() {
        let mut dl = DelayLine::new(64);
        dl.push(1.0);
        dl.push(2.0);
        dl.push(3.0);
        // Delay 0 is the newest sample pushed (3.0)
        assert!((dl.read_delayed(0) - 3.0).abs() < 1e-6);
        // Delay 1 is 2.0
        assert!((dl.read_delayed(1) - 2.0).abs() < 1e-6);
        // Delay 2 is 1.0
        assert!((dl.read_delayed(2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binaural_spatializer_azimuth() {
        let mut spatializer = BinauralSpatializer::new(48000.0);
        let listener = Listener::new(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let clear = crate::occlusion::VoiceOcclusion::clear();

        // Emitter to the right (+X)
        let (l, r) = spatializer.spatialize_sample(
            1.0,
            &listener,
            Vec3::new(5.0, 0.0, 0.0),
            1.0,
            32.0,
            1.0,
            &clear,
        );
        // Right ear should have significantly higher level than shadowed left ear
        assert!(r > l);

        // Reset and test emitter to the left (-X)
        spatializer.reset();
        let (l, r) = spatializer.spatialize_sample(
            1.0,
            &listener,
            Vec3::new(-5.0, 0.0, 0.0),
            1.0,
            32.0,
            1.0,
            &clear,
        );
        assert!(l > r);
    }
}
