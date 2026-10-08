//! Voxel terrain acoustic occlusion modeling.
//!
//! Simulates transmission loss and low-pass frequency muffling through solid intervening blocks.

/// Voxel acoustic occlusion calculator and filter parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceOcclusion {
    /// Occlusion factor in `[0.0, 1.0]`. 0.0 = clear line of sight, 1.0 = fully blocked.
    pub factor: f32,
    /// Amplitude transmission multiplier `[0.15, 1.0]`.
    pub transmission_gain: f32,
    /// Low-pass filter cutoff frequency in Hertz `[800.0, 20_000.0]`.
    pub cutoff_hz: f32,
}

impl VoiceOcclusion {
    /// Minimum low-pass cutoff frequency for fully occluded speech (800 Hz).
    pub const MIN_CUTOFF_HZ: f32 = 800.0;
    /// Maximum cutoff frequency for unoccluded speech (20 kHz).
    pub const MAX_CUTOFF_HZ: f32 = 20_000.0;
    /// Minimum transmission amplitude multiplier (15% volume through heavy stone).
    pub const MIN_TRANSMISSION: f32 = 0.15;

    /// Evaluates occlusion from the count of solid voxels obstructing the line of sight.
    #[must_use]
    pub fn from_blocking_voxels(blocking_count: usize) -> Self {
        if blocking_count == 0 {
            return Self::clear();
        }

        // Exponential decay: 1 voxel -> ~0.45 occlusion, 3 voxels -> ~0.83, 5+ -> >0.95
        let factor = (1.0 - (-0.6 * blocking_count as f32).exp()).clamp(0.0, 1.0);
        let transmission_gain = 1.0 - (1.0 - Self::MIN_TRANSMISSION) * factor;
        let cutoff_hz = Self::MAX_CUTOFF_HZ * (1.0 - factor) + Self::MIN_CUTOFF_HZ * factor;

        Self {
            factor,
            transmission_gain,
            cutoff_hz,
        }
    }

    /// Clear, unobstructed line of sight (no occlusion).
    #[must_use]
    pub const fn clear() -> Self {
        Self {
            factor: 0.0,
            transmission_gain: 1.0,
            cutoff_hz: Self::MAX_CUTOFF_HZ,
        }
    }
}

/// Simple single-pole low-pass filter for acoustic occlusion muffling and head shadowing.
#[derive(Debug, Clone, Copy)]
pub struct LowPassFilter {
    state: f32,
    alpha: f32,
}

impl LowPassFilter {
    /// Creates a new filter with a given cutoff frequency and sample rate.
    #[must_use]
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            state: 0.0,
            alpha: 0.0,
        };
        filter.set_cutoff(cutoff_hz, sample_rate);
        filter
    }

    /// Updates the filter cutoff frequency.
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        if cutoff_hz >= sample_rate * 0.49 {
            self.alpha = 0.0;
        } else {
            let dt = 1.0 / sample_rate;
            let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz.max(10.0));
            self.alpha = dt / (rc + dt);
        }
    }

    /// Processes a single sample through the filter.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.state += self.alpha * (input - self.state);
        self.state
    }

    /// Resets filter memory state.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occlusion_clear_vs_blocked() {
        let clear = VoiceOcclusion::from_blocking_voxels(0);
        assert!(clear.factor.abs() < 1e-6);
        assert!((clear.transmission_gain - 1.0).abs() < 1e-6);
        assert!((clear.cutoff_hz - VoiceOcclusion::MAX_CUTOFF_HZ).abs() < 1e-6);

        let one_block = VoiceOcclusion::from_blocking_voxels(1);
        assert!(one_block.factor > 0.4);
        assert!(one_block.transmission_gain < 0.7);
        assert!(one_block.cutoff_hz < 15_000.0);

        let five_blocks = VoiceOcclusion::from_blocking_voxels(5);
        assert!(five_blocks.factor > 0.9);
        assert!(five_blocks.transmission_gain <= 0.25);
        assert!(five_blocks.cutoff_hz < 2_500.0);
    }

    #[test]
    fn test_low_pass_filter() {
        let mut filter = LowPassFilter::new(1000.0, 48000.0);
        let mut out = 0.0;
        for _ in 0..100 {
            out = filter.process_sample(1.0);
        }
        // Step response asymptotically approaches 1.0
        assert!((out - 1.0).abs() < 1e-2);
    }
}
