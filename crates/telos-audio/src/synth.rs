//! Built-in procedural acoustic synthesizer for fallback and out-of-the-box gameplay sound effects.

use std::sync::Arc;

/// Sample rate used for all procedural audio synthesizers (standard CD quality).
pub const SYNTH_SAMPLE_RATE: u32 = 44_100;

/// Fast, deterministic 32-bit PRNG for procedural noise generation.
#[derive(Debug, Clone, Copy)]
struct SimpleRng(u32);

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x1234_5678 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Returns a uniform float in `[-1.0, 1.0]`.
    fn next_f32(&mut self) -> f32 {
        let val = (self.next_u32() >> 8) as f32 / 16_777_216.0; // [0, 1)
        val * 2.0 - 1.0
    }
}

/// Synthesizes a material footstep sound buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_footstep(material: &str, pitch: f32) -> Arc<[f32]> {
    let pitch = pitch.clamp(0.5, 2.0);
    let duration_sec = 0.12 / pitch;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = SimpleRng::new(0xABCD);

    let (base_freq, lpf_alpha, decay_rate, noise_ratio) = match material {
        "grass" | "plant" | "foliage" => (100.0, 0.16, 32.0, 0.85),
        "wood" | "log" | "planks" => (140.0, 0.28, 38.0, 0.70),
        "stone" | "cobblestone" => (180.0, 0.38, 44.0, 0.65),
        "gravel" | "dirt" => (90.0, 0.20, 34.0, 0.82),
        "sand" => (80.0, 0.14, 28.0, 0.88),
        "water" => (220.0, 0.30, 22.0, 0.60),
        _ => (120.0, 0.25, 35.0, 0.75),
    };

    let mut filtered_noise = 0.0f32;
    let mut phase = 0.0f32;

    for i in 0..num_samples {
        let t = i as f32 / SYNTH_SAMPLE_RATE as f32;
        let attack = (t / 0.003).min(1.0);
        let env = attack * (-decay_rate * pitch * t).exp();

        // Low-pass filtered noise for natural material texture
        let raw_noise = rng.next_f32();
        filtered_noise += lpf_alpha * (raw_noise - filtered_noise);

        // Pitch-drop transient for physical impact body
        let inst_freq = base_freq * (0.6 + 1.8 * (-90.0 * t).exp()) * pitch;
        phase += 2.0 * std::f32::consts::PI * inst_freq / SYNTH_SAMPLE_RATE as f32;
        let tone = phase.sin();

        let sample = (tone * (1.0 - noise_ratio) + filtered_noise * noise_ratio) * env * 0.75;
        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples.into()
}

/// Synthesizes a block crumbling and break sound buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_block_break(pitch: f32) -> Arc<[f32]> {
    let pitch = pitch.clamp(0.5, 2.0);
    let duration_sec = 0.20 / pitch;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = SimpleRng::new(0x9876);

    let mut filtered_noise = 0.0f32;
    let mut phase = 0.0f32;

    for i in 0..num_samples {
        let t = i as f32 / SYNTH_SAMPLE_RATE as f32;
        let env = (-20.0 * pitch * t).exp();

        // Multiple crumbling crack pulses
        let pulse1 = (-(45.0 * (t - 0.02).abs())).exp().max(0.0);
        let pulse2 = (-(50.0 * (t - 0.06).abs())).exp().max(0.0);
        let pulse3 = (-(55.0 * (t - 0.10).abs())).exp().max(0.0);
        let pulse_env = 0.3 + 0.7 * (pulse1 + pulse2 + pulse3);

        let raw_noise = rng.next_f32();
        filtered_noise += 0.32 * (raw_noise - filtered_noise);

        // Low thud transient
        let inst_freq = 90.0 * (0.5 + 1.5 * (-60.0 * t).exp()) * pitch;
        phase += 2.0 * std::f32::consts::PI * inst_freq / SYNTH_SAMPLE_RATE as f32;
        let thud = phase.sin() * 0.25;

        let sample = (filtered_noise * 0.85 + thud) * env * pulse_env * 0.80;
        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples.into()
}

/// Synthesizes a block placement thud sound buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_block_place(pitch: f32) -> Arc<[f32]> {
    let pitch = pitch.clamp(0.5, 2.0);
    let duration_sec = 0.10 / pitch;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = SimpleRng::new(0x5432);

    let mut filtered_noise = 0.0f32;
    let mut phase = 0.0f32;

    for i in 0..num_samples {
        let t = i as f32 / SYNTH_SAMPLE_RATE as f32;
        let attack = (t / 0.002).min(1.0);
        let env = attack * (-36.0 * pitch * t).exp();

        let raw_noise = rng.next_f32();
        filtered_noise += 0.35 * (raw_noise - filtered_noise);

        // Downward resonant thud from 180Hz down to 60Hz
        let inst_freq = 70.0 * (0.7 + 1.8 * (-110.0 * t).exp()) * pitch;
        phase += 2.0 * std::f32::consts::PI * inst_freq / SYNTH_SAMPLE_RATE as f32;
        let thud = phase.sin();

        let sample = (thud * 0.45 + filtered_noise * 0.55) * env * 0.80;
        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples.into()
}

/// Synthesizes a seamlessly loopable rain ambient sound buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_rain_loop() -> Arc<[f32]> {
    let duration_sec = 1.5;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = SimpleRng::new(0x2468);

    // Simple IIR low-pass filter to simulate gentle rain patter
    let mut filtered = 0.0f32;
    let alpha = 0.25;

    for _ in 0..num_samples {
        let raw = rng.next_f32();
        filtered = filtered + alpha * (raw - filtered);

        // Occasional droplet spike
        let spike = if rng.next_u32().is_multiple_of(250) {
            rng.next_f32() * 0.6
        } else {
            0.0
        };

        let sample = (filtered * 0.7 + spike) * 0.6;
        samples.push(sample.clamp(-1.0, 1.0));
    }

    // Apply quick crossfade at buffer ends for seamless loop
    let fade_len = (0.05 * SYNTH_SAMPLE_RATE as f32) as usize;
    for i in 0..fade_len {
        let factor = i as f32 / fade_len as f32;
        let start_val = samples[i];
        let end_idx = num_samples - fade_len + i;
        let end_val = samples[end_idx];

        samples[i] = end_val * (1.0 - factor) + start_val * factor;
        samples[end_idx] = end_val * (1.0 - factor) + start_val * factor;
    }

    samples.into()
}

/// Synthesizes an explosive lightning thunder strike buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_thunder() -> Arc<[f32]> {
    let duration_sec = 1.4;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = SimpleRng::new(0xFEDC);

    for i in 0..num_samples {
        let t = i as f32 / SYNTH_SAMPLE_RATE as f32;

        // Sharp initial crack at t = 0 followed by low-frequency rolling rumble
        let crack_env = (-(80.0 * t)).exp();
        let rumble_env = (-(3.5 * t)).exp();

        let crack = rng.next_f32() * crack_env * 0.95;

        // Low-frequency rumble (40 Hz with harmonics)
        let sub_bass = (2.0 * std::f32::consts::PI * 42.0 * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * 84.0 * t).sin();
        let rumble_noise = rng.next_f32() * 0.4;
        let rumble = (sub_bass * 0.6 + rumble_noise) * rumble_env * 0.75;

        let sample = (crack + rumble).clamp(-1.0, 1.0);
        samples.push(sample);
    }

    samples.into()
}

/// Synthesizes a damage hurt grunt sound buffer (mono, 44.1 kHz).
#[must_use]
pub fn synthesize_entity_hurt(pitch: f32) -> Arc<[f32]> {
    let pitch = pitch.clamp(0.5, 2.0);
    let duration_sec = 0.18 / pitch;
    let num_samples = (duration_sec * SYNTH_SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / SYNTH_SAMPLE_RATE as f32;
        let env = (-24.0 * pitch * t).exp();

        // Downward frequency sweep from 300Hz down to 120Hz
        let freq = (300.0 - 180.0 * (t / duration_sec)) * pitch;
        let tone = (2.0 * std::f32::consts::PI * freq * t).sin();

        // Soft distortion for gruffness
        let saturated = (tone * 1.5).tanh();
        let sample = saturated * env * 0.85;

        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footstep_bounds() {
        for mat in &["grass", "stone", "wood", "dirt", "water"] {
            let buf = synthesize_footstep(mat, 1.0);
            assert!(!buf.is_empty());
            for &s in buf.iter() {
                assert!(!s.is_nan());
                assert!((-1.0..=1.0).contains(&s));
            }
        }
    }

    #[test]
    fn test_rain_loop_seamless() {
        let buf = synthesize_rain_loop();
        assert!(!buf.is_empty());
        // Verify start and end crossfade continuity
        let diff = (buf[0] - buf[buf.len() - 1]).abs();
        assert!(diff < 0.05);
    }

    #[test]
    fn test_thunder_bounds() {
        let buf = synthesize_thunder();
        assert!(!buf.is_empty());
        for &s in buf.iter() {
            assert!(!s.is_nan());
            assert!((-1.0..=1.0).contains(&s));
        }
    }
}
