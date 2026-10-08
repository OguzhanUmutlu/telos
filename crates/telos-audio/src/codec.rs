//! Low-latency Opus voice encoder and decoder streams.
//!
//! Operates at 48,000 Hz mono with standard 20 ms frames (960 samples per frame).
//! Provides energy-based Voice Activity Detection (VAD) and Packet Loss Concealment (PLC).

use crate::error::AudioError;
use opus::{Application, Bitrate, Channels, Decoder, Encoder};

/// Sampling rate for voice chat audio streams (48 kHz).
pub const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Number of audio channels for voice chat (mono = 1).
pub const OPUS_CHANNELS: usize = 1;

/// Duration of a single voice frame in milliseconds (20 ms).
pub const OPUS_FRAME_MS: u32 = 20;

/// Number of audio samples per 20 ms frame at 48 kHz (960 samples).
pub const OPUS_FRAME_SIZE: usize = (OPUS_SAMPLE_RATE as usize * OPUS_FRAME_MS as usize) / 1000;

/// Default target voice bitrate in bits per second (32 kbps).
pub const DEFAULT_BITRATE_BPS: i32 = 32_000;

/// Maximum compressed byte size for a single 20 ms Opus frame.
pub const MAX_OPUS_PACKET_SIZE: usize = 1024;

/// Low-latency Opus audio encoder with Voice Activity Detection (VAD).
pub struct OpusVoiceEncoder {
    encoder: Encoder,
    vad_threshold_rms: f32,
    hangover_frames: u32,
    remaining_hangover: u32,
}

impl OpusVoiceEncoder {
    /// Creates a new encoder with default `VoIP` bitrate (32 kbps) and VAD sensitivity.
    pub fn new() -> Result<Self, AudioError> {
        Self::with_config(DEFAULT_BITRATE_BPS, 0.01, 10)
    }

    /// Creates a new encoder with custom bitrate, RMS VAD threshold, and hangover frame count.
    ///
    /// # Arguments
    /// * `bitrate_bps` - Target bitrate in bits per second (e.g. 32000).
    /// * `vad_threshold_rms` - Minimum RMS amplitude to trigger voice activity (e.g. 0.01).
    /// * `hangover_frames` - Trailing frames to transmit after speech falls below threshold (e.g. 10 frames = 200 ms).
    pub fn with_config(
        bitrate_bps: i32,
        vad_threshold_rms: f32,
        hangover_frames: u32,
    ) -> Result<Self, AudioError> {
        let mut encoder = Encoder::new(OPUS_SAMPLE_RATE, Channels::Mono, Application::Voip)
            .map_err(|e| AudioError::Codec(format!("Failed to create Opus encoder: {e}")))?;

        encoder
            .set_bitrate(Bitrate::Bits(bitrate_bps))
            .map_err(|e| AudioError::Codec(format!("Failed to set Opus bitrate: {e}")))?;

        Ok(Self {
            encoder,
            vad_threshold_rms: vad_threshold_rms.max(0.0),
            hangover_frames,
            remaining_hangover: 0,
        })
    }

    /// Sets the target bitrate in bits per second.
    pub fn set_bitrate(&mut self, bitrate_bps: i32) -> Result<(), AudioError> {
        self.encoder
            .set_bitrate(Bitrate::Bits(bitrate_bps))
            .map_err(|e| AudioError::Codec(format!("Failed to set Opus bitrate: {e}")))
    }

    /// Sets the Voice Activity Detection RMS energy threshold.
    pub fn set_vad_threshold(&mut self, threshold_rms: f32) {
        self.vad_threshold_rms = threshold_rms.max(0.0);
    }

    /// Encodes a 20 ms PCM float slice (must be exactly `OPUS_FRAME_SIZE` = 960 samples).
    ///
    /// Returns `Ok(Some(bytes_written))` if voice activity was detected or hangover frames remain,
    /// or `Ok(None)` if silence was suppressed.
    pub fn encode(&mut self, pcm: &[f32], out_buf: &mut [u8]) -> Result<Option<usize>, AudioError> {
        if pcm.len() != OPUS_FRAME_SIZE {
            return Err(AudioError::Codec(format!(
                "Invalid PCM frame length: expected {OPUS_FRAME_SIZE}, got {}",
                pcm.len()
            )));
        }

        let rms = calculate_rms(pcm);
        let active = rms >= self.vad_threshold_rms;

        if active {
            self.remaining_hangover = self.hangover_frames;
        }

        if active || self.remaining_hangover > 0 {
            if !active {
                self.remaining_hangover = self.remaining_hangover.saturating_sub(1);
            }
            let written = self
                .encoder
                .encode_float(pcm, out_buf)
                .map_err(|e| AudioError::Codec(format!("Opus encoding failed: {e}")))?;
            Ok(Some(written))
        } else {
            Ok(None)
        }
    }

    /// Force-encodes a frame bypassing VAD (for Push-To-Talk).
    pub fn encode_force(&mut self, pcm: &[f32], out_buf: &mut [u8]) -> Result<usize, AudioError> {
        if pcm.len() != OPUS_FRAME_SIZE {
            return Err(AudioError::Codec(format!(
                "Invalid PCM frame length: expected {OPUS_FRAME_SIZE}, got {}",
                pcm.len()
            )));
        }

        self.encoder
            .encode_float(pcm, out_buf)
            .map_err(|e| AudioError::Codec(format!("Opus encoding failed: {e}")))
    }
}

/// Low-latency Opus audio decoder with Packet Loss Concealment (PLC).
pub struct OpusVoiceDecoder {
    decoder: Decoder,
}

impl OpusVoiceDecoder {
    /// Creates a new Opus decoder operating at 48,000 Hz mono.
    pub fn new() -> Result<Self, AudioError> {
        let decoder = Decoder::new(OPUS_SAMPLE_RATE, Channels::Mono)
            .map_err(|e| AudioError::Codec(format!("Failed to create Opus decoder: {e}")))?;
        Ok(Self { decoder })
    }

    /// Decodes an Opus packet into floating point PCM samples.
    ///
    /// If `packet` is `Some(bytes)`, decodes standard wire frame.
    /// If `packet` is `None`, executes Packet Loss Concealment (PLC) to extrapolate speech.
    ///
    /// `out_pcm` must have space for at least `OPUS_FRAME_SIZE` (960) samples.
    /// Returns the number of samples decoded into `out_pcm`.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        out_pcm: &mut [f32],
    ) -> Result<usize, AudioError> {
        if out_pcm.len() < OPUS_FRAME_SIZE {
            return Err(AudioError::Codec(format!(
                "Output buffer too small: expected at least {OPUS_FRAME_SIZE}, got {}",
                out_pcm.len()
            )));
        }

        match packet {
            Some(bytes) => self
                .decoder
                .decode_float(bytes, out_pcm, false)
                .map_err(|e| AudioError::Codec(format!("Opus decode failed: {e}"))),
            None => {
                // Packet loss concealment: empty slice triggers PLC extrapolation in Opus
                self.decoder
                    .decode_float(&[], out_pcm, false)
                    .map_err(|e| AudioError::Codec(format!("Opus PLC decode failed: {e}")))
            }
        }
    }
}

/// Calculates the Root Mean Square (RMS) amplitude of a float PCM buffer.
#[must_use]
pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_roundtrip() {
        let mut encoder = OpusVoiceEncoder::new().expect("encoder");
        let mut decoder = OpusVoiceDecoder::new().expect("decoder");

        // Generate 960 samples of 440 Hz sine wave
        let mut input = vec![0.0f32; OPUS_FRAME_SIZE];
        for (i, sample) in input.iter_mut().enumerate() {
            let t = i as f32 / OPUS_SAMPLE_RATE as f32;
            *sample = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
        }

        let mut wire_buf = [0u8; MAX_OPUS_PACKET_SIZE];
        let enc_res = encoder.encode(&input, &mut wire_buf).expect("encode");
        let encoded_len = enc_res.expect("audio should be active");
        assert!(encoded_len > 0);
        assert!(encoded_len < MAX_OPUS_PACKET_SIZE);

        let mut output = vec![0.0f32; OPUS_FRAME_SIZE];
        let decoded_len = decoder
            .decode(Some(&wire_buf[..encoded_len]), &mut output)
            .expect("decode");
        assert_eq!(decoded_len, OPUS_FRAME_SIZE);

        // Check decoded RMS is reasonably close to input RMS
        let in_rms = calculate_rms(&input);
        let out_rms = calculate_rms(&output);
        assert!((in_rms - out_rms).abs() < 0.1);
    }

    #[test]
    fn test_opus_vad_silence() {
        let mut encoder =
            OpusVoiceEncoder::with_config(32_000, 0.05, 0).expect("encoder with 0 hangover");
        let silence = vec![0.001f32; OPUS_FRAME_SIZE];
        let mut wire_buf = [0u8; MAX_OPUS_PACKET_SIZE];

        let enc_res = encoder.encode(&silence, &mut wire_buf).expect("encode");
        assert!(enc_res.is_none(), "silence should be suppressed by VAD");
    }

    #[test]
    fn test_opus_packet_loss_concealment() {
        let mut decoder = OpusVoiceDecoder::new().expect("decoder");
        let mut output = vec![0.0f32; OPUS_FRAME_SIZE];

        // PLC call on None packet
        let plc_len = decoder.decode(None, &mut output).expect("plc decode");
        assert_eq!(plc_len, OPUS_FRAME_SIZE);
    }
}
