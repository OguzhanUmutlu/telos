//! Audio buffer representation and streamable source adapters.

use crate::error::AudioError;
use rodio::Source;
use std::io::Cursor;
use std::num::NonZero;
use std::sync::Arc;
use std::time::Duration;

/// In-memory PCM audio buffer holding uncompressed floating-point audio samples.
#[derive(Debug, Clone)]
pub struct SoundBuffer {
    /// Number of audio channels (1 for mono, 2 for stereo).
    pub channels: u16,
    /// Sampling frequency in Hertz (typically 44100).
    pub sample_rate: u32,
    /// Interleaved PCM audio samples in `[-1.0, 1.0]`.
    pub samples: Arc<[f32]>,
}

impl SoundBuffer {
    /// Creates a new mono audio buffer.
    #[must_use]
    pub fn from_mono(sample_rate: u32, samples: Arc<[f32]>) -> Self {
        Self {
            channels: 1,
            sample_rate,
            samples,
        }
    }

    /// Creates a new stereo audio buffer.
    #[must_use]
    pub fn from_stereo(sample_rate: u32, samples: Arc<[f32]>) -> Self {
        Self {
            channels: 2,
            sample_rate,
            samples,
        }
    }

    /// Decodes an in-memory OGG/Vorbis or WAV byte slice into a PCM `SoundBuffer`.
    pub fn decode_from_bytes(bytes: &[u8]) -> Result<Self, AudioError> {
        let cursor = Cursor::new(bytes.to_vec());
        let decoder = rodio::Decoder::try_from(cursor)
            .map_err(|e| AudioError::Decode(format!("Decoder init failed: {e}")))?;

        let channels = decoder.channels().get();
        let sample_rate = decoder.sample_rate().get();
        let samples: Vec<f32> = decoder.collect();

        Ok(Self {
            channels,
            sample_rate,
            samples: samples.into(),
        })
    }

    /// Returns a playable `rodio::Source` instance playing this buffer.
    #[must_use]
    pub fn as_source(&self) -> SoundBufferSource {
        SoundBufferSource {
            buffer: self.clone(),
            cursor: 0,
        }
    }

    /// Returns a playable 2-channel stereo source with spatial left/right gains.
    #[must_use]
    pub fn as_spatial_source(&self, left_gain: f32, right_gain: f32) -> SpatialBufferSource {
        SpatialBufferSource {
            buffer: self.clone(),
            cursor: 0,
            channel_toggle: false,
            left_gain,
            right_gain,
        }
    }
}

/// A linear playback stream over an in-memory `SoundBuffer`.
#[derive(Clone)]
pub struct SoundBufferSource {
    buffer: SoundBuffer,
    cursor: usize,
}

impl Iterator for SoundBufferSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.buffer.samples.len() {
            let sample = self.buffer.samples[self.cursor];
            self.cursor += 1;
            Some(sample)
        } else {
            None
        }
    }
}

impl Source for SoundBufferSource {
    fn current_span_len(&self) -> Option<usize> {
        Some(self.buffer.samples.len().saturating_sub(self.cursor))
    }

    fn channels(&self) -> NonZero<u16> {
        NonZero::new(self.buffer.channels).unwrap_or(NonZero::<u16>::MIN)
    }

    fn sample_rate(&self) -> NonZero<u32> {
        NonZero::new(self.buffer.sample_rate).unwrap_or(NonZero::new(44_100).unwrap())
    }

    fn total_duration(&self) -> Option<Duration> {
        let frames = self.buffer.samples.len() / (self.buffer.channels as usize).max(1);
        Some(Duration::from_secs_f64(
            frames as f64 / f64::from(self.buffer.sample_rate),
        ))
    }
}

/// A 2-channel spatialized playback stream applying ear panning and distance attenuation.
#[derive(Clone)]
pub struct SpatialBufferSource {
    buffer: SoundBuffer,
    cursor: usize,
    channel_toggle: bool,
    left_gain: f32,
    right_gain: f32,
}

impl Iterator for SpatialBufferSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.channels == 1 {
            // Mono input: emit left then right for each sample
            if self.cursor >= self.buffer.samples.len() {
                return None;
            }

            let sample = self.buffer.samples[self.cursor];
            if self.channel_toggle {
                self.channel_toggle = false;
                self.cursor += 1;
                Some(sample * self.right_gain)
            } else {
                self.channel_toggle = true;
                Some(sample * self.left_gain)
            }
        } else {
            // Stereo input: modulate left (even) and right (odd)
            if self.cursor >= self.buffer.samples.len() {
                return None;
            }
            let sample = self.buffer.samples[self.cursor];
            let gain = if self.cursor.is_multiple_of(2) {
                self.left_gain
            } else {
                self.right_gain
            };
            self.cursor += 1;
            Some(sample * gain)
        }
    }
}

impl Source for SpatialBufferSource {
    fn current_span_len(&self) -> Option<usize> {
        let remaining_frames = self.buffer.samples.len().saturating_sub(self.cursor);
        Some(remaining_frames * 2)
    }

    fn channels(&self) -> NonZero<u16> {
        NonZero::new(2).unwrap()
    }

    fn sample_rate(&self) -> NonZero<u32> {
        NonZero::new(self.buffer.sample_rate).unwrap_or(NonZero::new(44_100).unwrap())
    }

    fn total_duration(&self) -> Option<Duration> {
        let frames = self.buffer.samples.len() / (self.buffer.channels as usize).max(1);
        Some(Duration::from_secs_f64(
            frames as f64 / f64::from(self.buffer.sample_rate),
        ))
    }
}
