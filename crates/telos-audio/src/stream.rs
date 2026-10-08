//! Real-time spatial voice audio stream player connecting the jitter buffer and Opus decoder to rodio.

use crate::codec::{OPUS_FRAME_SIZE, OPUS_SAMPLE_RATE, OpusVoiceDecoder};
use crate::error::AudioError;
use crate::jitter::VoiceJitterBuffer;
use crate::occlusion::VoiceOcclusion;
use crate::spatial::{BinauralSpatializer, Listener};
use glam::Vec3;
use rodio::Source;
use std::collections::VecDeque;
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Spatial voice player managing jitter buffering, Opus decoding, and binaural HRTF spatialization for one remote speaker.
pub struct SpatialVoicePlayer {
    decoder: OpusVoiceDecoder,
    jitter: VoiceJitterBuffer,
    spatializer: BinauralSpatializer,
    occlusion: VoiceOcclusion,
    emitter_pos: Vec3,
    min_distance: f32,
    max_distance: f32,
    volume: f32,
    audio_fifo: VecDeque<f32>,
    temp_pcm: Vec<f32>,
    temp_stereo: Vec<f32>,
    idle_frames: usize,
}

impl SpatialVoicePlayer {
    /// Creates a new spatial voice player with specified near and far acoustic distance thresholds.
    pub fn new(min_distance: f32, max_distance: f32) -> Result<Self, AudioError> {
        let decoder = OpusVoiceDecoder::new()?;
        let spatializer = BinauralSpatializer::new(OPUS_SAMPLE_RATE as f32);

        Ok(Self {
            decoder,
            jitter: VoiceJitterBuffer::new(),
            spatializer,
            occlusion: VoiceOcclusion::clear(),
            emitter_pos: Vec3::ZERO,
            min_distance,
            max_distance,
            volume: 1.0,
            audio_fifo: VecDeque::with_capacity(OPUS_FRAME_SIZE * 8),
            temp_pcm: vec![0.0; OPUS_FRAME_SIZE],
            temp_stereo: vec![0.0; OPUS_FRAME_SIZE * 2],
            idle_frames: 0,
        })
    }

    /// Pushes an incoming voice frame into the player's jitter buffer and updates speaker position.
    pub fn push_voice_packet(&mut self, sequence: u64, data: Vec<u8>, pos: Vec3) {
        self.emitter_pos = pos;
        self.jitter.push(sequence, data);
        self.idle_frames = 0;
    }

    /// Updates speaker world coordinates and terrain acoustic occlusion.
    pub fn update_acoustics(&mut self, pos: Vec3, occlusion: VoiceOcclusion) {
        self.emitter_pos = pos;
        self.occlusion = occlusion;
    }

    /// Sets the voice playback volume multiplier `[0.0, 1.0+]`.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.max(0.0);
    }

    /// Returns the current playback volume.
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Returns the current 3D position of the speaker.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.emitter_pos
    }

    /// Decodes buffered frames from the jitter buffer and renders interleaved stereo samples.
    pub fn render_samples_if_needed(&mut self, listener: &Listener, target_buffered: usize) {
        while self.audio_fifo.len() < target_buffered {
            if let Some(pkt_opt) = self.jitter.pop() {
                // Decode Opus frame or trigger Packet Loss Concealment (PLC) on None
                let decoded_res = self.decoder.decode(pkt_opt.as_deref(), &mut self.temp_pcm);

                if let Ok(samples) = decoded_res {
                    // Spatialize mono frame into interleaved stereo
                    self.spatializer.spatialize_frame_interleaved(
                        &self.temp_pcm[..samples],
                        &mut self.temp_stereo[..samples * 2],
                        listener,
                        self.emitter_pos,
                        self.min_distance,
                        self.max_distance,
                        self.volume,
                        &self.occlusion,
                    );
                    self.audio_fifo.extend(&self.temp_stereo[..samples * 2]);
                }
            } else {
                break;
            }
        }
    }

    /// Pops the next interleaved stereo sample from the rendered FIFO.
    pub fn pop_sample(&mut self, listener: &Listener) -> Option<f32> {
        self.render_samples_if_needed(listener, OPUS_FRAME_SIZE * 2);
        if let Some(sample) = self.audio_fifo.pop_front() {
            Some(sample)
        } else {
            // Silence when buffer underruns
            self.idle_frames += 1;
            if self.idle_frames > OPUS_SAMPLE_RATE as usize * 10 {
                // Terminate after 10 seconds of silence to allow cleanup
                None
            } else {
                Some(0.0)
            }
        }
    }
}

/// Streamable `rodio::Source` adapter consuming from a shared `SpatialVoicePlayer`.
#[derive(Clone)]
pub struct SpatialVoiceSource {
    player: Arc<Mutex<SpatialVoicePlayer>>,
    listener: Arc<Mutex<Listener>>,
}

impl SpatialVoiceSource {
    /// Creates a new playable source for rodio mixing.
    pub fn new(player: Arc<Mutex<SpatialVoicePlayer>>, listener: Arc<Mutex<Listener>>) -> Self {
        Self { player, listener }
    }
}

impl Iterator for SpatialVoiceSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let listener = {
            let l = self.listener.lock().unwrap();
            *l
        };
        let mut player = self.player.lock().unwrap();
        player.pop_sample(&listener)
    }
}

impl Source for SpatialVoiceSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> NonZero<u16> {
        NonZero::new(2).unwrap()
    }

    fn sample_rate(&self) -> NonZero<u32> {
        NonZero::new(OPUS_SAMPLE_RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        None // Live stream with indefinite duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_voice_player_lifecycle() {
        let mut player = SpatialVoicePlayer::new(1.0, 32.0).expect("player");
        let listener = Listener::default();

        // Initially empty: should produce 0.0 (silence)
        let sample = player.pop_sample(&listener);
        assert_eq!(sample, Some(0.0));
    }
}
