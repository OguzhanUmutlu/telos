//! Core Audio Engine managing devices, playback channels, and spatial mixing.

use crate::error::AudioError;
use crate::source::SoundBuffer;
use crate::spatial::{Listener, calculate_spatial_gains};
use crate::synth::{
    SYNTH_SAMPLE_RATE, synthesize_arrow_hit, synthesize_block_break, synthesize_block_place,
    synthesize_bow_shoot, synthesize_entity_hurt, synthesize_footstep, synthesize_item_pickup,
    synthesize_rain_loop, synthesize_thunder,
};
use glam::Vec3;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use rodio::{Player, Source};
use std::collections::HashMap;
use telos_content::sound::SoundCategory;
use tracing::{info, warn};

const CATEGORY_COUNT: usize = 10;

/// Central audio manager controlling spatial listener state, playback channels, and volume busses.
pub struct AudioEngine {
    sink: Option<MixerDeviceSink>,
    listener: Listener,
    master_volume: f32,
    category_volumes: [f32; CATEGORY_COUNT],
    active_players: Vec<Player>,
    ambient_rain_player: Option<Player>,
    cached_buffers: HashMap<String, SoundBuffer>,
}

impl AudioEngine {
    /// Initializes the audio engine with the default OS audio output device.
    ///
    /// If no audio device is available (e.g. CI runner), gracefully degrades to mock mode.
    #[must_use]
    pub fn new() -> Self {
        let sink = match DeviceSinkBuilder::open_default_sink() {
            Ok(s) => {
                info!("Audio output device successfully initialized");
                Some(s)
            }
            Err(e) => {
                warn!("No audio output device found: {e}. Running in headless/mock audio mode");
                None
            }
        };

        let mut category_volumes = [1.0f32; CATEGORY_COUNT];
        category_volumes[SoundCategory::Weather as usize] = 0.85;
        category_volumes[SoundCategory::Blocks as usize] = 0.95;
        category_volumes[SoundCategory::Players as usize] = 1.0;

        Self {
            sink,
            listener: Listener::default(),
            master_volume: 1.0,
            category_volumes,
            active_players: Vec::with_capacity(32),
            ambient_rain_player: None,
            cached_buffers: HashMap::new(),
        }
    }

    /// Creates an audio engine in explicit mock mode without requesting audio hardware.
    #[must_use]
    pub fn mock() -> Self {
        Self {
            sink: None,
            listener: Listener::default(),
            master_volume: 1.0,
            category_volumes: [1.0f32; CATEGORY_COUNT],
            active_players: Vec::new(),
            ambient_rain_player: None,
            cached_buffers: HashMap::new(),
        }
    }

    /// Returns `true` if the engine is running in mock/silent mode without physical audio output.
    #[must_use]
    pub fn is_mock(&self) -> bool {
        self.sink.is_none()
    }

    /// Updates the 3D listener position, forward view vector, and up vector.
    pub fn set_listener(&mut self, position: Vec3, forward: Vec3, up: Vec3) {
        self.listener = Listener::new(position, forward, up);
    }

    /// Sets the master volume multiplier `[0.0, 1.0+]`.
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume.max(0.0);
    }

    /// Returns the current master volume.
    #[must_use]
    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }

    /// Sets the volume multiplier for a specific category `[0.0, 1.0+]`.
    pub fn set_category_volume(&mut self, category: SoundCategory, volume: f32) {
        let idx = category as usize;
        if idx < CATEGORY_COUNT {
            self.category_volumes[idx] = volume.max(0.0);
        }
    }

    /// Returns the volume for a specific category.
    #[must_use]
    pub fn category_volume(&self, category: SoundCategory) -> f32 {
        let idx = category as usize;
        if idx < CATEGORY_COUNT {
            self.category_volumes[idx]
        } else {
            1.0
        }
    }

    /// Calculates the effective volume for a category factoring in master volume.
    #[must_use]
    pub fn effective_volume(&self, category: SoundCategory) -> f32 {
        self.master_volume * self.category_volume(category)
    }

    /// Removes completed sound players to prevent unbounded memory growth.
    pub fn cleanup_finished_players(&mut self) {
        self.active_players.retain(|player| !player.empty());
    }

    /// Returns the count of currently active playing sound channels.
    #[must_use]
    pub fn active_players_count(&self) -> usize {
        self.active_players.len()
    }

    /// Plays a non-spatial 2D sound instance (e.g. GUI click, global music).
    pub fn play_sound_2d(
        &mut self,
        category: SoundCategory,
        buffer: &SoundBuffer,
        volume: f32,
        pitch: f32,
    ) {
        let eff_vol = self.effective_volume(category) * volume;
        if eff_vol <= 0.0 {
            return;
        }

        let Some(sink) = &self.sink else {
            return;
        };

        let player = Player::connect_new(sink.mixer());
        player.set_volume(eff_vol);
        player.set_speed(pitch.clamp(0.5, 2.0));
        player.append(buffer.as_source());
        self.active_players.push(player);
    }

    /// Plays a 3D positional sound instance with distance attenuation and ear panning.
    pub fn play_sound_3d(
        &mut self,
        category: SoundCategory,
        buffer: &SoundBuffer,
        pos: Vec3,
        volume: f32,
        pitch: f32,
        min_distance: f32,
        max_distance: f32,
    ) {
        let base_vol = self.effective_volume(category) * volume;
        if base_vol <= 0.0 {
            return;
        }

        let (left_gain, right_gain) =
            calculate_spatial_gains(&self.listener, pos, min_distance, max_distance, base_vol);

        if left_gain <= 1e-4 && right_gain <= 1e-4 {
            return;
        }

        let Some(sink) = &self.sink else {
            return;
        };

        let player = Player::connect_new(sink.mixer());
        player.set_speed(pitch.clamp(0.5, 2.0));
        player.append(buffer.as_spatial_source(left_gain, right_gain));
        self.active_players.push(player);
    }

    /// Dispatches a procedural material footstep sound at the specified coordinates.
    pub fn play_procedural_step(&mut self, material: &str, pos: Vec3, pitch: f32) {
        let samples = synthesize_footstep(material, pitch);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Players, &buffer, pos, 0.7, pitch, 1.0, 16.0);
    }

    /// Dispatches a procedural block crumble/break sound at the specified coordinates.
    pub fn play_procedural_break(&mut self, pos: Vec3, pitch: f32) {
        let samples = synthesize_block_break(pitch);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Blocks, &buffer, pos, 0.9, pitch, 1.0, 24.0);
    }

    /// Dispatches a procedural block placement thud sound at the specified coordinates.
    pub fn play_procedural_place(&mut self, pos: Vec3, pitch: f32) {
        let samples = synthesize_block_place(pitch);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Blocks, &buffer, pos, 0.85, pitch, 1.0, 20.0);
    }

    /// Dispatches a procedural mob damage hurt sound at the entity's position.
    pub fn play_procedural_hurt(&mut self, pos: Vec3, pitch: f32) {
        let samples = synthesize_entity_hurt(pitch);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Hostile, &buffer, pos, 1.0, pitch, 1.5, 32.0);
    }

    /// Updates or transitions the ambient looping rain soundscape.
    pub fn update_ambient_rain(&mut self, rain_intensity: f32) {
        let intensity = rain_intensity.clamp(0.0, 1.0);
        let target_vol = self.effective_volume(SoundCategory::Weather) * intensity * 0.75;

        let Some(sink) = &self.sink else {
            return;
        };

        if target_vol <= 1e-4 {
            if let Some(player) = &self.ambient_rain_player {
                player.pause();
                player.set_volume(0.0);
            }
            return;
        }

        if self.ambient_rain_player.is_none() {
            let player = Player::connect_new(sink.mixer());
            let rain_samples = synthesize_rain_loop();
            let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, rain_samples);
            player.append(buffer.as_source().repeat_infinite());
            player.play();
            self.ambient_rain_player = Some(player);
        }

        if let Some(player) = &self.ambient_rain_player {
            player.set_volume(target_vol);
            if player.is_paused() {
                player.play();
            }
        }
    }

    /// Dispatches a thunder strike explosion at lightning coordinates with distance falloff.
    pub fn play_thunder(&mut self, lightning_pos: Vec3) {
        let samples = synthesize_thunder();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(
            SoundCategory::Weather,
            &buffer,
            lightning_pos,
            1.2,
            1.0,
            16.0,
            160.0,
        );
    }

    /// Dispatches a procedural item pickup chime sound at the player position.
    pub fn play_item_pickup(&mut self, pos: Vec3) {
        let samples = synthesize_item_pickup();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Players, &buffer, pos, 0.8, 1.0, 1.0, 16.0);
    }

    /// Dispatches a bow release twang sound at the shooter position.
    pub fn play_bow_shoot(&mut self, pos: Vec3) {
        let samples = synthesize_bow_shoot();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Players, &buffer, pos, 0.9, 1.0, 1.0, 24.0);
    }

    /// Dispatches an arrow impact sound at the struck position.
    pub fn play_arrow_hit(&mut self, pos: Vec3, is_entity: bool) {
        let samples = synthesize_arrow_hit(is_entity);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        let category = if is_entity {
            SoundCategory::Hostile
        } else {
            SoundCategory::Blocks
        };
        self.play_sound_3d(category, &buffer, pos, 0.9, 1.0, 1.0, 24.0);
    }

    /// Loads and caches an in-memory OGG/WAV sound buffer.
    pub fn load_sound_bytes(&mut self, key: &str, bytes: &[u8]) -> Result<(), AudioError> {
        let buffer = SoundBuffer::decode_from_bytes(bytes)?;
        self.cached_buffers.insert(key.to_string(), buffer);
        Ok(())
    }

    /// Retrieves a cached sound buffer by key.
    #[must_use]
    pub fn get_cached_sound(&self, key: &str) -> Option<&SoundBuffer> {
        self.cached_buffers.get(key)
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}
