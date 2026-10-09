//! Core Audio Engine managing devices, playback channels, and spatial mixing.

use crate::error::AudioError;
use crate::occlusion::VoiceOcclusion;
use crate::source::SoundBuffer;
use crate::spatial::{Listener, calculate_spatial_gains};
use crate::stream::{SpatialVoicePlayer, SpatialVoiceSource};
use crate::synth::{
    SYNTH_SAMPLE_RATE, synthesize_advancement_chime, synthesize_arrow_hit, synthesize_block_break,
    synthesize_block_place, synthesize_bow_shoot, synthesize_chest_close, synthesize_chest_open,
    synthesize_entity_hurt, synthesize_footstep, synthesize_item_pickup, synthesize_rain_loop,
    synthesize_thunder, synthesize_underwater_ambience, synthesize_water_splash,
};
use glam::Vec3;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use rodio::{Player, Source};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use telos_content::sound::SoundCategory;
use tracing::{info, warn};

const CATEGORY_COUNT: usize = 10;

/// Central audio manager controlling spatial listener state, playback channels, and volume busses.
pub struct AudioEngine {
    sink: Option<MixerDeviceSink>,
    listener: Listener,
    shared_listener: Arc<Mutex<Listener>>,
    master_volume: f32,
    category_volumes: [f32; CATEGORY_COUNT],
    voice_chat_volume: f32,
    active_players: Vec<Player>,
    voice_players: HashMap<[u8; 16], (Arc<Mutex<SpatialVoicePlayer>>, Player)>,
    ambient_rain_player: Option<Player>,
    ambient_underwater_player: Option<Player>,
    is_underwater: bool,
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

        let listener = Listener::default();
        let shared_listener = Arc::new(Mutex::new(listener));

        Self {
            sink,
            listener,
            shared_listener,
            master_volume: 1.0,
            category_volumes,
            voice_chat_volume: 1.0,
            active_players: Vec::with_capacity(32),
            voice_players: HashMap::new(),
            ambient_rain_player: None,
            ambient_underwater_player: None,
            is_underwater: false,
            cached_buffers: HashMap::new(),
        }
    }

    /// Creates an audio engine in explicit mock mode without requesting audio hardware.
    #[must_use]
    pub fn mock() -> Self {
        let listener = Listener::default();
        let shared_listener = Arc::new(Mutex::new(listener));

        Self {
            sink: None,
            listener,
            shared_listener,
            master_volume: 1.0,
            category_volumes: [1.0f32; CATEGORY_COUNT],
            voice_chat_volume: 1.0,
            active_players: Vec::new(),
            voice_players: HashMap::new(),
            ambient_rain_player: None,
            ambient_underwater_player: None,
            is_underwater: false,
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
        if let Ok(mut l) = self.shared_listener.lock() {
            *l = self.listener;
        }
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

    /// Sets the voice chat volume multiplier `[0.0, 1.0+]`.
    pub fn set_voice_chat_volume(&mut self, volume: f32) {
        self.voice_chat_volume = volume.max(0.0);
        let eff_vol = self.master_volume * self.voice_chat_volume;
        for (player_arc, _) in self.voice_players.values() {
            if let Ok(mut p) = player_arc.lock() {
                p.set_volume(eff_vol);
            }
        }
    }

    /// Returns the voice chat volume.
    #[must_use]
    pub fn voice_chat_volume(&self) -> f32 {
        self.voice_chat_volume
    }

    /// Dispatches an incoming voice packet for a remote speaker.
    pub fn play_voice_packet(
        &mut self,
        speaker_uuid: [u8; 16],
        sequence: u64,
        data: Vec<u8>,
        pos: Vec3,
        occlusion: VoiceOcclusion,
    ) {
        let eff_vol = self.master_volume * self.voice_chat_volume;
        if eff_vol <= 0.0 {
            return;
        }

        if let Some((player_arc, _)) = self.voice_players.get(&speaker_uuid) {
            if let Ok(mut p) = player_arc.lock() {
                p.push_voice_packet(sequence, data, pos);
                p.update_acoustics(pos, occlusion);
                p.set_volume(eff_vol);
            }
            return;
        }

        let Some(sink) = &self.sink else {
            return;
        };

        if let Ok(mut player) = SpatialVoicePlayer::new(1.5, 32.0) {
            player.push_voice_packet(sequence, data, pos);
            player.update_acoustics(pos, occlusion);
            player.set_volume(eff_vol);

            let player_arc = Arc::new(Mutex::new(player));
            let source = SpatialVoiceSource::new(player_arc.clone(), self.shared_listener.clone());

            let rodio_player = Player::connect_new(sink.mixer());
            rodio_player.append(source);

            self.voice_players
                .insert(speaker_uuid, (player_arc, rodio_player));
        }
    }

    /// Removes a speaker voice stream when disconnected or despawned.
    pub fn remove_voice_player(&mut self, speaker_uuid: &[u8; 16]) {
        self.voice_players.remove(speaker_uuid);
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
        self.voice_players.retain(|_, (_, p)| !p.empty());
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

    /// Returns `true` if the listener / camera is currently submerged underwater.
    #[must_use]
    pub fn is_underwater(&self) -> bool {
        self.is_underwater
    }

    /// Updates or transitions the ambient underwater soundscape and acoustics.
    pub fn update_underwater(&mut self, is_underwater: bool) {
        self.is_underwater = is_underwater;
        let target_vol = if is_underwater {
            self.effective_volume(SoundCategory::Weather) * 0.85
        } else {
            0.0
        };

        let Some(sink) = &self.sink else {
            return;
        };

        if target_vol <= 1e-4 {
            if let Some(player) = &self.ambient_underwater_player {
                player.pause();
                player.set_volume(0.0);
            }
            return;
        }

        if self.ambient_underwater_player.is_none() {
            let player = Player::connect_new(sink.mixer());
            let underwater_samples = synthesize_underwater_ambience();
            let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, underwater_samples);
            player.append(buffer.as_source().repeat_infinite());
            player.play();
            self.ambient_underwater_player = Some(player);
        }

        if let Some(player) = &self.ambient_underwater_player {
            player.set_volume(target_vol);
            if player.is_paused() {
                player.play();
            }
        }
    }

    /// Dispatches a procedural water splash sound at the specified coordinates.
    pub fn play_water_splash(&mut self, pos: Vec3, volume: f32) {
        let samples = synthesize_water_splash(1.0);
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(
            SoundCategory::Players,
            &buffer,
            pos,
            volume.clamp(0.1, 2.0),
            1.0,
            1.0,
            24.0,
        );
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

    /// Dispatches a chest opening sound at the chest block position.
    pub fn play_chest_open(&mut self, pos: Vec3) {
        let samples = synthesize_chest_open();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Blocks, &buffer, pos, 0.9, 1.0, 1.0, 16.0);
    }

    /// Dispatches a chest closing sound at the chest block position.
    pub fn play_chest_close(&mut self, pos: Vec3) {
        let samples = synthesize_chest_close();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_3d(SoundCategory::Blocks, &buffer, pos, 0.9, 1.0, 1.0, 16.0);
    }

    /// Dispatches a celebratory procedural advancement fanfare chime (non-spatial 2D).
    pub fn play_advancement_chime(&mut self) {
        let samples = synthesize_advancement_chime();
        let buffer = SoundBuffer::from_mono(SYNTH_SAMPLE_RATE, samples);
        self.play_sound_2d(SoundCategory::Players, &buffer, 1.0, 1.0);
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
