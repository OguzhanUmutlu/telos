//! High-performance audio engine and 3D spatial acoustic simulation system.
//!
//! Provides distance attenuation, ear panning, procedural acoustic synthesis,
//! block material interaction audio, and ambient looping weather soundscapes.

pub mod codec;
pub mod engine;
pub mod error;
pub mod jitter;
pub mod occlusion;
pub mod source;
pub mod spatial;
pub mod stream;
pub mod synth;

pub use codec::{
    MAX_OPUS_PACKET_SIZE, OPUS_CHANNELS, OPUS_FRAME_MS, OPUS_FRAME_SIZE, OPUS_SAMPLE_RATE,
    OpusVoiceDecoder, OpusVoiceEncoder,
};
pub use engine::AudioEngine;
pub use error::AudioError;
pub use jitter::VoiceJitterBuffer;
pub use occlusion::{LowPassFilter, VoiceOcclusion};
pub use source::SoundBuffer;
pub use spatial::{BinauralSpatializer, DelayLine, Listener, calculate_spatial_gains};
pub use stream::{SpatialVoicePlayer, SpatialVoiceSource};
pub use synth::{
    SYNTH_SAMPLE_RATE, synthesize_advancement_chime, synthesize_arrow_hit, synthesize_block_break,
    synthesize_block_place, synthesize_bow_shoot, synthesize_chest_close, synthesize_chest_open,
    synthesize_entity_hurt, synthesize_footstep, synthesize_item_pickup, synthesize_rain_loop,
    synthesize_thunder, synthesize_underwater_ambience, synthesize_water_splash,
};
pub use telos_content::sound::{BlockSoundGroup, SoundCategory, SoundEvent};
