//! High-performance audio engine and 3D spatial acoustic simulation system.
//!
//! Provides distance attenuation, ear panning, procedural acoustic synthesis,
//! block material interaction audio, and ambient looping weather soundscapes.

pub mod engine;
pub mod error;
pub mod source;
pub mod spatial;
pub mod synth;

pub use engine::AudioEngine;
pub use error::AudioError;
pub use source::SoundBuffer;
pub use spatial::{Listener, calculate_spatial_gains};
pub use synth::{
    SYNTH_SAMPLE_RATE, synthesize_block_break, synthesize_block_place, synthesize_entity_hurt,
    synthesize_footstep, synthesize_rain_loop, synthesize_thunder,
};
pub use vx_content::sound::{BlockSoundGroup, SoundCategory, SoundEvent};
