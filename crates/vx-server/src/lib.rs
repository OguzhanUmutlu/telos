//! Authoritative server runtime, session management, and chunk delivery pipeline for voxel.
//!
//! Provides the core game server running at 20 TPS, managing player sessions,
//! prioritizing chunk transmission by distance and camera look angle, and driving
//! procedural terrain generation and lighting asynchronously.

pub mod config;
pub mod priority;
pub mod server;
pub mod session;
pub mod world;

pub use config::ServerConfig;
pub use priority::{QueuedChunk, compute_chunk_priority};
pub use server::Server;
pub use session::PlayerSession;
pub use world::ServerWorld;
