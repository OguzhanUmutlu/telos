//! Authoritative server runtime, session management, and chunk delivery pipeline for voxel.
//!
//! Provides the core game server running at 20 TPS, managing player sessions,
//! prioritizing chunk transmission by distance and camera look angle, and driving
//! procedural terrain generation and lighting asynchronously.

pub mod builder;
pub mod config;
pub mod error;
pub mod form;
pub mod multi_world;
pub mod priority;
pub mod server;
pub mod session;
pub mod storage;
pub mod usercache;
pub mod world;

pub use builder::ServerBuilder;
pub use config::{ConfigError, ServerConfig};
pub use error::ServerError;
pub use multi_world::{MultiWorldManager, WorldConfig, WorldError};
pub use priority::{QueuedChunk, compute_chunk_priority};
pub use server::Server;
pub use session::{ActiveContainerSession, PlayerSession};
pub use storage::WorldStorage;
pub use telos_worldgen::GeneratorKind;
pub use usercache::{IdentityConflictError, UserCache, UserCacheEntry};
pub use world::{ServerChunk, ServerWorld};
