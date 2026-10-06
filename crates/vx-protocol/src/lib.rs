//! # vx-protocol
//!
//! Pure CPU protocol domain crate for the voxel engine.
//! Enforces `#![forbid(unsafe_code)]` with zero I/O and zero async dependencies.
//!
//! Provides:
//! - Type-level bounds (`BoundedString`, `BoundedVec`) to eliminate unbounded allocations.
//! - `VarInt` / `VarLong` codecs (LEB128).
//! - Phase-isolated wire messages across `Hello`, `Login`, `Config`, and `Play`.
//! - Bounded length-prefixed framing and phase-aware message dispatch.

#![forbid(unsafe_code)]

pub mod bounded;
pub mod codec;
pub mod error;
pub mod messages;
pub mod varint;

pub use bounded::{BoundedString, BoundedVec};
pub use codec::{
    COMPRESSION_THRESHOLD, MAX_FRAME_SIZE, PACKET_HEADER_SIZE, PHASE_ANY, PHASE_CONFIG,
    PHASE_HELLO, PHASE_LOGIN, PHASE_PLAY, PROTOCOL_MAGIC, PROTOCOL_VERSION, PacketFlags,
    PacketHeader, decode_c2s, decode_s2c, encode_c2s, encode_s2c, peek_frame,
};
pub use error::{ProtocolError, Result};
pub use messages::{
    C2sMessage, ConnectionPhase, MSG_ID_DISCONNECT, S2cMessage,
    config::{C2sClientSettings, C2sConfigAck, C2sKnownRegistries, S2cConfigDone, S2cRegistryData},
    disconnect::{Disconnect, DisconnectReason},
    hello::{C2sHello, S2cHelloReply},
    login::{AuthMode, C2sLoginStart, S2cLoginSuccess},
    play::{
        C2sChatMessage, C2sInteractEntity, C2sKeepAlive, LodPayload, S2cChatMessage,
        S2cDespawnEntity, S2cEntityMove, S2cEntityStatus, S2cKeepAlive, S2cLodNodeData,
        S2cLodNodeUnload, S2cSpawnEntity, S2cUpdateWeather,
    },
};

/// Default build identifier string announced during handshake.
pub const BUILD_IDENTIFIER: &str = "voxel-0.1.0-dev";

/// ALPN identifier used for QUIC / TLS negotiation.
pub const ALPN_PROTOCOL: &[u8] = b"vx/1";
