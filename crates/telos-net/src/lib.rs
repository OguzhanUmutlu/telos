//! # telos-net
//!
//! Networking runtime and transport abstractions for the voxel engine.
//!
//! Provides:
//! - Priority `Lane`s (`Control`, `Chunk`, `Bulk`, `Unreliable`).
//! - Unified `Connection` trait across in-memory and network channels.
//! - `MemoryConnection`: Zero-copy, zero-socket transport for singleplayer.
//! - `LoopbackConnection`: Deterministic wire-serialized transport for testing.
//! - `QuicEndpoint`: Real remote QUIC/UDP transport over TLS 1.3 using Quinn.

pub mod error;
pub mod lan;
pub mod loopback;
pub mod memory;
pub mod nat;
/// QUIC transport and endpoint runtime over Quinn and TLS 1.3.
pub mod quic;
pub mod ticket;
pub mod transport;

pub use error::{NetError, Result};
pub use lan::{DiscoveredLanServer, LanBeacon, LanBeaconEmitter, LanDiscoveryListener};
pub use loopback::{
    ClientCodec, LoopbackClient, LoopbackConnection, LoopbackServer, ServerCodec, WireCodec,
    loopback_pair,
};
pub use memory::{DEFAULT_RELIABLE_CAPACITY, DEFAULT_UNRELIABLE_CAPACITY, MemoryConnection};
pub use nat::{
    CandidatePriority, HolePunchCoordinator, HolePunchPacket, NatCandidate, NatMappingResult,
    NatPmpClient, NatPmpMapping, PortMapper, StunClient, StunEndpoint, UpnpClient, UpnpGateway,
};
pub use quic::{
    QuicClientEndpoint, QuicConnection, QuicListener, QuicServerEndpoint, create_client_config,
    create_server_config, generate_self_signed_cert,
};
pub use ticket::{
    CandidateAddress, CandidateType, ConnectionTicket, INVITE_SCHEME_PREFIX, base64_url_decode,
    base64_url_encode,
};
pub use transport::{ConnStats, Connection, Incoming, Lane, Payload};
