//! Message definitions grouped by protocol lifecycle phases.

pub mod config;
pub mod disconnect;
pub mod hello;
pub mod login;
pub mod play;

pub use config::{
    C2sClientSettings, C2sConfigAck, C2sKnownRegistries, S2cConfigDone, S2cRegistryData,
};
pub use disconnect::{Disconnect, DisconnectReason};
pub use hello::{C2sHello, S2cHelloReply};
pub use login::{AuthMode, C2sLoginStart, S2cLoginSuccess};
pub use play::{
    C2sChatMessage, C2sKeepAlive, C2sPlayerPosition, ChunkPayload, S2cChatMessage, S2cChunkData,
    S2cChunkUnload, S2cJoinGame, S2cKeepAlive, S2cUniformChunk, decode_chunk_snapshot,
    encode_chunk_snapshot,
};

/// The protocol lifecycle phase of a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionPhase {
    /// Initial version and feature negotiation.
    Hello,
    /// Player identity verification and session start.
    Login,
    /// Registry synchronization and client settings negotiation.
    Config,
    /// Active in-game state and simulation.
    Play,
}

impl ConnectionPhase {
    /// Human-readable phase name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hello => "Hello",
            Self::Login => "Login",
            Self::Config => "Config",
            Self::Play => "Play",
        }
    }
}

/// Disconnect message wire ID across all phases.
pub const MSG_ID_DISCONNECT: u32 = 0xFF;

/// All client-to-server messages across connection phases.
#[derive(Debug, Clone, PartialEq)]
pub enum C2sMessage {
    /// Hello phase handshake initiation.
    Hello(C2sHello),
    /// Login phase authentication request.
    LoginStart(C2sLoginStart),
    /// Config phase client announcement of known registry hashes.
    KnownRegistries(C2sKnownRegistries),
    /// Config phase client render and simulation preferences.
    ClientSettings(C2sClientSettings),
    /// Config phase client acknowledgment to enter Play.
    ConfigAck(C2sConfigAck),
    /// Play phase heartbeat ping/pong response.
    KeepAlive(C2sKeepAlive),
    /// Play phase chat message or slash command.
    ChatMessage(C2sChatMessage),
    /// Play phase updated player position and orientation.
    PlayerPosition(C2sPlayerPosition),
    /// Termination message valid in any connection phase.
    Disconnect(Disconnect),
}

impl Eq for C2sMessage {}

impl C2sMessage {
    /// The lifecycle phase associated with this message.
    #[must_use]
    pub fn phase(&self) -> Option<ConnectionPhase> {
        match self {
            Self::Hello(_) => Some(ConnectionPhase::Hello),
            Self::LoginStart(_) => Some(ConnectionPhase::Login),
            Self::KnownRegistries(_) | Self::ClientSettings(_) | Self::ConfigAck(_) => {
                Some(ConnectionPhase::Config)
            }
            Self::KeepAlive(_) | Self::ChatMessage(_) | Self::PlayerPosition(_) => {
                Some(ConnectionPhase::Play)
            }
            Self::Disconnect(_) => None, // Valid in all phases
        }
    }

    /// Wire message ID within its lifecycle phase.
    #[must_use]
    pub const fn message_id(&self) -> u32 {
        match self {
            Self::Hello(_)
            | Self::LoginStart(_)
            | Self::KnownRegistries(_)
            | Self::KeepAlive(_) => 0,
            Self::ClientSettings(_) | Self::ChatMessage(_) => 1,
            Self::ConfigAck(_) | Self::PlayerPosition(_) => 2,
            Self::Disconnect(_) => MSG_ID_DISCONNECT,
        }
    }

    /// Serializes the message body (excluding message ID) into `buf`.
    pub fn encode_body(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Hello(m) => m.encode(buf),
            Self::LoginStart(m) => m.encode(buf),
            Self::KnownRegistries(m) => m.encode(buf),
            Self::ClientSettings(m) => m.encode(buf),
            Self::ConfigAck(m) => m.encode(buf),
            Self::KeepAlive(m) => m.encode(buf),
            Self::ChatMessage(m) => m.encode(buf),
            Self::PlayerPosition(m) => m.encode(buf),
            Self::Disconnect(m) => m.encode(buf),
        }
    }
}

/// All server-to-client messages across connection phases.
#[derive(Debug, Clone, PartialEq)]
pub enum S2cMessage {
    /// Hello phase handshake response.
    HelloReply(S2cHelloReply),
    /// Login phase authentication confirmation.
    LoginSuccess(S2cLoginSuccess),
    /// Config phase registry entry definitions.
    RegistryData(S2cRegistryData),
    /// Config phase server signaling configuration complete.
    ConfigDone(S2cConfigDone),
    /// Play phase heartbeat ping/pong challenge.
    KeepAlive(S2cKeepAlive),
    /// Play phase chat message or system alert broadcast.
    ChatMessage(S2cChatMessage),
    /// Play phase game join confirmation and spawn info.
    JoinGame(S2cJoinGame),
    /// Play phase chunk data and light.
    ChunkData(S2cChunkData),
    /// Play phase uniform chunk descriptor.
    UniformChunk(S2cUniformChunk),
    /// Play phase chunk unload notification.
    ChunkUnload(S2cChunkUnload),
    /// Termination message valid in any connection phase.
    Disconnect(Disconnect),
}

impl Eq for S2cMessage {}

impl S2cMessage {
    /// The lifecycle phase associated with this message.
    #[must_use]
    pub fn phase(&self) -> Option<ConnectionPhase> {
        match self {
            Self::HelloReply(_) => Some(ConnectionPhase::Hello),
            Self::LoginSuccess(_) => Some(ConnectionPhase::Login),
            Self::RegistryData(_) | Self::ConfigDone(_) => Some(ConnectionPhase::Config),
            Self::KeepAlive(_)
            | Self::ChatMessage(_)
            | Self::JoinGame(_)
            | Self::ChunkData(_)
            | Self::UniformChunk(_)
            | Self::ChunkUnload(_) => Some(ConnectionPhase::Play),
            Self::Disconnect(_) => None, // Valid in all phases
        }
    }

    /// Wire message ID within its lifecycle phase.
    #[must_use]
    pub const fn message_id(&self) -> u32 {
        match self {
            Self::HelloReply(_)
            | Self::LoginSuccess(_)
            | Self::RegistryData(_)
            | Self::KeepAlive(_) => 0,
            Self::ConfigDone(_) | Self::ChatMessage(_) => 1,
            Self::JoinGame(_) => 2,
            Self::ChunkData(_) => 3,
            Self::UniformChunk(_) => 4,
            Self::ChunkUnload(_) => 5,
            Self::Disconnect(_) => MSG_ID_DISCONNECT,
        }
    }

    /// Serializes the message body (excluding message ID) into `buf`.
    pub fn encode_body(&self, buf: &mut Vec<u8>) {
        match self {
            Self::HelloReply(m) => m.encode(buf),
            Self::LoginSuccess(m) => m.encode(buf),
            Self::RegistryData(m) => m.encode(buf),
            Self::ConfigDone(m) => m.encode(buf),
            Self::KeepAlive(m) => m.encode(buf),
            Self::ChatMessage(m) => m.encode(buf),
            Self::JoinGame(m) => m.encode(buf),
            Self::ChunkData(m) => m.encode(buf),
            Self::UniformChunk(m) => m.encode(buf),
            Self::ChunkUnload(m) => m.encode(buf),
            Self::Disconnect(m) => m.encode(buf),
        }
    }
}
