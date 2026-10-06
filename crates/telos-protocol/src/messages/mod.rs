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
    BlockActionKind, C2sBlockAction, C2sChatMessage, C2sCommandSuggest, C2sInteractEntity,
    C2sInventoryClick, C2sKeepAlive, C2sPlayerCommand, C2sPlayerInput, C2sPlayerPosition,
    C2sTeleportAck, ChunkPayload, InputFrame, LodPayload, PlayerCommandKind, S2cBlockActionAck,
    S2cBlockUpdate, S2cChatMessage, S2cChunkData, S2cChunkUnload, S2cCommandSuggestions,
    S2cDespawnEntity, S2cEntityMove, S2cEntityStatus, S2cInventoryBulk, S2cInventorySlot,
    S2cJoinGame, S2cKeepAlive, S2cLodNodeData, S2cLodNodeUnload, S2cPlayerMovementAck,
    S2cSpawnEntity, S2cUniformChunk, S2cUpdateStats, S2cUpdateTime, S2cUpdateWeather, SlotData,
    decode_chunk_snapshot, encode_chunk_snapshot, input_buttons,
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

    /// Converts this phase to its 1-byte wire code.
    #[must_use]
    pub const fn to_wire(self) -> u8 {
        match self {
            Self::Hello => 0,
            Self::Login => 1,
            Self::Config => 2,
            Self::Play => 3,
        }
    }

    /// Converts a 1-byte wire code to `ConnectionPhase`.
    #[must_use]
    pub const fn from_wire(wire: u8) -> Option<Self> {
        match wire {
            0 => Some(Self::Hello),
            1 => Some(Self::Login),
            2 => Some(Self::Config),
            3 => Some(Self::Play),
            _ => None,
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
    /// Play phase block break or place action.
    BlockAction(C2sBlockAction),
    /// Play phase inventory click action.
    InventoryClick(C2sInventoryClick),
    /// Play phase player debug / action command.
    PlayerCommand(C2sPlayerCommand),
    /// Play phase entity attack or interaction.
    InteractEntity(C2sInteractEntity),
    /// Play phase command auto-completion query.
    CommandSuggest(C2sCommandSuggest),
    /// Play phase movement inputs for authoritative client prediction.
    PlayerInput(C2sPlayerInput),
    /// Play phase acknowledgment of server-initiated teleport.
    TeleportAck(C2sTeleportAck),
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
            Self::KeepAlive(_)
            | Self::ChatMessage(_)
            | Self::PlayerPosition(_)
            | Self::BlockAction(_)
            | Self::InventoryClick(_)
            | Self::PlayerCommand(_)
            | Self::InteractEntity(_)
            | Self::CommandSuggest(_)
            | Self::PlayerInput(_)
            | Self::TeleportAck(_) => Some(ConnectionPhase::Play),
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
            Self::BlockAction(_) => 3,
            Self::InventoryClick(_) => 4,
            Self::PlayerCommand(_) => 5,
            Self::InteractEntity(_) => 6,
            Self::CommandSuggest(_) => 7,
            Self::PlayerInput(_) => 8,
            Self::TeleportAck(_) => 9,
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
            Self::BlockAction(m) => m.encode(buf),
            Self::InventoryClick(m) => m.encode(buf),
            Self::PlayerCommand(m) => m.encode(buf),
            Self::InteractEntity(m) => m.encode(buf),
            Self::CommandSuggest(m) => m.encode(buf),
            Self::PlayerInput(m) => m.encode(buf),
            Self::TeleportAck(m) => m.encode(buf),
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
    /// Play phase LOD node mesh and palette data.
    LodNodeData(S2cLodNodeData),
    /// Play phase LOD node unload notification.
    LodNodeUnload(S2cLodNodeUnload),
    /// Play phase world block state update.
    BlockUpdate(S2cBlockUpdate),
    /// Play phase acknowledgment of client block predictions.
    BlockActionAck(S2cBlockActionAck),
    /// Play phase world age and time-of-day synchronization.
    UpdateTime(S2cUpdateTime),
    /// Play phase player survival stats synchronization.
    UpdateStats(S2cUpdateStats),
    /// Play phase single inventory slot update.
    InventorySlot(S2cInventorySlot),
    /// Play phase full inventory bulk synchronization.
    InventoryBulk(S2cInventoryBulk),
    /// Play phase weather and atmospheric synchronization.
    UpdateWeather(S2cUpdateWeather),
    /// Play phase spawn new living entity.
    SpawnEntity(S2cSpawnEntity),
    /// Play phase despawn one or more entities.
    DespawnEntity(S2cDespawnEntity),
    /// Play phase entity position and rotation update.
    EntityMove(S2cEntityMove),
    /// Play phase entity event or animation status (hurt, death).
    EntityStatus(S2cEntityStatus),
    /// Play phase command auto-completion suggestions response.
    CommandSuggestions(S2cCommandSuggestions),
    /// Play phase authoritative player movement outcome acknowledgment.
    PlayerMovementAck(S2cPlayerMovementAck),
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
            | Self::ChunkUnload(_)
            | Self::LodNodeData(_)
            | Self::LodNodeUnload(_)
            | Self::BlockUpdate(_)
            | Self::BlockActionAck(_)
            | Self::UpdateTime(_)
            | Self::UpdateStats(_)
            | Self::InventorySlot(_)
            | Self::InventoryBulk(_)
            | Self::UpdateWeather(_)
            | Self::SpawnEntity(_)
            | Self::DespawnEntity(_)
            | Self::EntityMove(_)
            | Self::EntityStatus(_)
            | Self::CommandSuggestions(_)
            | Self::PlayerMovementAck(_) => Some(ConnectionPhase::Play),
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
            Self::LodNodeData(_) => 6,
            Self::LodNodeUnload(_) => 7,
            Self::BlockUpdate(_) => 8,
            Self::BlockActionAck(_) => 9,
            Self::UpdateTime(_) => 10,
            Self::UpdateStats(_) => 11,
            Self::InventorySlot(_) => 12,
            Self::InventoryBulk(_) => 13,
            Self::UpdateWeather(_) => 14,
            Self::SpawnEntity(_) => 15,
            Self::DespawnEntity(_) => 16,
            Self::EntityMove(_) => 17,
            Self::EntityStatus(_) => 18,
            Self::CommandSuggestions(_) => 19,
            Self::PlayerMovementAck(_) => 20,
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
            Self::LodNodeData(m) => m.encode(buf),
            Self::LodNodeUnload(m) => m.encode(buf),
            Self::BlockUpdate(m) => m.encode(buf),
            Self::BlockActionAck(m) => m.encode(buf),
            Self::UpdateTime(m) => m.encode(buf),
            Self::UpdateStats(m) => m.encode(buf),
            Self::InventorySlot(m) => m.encode(buf),
            Self::InventoryBulk(m) => m.encode(buf),
            Self::UpdateWeather(m) => m.encode(buf),
            Self::SpawnEntity(m) => m.encode(buf),
            Self::DespawnEntity(m) => m.encode(buf),
            Self::EntityMove(m) => m.encode(buf),
            Self::EntityStatus(m) => m.encode(buf),
            Self::CommandSuggestions(m) => m.encode(buf),
            Self::PlayerMovementAck(m) => m.encode(buf),
            Self::Disconnect(m) => m.encode(buf),
        }
    }
}
