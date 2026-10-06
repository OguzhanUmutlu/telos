//! Core transport abstraction, priority lanes, payloads, and connection trait.

use crate::error::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use telos_protocol::messages::DisconnectReason;

/// Communication lane determining transmission priority, reliability, and ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    /// Reliable, strictly ordered stream for lifecycle control, configuration, chat, and block edits.
    Control,
    /// Reliable, prioritized stream for voxel chunks. Higher priority chunks are sent first.
    Chunk {
        /// Streaming priority (higher values prioritized over lower values).
        priority: i32,
    },
    /// Reliable, background stream for resource packs, mods, and bulk asset downloads.
    Bulk,
    /// Unreliable RFC 9221 datagrams for high-frequency player inputs and entity snapshots.
    Unreliable,
}

/// Outgoing payload passing either typed messages or zero-copy shared buffers.
#[derive(Clone, Debug)]
pub enum Payload<M> {
    /// Strongly-typed protocol message.
    Msg(M),
    /// Zero-copy shared byte slice (e.g. compressed chunk payload or asset blob).
    Shared(Arc<Vec<u8>>),
}

impl<M> Payload<M> {
    /// Wraps a typed message.
    #[inline]
    pub fn msg(message: M) -> Self {
        Self::Msg(message)
    }

    /// Wraps a shared immutable buffer.
    #[inline]
    pub fn shared(buffer: Arc<Vec<u8>>) -> Self {
        Self::Shared(buffer)
    }

    /// Returns `true` if this payload holds an `Arc` shared buffer.
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }
}

impl<M> From<M> for Payload<M> {
    #[inline]
    fn from(msg: M) -> Self {
        Self::Msg(msg)
    }
}

/// Incoming data received from a connection.
#[derive(Clone, Debug)]
pub enum Incoming<M> {
    /// Strongly-typed deserialized message.
    Msg(M),
    /// Raw unparsed bytes from network datagram or bulk stream.
    Raw(Vec<u8>),
    /// Zero-copy shared buffer passed in-memory.
    Shared(Arc<Vec<u8>>),
}

impl<M> Incoming<M> {
    /// Attempts to consume this incoming item as a typed message.
    pub fn into_msg(self) -> Option<M> {
        match self {
            Self::Msg(m) => Some(m),
            _ => None,
        }
    }
}

/// Real-time connection metrics and telemetry counters.
#[derive(Debug, Default)]
pub struct ConnStatsCounters {
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    packets_sent: AtomicU64,
    packets_received: AtomicU64,
    rtt_ms: AtomicU32,
    packet_loss_percent: AtomicU8,
}

impl ConnStatsCounters {
    /// Increments sent byte counter.
    #[inline]
    pub fn add_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increments received byte counter.
    #[inline]
    pub fn add_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increments sent packet counter.
    #[inline]
    pub fn inc_packets_sent(&self) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments received packet counter.
    #[inline]
    pub fn inc_packets_received(&self) {
        self.packets_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Sets measured round-trip time in milliseconds.
    #[inline]
    pub fn set_rtt(&self, rtt_ms: u32) {
        self.rtt_ms.store(rtt_ms, Ordering::Relaxed);
    }

    /// Sets measured packet loss percentage (0-100).
    #[inline]
    pub fn set_packet_loss(&self, loss: u8) {
        self.packet_loss_percent.store(loss, Ordering::Relaxed);
    }

    /// Snapshots current counters into a `ConnStats` structure.
    pub fn snapshot(&self) -> ConnStats {
        ConnStats {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            packets_received: self.packets_received.load(Ordering::Relaxed),
            rtt_ms: self.rtt_ms.load(Ordering::Relaxed),
            packet_loss_percent: self.packet_loss_percent.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of connection statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConnStats {
    /// Total bytes sent over transport.
    pub bytes_sent: u64,
    /// Total bytes received from transport.
    pub bytes_received: u64,
    /// Total packets/messages transmitted.
    pub packets_sent: u64,
    /// Total packets/messages received.
    pub packets_received: u64,
    /// Current estimated round-trip latency in milliseconds.
    pub rtt_ms: u32,
    /// Current packet loss percentage (0..=100).
    pub packet_loss_percent: u8,
}

/// Unified bidirectional transport connection between client and server.
pub trait Connection<Tx, Rx>: Send + Sync + 'static {
    /// Transmits a payload across the specified lane.
    fn send(&self, lane: Lane, payload: Payload<Tx>) -> Result<()>;

    /// Synchronously retrieves the next incoming item without blocking.
    /// Returns `Ok(Some(item))` if ready, `Ok(None)` if pending, or `Err(NetError)` if closed.
    fn try_recv(&self) -> Result<Option<Incoming<Rx>>>;

    /// Returns point-in-time telemetry counters.
    fn stats(&self) -> ConnStats;

    /// Closes the connection with an explicit reason.
    fn close(&self, reason: DisconnectReason);

    /// Checks if this connection has been terminated locally or remotely.
    fn is_closed(&self) -> bool;
}
