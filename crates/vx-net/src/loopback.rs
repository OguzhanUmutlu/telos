//! In-process transport that forces full wire encoding and decoding for deterministic testing.

use crate::error::{NetError, Result};
use crate::transport::{ConnStats, ConnStatsCounters, Connection, Incoming, Lane, Payload};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use vx_protocol::codec::{decode_c2s, decode_s2c, encode_c2s, encode_s2c};
use vx_protocol::messages::{C2sMessage, ConnectionPhase, DisconnectReason, S2cMessage};

/// Encodes outgoing messages according to client/server direction.
pub trait WireCodec<Tx, Rx>: Send + Sync + 'static {
    /// Encodes `Tx` message into byte buffer.
    fn encode(msg: &Tx, buf: &mut Vec<u8>);

    /// Decodes `Rx` message from byte slice for the given phase.
    fn decode(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<Rx>;
}

/// Codec implementation for client endpoint (transmits C2S, receives S2C).
pub struct ClientCodec;

impl WireCodec<C2sMessage, S2cMessage> for ClientCodec {
    fn encode(msg: &C2sMessage, buf: &mut Vec<u8>) {
        encode_c2s(msg, buf);
    }

    fn decode(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<S2cMessage> {
        decode_s2c(phase, cursor).map_err(NetError::Protocol)
    }
}

/// Codec implementation for server endpoint (transmits S2C, receives C2S).
pub struct ServerCodec;

impl WireCodec<S2cMessage, C2sMessage> for ServerCodec {
    fn encode(msg: &S2cMessage, buf: &mut Vec<u8>) {
        encode_s2c(msg, buf);
    }

    fn decode(phase: ConnectionPhase, cursor: &mut &[u8]) -> Result<C2sMessage> {
        decode_c2s(phase, cursor).map_err(NetError::Protocol)
    }
}

struct RawQueues {
    reliable: Mutex<VecDeque<Vec<u8>>>,
    unreliable_ring: Mutex<VecDeque<Vec<u8>>>,
    capacity: usize,
}

impl RawQueues {
    fn new(capacity: usize) -> Self {
        Self {
            reliable: Mutex::new(VecDeque::with_capacity(32)),
            unreliable_ring: Mutex::new(VecDeque::with_capacity(64)),
            capacity,
        }
    }
}

/// In-process wire-serialized connection endpoint.
pub struct LoopbackConnection<Tx, Rx, C: WireCodec<Tx, Rx>> {
    phase: Mutex<ConnectionPhase>,
    local_closed: Arc<AtomicBool>,
    remote_closed: Arc<AtomicBool>,
    disconnect_reason: Arc<Mutex<Option<DisconnectReason>>>,
    stats: Arc<ConnStatsCounters>,
    peer_stats: Arc<ConnStatsCounters>,
    inbox: Arc<RawQueues>,
    outbox: Arc<RawQueues>,
    _marker: std::marker::PhantomData<(Tx, Rx, C)>,
}

impl<Tx, Rx, C: WireCodec<Tx, Rx>> LoopbackConnection<Tx, Rx, C> {
    /// Sets the active lifecycle phase for message parsing.
    pub fn set_phase(&self, phase: ConnectionPhase) {
        *self.phase.lock().unwrap() = phase;
    }

    /// Gets the current lifecycle phase.
    pub fn phase(&self) -> ConnectionPhase {
        *self.phase.lock().unwrap()
    }
}

/// Convenient type alias for client-side loopback connection.
pub type LoopbackClient = LoopbackConnection<C2sMessage, S2cMessage, ClientCodec>;

/// Convenient type alias for server-side loopback connection.
pub type LoopbackServer = LoopbackConnection<S2cMessage, C2sMessage, ServerCodec>;

/// Creates a connected pair of wire-serialized loopback endpoints.
pub fn loopback_pair(capacity: usize) -> (LoopbackClient, LoopbackServer) {
    let a_closed = Arc::new(AtomicBool::new(false));
    let b_closed = Arc::new(AtomicBool::new(false));
    let disconnect_reason = Arc::new(Mutex::new(None));

    let a_stats = Arc::new(ConnStatsCounters::default());
    let b_stats = Arc::new(ConnStatsCounters::default());

    let a_inbox = Arc::new(RawQueues::new(capacity));
    let b_inbox = Arc::new(RawQueues::new(capacity));

    let client = LoopbackConnection {
        phase: Mutex::new(ConnectionPhase::Hello),
        local_closed: a_closed.clone(),
        remote_closed: b_closed.clone(),
        disconnect_reason: disconnect_reason.clone(),
        stats: a_stats.clone(),
        peer_stats: b_stats.clone(),
        inbox: a_inbox.clone(),
        outbox: b_inbox.clone(),
        _marker: std::marker::PhantomData,
    };

    let server = LoopbackConnection {
        phase: Mutex::new(ConnectionPhase::Hello),
        local_closed: b_closed,
        remote_closed: a_closed,
        disconnect_reason,
        stats: b_stats,
        peer_stats: a_stats,
        inbox: b_inbox,
        outbox: a_inbox,
        _marker: std::marker::PhantomData,
    };

    (client, server)
}

impl<Tx: Send + Sync + 'static, Rx: Send + Sync + 'static, C: WireCodec<Tx, Rx>> Connection<Tx, Rx>
    for LoopbackConnection<Tx, Rx, C>
{
    fn send(&self, lane: Lane, payload: Payload<Tx>) -> Result<()> {
        if self.is_closed() {
            return Err(NetError::Closed);
        }

        let mut buf = Vec::new();
        match payload {
            Payload::Msg(ref m) => {
                C::encode(m, &mut buf);
            }
            Payload::Shared(ref s) => {
                buf.extend_from_slice(s);
            }
        }

        let byte_len = buf.len() as u64;

        match lane {
            Lane::Unreliable => {
                let mut ring = self.outbox.unreliable_ring.lock().unwrap();
                if ring.len() >= 64 {
                    ring.pop_front();
                }
                ring.push_back(buf);
            }
            Lane::Control | Lane::Chunk { .. } | Lane::Bulk => {
                let mut queue = self.outbox.reliable.lock().unwrap();
                if queue.len() >= self.outbox.capacity {
                    return Err(NetError::ChannelFull);
                }
                queue.push_back(buf);
            }
        }

        self.stats.inc_packets_sent();
        self.stats.add_bytes_sent(byte_len);
        self.peer_stats.inc_packets_received();
        self.peer_stats.add_bytes_received(byte_len);

        Ok(())
    }

    fn try_recv(&self) -> Result<Option<Incoming<Rx>>> {
        let raw_opt = {
            let mut queue = self.inbox.reliable.lock().unwrap();
            queue.pop_front()
        }
        .or_else(|| {
            let mut ring = self.inbox.unreliable_ring.lock().unwrap();
            ring.pop_front()
        });

        if let Some(bytes) = raw_opt {
            let phase = self.phase();
            let mut cursor = &bytes[..];
            let msg = C::decode(phase, &mut cursor)?;
            return Ok(Some(Incoming::Msg(msg)));
        }

        if self.local_closed.load(Ordering::Acquire) {
            return Err(NetError::Closed);
        }
        if self.remote_closed.load(Ordering::Acquire) {
            let reason = self
                .disconnect_reason
                .lock()
                .unwrap()
                .unwrap_or(DisconnectReason::Normal);
            return Err(NetError::Disconnected {
                reason,
                message: "Closed by peer".to_string(),
            });
        }

        Ok(None)
    }

    fn stats(&self) -> ConnStats {
        self.stats.snapshot()
    }

    fn close(&self, reason: DisconnectReason) {
        self.local_closed.store(true, Ordering::Release);
        let mut r = self.disconnect_reason.lock().unwrap();
        if r.is_none() {
            *r = Some(reason);
        }
    }

    fn is_closed(&self) -> bool {
        self.local_closed.load(Ordering::Relaxed) || self.remote_closed.load(Ordering::Relaxed)
    }
}
