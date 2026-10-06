//! In-memory zero-copy transport for singleplayer client-server communication.

use crate::error::{NetError, Result};
use crate::transport::{ConnStats, ConnStatsCounters, Connection, Incoming, Lane, Payload};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use vx_protocol::messages::DisconnectReason;

/// Default capacity for reliable queued messages before applying backpressure.
pub const DEFAULT_RELIABLE_CAPACITY: usize = 4096;

/// Default capacity for unreliable datagram ring buffer before dropping oldest.
pub const DEFAULT_UNRELIABLE_CAPACITY: usize = 64;

struct EndpointQueues<T> {
    reliable: Mutex<VecDeque<Incoming<T>>>,
    unreliable_ring: Mutex<VecDeque<Incoming<T>>>,
    reliable_capacity: usize,
    unreliable_capacity: usize,
}

impl<T> EndpointQueues<T> {
    fn new(reliable_cap: usize, unreliable_cap: usize) -> Self {
        Self {
            reliable: Mutex::new(VecDeque::with_capacity(64)),
            unreliable_ring: Mutex::new(VecDeque::with_capacity(unreliable_cap)),
            reliable_capacity: reliable_cap,
            unreliable_capacity: unreliable_cap,
        }
    }
}

/// Bidirectional in-memory connection endpoint.
pub struct MemoryConnection<Tx, Rx> {
    local_closed: Arc<AtomicBool>,
    remote_closed: Arc<AtomicBool>,
    disconnect_reason: Arc<Mutex<Option<DisconnectReason>>>,
    stats: Arc<ConnStatsCounters>,
    peer_stats: Arc<ConnStatsCounters>,
    inbox: Arc<EndpointQueues<Rx>>,
    outbox: Arc<EndpointQueues<Tx>>,
}

impl<Tx: Send + 'static, Rx: Send + 'static> MemoryConnection<Tx, Rx> {
    /// Creates a connected bidirectional pair of in-memory connections.
    ///
    /// One endpoint transmits `A` and receives `B`; the other transmits `B` and receives `A`.
    pub fn pair(
        reliable_cap: usize,
        unreliable_cap: usize,
    ) -> (MemoryConnection<Tx, Rx>, MemoryConnection<Rx, Tx>) {
        let a_closed = Arc::new(AtomicBool::new(false));
        let b_closed = Arc::new(AtomicBool::new(false));
        let disconnect_reason = Arc::new(Mutex::new(None));

        let a_stats = Arc::new(ConnStatsCounters::default());
        let b_stats = Arc::new(ConnStatsCounters::default());

        let a_inbox = Arc::new(EndpointQueues::<Rx>::new(reliable_cap, unreliable_cap));
        let b_inbox = Arc::new(EndpointQueues::<Tx>::new(reliable_cap, unreliable_cap));

        let conn_a = MemoryConnection {
            local_closed: a_closed.clone(),
            remote_closed: b_closed.clone(),
            disconnect_reason: disconnect_reason.clone(),
            stats: a_stats.clone(),
            peer_stats: b_stats.clone(),
            inbox: a_inbox.clone(),
            outbox: b_inbox.clone(),
        };

        let conn_b = MemoryConnection {
            local_closed: b_closed,
            remote_closed: a_closed,
            disconnect_reason,
            stats: b_stats,
            peer_stats: a_stats,
            inbox: b_inbox,
            outbox: a_inbox,
        };

        (conn_a, conn_b)
    }

    /// Creates a bidirectional pair using default capacities.
    pub fn pair_default() -> (MemoryConnection<Tx, Rx>, MemoryConnection<Rx, Tx>) {
        Self::pair(DEFAULT_RELIABLE_CAPACITY, DEFAULT_UNRELIABLE_CAPACITY)
    }
}

fn convert_payload<M>(payload: Payload<M>) -> Incoming<M> {
    match payload {
        Payload::Msg(m) => Incoming::Msg(m),
        Payload::Shared(s) => Incoming::Shared(s),
    }
}

impl<Tx: Send + Sync + 'static, Rx: Send + Sync + 'static> Connection<Tx, Rx>
    for MemoryConnection<Tx, Rx>
{
    fn send(&self, lane: Lane, payload: Payload<Tx>) -> Result<()> {
        if self.is_closed() {
            return Err(NetError::Closed);
        }

        let incoming = convert_payload(payload);

        match lane {
            Lane::Unreliable => {
                let mut ring = self.outbox.unreliable_ring.lock().unwrap();
                if ring.len() >= self.outbox.unreliable_capacity {
                    // Ring buffer full: drop oldest datagram to preserve real-time semantics
                    ring.pop_front();
                }
                ring.push_back(incoming);
            }
            Lane::Control | Lane::Chunk { .. } | Lane::Bulk => {
                let mut queue = self.outbox.reliable.lock().unwrap();
                if queue.len() >= self.outbox.reliable_capacity {
                    return Err(NetError::ChannelFull);
                }
                queue.push_back(incoming);
            }
        }

        self.stats.inc_packets_sent();
        self.peer_stats.inc_packets_received();

        Ok(())
    }

    fn try_recv(&self) -> Result<Option<Incoming<Rx>>> {
        // First check reliable queue
        {
            let mut queue = self.inbox.reliable.lock().unwrap();
            if let Some(item) = queue.pop_front() {
                return Ok(Some(item));
            }
        }

        // Then check unreliable ring
        {
            let mut ring = self.inbox.unreliable_ring.lock().unwrap();
            if let Some(item) = ring.pop_front() {
                return Ok(Some(item));
            }
        }

        // Both queues empty: check connection state
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
                message: "Connection closed by remote party".to_string(),
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
