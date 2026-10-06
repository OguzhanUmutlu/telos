//! Game event definitions and buffered event queue for simulation and mod dispatch.

use telos_core::coords::BlockPos;
use telos_voxel::state::BlockStateId;

/// An individual game event emitted during server simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum GameEvent {
    /// A block was broken at world coordinates.
    BlockBroken {
        /// Voxel block coordinates.
        pos: BlockPos,
        /// State ID of the block that was broken.
        old_state: BlockStateId,
        /// Optional actor (player/entity net ID) that broke the block.
        actor_net_id: Option<u64>,
    },
    /// A block was placed at world coordinates.
    BlockPlaced {
        /// Voxel block coordinates.
        pos: BlockPos,
        /// State ID of the placed block.
        new_state: BlockStateId,
        /// Optional actor (player/entity net ID) that placed the block.
        actor_net_id: Option<u64>,
    },
    /// An entity took damage.
    EntityDamage {
        /// Network ID of target entity that took damage.
        target_net_id: u32,
        /// Amount of damage applied.
        damage: f32,
        /// Network ID of attacker entity, if any.
        attacker_net_id: Option<u32>,
    },
    /// A player joined the server.
    PlayerJoined {
        /// Network entity ID assigned to the player.
        entity_net_id: u32,
        /// Username of the joining player.
        username: String,
    },
    /// A player left the server.
    PlayerLeft {
        /// Network entity ID of the leaving player.
        entity_net_id: u32,
        /// Username of the leaving player.
        username: String,
    },
    /// Server tick milestone.
    Tick {
        /// Current server tick number.
        tick: u64,
    },
}

bitflags::bitflags! {
    /// Bitflag categories of game events that mods can subscribe to.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct EventFilter: u32 {
        /// Listen to block broken events.
        const BLOCK_BROKEN = 1 << 0;
        /// Listen to block placed events.
        const BLOCK_PLACED = 1 << 1;
        /// Listen to entity damage events.
        const ENTITY_DAMAGE = 1 << 2;
        /// Listen to player join/leave events.
        const PLAYER_CONNECTION = 1 << 3;
        /// Listen to tick events.
        const TICK = 1 << 4;
    }
}

impl EventFilter {
    /// Checks if a given `GameEvent` matches this filter.
    #[must_use]
    pub fn matches(&self, event: &GameEvent) -> bool {
        match event {
            GameEvent::BlockBroken { .. } => self.contains(Self::BLOCK_BROKEN),
            GameEvent::BlockPlaced { .. } => self.contains(Self::BLOCK_PLACED),
            GameEvent::EntityDamage { .. } => self.contains(Self::ENTITY_DAMAGE),
            GameEvent::PlayerJoined { .. } | GameEvent::PlayerLeft { .. } => {
                self.contains(Self::PLAYER_CONNECTION)
            }
            GameEvent::Tick { .. } => self.contains(Self::TICK),
        }
    }
}

/// Bounded FIFO queue buffering simulation events during a tick.
#[derive(Debug)]
pub struct EventQueue {
    events: Vec<GameEvent>,
    max_capacity: usize,
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new(4096)
    }
}

impl EventQueue {
    /// Creates a new `EventQueue` with a maximum capacity.
    #[must_use]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_capacity.min(1024)),
            max_capacity,
        }
    }

    /// Pushes an event into the queue, dropping the oldest if capacity is exceeded.
    pub fn push(&mut self, event: GameEvent) {
        if self.events.len() >= self.max_capacity && !self.events.is_empty() {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Drains all buffered events, resetting the queue.
    #[must_use]
    pub fn drain(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns the number of events currently queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_queue_and_filter() {
        let mut queue = EventQueue::new(3);
        queue.push(GameEvent::Tick { tick: 1 });
        queue.push(GameEvent::Tick { tick: 2 });
        queue.push(GameEvent::Tick { tick: 3 });
        queue.push(GameEvent::Tick { tick: 4 });

        // First event (tick 1) should be dropped due to capacity 3
        assert_eq!(queue.len(), 3);
        let drained = queue.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0], GameEvent::Tick { tick: 2 });
        assert!(queue.is_empty());

        let filter = EventFilter::TICK;
        assert!(filter.matches(&GameEvent::Tick { tick: 5 }));
        assert!(!filter.matches(&GameEvent::BlockBroken {
            pos: BlockPos::new(0, 0, 0),
            old_state: BlockStateId::AIR,
            actor_net_id: None,
        }));
    }
}
