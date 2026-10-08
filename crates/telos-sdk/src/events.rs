//! Event types, decode routines, and filter bitmasks for guest mods.

/// High-level event dispatched to mod subscribers.
#[derive(Debug, Clone, PartialEq)]
pub enum ModEvent {
    /// A block was broken in the world.
    BlockBroken {
        /// Coordinates of the broken voxel (x, y, z).
        pos: (i32, i32, i32),
        /// Previous block state ID.
        old_state: u32,
        /// Actor entity net ID if broken by a player/mob.
        actor_id: Option<u32>,
    },
    /// A block was placed in the world.
    BlockPlaced {
        /// Coordinates of the placed voxel (x, y, z).
        pos: (i32, i32, i32),
        /// New block state ID written.
        new_state: u32,
        /// Actor entity net ID if placed by a player/mob.
        actor_id: Option<u32>,
    },
    /// An entity took damage.
    EntityDamage {
        /// Net ID of the entity that took damage.
        target_id: u32,
        /// Amount of damage dealt.
        damage: f32,
        /// Attacker entity net ID if known.
        attacker_id: Option<u32>,
    },
    /// Server / world simulation tick advancement.
    Tick {
        /// Monotonically increasing tick counter.
        tick: u64,
    },
    /// A player joined the server.
    PlayerJoined {
        /// Net ID of the player entity.
        player_id: u32,
    },
    /// A player disconnected from the server.
    PlayerLeft {
        /// Net ID of the player entity.
        player_id: u32,
    },
}

impl ModEvent {
    /// Decodes an ABI event tuple `(event_id, p1, p2, p3, p4)` into a `ModEvent`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_abi(event_id: i32, p1: i64, p2: i64, p3: i64, p4: i64) -> Option<Self> {
        match event_id {
            1 => Some(Self::BlockBroken {
                pos: (p1 as i32, p2 as i32, p3 as i32),
                old_state: p4.unsigned_abs() as u32,
                actor_id: None,
            }),
            2 => Some(Self::BlockPlaced {
                pos: (p1 as i32, p2 as i32, p3 as i32),
                new_state: p4.unsigned_abs() as u32,
                actor_id: None,
            }),
            3 => Some(Self::EntityDamage {
                target_id: p1.unsigned_abs() as u32,
                damage: f32::from_bits(p2.unsigned_abs() as u32),
                attacker_id: if (p3 as u32) == u32::MAX {
                    None
                } else {
                    Some(p3.unsigned_abs() as u32)
                },
            }),
            4 => Some(Self::Tick {
                tick: p1.unsigned_abs(),
            }),
            5 => Some(Self::PlayerJoined {
                player_id: p1.unsigned_abs() as u32,
            }),
            6 => Some(Self::PlayerLeft {
                player_id: p1.unsigned_abs() as u32,
            }),
            _ => None,
        }
    }
}

/// Bitmask filter for subscribing to specific event subsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EventFilter(pub u32);

impl EventFilter {
    /// Filter matching block broken events.
    pub const BLOCK_BROKEN: Self = Self(1 << 0);
    /// Filter matching block placed events.
    pub const BLOCK_PLACED: Self = Self(1 << 1);
    /// Filter matching entity damage events.
    pub const ENTITY_DAMAGE: Self = Self(1 << 2);
    /// Filter matching periodic ticks.
    pub const TICK: Self = Self(1 << 3);
    /// Filter matching player joined events.
    pub const PLAYER_JOINED: Self = Self(1 << 4);
    /// Filter matching player left events.
    pub const PLAYER_LEFT: Self = Self(1 << 5);
    /// Filter matching all events.
    pub const ALL: Self = Self(0b0011_1111);

    /// Combines two event filters.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Checks if a filter contains the given flag.
    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }
}
