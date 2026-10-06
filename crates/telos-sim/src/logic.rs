//! High-performance, deterministic logic & signal propagation engine.
//!
//! Eliminates legacy locational/directional update ordering quirks, quasi-connectivity,
//! and 0-tick exploits. Uses chunk bitboard masks and multi-source decreasing wavefront
//! relaxation for deterministic, orientation-invariant signal transmission.
//!
//! Powered visuals are decoupled from dynamic block lighting to avoid cascade lighting lag.

use hashbrown::{HashMap, HashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use telos_core::coords::{BlockPos, ChunkPos, Face};
use telos_voxel::coords::{LocalIdx, split_block_pos};

/// Maximum transmission distance of a logic wire before signal drops to 0.
pub const MAX_SIGNAL_DISTANCE: u8 = 15;

/// Type and state classification of a logic component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicKind {
    /// Signal transmission wire. Power level in `0..=15`.
    Wire,
    /// Solid power block emitting constant power 15 in all 6 directions.
    PowerBlock,
    /// Toggleable lever / switch. Emits power 15 when `powered` is true.
    Lever {
        /// Whether the switch is currently closed/providing power.
        powered: bool,
    },
    /// Signal sink (lamp). Lit when receiving power $\ge 1$.
    Lamp {
        /// Whether the lamp is currently illuminated.
        lit: bool,
    },
    /// Directional amplifier / diode with configurable delay (1..=4 ticks).
    Repeater {
        /// Output forward direction.
        facing: Face,
        /// Tick delay before state transition (1..=4).
        delay: u8,
        /// Whether currently emitting power 15 forward.
        powered: bool,
    },
    /// Directional logic inverter (NOT gate). 1-tick delay.
    Inverter {
        /// Output direction.
        facing: Face,
        /// Whether currently emitting power 15.
        powered: bool,
    },
    /// Directional zero-delay diode (passes signal rear -> front, blocks reverse).
    Diode {
        /// Output forward direction.
        facing: Face,
    },
}

impl LogicKind {
    /// Returns `true` if this component is a wire.
    #[inline]
    #[must_use]
    pub const fn is_wire(self) -> bool {
        matches!(self, Self::Wire)
    }

    /// Returns `true` if this component actively provides a signal source.
    #[inline]
    #[must_use]
    pub const fn is_active_source(self) -> bool {
        match self {
            Self::PowerBlock => true,
            Self::Lever { powered }
            | Self::Repeater { powered, .. }
            | Self::Inverter { powered, .. } => powered,
            Self::Wire | Self::Lamp { .. } | Self::Diode { .. } => false,
        }
    }
}

/// A registered logic component instance in the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicComponent {
    /// Type and specific attributes of the component.
    pub kind: LogicKind,
    /// Current evaluated signal power level (0..=15).
    pub power: u8,
}

impl LogicComponent {
    /// Creates a new `LogicComponent`.
    #[inline]
    #[must_use]
    pub const fn new(kind: LogicKind, power: u8) -> Self {
        Self { kind, power }
    }
}

/// A scheduled state change for tick-delayed components (inverters and repeaters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledLogicTick {
    /// Absolute simulation tick at which this change executes.
    pub tick: u64,
    /// World position of the component.
    pub pos: BlockPos,
    /// Desired new active/power state (e.g. 15 for on, 0 for off).
    pub target_state: bool,
}

impl Ord for ScheduledLogicTick {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap ordering: smaller tick fires first.
        // Tie-breaker on BlockPos coordinates ensures deterministic execution order.
        self.tick
            .cmp(&other.tick)
            .then_with(|| self.pos.x().cmp(&other.pos.x()))
            .then_with(|| self.pos.y().cmp(&other.pos.y()))
            .then_with(|| self.pos.z().cmp(&other.pos.z()))
    }
}

impl PartialOrd for ScheduledLogicTick {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 32³ chunk container maintaining bitboard occupancy masks and component states.
#[derive(Debug, Clone)]
pub struct LogicChunk {
    /// World position of this chunk.
    pub chunk_pos: ChunkPos,
    /// 32×32 vertical column bitboard of logic component presence: index `(z << 5) | x`, bit $y$.
    pub component_mask: Box<[u32; 1024]>,
    /// 32×32 vertical column bitboard of active power sources: index `(z << 5) | x`, bit $y$.
    pub source_mask: Box<[u32; 1024]>,
    /// Sparse map of registered components within this chunk.
    pub components: HashMap<LocalIdx, LogicComponent>,
}

impl LogicChunk {
    /// Creates a new empty `LogicChunk`.
    #[must_use]
    pub fn new(chunk_pos: ChunkPos) -> Self {
        Self {
            chunk_pos,
            component_mask: vec![0u32; 1024].into_boxed_slice().try_into().unwrap(),
            source_mask: vec![0u32; 1024].into_boxed_slice().try_into().unwrap(),
            components: HashMap::new(),
        }
    }

    /// Returns `true` if there are no logic components in this chunk.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Sets or updates a component at the specified local index.
    pub fn set(&mut self, local_idx: LocalIdx, comp: LogicComponent) {
        let x = local_idx.x() as usize;
        let y = local_idx.y() as usize;
        let z = local_idx.z() as usize;
        let col = (z << 5) | x;
        let bit = 1u32 << y;

        self.component_mask[col] |= bit;
        if comp.kind.is_active_source() {
            self.source_mask[col] |= bit;
        } else {
            self.source_mask[col] &= !bit;
        }

        self.components.insert(local_idx, comp);
    }

    /// Removes a component at the specified local index.
    pub fn remove(&mut self, local_idx: LocalIdx) -> Option<LogicComponent> {
        let x = local_idx.x() as usize;
        let y = local_idx.y() as usize;
        let z = local_idx.z() as usize;
        let col = (z << 5) | x;
        let bit = 1u32 << y;

        self.component_mask[col] &= !bit;
        self.source_mask[col] &= !bit;

        self.components.remove(&local_idx)
    }

    /// Gets a component at the specified local index.
    #[inline]
    #[must_use]
    pub fn get(&self, local_idx: LocalIdx) -> Option<&LogicComponent> {
        let x = local_idx.x() as usize;
        let y = local_idx.y() as usize;
        let z = local_idx.z() as usize;
        let col = (z << 5) | x;
        let bit = 1u32 << y;

        if (self.component_mask[col] & bit) == 0 {
            return None;
        }
        self.components.get(&local_idx)
    }

    /// Gets a mutable reference to a component at the specified local index.
    #[inline]
    pub fn get_mut(&mut self, local_idx: LocalIdx) -> Option<&mut LogicComponent> {
        let x = local_idx.x() as usize;
        let y = local_idx.y() as usize;
        let z = local_idx.z() as usize;
        let col = (z << 5) | x;
        let bit = 1u32 << y;

        if (self.component_mask[col] & bit) == 0 {
            return None;
        }
        self.components.get_mut(&local_idx)
    }
}

/// Deterministic, server-authoritative logic simulation engine.
#[derive(Debug, Default)]
pub struct LogicEngine {
    chunks: HashMap<ChunkPos, LogicChunk>,
    scheduled_ticks: BinaryHeap<Reverse<ScheduledLogicTick>>,
}

impl LogicEngine {
    /// Creates a new empty `LogicEngine`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            scheduled_ticks: BinaryHeap::new(),
        }
    }

    /// Registers or updates a component at a world position.
    pub fn set_component(&mut self, pos: BlockPos, kind: LogicKind) {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        let chunk = self
            .chunks
            .entry(chunk_pos)
            .or_insert_with(|| LogicChunk::new(chunk_pos));

        let initial_power = if kind.is_active_source() { 15 } else { 0 };
        chunk.set(local_idx, LogicComponent::new(kind, initial_power));

        self.propagate_around(pos);
    }

    /// Removes a component at a world position.
    pub fn remove_component(&mut self, pos: BlockPos) {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        if let Some(chunk) = self.chunks.get_mut(&chunk_pos) {
            chunk.remove(local_idx);
            if chunk.is_empty() {
                self.chunks.remove(&chunk_pos);
            }
        }
        self.propagate_around(pos);
    }

    /// Looks up a component at a world position.
    #[must_use]
    pub fn get_component(&self, pos: BlockPos) -> Option<LogicComponent> {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        self.chunks.get(&chunk_pos)?.get(local_idx).copied()
    }

    /// Returns the evaluated signal power at a world position (0..=15).
    #[must_use]
    pub fn get_power(&self, pos: BlockPos) -> u8 {
        self.get_component(pos).map_or(0, |c| c.power)
    }

    /// Sets the lever state at the given position and updates power.
    pub fn set_lever(&mut self, pos: BlockPos, powered: bool) -> bool {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        let changed = {
            let Some(chunk) = self.chunks.get_mut(&chunk_pos) else {
                return false;
            };
            let Some(comp) = chunk.get_mut(local_idx) else {
                return false;
            };

            if let LogicKind::Lever {
                powered: current_powered,
            } = &mut comp.kind
            {
                if *current_powered == powered {
                    false
                } else {
                    *current_powered = powered;
                    comp.power = if powered { 15 } else { 0 };
                    let is_src = comp.kind.is_active_source();
                    let x = local_idx.x() as usize;
                    let y = local_idx.y() as usize;
                    let z = local_idx.z() as usize;
                    let col = (z << 5) | x;
                    let bit = 1u32 << y;
                    if is_src {
                        chunk.source_mask[col] |= bit;
                    } else {
                        chunk.source_mask[col] &= !bit;
                    }
                    true
                }
            } else {
                false
            }
        };

        if changed {
            self.propagate_around(pos);
            true
        } else {
            false
        }
    }

    /// Toggles a lever switch between on and off.
    pub fn toggle_lever(&mut self, pos: BlockPos) -> Option<bool> {
        let comp = self.get_component(pos)?;
        if let LogicKind::Lever { powered } = comp.kind {
            let new_state = !powered;
            self.set_lever(pos, new_state);
            Some(new_state)
        } else {
            None
        }
    }

    /// Schedules a state change for a delayed component.
    pub fn schedule_tick(&mut self, tick: u64, pos: BlockPos, target_state: bool) {
        self.scheduled_ticks.push(Reverse(ScheduledLogicTick {
            tick,
            pos,
            target_state,
        }));
    }

    /// Advances simulation to `current_tick`, processing expired scheduled events
    /// and propagating signals. Returns all positions whose visual state changed.
    pub fn tick(&mut self, current_tick: u64) -> Vec<(BlockPos, LogicComponent)> {
        let mut affected_positions = HashSet::new();

        // 1. Drain all events scheduled for current_tick or earlier
        while let Some(Reverse(event)) = self.scheduled_ticks.peek() {
            if event.tick > current_tick {
                break;
            }
            let event = self.scheduled_ticks.pop().unwrap().0;

            let (chunk_pos, local_idx) = split_block_pos(event.pos);
            if let Some(chunk) = self.chunks.get_mut(&chunk_pos)
                && let Some(comp) = chunk.get_mut(local_idx)
            {
                match &mut comp.kind {
                    LogicKind::Repeater { powered, .. } | LogicKind::Inverter { powered, .. }
                        if *powered != event.target_state =>
                    {
                        *powered = event.target_state;
                        comp.power = if event.target_state { 15 } else { 0 };
                        let is_src = comp.kind.is_active_source();
                        let x = local_idx.x() as usize;
                        let y = local_idx.y() as usize;
                        let z = local_idx.z() as usize;
                        let col = (z << 5) | x;
                        let bit = 1u32 << y;
                        if is_src {
                            chunk.source_mask[col] |= bit;
                        } else {
                            chunk.source_mask[col] &= !bit;
                        }
                        affected_positions.insert(event.pos);
                    }
                    _ => {}
                }
            }
        }

        // 2. Propagate signals around all modified component positions
        let mut changes = Vec::new();
        for pos in affected_positions {
            let mut network_changes = self.propagate_network_around(pos, current_tick);
            changes.append(&mut network_changes);
        }

        changes
    }

    /// Propagates signal across the network connected to `origin`.
    pub fn propagate_around(&mut self, origin: BlockPos) -> Vec<(BlockPos, LogicComponent)> {
        self.propagate_network_around(origin, 0)
    }

    /// Internal network propagation using multi-source decreasing wavefront relaxation.
    #[allow(clippy::too_many_lines)]
    fn propagate_network_around(
        &mut self,
        origin: BlockPos,
        current_tick: u64,
    ) -> Vec<(BlockPos, LogicComponent)> {
        // 1. Discover the connected component network
        let network = self.find_connected_network(origin);
        if network.is_empty() {
            return Vec::new();
        }

        // 2. Clear power on all wires and lamps in the network
        for &pos in &network {
            let (chunk_pos, local_idx) = split_block_pos(pos);
            if let Some(chunk) = self.chunks.get_mut(&chunk_pos)
                && let Some(comp) = chunk.get_mut(local_idx)
            {
                if comp.kind.is_wire() {
                    comp.power = 0;
                } else if let LogicKind::Lamp { lit } = &mut comp.kind {
                    *lit = false;
                    comp.power = 0;
                }
            }
        }

        // 3. Multi-source bucket queue: buckets[15] down to buckets[1]
        let mut buckets: [Vec<BlockPos>; 16] = Default::default();

        // Collect all active sources connected to or within this network
        for &pos in &network {
            if let Some(comp) = self.get_component(pos)
                && comp.kind.is_active_source()
            {
                // Power block, enabled lever, powered repeater, or active inverter
                buckets[15].push(pos);
            }
        }

        // Also check external immediate neighbors of the network that are active sources
        let mut visited_sources = HashSet::new();
        for &pos in &network {
            for neighbor in cardinal_neighbors(pos) {
                if !network.contains(&neighbor)
                    && visited_sources.insert(neighbor)
                    && let Some(comp) = self.get_component(neighbor)
                    && comp.kind.is_active_source()
                    && can_source_power_into(neighbor, comp, pos)
                {
                    buckets[15].push(neighbor);
                }
            }
        }

        // 4. Wavefront BFS: propagate decreasing power
        for p in (2..=15).rev() {
            let current_bucket = std::mem::take(&mut buckets[p]);
            for src_pos in current_bucket {
                let src_power = self.get_power(src_pos);
                if src_power < p as u8 && !self.is_constant_source(src_pos) {
                    continue;
                }

                let target_power = (p - 1) as u8;

                let (neighbor_buf, count) = match self.get_component(src_pos).map(|c| c.kind) {
                    Some(
                        LogicKind::Repeater { facing, .. } | LogicKind::Inverter { facing, .. },
                    ) => (
                        [
                            src_pos.offset(facing),
                            src_pos,
                            src_pos,
                            src_pos,
                            src_pos,
                            src_pos,
                        ],
                        1,
                    ),
                    _ => (cardinal_neighbors(src_pos), 6),
                };

                for &neighbor in &neighbor_buf[..count] {
                    if !network.contains(&neighbor) {
                        continue;
                    }

                    if let Some(comp) = self.get_component(neighbor) {
                        match comp.kind {
                            LogicKind::Wire => {
                                if comp.power < target_power {
                                    self.set_power_internal(neighbor, target_power);
                                    if target_power > 1 {
                                        buckets[target_power as usize].push(neighbor);
                                    }
                                }
                            }
                            LogicKind::Lamp { .. } => {
                                if target_power >= 1 {
                                    self.set_lamp_lit_internal(neighbor, true);
                                }
                            }
                            LogicKind::Diode { facing }
                                if neighbor.offset(facing.opposite()) == src_pos =>
                            {
                                let diode_out_power = target_power;
                                self.set_power_internal(neighbor, diode_out_power);
                                let out_pos = neighbor.offset(facing);
                                if network.contains(&out_pos) && diode_out_power > 1 {
                                    buckets[diode_out_power as usize].push(neighbor);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // 5. Evaluate gate inputs (Repeaters & Inverters) and schedule delayed state changes
        for &pos in &network {
            if let Some(comp) = self.get_component(pos) {
                match comp.kind {
                    LogicKind::Repeater {
                        facing,
                        delay,
                        powered,
                    } => {
                        let input_pos = pos.offset(facing.opposite());
                        let input_power = self.get_power(input_pos);
                        let should_be_on = input_power >= 1;

                        if should_be_on != powered {
                            self.schedule_tick(current_tick + u64::from(delay), pos, should_be_on);
                        }
                    }
                    LogicKind::Inverter { facing, powered } => {
                        let input_pos = pos.offset(facing.opposite());
                        let input_power = self.get_power(input_pos);
                        // Inverter (NOT gate): on when input is 0, off when input >= 1
                        let should_be_on = input_power == 0;

                        if should_be_on != powered {
                            self.schedule_tick(current_tick + 1, pos, should_be_on);
                        }
                    }
                    _ => {}
                }
            }
        }

        // 6. Return all components in the network with their new evaluated state
        network
            .into_iter()
            .filter_map(|pos| self.get_component(pos).map(|c| (pos, c)))
            .collect()
    }

    /// Finds all contiguous logic components reachable from `origin`.
    fn find_connected_network(&self, origin: BlockPos) -> HashSet<BlockPos> {
        let mut visited = HashSet::new();
        let mut queue = Vec::new();

        if self.get_component(origin).is_some() {
            visited.insert(origin);
            queue.push(origin);
        } else {
            // If origin is not a component (e.g. air after break), seed queue with adjacent components
            for neighbor in cardinal_neighbors(origin) {
                if self.get_component(neighbor).is_some() && visited.insert(neighbor) {
                    queue.push(neighbor);
                }
            }
        }

        while let Some(current) = queue.pop() {
            for neighbor in cardinal_neighbors(current) {
                if visited.contains(&neighbor) {
                    continue;
                }
                if self.get_component(neighbor).is_some() {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        visited
    }

    fn is_constant_source(&self, pos: BlockPos) -> bool {
        self.get_component(pos)
            .is_some_and(|c| c.kind.is_active_source())
    }

    fn set_power_internal(&mut self, pos: BlockPos, power: u8) {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        if let Some(comp) = self
            .chunks
            .get_mut(&chunk_pos)
            .and_then(|chunk| chunk.get_mut(local_idx))
        {
            comp.power = power;
        }
    }

    fn set_lamp_lit_internal(&mut self, pos: BlockPos, lit: bool) {
        let (chunk_pos, local_idx) = split_block_pos(pos);
        if let Some(comp) = self
            .chunks
            .get_mut(&chunk_pos)
            .and_then(|chunk| chunk.get_mut(local_idx))
            && let LogicKind::Lamp { lit: current_lit } = &mut comp.kind
        {
            *current_lit = lit;
            comp.power = if lit { 15 } else { 0 };
        }
    }
}

fn can_source_power_into(
    src_pos: BlockPos,
    src_comp: LogicComponent,
    target_pos: BlockPos,
) -> bool {
    match src_comp.kind {
        LogicKind::PowerBlock | LogicKind::Lever { powered: true } => true,
        LogicKind::Repeater {
            facing,
            powered: true,
            ..
        }
        | LogicKind::Inverter {
            facing,
            powered: true,
        } => src_pos.offset(facing) == target_pos,
        _ => false,
    }
}

/// Helper returning 6 cardinal adjacent block positions.
#[must_use]
pub fn cardinal_neighbors(pos: BlockPos) -> [BlockPos; 6] {
    [
        pos.offset(Face::North),
        pos.offset(Face::South),
        pos.offset(Face::East),
        pos.offset(Face::West),
        pos.offset(Face::Up),
        pos.offset(Face::Down),
    ]
}

/// Helper to step multiple units in a cardinal direction.
#[must_use]
pub fn step_in_dir(pos: BlockPos, face: Face, dist: i32) -> BlockPos {
    let n = face.normal();
    BlockPos::new(
        pos.x() + n.0 * dist,
        pos.y() + n.1 * dist,
        pos.z() + n.2 * dist,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_engine_rejection() {
        let engine = LogicEngine::new();
        assert_eq!(engine.get_power(BlockPos::new(0, 0, 0)), 0);
        assert_eq!(engine.chunks.len(), 0);
    }

    #[test]
    fn test_wire_linear_attenuation() {
        let mut engine = LogicEngine::new();

        // Place power block at (0, 0, 0)
        let source_pos = BlockPos::new(0, 0, 0);
        engine.set_component(source_pos, LogicKind::PowerBlock);

        // Place a line of 16 wires along +X: (1, 0, 0) .. (16, 0, 0)
        for x in 1..=16 {
            let pos = BlockPos::new(x, 0, 0);
            engine.set_component(pos, LogicKind::Wire);
        }

        // Verify power attenuation from 15 to 0
        assert_eq!(engine.get_power(source_pos), 15);
        assert_eq!(engine.get_power(BlockPos::new(1, 0, 0)), 14);
        assert_eq!(engine.get_power(BlockPos::new(2, 0, 0)), 13);
        assert_eq!(engine.get_power(BlockPos::new(14, 0, 0)), 1);
        // Wire at distance 15 from power block has power 0 (signal extinguished)
        assert_eq!(engine.get_power(BlockPos::new(15, 0, 0)), 0);
        assert_eq!(engine.get_power(BlockPos::new(16, 0, 0)), 0);
    }

    #[test]
    fn test_directional_symmetry_invariance() {
        // Wire facing North vs South vs East vs West must produce bit-identical power
        for face in [Face::North, Face::South, Face::East, Face::West] {
            let mut engine = LogicEngine::new();
            let source_pos = BlockPos::new(100, 64, 100);
            engine.set_component(source_pos, LogicKind::PowerBlock);

            for dist in 1..=10 {
                let wire_pos = step_in_dir(source_pos, face, dist);
                engine.set_component(wire_pos, LogicKind::Wire);
            }

            for dist in 1..=10 {
                let wire_pos = step_in_dir(source_pos, face, dist);
                let expected_power = 15 - dist as u8;
                assert_eq!(
                    engine.get_power(wire_pos),
                    expected_power,
                    "Mismatch for face {face:?} at dist {dist}"
                );
            }
        }
    }

    #[test]
    fn test_lever_toggle_and_depower() {
        let mut engine = LogicEngine::new();
        let lever_pos = BlockPos::new(0, 0, 0);
        let wire_pos = BlockPos::new(1, 0, 0);
        let lamp_pos = BlockPos::new(2, 0, 0);

        engine.set_component(lever_pos, LogicKind::Lever { powered: false });
        engine.set_component(wire_pos, LogicKind::Wire);
        engine.set_component(lamp_pos, LogicKind::Lamp { lit: false });

        // Initially unpowered
        assert_eq!(engine.get_power(wire_pos), 0);
        let lamp = engine.get_component(lamp_pos).unwrap();
        assert_eq!(lamp.kind, LogicKind::Lamp { lit: false });

        // Toggle lever ON
        let new_state = engine.toggle_lever(lever_pos).unwrap();
        assert!(new_state);
        assert_eq!(engine.get_power(wire_pos), 14);
        let lamp = engine.get_component(lamp_pos).unwrap();
        assert_eq!(lamp.kind, LogicKind::Lamp { lit: true });

        // Toggle lever OFF
        let new_state = engine.toggle_lever(lever_pos).unwrap();
        assert!(!new_state);
        assert_eq!(engine.get_power(wire_pos), 0);
        let lamp = engine.get_component(lamp_pos).unwrap();
        assert_eq!(lamp.kind, LogicKind::Lamp { lit: false });
    }

    #[test]
    fn test_cross_chunk_boundary_propagation() {
        let mut engine = LogicEngine::new();

        // Wire starts in chunk (0, 0, 0) at x = 30 and crosses into chunk (1, 0, 0) at x = 32
        let source_pos = BlockPos::new(30, 0, 0);
        engine.set_component(source_pos, LogicKind::PowerBlock);

        for x in 31..=35 {
            engine.set_component(BlockPos::new(x, 0, 0), LogicKind::Wire);
        }

        // Source: 15
        // x = 31: 14 (chunk 0)
        // x = 32: 13 (chunk 1)
        // x = 33: 12 (chunk 1)
        // x = 34: 11 (chunk 1)
        // x = 35: 10 (chunk 1)
        assert_eq!(engine.get_power(BlockPos::new(31, 0, 0)), 14);
        assert_eq!(engine.get_power(BlockPos::new(32, 0, 0)), 13);
        assert_eq!(engine.get_power(BlockPos::new(33, 0, 0)), 12);
        assert_eq!(engine.get_power(BlockPos::new(34, 0, 0)), 11);
        assert_eq!(engine.get_power(BlockPos::new(35, 0, 0)), 10);
    }

    #[test]
    fn test_diode_one_way_signal() {
        let mut engine = LogicEngine::new();
        let src = BlockPos::new(0, 0, 0);
        let rear = BlockPos::new(1, 0, 0);
        let diode = BlockPos::new(2, 0, 0);
        let front = BlockPos::new(3, 0, 0);

        engine.set_component(src, LogicKind::PowerBlock);
        engine.set_component(rear, LogicKind::Wire);
        // Diode pointing East (+X), rear is West (-X)
        engine.set_component(diode, LogicKind::Diode { facing: Face::East });
        engine.set_component(front, LogicKind::Wire);

        // Power flows: src(15) -> rear(14) -> diode(13) -> front(12)
        assert_eq!(engine.get_power(rear), 14);
        assert_eq!(engine.get_power(diode), 13);
        assert_eq!(engine.get_power(front), 12);

        // Now remove source and put source on FRONT side
        engine.remove_component(src);
        assert_eq!(engine.get_power(rear), 0);
        assert_eq!(engine.get_power(front), 0);

        let reverse_src = BlockPos::new(4, 0, 0);
        engine.set_component(reverse_src, LogicKind::PowerBlock);

        // Reverse flow blocked by diode!
        assert_eq!(engine.get_power(front), 14);
        assert_eq!(
            engine.get_power(rear),
            0,
            "Diode must block reverse signal flow!"
        );
    }

    #[test]
    fn test_inverter_scheduled_delay() {
        let mut engine = LogicEngine::new();
        let lever_pos = BlockPos::new(0, 0, 0);
        // Inverter facing East, input is West (lever)
        let inverter_pos = BlockPos::new(1, 0, 0);
        let out_wire = BlockPos::new(2, 0, 0);

        engine.set_component(lever_pos, LogicKind::Lever { powered: false });
        engine.set_component(
            inverter_pos,
            LogicKind::Inverter {
                facing: Face::East,
                powered: true,
            },
        );
        engine.set_component(out_wire, LogicKind::Wire);

        // Initial state: input 0, inverter output 15, wire 14
        assert_eq!(engine.get_power(out_wire), 14);

        // Turn lever ON at tick 10
        engine.set_lever(lever_pos, true);
        // Immediately within tick 10, inverter is still on until scheduled tick arrives
        assert_eq!(engine.get_power(out_wire), 14);

        // Tick 11 arrives (1-tick inverter delay): inverter turns OFF
        let changes = engine.tick(11);
        assert!(!changes.is_empty());
        assert_eq!(engine.get_power(out_wire), 0);
    }

    #[test]
    fn test_repeater_amplification_and_delay() {
        let mut engine = LogicEngine::new();
        let src = BlockPos::new(0, 0, 0);
        let wire1 = BlockPos::new(1, 0, 0);
        let repeater = BlockPos::new(2, 0, 0);
        let wire2 = BlockPos::new(3, 0, 0);

        engine.set_component(src, LogicKind::PowerBlock);
        engine.set_component(wire1, LogicKind::Wire);
        // Repeater facing East, delay 2 ticks, currently unpowered
        engine.set_component(
            repeater,
            LogicKind::Repeater {
                facing: Face::East,
                delay: 2,
                powered: false,
            },
        );
        engine.set_component(wire2, LogicKind::Wire);

        // Wire 1 has power 14, wire 2 has power 0 (repeater waiting for delay)
        assert_eq!(engine.get_power(wire1), 14);
        assert_eq!(engine.get_power(wire2), 0);

        // Tick 1: still waiting (delay 2)
        engine.tick(1);
        assert_eq!(engine.get_power(wire2), 0);

        // Tick 2: repeater turns ON and amplifies to 15! Wire 2 receives 14!
        engine.tick(2);
        assert_eq!(engine.get_power(repeater), 15);
        assert_eq!(engine.get_power(wire2), 14);
    }
}
