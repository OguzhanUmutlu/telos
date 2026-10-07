//! High-performance, deterministic cellular automata fluid simulation engine.
//!
//! Implements flowing water and lava mechanics:
//! - Viscous scheduled ticks: Water (5 ticks/step, 250 ms), Lava (30 ticks/step, 1.5 s).
//! - Downward flow priority forming vertical falling columns (`falling: true`).
//! - Slope-seeking drop detection probing up to 4 blocks for cliffs.
//! - Horizontal decay levels (1..=7 for water, 1..=4 for lava).
//! - Infinite water source renewal (2 adjacent water sources over solid/water base).
//! - Receding/drying cleanup wavefront when sources are removed.
//! - Water-lava fluid reaction matrix (Obsidian, Cobblestone, Stone).

use hashbrown::HashSet;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use telos_core::coords::{BlockPos, Face};
pub use telos_voxel::fluid::{FluidKind, FluidState};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

/// Update interval for water in simulation ticks (5 ticks = 250 ms at 20 TPS).
pub const WATER_TICK_RATE: u64 = 5;

/// Update interval for lava in simulation ticks (30 ticks = 1.5 s at 20 TPS).
pub const LAVA_TICK_RATE: u64 = 30;

/// Maximum horizontal spread distance / decay level for water (1..=7).
pub const WATER_MAX_DECAY: u8 = 7;

/// Maximum horizontal spread distance / decay level for lava in the overworld (1..=4).
pub const LAVA_MAX_DECAY: u8 = 4;

/// Maximum search distance when probing for drop-offs (cliffs/holes).
pub const SLOPE_SEARCH_DISTANCE: i32 = 4;

/// Resulting solid block when water and lava interact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidReaction {
    /// Water touches lava source block -> Obsidian.
    Obsidian,
    /// Water touches flowing lava -> Cobblestone.
    Cobblestone,
    /// Lava flows downward on top of water -> Stone.
    Stone,
}

/// A scheduled fluid simulation step for a block position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledFluidTick {
    /// Simulation tick at which this update must execute.
    pub tick: u64,
    /// World position of the fluid block.
    pub pos: BlockPos,
    /// Fluid kind (water or lava).
    pub kind: FluidKind,
}

impl Ord for ScheduledFluidTick {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tick.cmp(&other.tick).then_with(|| {
            self.pos
                .x()
                .cmp(&other.pos.x())
                .then_with(|| self.pos.y().cmp(&other.pos.y()))
                .then_with(|| self.pos.z().cmp(&other.pos.z()))
        })
    }
}

impl PartialOrd for ScheduledFluidTick {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Trait abstracting world block queries for the fluid simulation engine.
pub trait FluidWorldReader {
    /// Returns the current block state at the specified world coordinates.
    fn get_block(&self, pos: BlockPos) -> BlockStateId;

    /// Returns a reference to the active block registry.
    fn registry(&self) -> &BlockRegistry;
}

/// Real-time cellular automata fluid simulation engine.
#[derive(Debug, Default)]
pub struct FluidEngine {
    scheduled_ticks: BinaryHeap<Reverse<ScheduledFluidTick>>,
    scheduled_set: HashSet<BlockPos>,
}

impl FluidEngine {
    /// Creates a new empty `FluidEngine`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scheduled_ticks: BinaryHeap::new(),
            scheduled_set: HashSet::new(),
        }
    }

    /// Returns the tick rate for the given fluid kind.
    #[inline]
    #[must_use]
    pub const fn tick_rate(kind: FluidKind) -> u64 {
        match kind {
            FluidKind::Water => WATER_TICK_RATE,
            FluidKind::Lava => LAVA_TICK_RATE,
        }
    }

    /// Returns the maximum horizontal decay level for the given fluid kind.
    #[inline]
    #[must_use]
    pub const fn max_decay(kind: FluidKind) -> u8 {
        match kind {
            FluidKind::Water => WATER_MAX_DECAY,
            FluidKind::Lava => LAVA_MAX_DECAY,
        }
    }

    /// Returns the count of pending scheduled fluid ticks.
    #[inline]
    #[must_use]
    pub fn pending_ticks_count(&self) -> usize {
        self.scheduled_ticks.len()
    }

    /// Schedules a fluid update tick for a specific position if not already queued.
    pub fn schedule_tick(&mut self, tick: u64, pos: BlockPos, kind: FluidKind) {
        if self.scheduled_set.insert(pos) {
            self.scheduled_ticks
                .push(Reverse(ScheduledFluidTick { tick, pos, kind }));
        }
    }

    /// Notifies the fluid engine that a block at `pos` was placed, broken, or modified.
    ///
    /// Schedules `pos` (if it is fluid) and all 6 adjacent cardinal neighbors if they are fluid.
    pub fn on_block_changed<R: FluidWorldReader>(
        &mut self,
        pos: BlockPos,
        current_tick: u64,
        world: &R,
    ) {
        let reg = world.registry();
        let cur_state = world.get_block(pos);

        if let Some(fluid) = reg.fluid_state(cur_state) {
            self.schedule_tick(current_tick + Self::tick_rate(fluid.kind), pos, fluid.kind);
        }

        // Check 6 adjacent neighbors (Up, Down, North, South, East, West)
        for &face in &Face::ALL {
            let n_pos = pos.offset(face);
            let n_state = world.get_block(n_pos);
            if let Some(fluid) = reg.fluid_state(n_state) {
                self.schedule_tick(
                    current_tick + Self::tick_rate(fluid.kind),
                    n_pos,
                    fluid.kind,
                );
            }
        }
    }

    /// Evaluates if water source renewal conditions are met for `pos`.
    ///
    /// If at least 2 horizontally adjacent neighbors are water source blocks
    /// and the block below is solid or water, water source block renewal triggers.
    pub fn check_water_source_creation<R: FluidWorldReader>(
        &self,
        pos: BlockPos,
        world: &R,
    ) -> bool {
        let reg = world.registry();
        let below_pos = pos.offset(Face::Down);
        let below_state = world.get_block(below_pos);

        // Base must be solid or water
        let is_solid_base = reg
            .flags(below_state)
            .contains(telos_voxel::state::StateFlags::OPAQUE_FULL)
            || reg.is_fluid(below_state);
        if !is_solid_base {
            return false;
        }

        let mut source_count = 0;
        for &face in &[Face::North, Face::South, Face::East, Face::West] {
            let n_pos = pos.offset(face);
            let n_state = world.get_block(n_pos);
            if let Some(fluid) = reg.fluid_state(n_state)
                && fluid.kind == FluidKind::Water
                && fluid.is_source()
            {
                source_count += 1;
            }
        }

        source_count >= 2
    }

    /// Evaluates water and lava fluid interactions for `pos`.
    ///
    /// Returns the target position and reaction kind if an interaction occurs.
    pub fn check_reaction<R: FluidWorldReader>(
        &self,
        pos: BlockPos,
        kind: FluidKind,
        world: &R,
    ) -> Option<(BlockPos, FluidReaction)> {
        let reg = world.registry();

        match kind {
            FluidKind::Water => {
                // Check all 6 adjacent blocks for lava
                for &face in &Face::ALL {
                    let n_pos = pos.offset(face);
                    let n_state = world.get_block(n_pos);
                    if let Some(n_fluid) = reg.fluid_state(n_state)
                        && n_fluid.kind == FluidKind::Lava
                    {
                        if n_fluid.is_source() {
                            return Some((n_pos, FluidReaction::Obsidian));
                        }
                        return Some((n_pos, FluidReaction::Cobblestone));
                    }
                }
                None
            }
            FluidKind::Lava => {
                // Downward check: lava flowing onto water -> Stone
                let down_pos = pos.offset(Face::Down);
                let down_state = world.get_block(down_pos);
                if let Some(down_fluid) = reg.fluid_state(down_state)
                    && down_fluid.kind == FluidKind::Water
                {
                    return Some((down_pos, FluidReaction::Stone));
                }

                // Horizontal check: if flowing lava touches water -> Cobblestone
                let this_state = world.get_block(pos);
                let this_fluid = reg.fluid_state(this_state)?;
                for &face in &[Face::North, Face::South, Face::East, Face::West] {
                    let n_pos = pos.offset(face);
                    let n_state = world.get_block(n_pos);
                    if let Some(n_fluid) = reg.fluid_state(n_state)
                        && n_fluid.kind == FluidKind::Water
                    {
                        if this_fluid.is_source() {
                            return Some((pos, FluidReaction::Obsidian));
                        }
                        return Some((pos, FluidReaction::Cobblestone));
                    }
                }
                None
            }
        }
    }

    /// Probes horizontal directions for drop-offs (cliffs/holes) up to `SLOPE_SEARCH_DISTANCE`.
    ///
    /// If drops are found, returns only the direction(s) that reach a drop in the minimum distance.
    /// Otherwise, returns all unblocked horizontal directions.
    pub fn find_flow_directions<R: FluidWorldReader>(
        &self,
        pos: BlockPos,
        kind: FluidKind,
        world: &R,
    ) -> ([Face; 4], usize) {
        let reg = world.registry();
        let cardinals = [Face::North, Face::South, Face::East, Face::West];

        let mut min_distance = i32::MAX;
        let mut drop_dirs = [Face::North; 4];
        let mut drop_count = 0;
        let mut open_dirs = [Face::North; 4];
        let mut open_count = 0;

        for &face in &cardinals {
            let n_pos = pos.offset(face);
            let n_state = world.get_block(n_pos);

            // Cannot flow into non-replaceable solid block
            let can_flow_into = n_state.is_air()
                || reg.is_replaceable(n_state)
                || reg.fluid_state(n_state).is_some_and(|f| f.kind == kind);
            if !can_flow_into {
                continue;
            }
            open_dirs[open_count] = face;
            open_count += 1;

            // Probe forward for cliff/drop
            let mut curr = n_pos;
            let mut dist = 1;
            while dist <= SLOPE_SEARCH_DISTANCE {
                let below = curr.offset(Face::Down);
                let below_state = world.get_block(below);

                // If block below is open air or lower fluid, we found a drop!
                let is_drop = below_state.is_air()
                    || reg.is_replaceable(below_state)
                    || reg
                        .fluid_state(below_state)
                        .is_some_and(|f| f.kind == kind && f.level > 0);
                if is_drop {
                    if dist < min_distance {
                        min_distance = dist;
                        drop_dirs[0] = face;
                        drop_count = 1;
                    } else if dist == min_distance {
                        drop_dirs[drop_count] = face;
                        drop_count += 1;
                    }
                    break;
                }

                // Advance in this direction
                curr = curr.offset(face);
                let step_state = world.get_block(curr);
                let step_open = step_state.is_air()
                    || reg.is_replaceable(step_state)
                    || reg.fluid_state(step_state).is_some_and(|f| f.kind == kind);
                if !step_open {
                    break;
                }
                dist += 1;
            }
        }

        if drop_count > 0 {
            (drop_dirs, drop_count)
        } else {
            (open_dirs, open_count)
        }
    }

    /// Advances the fluid cellular automata simulation for the specified simulation tick.
    ///
    /// Evaluates scheduled fluid ticks, computes falling columns and horizontal spreads,
    /// applies fluid reaction conversions (Obsidian, Cobblestone, Stone), and returns
    /// all block mutations `(BlockPos, BlockStateId)`.
    fn compute_target_state<R: FluidWorldReader>(
        &self,
        pos: BlockPos,
        kind: FluidKind,
        fluid: FluidState,
        world: &R,
    ) -> Option<FluidState> {
        let reg = world.registry();
        if fluid.is_source() {
            return Some(fluid);
        }

        // Check if fluid is falling directly from above
        let above_pos = pos.offset(Face::Up);
        let above_state = world.get_block(above_pos);
        if let Some(above_fluid) = reg.fluid_state(above_state)
            && above_fluid.kind == kind
        {
            return Some(FluidState::new(kind, 1, true));
        }
        if kind == FluidKind::Water && self.check_water_source_creation(pos, world) {
            return Some(FluidState::new(kind, 0, false));
        }

        // Check horizontal incoming fluid levels
        let mut min_incoming = u8::MAX;
        for &face in &[Face::North, Face::South, Face::East, Face::West] {
            let n_pos = pos.offset(face);
            let n_state = world.get_block(n_pos);
            if let Some(n_fluid) = reg.fluid_state(n_state)
                && n_fluid.kind == kind
            {
                let effective_level = if n_fluid.falling {
                    1
                } else {
                    n_fluid.level.saturating_add(1)
                };
                if effective_level <= Self::max_decay(kind) {
                    min_incoming = min_incoming.min(effective_level);
                }
            }
        }

        if min_incoming <= Self::max_decay(kind) {
            Some(FluidState::new(kind, min_incoming, false))
        } else {
            // Fluid decays to air (evaporation/drain)
            None
        }
    }

    fn spread_downward<R: FluidWorldReader>(
        &mut self,
        pos: BlockPos,
        kind: FluidKind,
        down_fluid: Option<FluidState>,
        current_tick: u64,
        world: &R,
        updates: &mut Vec<(BlockPos, BlockStateId)>,
    ) {
        let reg = world.registry();
        if let Some(df) = down_fluid
            && df.kind != kind
        {
            if let Some((react_pos, reaction)) = self.check_reaction(pos, kind, world) {
                let reaction_state = reaction_to_block_state(reg, reaction);
                updates.push((react_pos, reaction_state));
            }
        } else {
            // Spread straight down as a falling vertical column
            let falling_id = reg.fluid_state_id(kind, 1, true);
            let down_pos = pos.offset(Face::Down);
            let down_state = world.get_block(down_pos);
            if down_state != falling_id {
                updates.push((down_pos, falling_id));
                self.schedule_tick(current_tick + Self::tick_rate(kind), down_pos, kind);
            }
        }
    }

    fn spread_horizontal<R: FluidWorldReader>(
        &mut self,
        pos: BlockPos,
        kind: FluidKind,
        active_fluid: FluidState,
        current_tick: u64,
        world: &R,
        updates: &mut Vec<(BlockPos, BlockStateId)>,
    ) {
        let reg = world.registry();
        let next_level = if active_fluid.falling {
            1
        } else {
            active_fluid.level.saturating_add(1)
        };

        if next_level > Self::max_decay(kind) {
            return;
        }

        let (flow_dirs, flow_count) = self.find_flow_directions(pos, kind, world);
        for &face in &flow_dirs[..flow_count] {
            let target_pos = pos.offset(face);
            let target_state = world.get_block(target_pos);
            let target_fluid = reg.fluid_state(target_state);

            if let Some(tf) = target_fluid {
                if tf.kind != kind {
                    // Fluid reaction
                    if let Some((react_pos, reaction)) = self.check_reaction(pos, kind, world) {
                        let reaction_state = reaction_to_block_state(reg, reaction);
                        updates.push((react_pos, reaction_state));
                    }
                } else if !tf.falling && tf.level > next_level {
                    // Spreading fluid with stronger level replaces weaker decaying fluid
                    let new_fluid_id = reg.fluid_state_id(kind, next_level, false);
                    updates.push((target_pos, new_fluid_id));
                    self.schedule_tick(current_tick + Self::tick_rate(kind), target_pos, kind);
                }
            } else if target_state.is_air() || reg.is_replaceable(target_state) {
                let new_fluid_id = reg.fluid_state_id(kind, next_level, false);
                updates.push((target_pos, new_fluid_id));
                self.schedule_tick(current_tick + Self::tick_rate(kind), target_pos, kind);
            }
        }
    }

    /// Advances cellular automata fluid mechanics by one step at `current_tick`.
    ///
    /// Evaluates scheduled fluid ticks, computes falling columns and horizontal spreads,
    /// applies fluid reaction conversions (Obsidian, Cobblestone, Stone), and returns
    /// all block mutations `(BlockPos, BlockStateId)`.
    pub fn tick<R: FluidWorldReader>(
        &mut self,
        current_tick: u64,
        world: &R,
    ) -> Vec<(BlockPos, BlockStateId)> {
        let reg = world.registry();
        let mut updates = Vec::new();
        let mut ready = Vec::new();

        // 1. Pop all scheduled ticks ready at or before current_tick
        while let Some(Reverse(entry)) = self.scheduled_ticks.peek() {
            if entry.tick <= current_tick {
                let Reverse(entry) = self.scheduled_ticks.pop().unwrap();
                self.scheduled_set.remove(&entry.pos);
                ready.push(entry);
            } else {
                break;
            }
        }

        // 2. Process each fluid update
        for entry in ready {
            let pos = entry.pos;
            let kind = entry.kind;
            let cur_state = world.get_block(pos);
            let cur_fluid = reg.fluid_state(cur_state);

            // If the position is no longer fluid, skip
            let Some(fluid) = cur_fluid else {
                continue;
            };

            // 2.1 Check fluid reaction matrix (water meeting lava)
            if let Some((target_pos, reaction)) = self.check_reaction(pos, kind, world) {
                let reaction_state = reaction_to_block_state(reg, reaction);
                updates.push((target_pos, reaction_state));
                for &face in &Face::ALL {
                    let n_pos = target_pos.offset(face);
                    let n_state = world.get_block(n_pos);
                    if let Some(nf) = reg.fluid_state(n_state) {
                        self.schedule_tick(current_tick + Self::tick_rate(nf.kind), n_pos, nf.kind);
                    }
                }
                continue;
            }

            // 2.2 Determine target level for this block (if not a permanent source block)
            let mut target_state = self.compute_target_state(pos, kind, fluid, world);

            // If target state differs from current state, mutate and notify
            let mut state_changed = false;
            if let Some(target) = target_state {
                if target != fluid {
                    let new_id = reg.fluid_state_id(target.kind, target.level, target.falling);
                    updates.push((pos, new_id));
                    state_changed = true;
                }
            } else {
                // Disappear to air
                updates.push((pos, BlockStateId::AIR));
                state_changed = true;
                target_state = None;
            }

            if state_changed {
                for &face in &Face::ALL {
                    let n_pos = pos.offset(face);
                    let n_state = world.get_block(n_pos);
                    if let Some(nf) = reg.fluid_state(n_state) {
                        self.schedule_tick(current_tick + Self::tick_rate(nf.kind), n_pos, nf.kind);
                    }
                }
            }

            // 2.3 Spread fluid downwards and horizontally if it still exists
            if let Some(active_fluid) = target_state {
                let down_pos = pos.offset(Face::Down);
                let down_state = world.get_block(down_pos);
                let down_fluid = reg.fluid_state(down_state);

                let can_flow_down = down_state.is_air()
                    || reg.is_replaceable(down_state)
                    || down_fluid.is_some_and(|f| f.kind == kind && (!f.falling || f.level > 1));

                if can_flow_down {
                    self.spread_downward(pos, kind, down_fluid, current_tick, world, &mut updates);
                } else {
                    self.spread_horizontal(
                        pos,
                        kind,
                        active_fluid,
                        current_tick,
                        world,
                        &mut updates,
                    );
                }
            }
        }

        updates
    }
}

fn reaction_to_block_state(reg: &BlockRegistry, reaction: FluidReaction) -> BlockStateId {
    match reaction {
        FluidReaction::Obsidian => {
            let id = telos_core::ident::Identifier::new("telos", "obsidian").unwrap();
            reg.get(&id)
                .map_or(BlockStateId::new(36), telos_voxel::Block::default_state)
        }
        FluidReaction::Cobblestone => {
            let id = telos_core::ident::Identifier::new("telos", "cobblestone").unwrap();
            reg.get(&id)
                .map_or(BlockStateId::new(35), telos_voxel::Block::default_state)
        }
        FluidReaction::Stone => BlockStateId::new(1),
    }
}
