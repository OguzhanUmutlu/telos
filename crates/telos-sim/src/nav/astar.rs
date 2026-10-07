//! High-performance local 3D A* search on voxel walkability graphs.

use hashbrown::{HashMap, HashSet};
use std::collections::BinaryHeap;
use telos_core::coords::BlockPos;

use super::path::NavPath;
use super::profile::PathProfile;
use super::reader::NavWorldReader;
use super::walkability::{is_passable, is_walkable_node};

/// Converts an f32 into an unsigned 32-bit integer preserving lexicographical ordering.
#[inline]
fn f32_to_sortable_u32(val: f32) -> u32 {
    let bits = val.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct OpenNode {
    f_score_bits: u32,
    pos: BlockPos,
}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap ordering: lowest f_score has highest priority
        other.f_score_bits.cmp(&self.f_score_bits)
    }
}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 3D octile distance heuristic with vertical travel bias.
#[inline]
#[must_use]
pub fn octile_heuristic(a: BlockPos, b: BlockPos) -> f32 {
    let dx = (a.x() - b.x()).abs() as f32;
    let dy = (a.y() - b.y()).abs() as f32;
    let dz = (a.z() - b.z()).abs() as f32;
    let (min_h, max_h) = if dx < dz { (dx, dz) } else { (dz, dx) };
    max_h + (std::f32::consts::SQRT_2 - 1.0) * min_h + dy * 1.2
}

/// High-performance 3D A* pathfinder for local voxel navigation.
#[derive(Debug, Clone)]
pub struct LocalAStar {
    max_expanded_nodes: usize,
}

impl Default for LocalAStar {
    fn default() -> Self {
        Self {
            max_expanded_nodes: 1000,
        }
    }
}

impl LocalAStar {
    /// Creates a new `LocalAStar` pathfinder with a configured expansion limit.
    #[must_use]
    pub const fn new(max_expanded_nodes: usize) -> Self {
        Self { max_expanded_nodes }
    }

    /// Searches for a 3D walkability path from `start` to `goal`.
    ///
    /// Returns `Some(NavPath)` if a complete or partial path was found, or `None` if completely unreachable.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn find_path(
        &self,
        start: BlockPos,
        goal: BlockPos,
        target_reach: f32,
        profile: &PathProfile,
        world: &impl NavWorldReader,
    ) -> Option<NavPath> {
        let registry = world.registry();

        // Validate or resolve start position to nearest supporting node
        let initial_node = if is_walkable_node(start, profile, world) {
            start
        } else if is_walkable_node(start.down(1), profile, world) {
            start.down(1)
        } else if is_walkable_node(start.up(1), profile, world) {
            start.up(1)
        } else {
            start
        };

        let reach_sq = target_reach * target_reach;

        // Check immediate arrival
        let dx = (initial_node.x() - goal.x()) as f32;
        let dy = (initial_node.y() - goal.y()) as f32;
        let dz = (initial_node.z() - goal.z()) as f32;
        if dx * dx + dy * dy + dz * dz <= reach_sq {
            return Some(NavPath::new(vec![initial_node], goal));
        }

        let mut open_set = BinaryHeap::with_capacity(128);
        let mut visited: HashMap<BlockPos, (f32, Option<BlockPos>)> = HashMap::with_capacity(256);
        let mut closed: HashSet<BlockPos> = HashSet::with_capacity(256);

        let initial_h = octile_heuristic(initial_node, goal);
        open_set.push(OpenNode {
            f_score_bits: f32_to_sortable_u32(initial_h),
            pos: initial_node,
        });
        visited.insert(initial_node, (0.0, None));

        let mut best_node = initial_node;
        let mut best_h = initial_h;
        let mut expansions = 0;

        let cardinals = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut neighbors: Vec<(BlockPos, f32)> = Vec::with_capacity(16);

        while let Some(current) = open_set.pop() {
            let u = current.pos;
            if !closed.insert(u) {
                continue;
            }

            let Some(&(current_g, _)) = visited.get(&u) else {
                continue;
            };

            // Check goal condition
            let d_goal_sq = {
                let gx = (u.x() - goal.x()) as f32;
                let gy = (u.y() - goal.y()) as f32;
                let gz = (u.z() - goal.z()) as f32;
                gx * gx + gy * gy + gz * gz
            };

            if d_goal_sq <= reach_sq {
                best_node = u;
                break;
            }

            expansions += 1;
            if expansions >= self.max_expanded_nodes {
                break;
            }

            neighbors.clear();

            // 1. Cardinal transitions
            for &(cdx, cdz) in &cardinals {
                Self::evaluate_directional_transitions(
                    u,
                    cdx,
                    cdz,
                    1.0,
                    profile,
                    world,
                    &mut neighbors,
                );
            }

            // 2. Diagonal transitions (strict corner-cutting avoidance)
            for &(ddx, ddz) in &diagonals {
                // Both adjacent cardinals must have passable foot & head clearance
                let card1 = BlockPos::new(u.x() + ddx, u.y(), u.z());
                let card2 = BlockPos::new(u.x(), u.y(), u.z() + ddz);

                let mut diagonal_clear = true;
                for h in 0..profile.height {
                    let c1 = card1.up(h.cast_signed());
                    let c2 = card2.up(h.cast_signed());
                    if !is_passable(world.get_block(c1), profile, registry)
                        || !is_passable(world.get_block(c2), profile, registry)
                    {
                        diagonal_clear = false;
                        break;
                    }
                }

                if diagonal_clear {
                    Self::evaluate_directional_transitions(
                        u,
                        ddx,
                        ddz,
                        std::f32::consts::SQRT_2,
                        profile,
                        world,
                        &mut neighbors,
                    );
                }
            }

            // Process evaluated neighbor transitions
            for &(v, edge_cost) in &neighbors {
                if closed.contains(&v) {
                    continue;
                }

                let tentative_g = current_g + edge_cost;

                let is_better = match visited.get(&v) {
                    Some(&(existing_g, _)) => tentative_g < existing_g,
                    None => true,
                };

                if is_better {
                    visited.insert(v, (tentative_g, Some(u)));
                    let h = octile_heuristic(v, goal);
                    if h < best_h {
                        best_h = h;
                        best_node = v;
                    }
                    let f = tentative_g + h;
                    open_set.push(OpenNode {
                        f_score_bits: f32_to_sortable_u32(f),
                        pos: v,
                    });
                }
            }
        }

        // Reconstruct path from best_node to initial_node
        if best_node == initial_node {
            return None;
        }

        let mut waypoints = Vec::new();
        let mut curr = Some(best_node);
        while let Some(node) = curr {
            waypoints.push(node);
            curr = visited.get(&node).and_then(|&(_, parent)| parent);
            if curr == Some(initial_node) {
                break;
            }
        }

        waypoints.reverse();
        Some(NavPath::new(waypoints, goal))
    }

    /// Evaluates flat, jump-up, and drop-down transitions along a horizontal direction `(dx, dz)`.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_directional_transitions(
        u: BlockPos,
        dx: i32,
        dz: i32,
        base_cost: f32,
        profile: &PathProfile,
        world: &impl NavWorldReader,
        out: &mut Vec<(BlockPos, f32)>,
    ) {
        let registry = world.registry();

        // A. Flat walk (dy = 0)
        let flat_v = BlockPos::new(u.x() + dx, u.y(), u.z() + dz);
        if is_walkable_node(flat_v, profile, world) {
            out.push((flat_v, base_cost));
        }

        // B. Jump-up 1 block (dy = +1)
        if profile.max_step_up >= 1 {
            // Must have jump headroom clearance above current position
            let jump_headroom = u.up(profile.height.cast_signed());
            if is_passable(world.get_block(jump_headroom), profile, registry) {
                let jump_v = BlockPos::new(u.x() + dx, u.y() + 1, u.z() + dz);
                if is_walkable_node(jump_v, profile, world) {
                    out.push((jump_v, base_cost + 0.5));
                }
            }
        }

        // C. Safe drop-down (dy = -1..=-max_drop_down)
        // Ensure entity can step horizontally over the edge
        let step_cell = BlockPos::new(u.x() + dx, u.y(), u.z() + dz);
        let mut column_clear = true;
        for h in 0..profile.height {
            if !is_passable(
                world.get_block(step_cell.up(h.cast_signed())),
                profile,
                registry,
            ) {
                column_clear = false;
                break;
            }
        }

        if column_clear {
            for drop in 1..=profile.max_drop_down {
                let drop_v = BlockPos::new(u.x() + dx, u.y() - drop, u.z() + dz);
                // Verify drop space is passable
                if !is_passable(world.get_block(drop_v), profile, registry) {
                    break; // Blocked by solid obstacle in the fall column
                }

                if is_walkable_node(drop_v, profile, world) {
                    out.push((drop_v, base_cost + 0.2 * (drop as f32)));
                    break; // Landed on first solid ground surface
                }
            }
        }
    }
}

/// Evaluates and assigns 3D A* navigation paths for all active mobs pursuing players or wandering.
#[allow(clippy::cast_possible_truncation)]
pub fn update_mob_navigation_paths<W: NavWorldReader>(
    world: &W,
    ecs_world: &mut bevy_ecs::world::World,
) {
    let player_list = ecs_world
        .get_resource::<crate::entity::PlayerPositions>()
        .map(|p| p.0.clone())
        .unwrap_or_default();

    let astar = LocalAStar::default();

    let mut query = ecs_world.query_filtered::<(
        &crate::entity::Mob,
        &mut crate::entity::PathFollower,
        &crate::entity::Position,
    ), bevy_ecs::query::Without<crate::entity::SimulationFrozen>>();

    for (mob, mut follower, pos) in query.iter_mut(ecs_world) {
        if follower.repath_cooldown > 0 {
            follower.repath_cooldown -= 1;
        }

        match mob.kind {
            crate::entity::MobKind::Hostile => {
                // Find nearest targetable player within 24 blocks
                let mut target_player = None;
                let mut min_dsq = 24.0 * 24.0;

                for player in &player_list {
                    if !player.targetable {
                        continue;
                    }
                    let dsq = pos.0.distance_squared(player.pos);
                    if dsq < min_dsq {
                        min_dsq = dsq;
                        target_player = Some((player.net_id, player.pos));
                    }
                }

                if let Some((target_id, target_pos)) = target_player {
                    follower.target_entity = Some(target_id);

                    let target_block = BlockPos::new(
                        target_pos.x.floor() as i32,
                        target_pos.y.floor() as i32,
                        target_pos.z.floor() as i32,
                    );

                    let need_repath = follower.path.is_none()
                        || follower.repath_cooldown == 0
                        || follower.last_target_pos.is_none_or(|p| {
                            let dx = (p.x() - target_block.x()).abs();
                            let dy = (p.y() - target_block.y()).abs();
                            let dz = (p.z() - target_block.z()).abs();
                            dx + dy + dz > 3
                        });

                    if need_repath {
                        let mob_block = BlockPos::new(
                            pos.0.x.floor() as i32,
                            pos.0.y.floor() as i32,
                            pos.0.z.floor() as i32,
                        );

                        let profile = PathProfile::humanoid();
                        let path = astar.find_path(mob_block, target_block, 1.5, &profile, world);
                        follower.path = path;
                        follower.last_target_pos = Some(target_block);
                        follower.repath_cooldown = 15;
                    }
                } else {
                    follower.target_entity = None;
                }
            }
            crate::entity::MobKind::Passive => {
                if follower.repath_cooldown == 0
                    && follower.path.is_none()
                    && let crate::entity::AiState::Wandering { target, .. } = mob.ai_state
                {
                    let mob_block = BlockPos::new(
                        pos.0.x.floor() as i32,
                        pos.0.y.floor() as i32,
                        pos.0.z.floor() as i32,
                    );
                    let target_block = BlockPos::new(
                        target.x.floor() as i32,
                        target.y.floor() as i32,
                        target.z.floor() as i32,
                    );
                    let profile = PathProfile::small_animal();
                    let path = astar.find_path(mob_block, target_block, 1.0, &profile, world);
                    follower.path = path;
                    follower.last_target_pos = Some(target_block);
                    follower.repath_cooldown = 20;
                }
            }
        }
    }
}
