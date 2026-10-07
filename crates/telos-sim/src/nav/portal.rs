//! Chunk-portal boundary graph and hierarchical macro-navigation across 32³ cubic chunks.

use hashbrown::{HashMap, HashSet};
use std::collections::BinaryHeap;
use telos_core::coords::{BlockPos, ChunkPos};

use super::profile::PathProfile;
use super::reader::NavWorldReader;
use super::walkability::is_walkable_node;

/// Boundary portals across the faces of a single 32³ cubic chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPortals {
    /// Chunk position in 32³ coordinates.
    pub chunk_pos: ChunkPos,
    /// Walkable boundary nodes that connect seamlessly to neighboring chunks.
    pub boundary_nodes: Vec<BlockPos>,
}

impl ChunkPortals {
    /// Extracts walkable boundary nodes on the horizontal borders of the chunk.
    #[must_use]
    pub fn extract(
        chunk_pos: ChunkPos,
        profile: &PathProfile,
        world: &impl NavWorldReader,
    ) -> Self {
        let base_x = chunk_pos.x() * 32;
        let base_y = chunk_pos.y() * 32;
        let base_z = chunk_pos.z() * 32;

        let mut boundary_nodes = Vec::new();

        // Sample horizontal boundary seams at regular intervals
        for local_coord in (0..32).step_by(2) {
            for local_y in 0..32 {
                let y = base_y + local_y;

                // West face (x = 0) and East face (x = 31)
                let west = BlockPos::new(base_x, y, base_z + local_coord);
                if is_walkable_node(west, profile, world)
                    && is_walkable_node(west.west(1), profile, world)
                {
                    boundary_nodes.push(west);
                }

                let east = BlockPos::new(base_x + 31, y, base_z + local_coord);
                if is_walkable_node(east, profile, world)
                    && is_walkable_node(east.east(1), profile, world)
                {
                    boundary_nodes.push(east);
                }

                // North face (z = 0) and South face (z = 31)
                let north = BlockPos::new(base_x + local_coord, y, base_z);
                if is_walkable_node(north, profile, world)
                    && is_walkable_node(north.north(1), profile, world)
                {
                    boundary_nodes.push(north);
                }

                let south = BlockPos::new(base_x + local_coord, y, base_z + 31);
                if is_walkable_node(south, profile, world)
                    && is_walkable_node(south.south(1), profile, world)
                {
                    boundary_nodes.push(south);
                }
            }
        }

        Self {
            chunk_pos,
            boundary_nodes,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct ChunkNode {
    f_score: i32,
    chunk: ChunkPos,
}

impl Ord for ChunkNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f_score.cmp(&self.f_score)
    }
}

impl PartialOrd for ChunkNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Macro chunk corridor navigation solver.
#[derive(Debug, Default)]
pub struct ChunkPortalGraph;

impl ChunkPortalGraph {
    /// Finds a sequence of chunk positions forming a macro corridor from `start_chunk` to `goal_chunk`.
    #[must_use]
    pub fn find_chunk_corridor(start_chunk: ChunkPos, goal_chunk: ChunkPos) -> Vec<ChunkPos> {
        if start_chunk == goal_chunk {
            return vec![start_chunk];
        }

        let heuristic = |a: ChunkPos, b: ChunkPos| -> i32 {
            (a.x() - b.x()).abs() + (a.y() - b.y()).abs() + (a.z() - b.z()).abs()
        };

        let mut open_set = BinaryHeap::new();
        let mut visited: HashMap<ChunkPos, (i32, Option<ChunkPos>)> = HashMap::new();
        let mut closed_set = HashSet::new();

        open_set.push(ChunkNode {
            f_score: heuristic(start_chunk, goal_chunk),
            chunk: start_chunk,
        });
        visited.insert(start_chunk, (0, None));

        let directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        while let Some(current) = open_set.pop() {
            let u = current.chunk;
            if u == goal_chunk {
                break;
            }

            if !closed_set.insert(u) {
                continue;
            }

            let Some(&(current_g, _)) = visited.get(&u) else {
                continue;
            };

            for &(dx, dy, dz) in &directions {
                let neighbor = ChunkPos::new(u.x() + dx, u.y() + dy, u.z() + dz);
                let tentative_g = current_g + 1;

                let is_better = match visited.get(&neighbor) {
                    Some(&(existing_g, _)) => tentative_g < existing_g,
                    None => true,
                };

                if is_better {
                    visited.insert(neighbor, (tentative_g, Some(u)));
                    let f = tentative_g + heuristic(neighbor, goal_chunk);
                    open_set.push(ChunkNode {
                        f_score: f,
                        chunk: neighbor,
                    });
                }
            }
        }

        let mut corridor = Vec::new();
        let mut curr = Some(goal_chunk);
        while let Some(chunk) = curr {
            corridor.push(chunk);
            curr = visited.get(&chunk).and_then(|&(_, parent)| parent);
            if curr == Some(start_chunk) {
                corridor.push(start_chunk);
                break;
            }
        }

        corridor.reverse();
        if corridor.first() == Some(&start_chunk) {
            corridor
        } else {
            vec![start_chunk]
        }
    }
}
