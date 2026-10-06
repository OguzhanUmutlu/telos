//! Chunk delivery priority scoring and distance/view-cone calculations.

use glam::Vec3;
use telos_core::coords::ChunkPos;

/// Computes chunk transmission priority based on Euclidean distance and view cone.
///
/// Implements `networking/SKILL.md` §9:
/// ```text
/// score = dist_chunks * (1.6 - 0.6 * max(0, cos_angle)) + 4 * max(0, dy_chunks - 1)
/// priority = 1000 - floor(score * 10)
/// ```
#[must_use]
pub fn compute_chunk_priority(player_chunk: ChunkPos, look_dir: Vec3, target: ChunkPos) -> i32 {
    let dx = (target.x() - player_chunk.x()) as f32;
    let dy = (target.y() - player_chunk.y()) as f32;
    let dz = (target.z() - player_chunk.z()) as f32;

    let dist_chunks = (dx * dx + dy * dy + dz * dz).sqrt();
    let to_chunk_dir = if dist_chunks > 0.001 {
        Vec3::new(dx, dy, dz) / dist_chunks
    } else {
        look_dir
    };

    let cos_angle = look_dir.dot(to_chunk_dir).max(0.0);
    let dy_penalty = 4.0 * (dy.abs() - 1.0).max(0.0);

    let score = dist_chunks * (1.6 - 0.6 * cos_angle) + dy_penalty;
    1000 - (score * 10.0) as i32
}

/// A chunk queued for transmission, ordered in a max-heap by priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuedChunk {
    /// Coordinates of the queued chunk.
    pub pos: ChunkPos,
    /// Computed streaming priority (higher is sent sooner).
    pub priority: i32,
}

impl Ord for QueuedChunk {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for QueuedChunk {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_priority_ordering() {
        let player = ChunkPos::new(0, 0, 0);
        let forward = Vec3::new(0.0, 0.0, 1.0); // Facing +Z

        // Chunk right in front of player
        let p_front = compute_chunk_priority(player, forward, ChunkPos::new(0, 0, 2));
        // Chunk right behind player
        let p_behind = compute_chunk_priority(player, forward, ChunkPos::new(0, 0, -2));

        assert!(
            p_front > p_behind,
            "Chunk in view cone must have higher priority: front={p_front} behind={p_behind}"
        );
    }
}
