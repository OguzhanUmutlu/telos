//! Amanatides & Woo (1987) fast voxel traversal algorithm (DDA raycast).

use glam::Vec3;

use crate::coords::{BlockPos, Face};

/// Result of a successful voxel raycast traversal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastHit {
    /// The block coordinate that was struck by the ray.
    pub pos: BlockPos,
    /// The face through which the ray entered the block.
    pub face: Face,
    /// The exact world-space intersection point.
    pub hit_point: Vec3,
    /// Distance from ray origin to the intersection point.
    pub distance: f32,
}

/// Casts a ray through a voxel grid using the Amanatides-Woo DDA algorithm.
///
/// Traverses discrete voxel cells along the ray `origin + t * direction` up to `max_distance`.
/// For each cell traversed, `is_hit` is called with the cell's `BlockPos`.
/// Traversal halts at the first cell where `is_hit` returns `true`.
///
/// # Returns
/// `Some(RaycastHit)` containing the hit cell, entry face, hit point, and distance, or `None`
/// if no cell was hit within `max_distance`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn raycast_voxels<F>(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    mut is_hit: F,
) -> Option<RaycastHit>
where
    F: FnMut(BlockPos) -> bool,
{
    if max_distance <= 0.0 || direction.length_squared() < 1e-6 {
        return None;
    }

    let dir = direction.normalize();

    // 1. Initial cell containing the origin
    let mut current_x = origin.x.floor() as i32;
    let mut current_y = origin.y.floor() as i32;
    let mut current_z = origin.z.floor() as i32;

    let initial_pos = BlockPos::new(current_x, current_y, current_z);
    if is_hit(initial_pos) {
        // Ray started inside a solid block
        let face = if dir.y > 0.0 {
            Face::Down
        } else if dir.y < 0.0 {
            Face::Up
        } else if dir.x > 0.0 {
            Face::West
        } else {
            Face::East
        };
        return Some(RaycastHit {
            pos: initial_pos,
            face,
            hit_point: origin,
            distance: 0.0,
        });
    }

    // 2. Step directions along each axis (-1, 0, or +1)
    let step_x = if dir.x > 0.0 {
        1
    } else if dir.x < 0.0 {
        -1
    } else {
        0
    };
    let step_y = if dir.y > 0.0 {
        1
    } else if dir.y < 0.0 {
        -1
    } else {
        0
    };
    let step_z = if dir.z > 0.0 {
        1
    } else if dir.z < 0.0 {
        -1
    } else {
        0
    };

    // 3. Distance along ray between adjacent grid boundaries
    let t_delta_x = if step_x != 0 {
        (1.0 / dir.x).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_y = if step_y != 0 {
        (1.0 / dir.y).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_z = if step_z != 0 {
        (1.0 / dir.z).abs()
    } else {
        f32::INFINITY
    };

    // 4. Distance to first grid boundary along each axis
    let next_boundary_x = match step_x.cmp(&0) {
        std::cmp::Ordering::Greater => (current_x + 1) as f32,
        std::cmp::Ordering::Less => current_x as f32,
        std::cmp::Ordering::Equal => 0.0,
    };
    let next_boundary_y = match step_y.cmp(&0) {
        std::cmp::Ordering::Greater => (current_y + 1) as f32,
        std::cmp::Ordering::Less => current_y as f32,
        std::cmp::Ordering::Equal => 0.0,
    };
    let next_boundary_z = match step_z.cmp(&0) {
        std::cmp::Ordering::Greater => (current_z + 1) as f32,
        std::cmp::Ordering::Less => current_z as f32,
        std::cmp::Ordering::Equal => 0.0,
    };

    let mut t_max_x = if step_x != 0 {
        (next_boundary_x - origin.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if step_y != 0 {
        (next_boundary_y - origin.y) / dir.y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if step_z != 0 {
        (next_boundary_z - origin.z) / dir.z
    } else {
        f32::INFINITY
    };

    // Max traversal iterations to guarantee loop termination
    let max_steps = (max_distance * 3.0).ceil() as usize + 8;
    let mut last_face;
    let mut t_traversed;

    for _ in 0..max_steps {
        if t_max_x < t_max_y && t_max_x < t_max_z {
            if t_max_x > max_distance {
                return None;
            }
            current_x += step_x;
            t_traversed = t_max_x;
            t_max_x += t_delta_x;
            last_face = if step_x > 0 { Face::West } else { Face::East };
        } else if t_max_y < t_max_z {
            if t_max_y > max_distance {
                return None;
            }
            current_y += step_y;
            t_traversed = t_max_y;
            t_max_y += t_delta_y;
            last_face = if step_y > 0 { Face::Down } else { Face::Up };
        } else {
            if t_max_z > max_distance {
                return None;
            }
            current_z += step_z;
            t_traversed = t_max_z;
            t_max_z += t_delta_z;
            last_face = if step_z > 0 { Face::North } else { Face::South };
        }

        let pos = BlockPos::new(current_x, current_y, current_z);
        if is_hit(pos) {
            let hit_point = origin + dir * t_traversed;
            return Some(RaycastHit {
                pos,
                face: last_face,
                hit_point,
                distance: t_traversed,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raycast_direct_hit_x() {
        let origin = Vec3::new(0.5, 0.5, 0.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let hit = raycast_voxels(origin, direction, 10.0, |pos| pos == BlockPos::new(3, 0, 0));

        let hit = hit.expect("Should hit block at (3, 0, 0)");
        assert_eq!(hit.pos, BlockPos::new(3, 0, 0));
        assert_eq!(hit.face, Face::West);
        assert!((hit.distance - 2.5).abs() < 1e-4);
        assert!((hit.hit_point.x - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_raycast_direct_hit_negative_y() {
        let origin = Vec3::new(0.5, 5.5, 0.5);
        let direction = Vec3::new(0.0, -1.0, 0.0);
        let hit = raycast_voxels(origin, direction, 10.0, |pos| pos == BlockPos::new(0, 1, 0));

        let hit = hit.expect("Should hit block at (0, 1, 0)");
        assert_eq!(hit.pos, BlockPos::new(0, 1, 0));
        assert_eq!(hit.face, Face::Up);
        assert!((hit.distance - 3.5).abs() < 1e-4);
        assert!((hit.hit_point.y - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_raycast_diagonal_across_negative_origin() {
        let origin = Vec3::new(-2.5, 1.5, -2.5);
        let direction = Vec3::new(1.0, 0.0, 1.0).normalize();
        let hit = raycast_voxels(origin, direction, 10.0, |pos| pos == BlockPos::new(0, 1, 0));

        let hit = hit.expect("Should hit block at (0, 1, 0)");
        assert_eq!(hit.pos, BlockPos::new(0, 1, 0));
    }

    #[test]
    fn test_raycast_miss_out_of_reach() {
        let origin = Vec3::new(0.5, 0.5, 0.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let hit = raycast_voxels(origin, direction, 2.0, |pos| pos == BlockPos::new(5, 0, 0));
        assert!(hit.is_none());
    }

    #[test]
    fn test_raycast_inside_solid() {
        let origin = Vec3::new(1.2, 3.4, 5.6);
        let direction = Vec3::new(0.0, 1.0, 0.0);
        let hit = raycast_voxels(origin, direction, 5.0, |pos| pos == BlockPos::new(1, 3, 5));

        let hit = hit.expect("Should hit enclosing block");
        assert_eq!(hit.pos, BlockPos::new(1, 3, 5));
        assert!(hit.distance.abs() < f32::EPSILON);
    }
}
