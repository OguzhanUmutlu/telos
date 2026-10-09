//! Fluid types, fluid state representation, and level classification.

/// Classification of fluid type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FluidKind {
    /// Water fluid.
    Water,
    /// Lava fluid (viscous, emissive).
    Lava,
}

/// Evaluated fluid properties for a block state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidState {
    /// Fluid kind (water or lava).
    pub kind: FluidKind,
    /// Fluid decay level: 0 = full source, 1..=7 = flowing.
    pub level: u8,
    /// Whether the fluid block represents a vertical falling column.
    pub falling: bool,
}

impl FluidState {
    /// Creates a new `FluidState`.
    #[inline]
    #[must_use]
    pub const fn new(kind: FluidKind, level: u8, falling: bool) -> Self {
        Self {
            kind,
            level,
            falling,
        }
    }

    /// Returns `true` if this fluid state is a source block.
    #[inline]
    #[must_use]
    pub const fn is_source(self) -> bool {
        self.level == 0 && !self.falling
    }
}

use glam::Vec3;
use telos_core::coords::BlockPos;

/// Evaluates the 3D fluid flow velocity vector for a fluid block at `pos`.
///
/// If `pos` is falling fluid, returns downward flow `(0.0, -1.0, 0.0)`.
/// Otherwise, evaluates the horizontal gradient from the 4 cardinal neighbors (North, South, West, East)
/// 4 Cardinal horizontal directions: (dx, dz)
/// +X (East), -X (West), +Z (South), -Z (North)
const CARDINAL_OFFSETS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Evaluates the 3D fluid flow velocity vector for a fluid block at `pos`.
///
/// If `pos` is falling fluid, returns downward flow `(0.0, -1.0, 0.0)`.
/// Otherwise, evaluates the horizontal gradient from the 4 cardinal neighbors (North, South, West, East)
/// and potential drops (unobstructed drops to air/water below neighboring blocks).
///
/// Returns a normalized 3D direction vector scaled by flow intensity, or `Vec3::ZERO` if stagnant or not fluid.
#[allow(clippy::cast_precision_loss)]
pub fn calculate_fluid_flow<F>(pos: BlockPos, mut get_fluid: F) -> Vec3
where
    F: FnMut(BlockPos) -> Option<FluidState>,
{
    let Some(current) = get_fluid(pos) else {
        return Vec3::ZERO;
    };

    if current.falling {
        return Vec3::new(0.0, -1.0, 0.0);
    }

    let cur_eff_level = if current.is_source() {
        0i32
    } else {
        i32::from(current.level)
    };
    let mut flow_x = 0.0f32;
    let mut flow_z = 0.0f32;

    for (dx, dz) in CARDINAL_OFFSETS {
        let neighbor_pos = BlockPos::new(pos.x() + dx, pos.y(), pos.z() + dz);
        let neighbor_fluid = get_fluid(neighbor_pos);

        if let Some(n_state) = neighbor_fluid {
            if n_state.kind == current.kind {
                if n_state.falling {
                    // Falling neighbor acts like a powerful source block
                    flow_x += (dx as f32) * -1.5;
                    flow_z += (dz as f32) * -1.5;
                } else {
                    let n_eff_level = if n_state.is_source() {
                        0i32
                    } else {
                        i32::from(n_state.level)
                    };
                    // Lower level number means higher fluid height (level 0 is source).
                    // Flow moves towards higher level number (lower height):
                    let diff = n_eff_level - cur_eff_level;
                    flow_x += (dx as f32) * (diff as f32);
                    flow_z += (dz as f32) * (diff as f32);
                }
            }
        } else {
            // Neighbor is not fluid. Check if below neighbor is an open cliff/drop
            let below_neighbor = BlockPos::new(pos.x() + dx, pos.y() - 1, pos.z() + dz);
            if let Some(below_fluid) = get_fluid(below_neighbor)
                && below_fluid.kind == current.kind
            {
                // Water flows aggressively towards the drop cliff
                flow_x += (dx as f32) * 3.0;
                flow_z += (dz as f32) * 3.0;
            }
        }
    }

    let horiz_len_sq = flow_x * flow_x + flow_z * flow_z;
    if horiz_len_sq > 1e-4 {
        let inv_len = 1.0 / horiz_len_sq.sqrt();
        Vec3::new(flow_x * inv_len, 0.0, flow_z * inv_len)
    } else {
        Vec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stagnant_source_flow() {
        let pos = BlockPos::new(10, 64, 10);
        // All neighbors are identical source blocks
        let flow = calculate_fluid_flow(pos, |_| Some(FluidState::new(FluidKind::Water, 0, false)));
        assert_eq!(flow, Vec3::ZERO);
    }

    #[test]
    fn test_falling_water_flow() {
        let pos = BlockPos::new(10, 64, 10);
        let flow = calculate_fluid_flow(pos, |_| Some(FluidState::new(FluidKind::Water, 1, true)));
        assert_eq!(flow, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn test_directional_water_flow() {
        let pos = BlockPos::new(10, 64, 10);
        // East neighbor (+X) is decaying water (level 2), West is source (level 0), North/South identical
        let flow = calculate_fluid_flow(pos, |p| {
            if p == pos {
                Some(FluidState::new(FluidKind::Water, 1, false))
            } else if p == BlockPos::new(11, 64, 10) {
                Some(FluidState::new(FluidKind::Water, 2, false))
            } else if p == BlockPos::new(9, 64, 10) {
                Some(FluidState::new(FluidKind::Water, 0, false))
            } else {
                Some(FluidState::new(FluidKind::Water, 1, false))
            }
        });
        assert!(flow.x > 0.0, "Flow should be directed towards East (+X)");
        assert!((flow.y).abs() < 1e-4);
        assert!((flow.z).abs() < 1e-4);
    }

    #[test]
    fn test_waterfall_drop_cliff_flow() {
        let pos = BlockPos::new(10, 64, 10);
        // North neighbor (Z - 1) has a drop to water below (Z - 1, Y - 1)
        let flow = calculate_fluid_flow(pos, |p| {
            if p == pos {
                Some(FluidState::new(FluidKind::Water, 0, false))
            } else if p == BlockPos::new(10, 63, 9) {
                Some(FluidState::new(FluidKind::Water, 1, true))
            } else {
                None
            }
        });
        assert!(
            flow.z < 0.0,
            "Flow should be drawn toward the waterfall drop (Z - 1)"
        );
    }
}
