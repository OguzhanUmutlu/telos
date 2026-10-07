//! Voxel walkability evaluation, standing node classification, and hazard detection.

use telos_core::coords::BlockPos;
use telos_voxel::fluid::FluidKind;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::shape::BlockShape;
use telos_voxel::state::{BlockStateId, StateFlags};

use super::profile::PathProfile;
use super::reader::NavWorldReader;

/// Returns `true` if the block state represents an environmental hazard (e.g. lava, fire).
#[must_use]
pub fn is_hazard(id: BlockStateId, registry: &BlockRegistry) -> bool {
    if id.is_air() {
        return false;
    }
    if let Some(fluid) = registry.fluid_state(id)
        && fluid.kind == FluidKind::Lava
    {
        return true;
    }
    if let Some(ident) = registry.identifier(id)
        && ident.path() == "fire"
    {
        return true;
    }
    false
}

/// Returns `true` if the block state provides solid, load-bearing ground support.
#[must_use]
pub fn is_solid_ground(id: BlockStateId, registry: &BlockRegistry) -> bool {
    if id.is_air() {
        return false;
    }
    let flags = registry.flags(id);
    if flags.contains(StateFlags::FLUID) {
        return false;
    }
    let shape = registry.shape(id);
    match shape {
        BlockShape::Cube => flags.contains(StateFlags::NON_EMPTY),
        BlockShape::Boxes(boxes) => boxes.iter().any(|b| b.max[1] >= 8),
        _ => false,
    }
}

/// Returns `true` if an entity can enter or stand inside this block without collision blocking.
#[must_use]
pub fn is_passable(id: BlockStateId, profile: &PathProfile, registry: &BlockRegistry) -> bool {
    if id.is_air() {
        return true;
    }
    if profile.avoid_hazards && is_hazard(id, registry) {
        return false;
    }
    if let Some(fluid) = registry.fluid_state(id) {
        return profile.can_swim && fluid.kind == FluidKind::Water;
    }
    registry.is_replaceable(id)
}

/// Evaluates whether an entity with `profile` can stand at `pos` (where `pos` is foot level).
///
/// Requires:
/// 1. Supporting floor at `pos.down(1)` (solid ground, or water if `can_swim`).
/// 2. Headroom clearance for all `profile.height` blocks above feet.
/// 3. Zero hazardous blocks intersecting feet or headroom.
#[must_use]
pub fn is_walkable_node(pos: BlockPos, profile: &PathProfile, world: &impl NavWorldReader) -> bool {
    let registry = world.registry();

    // 1. Supporting ground check
    let ground_pos = pos.down(1);
    let ground_id = world.get_block(ground_pos);

    if profile.avoid_hazards && is_hazard(ground_id, registry) {
        return false;
    }

    let ground_is_water = profile.can_swim
        && registry
            .fluid_state(ground_id)
            .is_some_and(|f| f.kind == FluidKind::Water);

    if !ground_is_water && !is_solid_ground(ground_id, registry) {
        return false;
    }

    // 2. Headroom & clearance check
    for h in 0..profile.height {
        let cell = pos.up(h.cast_signed());
        let cell_id = world.get_block(cell);
        if !is_passable(cell_id, profile, registry) {
            return false;
        }
    }

    true
}
