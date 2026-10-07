//! Procedural structure generation engine.
//!
//! Generates multi-chunk subterranean dungeons and surface ruins with jigsaw placement,
//! bounding box collision indexing, loot tables, and mob spawners.

pub mod bounding_box;
pub mod dungeon;
pub mod jigsaw;
pub mod loot;
pub mod ruin;

pub use bounding_box::StructureBoundingBox;
pub use dungeon::apply_dungeons;
pub use jigsaw::{JigsawAssembler, JigsawDirection, JigsawJoint, PieceType, StructurePiece};
pub use loot::{LootItem, LootTableKind, roll_chest_slots, roll_loot};
pub use ruin::apply_ruins;

use crate::aquifer::AquiferSampler;
use crate::surface::ResolvedBlocks;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::state::BlockStateId;

/// Evaluates and applies all procedural structures (subterranean dungeons and surface ruins)
/// intersecting the chunk at `pos`.
pub fn apply_structures(
    seed: u64,
    pos: ChunkPos,
    blocks: &ResolvedBlocks,
    aquifer: &AquiferSampler,
    dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    // 1. Subterranean dungeons (monster chambers, mossy cobblestone, spawners, chests)
    apply_dungeons(seed, pos, blocks, aquifer, dense);

    // 2. Surface ruins (outposts, shrines, cellars, chests)
    apply_ruins(seed, pos, blocks, dense);
}
