//! Top-down column surface rules and strata material assignment.

use crate::biome::BiomeId;
use vx_core::coords::{CHUNK_VOLUME, ChunkPos};
use vx_voxel::registry::BlockRegistry;
use vx_voxel::state::BlockStateId;

/// Resolved block state IDs needed by surface rules.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedBlocks {
    /// Air block state ID.
    pub air: BlockStateId,
    /// Stone block state ID.
    pub stone: BlockStateId,
    /// Dirt block state ID.
    pub dirt: BlockStateId,
    /// Grass block state ID.
    pub grass: BlockStateId,
    /// Bedrock block state ID.
    pub bedrock: BlockStateId,
    /// Sand block state ID.
    pub sand: BlockStateId,
    /// Water block state ID.
    pub water: BlockStateId,
    /// Poppy flower block state ID.
    pub poppy: BlockStateId,
    /// Dandelion flower block state ID.
    pub dandelion: BlockStateId,
    /// Short grass foliage block state ID.
    pub short_grass: BlockStateId,
    /// Fern foliage block state ID.
    pub fern: BlockStateId,
    /// Dead bush arid block state ID.
    pub dead_bush: BlockStateId,
}

impl ResolvedBlocks {
    /// Resolves required block state IDs from the registry.
    #[must_use]
    pub fn resolve(registry: &BlockRegistry) -> Self {
        let air = BlockStateId::AIR;
        let stone = registry
            .get(&vx_core::ident::Identifier::new("voxel", "stone").unwrap())
            .map_or(BlockStateId::new(1), vx_voxel::Block::default_state);
        let dirt = registry
            .get(&vx_core::ident::Identifier::new("voxel", "dirt").unwrap())
            .map_or(BlockStateId::new(2), vx_voxel::Block::default_state);
        let grass = registry
            .get(&vx_core::ident::Identifier::new("voxel", "grass_block").unwrap())
            .map_or(BlockStateId::new(3), vx_voxel::Block::default_state);
        let bedrock = registry
            .get(&vx_core::ident::Identifier::new("voxel", "bedrock").unwrap())
            .map_or(BlockStateId::new(4), vx_voxel::Block::default_state);
        let sand = registry
            .get(&vx_core::ident::Identifier::new("voxel", "sand").unwrap())
            .map_or(BlockStateId::new(5), vx_voxel::Block::default_state);
        let water = registry
            .get(&vx_core::ident::Identifier::new("voxel", "water").unwrap())
            .map_or(BlockStateId::new(6), vx_voxel::Block::default_state);
        let poppy = registry
            .get(&vx_core::ident::Identifier::new("voxel", "poppy").unwrap())
            .map_or(BlockStateId::new(12), vx_voxel::Block::default_state);
        let dandelion = registry
            .get(&vx_core::ident::Identifier::new("voxel", "dandelion").unwrap())
            .map_or(BlockStateId::new(13), vx_voxel::Block::default_state);
        let short_grass = registry
            .get(&vx_core::ident::Identifier::new("voxel", "short_grass").unwrap())
            .map_or(BlockStateId::new(16), vx_voxel::Block::default_state);
        let fern = registry
            .get(&vx_core::ident::Identifier::new("voxel", "fern").unwrap())
            .map_or(BlockStateId::new(17), vx_voxel::Block::default_state);
        let dead_bush = registry
            .get(&vx_core::ident::Identifier::new("voxel", "dead_bush").unwrap())
            .map_or(BlockStateId::new(18), vx_voxel::Block::default_state);

        Self {
            air,
            stone,
            dirt,
            grass,
            bedrock,
            sand,
            water,
            poppy,
            dandelion,
            short_grass,
            fern,
            dead_bush,
        }
    }
}

/// Applies surface rules to populate a dense 32³ block array from solid occupancy.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
pub fn apply_surface_rules(
    _seed: u64,
    chunk_pos: ChunkPos,
    occupancy: &[bool; CHUNK_VOLUME],
    biomes: &[BiomeId; 64],
    blocks: &ResolvedBlocks,
    out_dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let origin_y = chunk_pos.y() * 32;

    for z in 0usize..32 {
        let cz = z >> 2;
        for x in 0usize..32 {
            let cx = x >> 2;
            let biome = biomes[cz * 8 + cx];

            let mut depth_from_surface = 0usize;
            let mut is_under_solid = false;

            // Scan column top to bottom
            for y in (0usize..32).rev() {
                let wy = origin_y + y as i32;
                let idx = (y << 10) | (z << 5) | x;

                if occupancy[idx] {
                    // Bedrock at the bottom of the world
                    if wy <= -1020 {
                        out_dense[idx] = blocks.bedrock;
                        depth_from_surface += 1;
                        is_under_solid = true;
                        continue;
                    }

                    if !is_under_solid {
                        // Top surface voxel
                        is_under_solid = true;
                        depth_from_surface = 0;

                        match biome {
                            BiomeId::Desert | BiomeId::Ocean => {
                                out_dense[idx] = blocks.sand;
                            }
                            BiomeId::Mountains => {
                                if wy >= 90 {
                                    out_dense[idx] = blocks.stone;
                                } else {
                                    out_dense[idx] = blocks.grass;
                                }
                            }
                            BiomeId::Plains | BiomeId::Forest => {
                                out_dense[idx] = blocks.grass;
                            }
                        }
                    } else if depth_from_surface < 4 {
                        // Subsurface layers (depth 1..3)
                        match biome {
                            BiomeId::Desert => {
                                out_dense[idx] = blocks.sand;
                            }
                            _ => {
                                out_dense[idx] = blocks.dirt;
                            }
                        }
                    } else {
                        // Deep rock
                        out_dense[idx] = blocks.stone;
                    }

                    depth_from_surface += 1;
                } else {
                    // Air or water
                    is_under_solid = false;
                    depth_from_surface = 0;

                    if wy <= 0 {
                        // Below or at sea level in empty cavity
                        out_dense[idx] = blocks.water;
                    } else {
                        out_dense[idx] = blocks.air;
                    }
                }
            }
        }
    }
}
