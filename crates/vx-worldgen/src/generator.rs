//! Central terrain generator coordinator producing standard 32³ cubic chunks.

use crate::density::CoarseGrid;
use crate::surface::{ResolvedBlocks, apply_surface_rules};
use vx_core::coords::{CHUNK_VOLUME, ChunkPos};
use vx_voxel::chunk::Chunk;
use vx_voxel::registry::BlockRegistry;
use vx_voxel::state::BlockStateId;
use vx_voxel::storage::bulk;

/// Deterministic, server-side procedural world generator.
#[derive(Debug, Clone)]
pub struct WorldGenerator {
    seed: u64,
    blocks: ResolvedBlocks,
}

impl WorldGenerator {
    /// Creates a new `WorldGenerator` with the given 64-bit seed and block registry.
    #[must_use]
    pub fn new(seed: u64, registry: &BlockRegistry) -> Self {
        let blocks = ResolvedBlocks::resolve(registry);
        Self { seed, blocks }
    }

    /// World seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Generates a fully lit and paletted 32³ cubic chunk at `pos`.
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub fn generate_chunk(&self, pos: ChunkPos) -> Chunk {
        // 1. Evaluate 405 corner samples on 4x8x4 coarse grid
        let grid = CoarseGrid::evaluate(self.seed, pos);

        // 2. Exact sign skip and trilinear cell interpolation into 32³ occupancy
        let mut occupancy = [false; CHUNK_VOLUME];
        grid.fill_occupancy(&mut occupancy);

        // 3. Apply column surface rules and strata into dense array
        let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
        apply_surface_rules(
            self.seed,
            pos,
            &occupancy,
            grid.biomes(),
            &self.blocks,
            &mut dense,
        );

        // 4. Construct compact paletted chunk representation with uniform elision
        let packed_blocks = bulk::from_dense(&dense);

        let water_id = self.blocks.water;
        Chunk::from_blocks(pos, packed_blocks, move |state| {
            state != BlockStateId::AIR && state != water_id
        })
    }
}
