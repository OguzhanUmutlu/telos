//! Central terrain generator coordinator producing standard 32³ cubic chunks.

use crate::aquifer::AquiferSampler;
use crate::decoration::apply_surface_decorations;
use crate::density::CoarseGrid;
use crate::surface::{ResolvedBlocks, apply_surface_rules};
use serde::{Deserialize, Serialize};
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::chunk::Chunk;
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;
use telos_voxel::storage::bulk;

/// Deterministic generator algorithm kind for chunk generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorKind {
    /// Multi-noise continuous climate, coarse 3D density, caves, and strata surface rules.
    #[default]
    Standard,
    /// Flat world: 1 layer bedrock at y=0, 3 layers dirt at y=1..=3, 1 layer grass at y=4, air elsewhere.
    Flat,
    /// Completely empty void world (all air).
    Void,
}

/// Deterministic, server-side procedural world generator.
#[derive(Debug, Clone)]
pub struct WorldGenerator {
    seed: u64,
    kind: GeneratorKind,
    blocks: ResolvedBlocks,
    aquifer: AquiferSampler,
}

impl WorldGenerator {
    /// Creates a new `WorldGenerator` with the given 64-bit seed and block registry using `Standard` generation.
    #[must_use]
    pub fn new(seed: u64, registry: &BlockRegistry) -> Self {
        Self::with_kind(seed, registry, GeneratorKind::Standard)
    }

    /// Creates a new `WorldGenerator` with a specific `GeneratorKind`.
    #[must_use]
    pub fn with_kind(seed: u64, registry: &BlockRegistry, kind: GeneratorKind) -> Self {
        let blocks = ResolvedBlocks::resolve(registry);
        let aquifer = AquiferSampler::new(seed);
        Self {
            seed,
            kind,
            blocks,
            aquifer,
        }
    }

    /// World seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Generator algorithm kind.
    #[must_use]
    pub const fn kind(&self) -> GeneratorKind {
        self.kind
    }

    /// Reference to the underlying aquifer sampler.
    #[must_use]
    pub const fn aquifer(&self) -> &AquiferSampler {
        &self.aquifer
    }

    /// Generates a fully lit and paletted 32³ cubic chunk at `pos`.
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub fn generate_chunk(&self, pos: ChunkPos) -> Chunk {
        let water_id = self.blocks.water;
        let lava_id = self.blocks.lava;
        match self.kind {
            GeneratorKind::Void => Chunk::new_uniform(pos, BlockStateId::AIR, false),
            GeneratorKind::Flat => {
                if pos.y() != 0 {
                    return Chunk::new_uniform(pos, BlockStateId::AIR, false);
                }

                let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
                for y in 0..32usize {
                    let block = if y == 0 {
                        self.blocks.bedrock
                    } else if (1..=3).contains(&y) {
                        self.blocks.dirt
                    } else if y == 4 {
                        self.blocks.grass
                    } else {
                        BlockStateId::AIR
                    };

                    if block != BlockStateId::AIR {
                        let y_offset = y << 10;
                        for z in 0..32usize {
                            let z_offset = y_offset | (z << 5);
                            for x in 0..32usize {
                                dense[z_offset | x] = block;
                            }
                        }
                    }
                }

                let packed_blocks = bulk::from_dense(&dense);
                Chunk::from_blocks(pos, packed_blocks, move |state| {
                    state != BlockStateId::AIR && state != water_id && state != lava_id
                })
            }
            GeneratorKind::Standard => {
                // 1. Evaluate 405 corner samples on 4x8x4 coarse grid
                let grid = CoarseGrid::evaluate(self.seed, pos);

                // 2. Exact sign skip and trilinear cell interpolation into 32³ occupancy
                let mut occupancy = [false; CHUNK_VOLUME];
                grid.fill_occupancy(&mut occupancy);

                // 3. Precompute 3D aquifer Voronoi cells covering the chunk volume
                let aquifer_cache = self.aquifer.prepare_chunk(pos);

                // 4. Apply column surface rules, 3D aquifers, and strata into dense array
                let mut dense = [BlockStateId::AIR; CHUNK_VOLUME];
                apply_surface_rules(
                    self.seed,
                    pos,
                    &occupancy,
                    grid.biomes(),
                    &self.blocks,
                    &self.aquifer,
                    &aquifer_cache,
                    &mut dense,
                );

                // 5. Apply procedural 3D blob ores & large sinuous ore veins
                crate::ore::apply_ores(self.seed, pos, &self.blocks, &mut dense);

                // 6. Apply procedural surface foliage and floral patch decoration
                apply_surface_decorations(self.seed, pos, grid.biomes(), &self.blocks, &mut dense);

                // 6b. Apply procedural multi-stage trees and foliage canopies
                crate::tree::apply_trees(self.seed, pos, grid.biomes(), &self.blocks, &mut dense);

                // 7. Construct compact paletted chunk representation with uniform elision
                let packed_blocks = bulk::from_dense(&dense);

                Chunk::from_blocks(pos, packed_blocks, move |state| {
                    state != BlockStateId::AIR && state != water_id && state != lava_id
                })
            }
        }
    }
}
