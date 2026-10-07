//! Top-down column surface rules and strata material assignment.

use crate::biome::BiomeId;
use telos_core::coords::{CHUNK_VOLUME, ChunkPos};
use telos_voxel::registry::BlockRegistry;
use telos_voxel::state::BlockStateId;

/// Resolved block state IDs needed by surface rules and ore generation.
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
    /// Lava block state ID.
    pub lava: BlockStateId,
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

    // Phase 45 Ore & Subterranean Strata Blocks
    /// Deepslate subterranean host rock.
    pub deepslate: BlockStateId,
    /// Coal ore block state ID.
    pub coal_ore: BlockStateId,
    /// Iron ore block state ID.
    pub iron_ore: BlockStateId,
    /// Copper ore block state ID.
    pub copper_ore: BlockStateId,
    /// Gold ore block state ID.
    pub gold_ore: BlockStateId,
    /// Redstone ore block state ID.
    pub redstone_ore: BlockStateId,
    /// Lapis lazuli ore block state ID.
    pub lapis_ore: BlockStateId,
    /// Diamond ore block state ID.
    pub diamond_ore: BlockStateId,
    /// Emerald ore block state ID.
    pub emerald_ore: BlockStateId,

    /// Deepslate coal ore block state ID.
    pub deepslate_coal_ore: BlockStateId,
    /// Deepslate iron ore block state ID.
    pub deepslate_iron_ore: BlockStateId,
    /// Deepslate copper ore block state ID.
    pub deepslate_copper_ore: BlockStateId,
    /// Deepslate gold ore block state ID.
    pub deepslate_gold_ore: BlockStateId,
    /// Deepslate redstone ore block state ID.
    pub deepslate_redstone_ore: BlockStateId,
    /// Deepslate lapis lazuli ore block state ID.
    pub deepslate_lapis_ore: BlockStateId,
    /// Deepslate diamond ore block state ID.
    pub deepslate_diamond_ore: BlockStateId,
    /// Deepslate emerald ore block state ID.
    pub deepslate_emerald_ore: BlockStateId,

    /// Granite filler rock block state ID.
    pub granite: BlockStateId,
    /// Diorite filler rock block state ID.
    pub diorite: BlockStateId,
    /// Andesite filler rock block state ID.
    pub andesite: BlockStateId,
    /// Tuff filler rock block state ID.
    pub tuff: BlockStateId,

    /// Block of raw iron block state ID.
    pub raw_iron_block: BlockStateId,
    /// Block of raw copper block state ID.
    pub raw_copper_block: BlockStateId,

    // Phase 47 Trees & Foliage Blocks
    /// Oak tree wood log.
    pub oak_log: BlockStateId,
    /// Birch tree wood log.
    pub birch_log: BlockStateId,
    /// Spruce tree wood log.
    pub spruce_log: BlockStateId,
    /// Oak tree leaves.
    pub oak_leaves: BlockStateId,
    /// Birch tree leaves.
    pub birch_leaves: BlockStateId,
    /// Spruce tree leaves.
    pub spruce_leaves: BlockStateId,

    // Phase 49 Structures, Dungeons & Ruins
    /// Cobblestone block state ID.
    pub cobblestone: BlockStateId,
    /// Mossy cobblestone block state ID.
    pub mossy_cobblestone: BlockStateId,
    /// Monster spawner block state ID.
    pub monster_spawner: BlockStateId,
    /// Chest container block state ID.
    pub chest: BlockStateId,
}

impl ResolvedBlocks {
    /// Resolves required block state IDs from the registry.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn resolve(registry: &BlockRegistry) -> Self {
        let resolve_block = |name: &str, default_id: u32| {
            registry
                .get(&telos_core::ident::Identifier::new("telos", name).unwrap())
                .map_or(
                    BlockStateId::new(default_id),
                    telos_voxel::Block::default_state,
                )
        };

        let air = BlockStateId::AIR;
        let stone = resolve_block("stone", 1);
        let dirt = resolve_block("dirt", 2);
        let grass = resolve_block("grass_block", 3);
        let bedrock = resolve_block("bedrock", 4);
        let sand = resolve_block("sand", 5);
        let water = resolve_block("water", 6);
        let lava = resolve_block("lava", 33);
        let poppy = resolve_block("poppy", 12);
        let dandelion = resolve_block("dandelion", 13);
        let short_grass = resolve_block("short_grass", 16);
        let fern = resolve_block("fern", 17);
        let dead_bush = resolve_block("dead_bush", 18);

        let deepslate = resolve_block("deepslate", 57);
        let coal_ore = resolve_block("coal_ore", 49);
        let iron_ore = resolve_block("iron_ore", 50);
        let copper_ore = resolve_block("copper_ore", 51);
        let gold_ore = resolve_block("gold_ore", 52);
        let redstone_ore = resolve_block("redstone_ore", 53);
        let lapis_ore = resolve_block("lapis_ore", 54);
        let diamond_ore = resolve_block("diamond_ore", 55);
        let emerald_ore = resolve_block("emerald_ore", 56);

        let deepslate_coal_ore = resolve_block("deepslate_coal_ore", 58);
        let deepslate_iron_ore = resolve_block("deepslate_iron_ore", 59);
        let deepslate_copper_ore = resolve_block("deepslate_copper_ore", 60);
        let deepslate_gold_ore = resolve_block("deepslate_gold_ore", 61);
        let deepslate_redstone_ore = resolve_block("deepslate_redstone_ore", 62);
        let deepslate_lapis_ore = resolve_block("deepslate_lapis_ore", 63);
        let deepslate_diamond_ore = resolve_block("deepslate_diamond_ore", 64);
        let deepslate_emerald_ore = resolve_block("deepslate_emerald_ore", 65);

        let granite = resolve_block("granite", 66);
        let diorite = resolve_block("diorite", 67);
        let andesite = resolve_block("andesite", 68);
        let tuff = resolve_block("tuff", 69);

        let raw_iron_block = resolve_block("raw_iron_block", 70);
        let raw_copper_block = resolve_block("raw_copper_block", 71);

        let oak_log = resolve_block("oak_log", 74);
        let birch_log = resolve_block("birch_log", 75);
        let spruce_log = resolve_block("spruce_log", 76);
        let oak_leaves = resolve_block("oak_leaves", 8);
        let birch_leaves = resolve_block("birch_leaves", 77);
        let spruce_leaves = resolve_block("spruce_leaves", 78);

        let cobblestone = resolve_block("cobblestone", 4);
        let mossy_cobblestone = resolve_block("mossy_cobblestone", 79);
        let monster_spawner = resolve_block("monster_spawner", 80);
        let chest = resolve_block("chest", 81);

        Self {
            air,
            stone,
            dirt,
            grass,
            bedrock,
            sand,
            water,
            lava,
            poppy,
            dandelion,
            short_grass,
            fern,
            dead_bush,
            deepslate,
            coal_ore,
            iron_ore,
            copper_ore,
            gold_ore,
            redstone_ore,
            lapis_ore,
            diamond_ore,
            emerald_ore,
            deepslate_coal_ore,
            deepslate_iron_ore,
            deepslate_copper_ore,
            deepslate_gold_ore,
            deepslate_redstone_ore,
            deepslate_lapis_ore,
            deepslate_diamond_ore,
            deepslate_emerald_ore,
            granite,
            diorite,
            andesite,
            tuff,
            raw_iron_block,
            raw_copper_block,
            oak_log,
            birch_log,
            spruce_log,
            oak_leaves,
            birch_leaves,
            spruce_leaves,
            cobblestone,
            mossy_cobblestone,
            monster_spawner,
            chest,
        }
    }
}

/// Helper to determine the ambient rock type (stone vs deepslate) at world height `wy`.
#[inline]
#[must_use]
pub fn rock_at_height(wx: i32, wy: i32, wz: i32, blocks: &ResolvedBlocks) -> BlockStateId {
    if wy > 0 {
        blocks.stone
    } else if wy <= -16 {
        blocks.deepslate
    } else {
        // Dithered transition between 0 and -16
        let h = crate::math::hash3(0xDEE9_51A7_E000_0001, wx, wy, wz);
        let roll = crate::math::unit_f32(h);
        #[allow(clippy::cast_precision_loss)]
        let threshold = (-wy as f32) * (1.0 / 16.0);
        if roll < threshold {
            blocks.deepslate
        } else {
            blocks.stone
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
    aquifer: &crate::aquifer::AquiferSampler,
    aquifer_cache: &crate::aquifer::ChunkAquiferCache,
    out_dense: &mut [BlockStateId; CHUNK_VOLUME],
) {
    let origin_x = chunk_pos.x() * 32;
    let origin_y = chunk_pos.y() * 32;
    let origin_z = chunk_pos.z() * 32;

    for z in 0usize..32 {
        let wz = origin_z + z as i32;
        let cz = z >> 2;
        for x in 0usize..32 {
            let wx = origin_x + x as i32;
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
                        // Deep rock: stone or deepslate depending on depth
                        out_dense[idx] = rock_at_height(wx, wy, wz, blocks);
                    }

                    depth_from_surface += 1;
                } else {
                    // Empty cavity: evaluate 3D noise-modulated aquifer level & fluid barriers
                    match aquifer.sample(aquifer_cache, wx, wy, wz) {
                        crate::aquifer::AquiferSample::Barrier => {
                            out_dense[idx] = rock_at_height(wx, wy, wz, blocks);
                            depth_from_surface += 1;
                            is_under_solid = true;
                        }
                        crate::aquifer::AquiferSample::Fluid(crate::aquifer::FluidKind::Water) => {
                            out_dense[idx] = blocks.water;
                            depth_from_surface = 0;
                            is_under_solid = false;
                        }
                        crate::aquifer::AquiferSample::Fluid(crate::aquifer::FluidKind::Lava) => {
                            out_dense[idx] = blocks.lava;
                            depth_from_surface = 0;
                            is_under_solid = false;
                        }
                        crate::aquifer::AquiferSample::Air
                        | crate::aquifer::AquiferSample::Fluid(crate::aquifer::FluidKind::None) => {
                            out_dense[idx] = blocks.air;
                            depth_from_surface = 0;
                            is_under_solid = false;
                        }
                    }
                }
            }
        }
    }
}
