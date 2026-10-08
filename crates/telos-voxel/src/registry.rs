//! Block and block-state registry providing stable namespaced ID mapping.

use hashbrown::HashMap;
use telos_core::ident::Identifier;

use crate::state::{BlockStateId, StateFlags};

/// Definition of a registered block type.
#[derive(Debug, Clone)]
pub struct Block {
    identifier: Identifier,
    default_state: BlockStateId,
    states: Vec<BlockStateId>,
    flags: StateFlags,
}

impl Block {
    /// The namespaced identifier of this block.
    #[must_use]
    pub const fn identifier(&self) -> &Identifier {
        &self.identifier
    }

    /// The default `BlockStateId` for this block.
    #[must_use]
    pub const fn default_state(&self) -> BlockStateId {
        self.default_state
    }

    /// All state IDs associated with this block type.
    #[must_use]
    pub fn states(&self) -> &[BlockStateId] {
        &self.states
    }

    /// The flags for this block.
    #[must_use]
    pub const fn flags(&self) -> StateFlags {
        self.flags
    }
}

/// Central registry mapping namespaced block identifiers to dense numeric `BlockStateId`s.
#[derive(Debug, Clone)]
pub struct BlockRegistry {
    blocks: Vec<Block>,
    by_identifier: HashMap<Identifier, usize>,
    state_to_flags: Vec<StateFlags>,
    state_to_shape: Vec<crate::shape::BlockShape>,
    state_to_block: Vec<usize>,
    is_frozen: bool,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        let mut registry = Self {
            blocks: Vec::new(),
            by_identifier: HashMap::new(),
            state_to_flags: Vec::new(),
            state_to_shape: Vec::new(),
            state_to_block: Vec::new(),
            is_frozen: false,
        };

        // Register default built-in air state (ID 0)
        let air_id = Identifier::new("telos", "air").expect("Valid identifier");
        registry.register_with_shape(air_id, StateFlags::AIR, crate::shape::BlockShape::Empty);

        registry
    }
}

impl BlockRegistry {
    /// Creates a new block registry pre-populated with standard air.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new simple block type with default cube shape.
    ///
    /// # Panics
    pub fn register(&mut self, identifier: Identifier, flags: StateFlags) -> BlockStateId {
        let default_shape = if flags.contains(StateFlags::TRANSLUCENT) {
            crate::shape::BlockShape::Fluid {
                level: 0,
                falling: false,
            }
        } else {
            crate::shape::BlockShape::Cube
        };
        self.register_with_shape(identifier, flags, default_shape)
    }

    /// Registers a new block type with explicit shape geometry.
    ///
    /// # Panics
    /// Panics if the registry is already frozen or the identifier is registered.
    pub fn register_with_shape(
        &mut self,
        identifier: Identifier,
        flags: StateFlags,
        shape: crate::shape::BlockShape,
    ) -> BlockStateId {
        assert!(!self.is_frozen, "Cannot register block: registry is frozen");
        assert!(
            !self.by_identifier.contains_key(&identifier),
            "Block identifier '{identifier}' is already registered"
        );

        #[allow(clippy::cast_possible_truncation)]
        let state_id = BlockStateId::new(self.state_to_flags.len() as u32);
        let block_index = self.blocks.len();

        self.state_to_flags.push(flags);
        self.state_to_shape.push(shape);
        self.state_to_block.push(block_index);

        let block = Block {
            identifier: identifier.clone(),
            default_state: state_id,
            states: vec![state_id],
            flags,
        };

        self.by_identifier.insert(identifier, block_index);
        self.blocks.push(block);

        state_id
    }

    /// Freezes the registry, preventing further block registrations.
    pub fn freeze(&mut self) {
        self.is_frozen = true;
    }

    /// Returns `true` if the registry has been frozen.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Returns the block definition for a given identifier, if registered.
    #[must_use]
    pub fn get(&self, identifier: &Identifier) -> Option<&Block> {
        let index = self.by_identifier.get(identifier)?;
        self.blocks.get(*index)
    }

    /// Looks up the properties and `StateFlags` for a given `BlockStateId`.
    #[inline]
    #[must_use]
    pub fn flags(&self, id: BlockStateId) -> StateFlags {
        self.state_to_flags
            .get(id.as_usize())
            .copied()
            .unwrap_or(StateFlags::AIR)
    }

    /// Looks up the `BlockShape` for a given `BlockStateId`.
    #[inline]
    #[must_use]
    pub fn shape(&self, id: BlockStateId) -> &crate::shape::BlockShape {
        static EMPTY_SHAPE: crate::shape::BlockShape = crate::shape::BlockShape::Empty;
        self.state_to_shape
            .get(id.as_usize())
            .unwrap_or(&EMPTY_SHAPE)
    }

    /// Computes the 256-bit face occlusion mask for a given `BlockStateId` and direction.
    #[inline]
    #[must_use]
    pub fn occlusion_mask(
        &self,
        id: BlockStateId,
        face: telos_core::coords::Face,
    ) -> crate::shape::FaceOcclusionMask {
        self.shape(id).occlusion_mask(face)
    }

    /// Total number of registered block states.
    #[must_use]
    pub fn total_states(&self) -> usize {
        self.state_to_flags.len()
    }

    /// Looks up the block identifier for a given `BlockStateId`.
    #[must_use]
    pub fn identifier(&self, id: BlockStateId) -> Option<&Identifier> {
        let &block_idx = self.state_to_block.get(id.as_usize())?;
        let block = self.blocks.get(block_idx)?;
        Some(&block.identifier)
    }

    /// Creates a standard registry populated with baseline voxel blocks (stone, dirt, grass, etc.).
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    #[must_use]
    pub fn standard() -> Self {
        let mut reg = Self::new();

        let stone_id = Identifier::new("telos", "stone").unwrap();
        reg.register(stone_id, StateFlags::OPAQUE_CUBE);

        let dirt_id = Identifier::new("telos", "dirt").unwrap();
        reg.register(dirt_id, StateFlags::OPAQUE_CUBE);

        let grass_id = Identifier::new("telos", "grass_block").unwrap();
        reg.register(grass_id, StateFlags::OPAQUE_CUBE);

        let bedrock_id = Identifier::new("telos", "bedrock").unwrap();
        reg.register(bedrock_id, StateFlags::OPAQUE_CUBE);

        let sand_id = Identifier::new("telos", "sand").unwrap();
        reg.register(sand_id, StateFlags::OPAQUE_CUBE);

        let water_id = Identifier::new("telos", "water").unwrap();
        reg.register(
            water_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::TRANSLUCENT.bits()
                    | StateFlags::FLUID.bits(),
            ),
        );

        let planks_id = Identifier::new("telos", "oak_planks").unwrap();
        reg.register(planks_id, StateFlags::OPAQUE_CUBE);

        let leaves_id = Identifier::new("telos", "oak_leaves").unwrap();
        reg.register(
            leaves_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::CUTOUT.bits()
                    | StateFlags::LIGHT_BLOCKING.bits()
                    | StateFlags::TICKABLE.bits(),
            ),
        );

        let glass_id = Identifier::new("telos", "glass").unwrap();
        reg.register(glass_id, StateFlags::CUTOUT_CUBE);

        let slab_id = Identifier::new("telos", "stone_slab").unwrap();
        reg.register_with_shape(
            slab_id,
            StateFlags::NON_EMPTY,
            crate::shape::BlockShape::bottom_slab(),
        );

        let stairs_id = Identifier::new("telos", "oak_stairs").unwrap();
        reg.register_with_shape(
            stairs_id,
            StateFlags::NON_EMPTY,
            crate::shape::BlockShape::stairs(telos_core::coords::Face::North, false),
        );

        let poppy_id = Identifier::new("telos", "poppy").unwrap();
        reg.register_with_shape(
            poppy_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT,
            crate::shape::BlockShape::cross(),
        );

        let dandelion_id = Identifier::new("telos", "dandelion").unwrap();
        reg.register_with_shape(
            dandelion_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT,
            crate::shape::BlockShape::cross(),
        );

        let torch_id = Identifier::new("telos", "torch").unwrap();
        reg.register_with_shape(
            torch_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT | StateFlags::EMISSIVE,
            crate::shape::BlockShape::torch(None),
        );

        let flowing_water_id = Identifier::new("telos", "flowing_water").unwrap();
        reg.register_with_shape(
            flowing_water_id,
            StateFlags::NON_EMPTY | StateFlags::TRANSLUCENT | StateFlags::FLUID,
            crate::shape::BlockShape::fluid(1, false),
        );

        let short_grass_id = Identifier::new("telos", "short_grass").unwrap();
        reg.register_with_shape(
            short_grass_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT,
            crate::shape::BlockShape::cross(),
        );

        let fern_id = Identifier::new("telos", "fern").unwrap();
        reg.register_with_shape(
            fern_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT,
            crate::shape::BlockShape::cross(),
        );

        let dead_bush_id = Identifier::new("telos", "dead_bush").unwrap();
        reg.register_with_shape(
            dead_bush_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT,
            crate::shape::BlockShape::cross(),
        );

        // Logic & signal transmission components (Phase 36)
        let logic_wire_id = Identifier::new("telos", "logic_wire").unwrap();
        reg.register_with_shape(
            logic_wire_id,
            StateFlags::NON_EMPTY | StateFlags::CUTOUT | StateFlags::LOGIC_COMPONENT,
            crate::shape::BlockShape::flat_plate(),
        );

        let logic_wire_powered_id = Identifier::new("telos", "logic_wire_powered").unwrap();
        reg.register_with_shape(
            logic_wire_powered_id,
            StateFlags::NON_EMPTY
                | StateFlags::CUTOUT
                | StateFlags::LOGIC_COMPONENT
                | StateFlags::LOGIC_POWERED,
            crate::shape::BlockShape::flat_plate(),
        );

        let logic_power_block_id = Identifier::new("telos", "logic_power_block").unwrap();
        reg.register(
            logic_power_block_id,
            StateFlags::OPAQUE_CUBE | StateFlags::LOGIC_COMPONENT | StateFlags::LOGIC_POWERED,
        );

        let logic_lever_id = Identifier::new("telos", "logic_lever").unwrap();
        reg.register_with_shape(
            logic_lever_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT,
            crate::shape::BlockShape::lever(false),
        );

        let logic_lever_on_id = Identifier::new("telos", "logic_lever_on").unwrap();
        reg.register_with_shape(
            logic_lever_on_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT | StateFlags::LOGIC_POWERED,
            crate::shape::BlockShape::lever(true),
        );

        let logic_lamp_id = Identifier::new("telos", "logic_lamp").unwrap();
        reg.register(
            logic_lamp_id,
            StateFlags::OPAQUE_CUBE | StateFlags::LOGIC_COMPONENT,
        );

        let logic_lamp_lit_id = Identifier::new("telos", "logic_lamp_lit").unwrap();
        reg.register(
            logic_lamp_lit_id,
            StateFlags::OPAQUE_CUBE | StateFlags::LOGIC_COMPONENT | StateFlags::LOGIC_POWERED,
        );

        let logic_repeater_id = Identifier::new("telos", "logic_repeater").unwrap();
        reg.register_with_shape(
            logic_repeater_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT,
            crate::shape::BlockShape::flat_plate(),
        );

        let logic_repeater_powered_id = Identifier::new("telos", "logic_repeater_powered").unwrap();
        reg.register_with_shape(
            logic_repeater_powered_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT | StateFlags::LOGIC_POWERED,
            crate::shape::BlockShape::flat_plate(),
        );

        let logic_inverter_id = Identifier::new("telos", "logic_inverter").unwrap();
        reg.register_with_shape(
            logic_inverter_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT | StateFlags::LOGIC_POWERED,
            crate::shape::BlockShape::post(),
        );

        let logic_inverter_off_id = Identifier::new("telos", "logic_inverter_off").unwrap();
        reg.register_with_shape(
            logic_inverter_off_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT,
            crate::shape::BlockShape::post(),
        );

        let logic_diode_id = Identifier::new("telos", "logic_diode").unwrap();
        reg.register_with_shape(
            logic_diode_id,
            StateFlags::NON_EMPTY | StateFlags::LOGIC_COMPONENT,
            crate::shape::BlockShape::flat_plate(),
        );

        // Animated fluids and multi-layer surfaces (Phase 37)
        let lava_id = Identifier::new("telos", "lava").unwrap();
        reg.register_with_shape(
            lava_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::TRANSLUCENT.bits()
                    | StateFlags::EMISSIVE.bits()
                    | StateFlags::FLUID.bits(),
            ),
            crate::shape::BlockShape::fluid(0, false),
        );

        let flowing_lava_id = Identifier::new("telos", "flowing_lava").unwrap();
        reg.register_with_shape(
            flowing_lava_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::TRANSLUCENT.bits()
                    | StateFlags::EMISSIVE.bits()
                    | StateFlags::FLUID.bits(),
            ),
            crate::shape::BlockShape::fluid(1, false),
        );

        let fire_id = Identifier::new("telos", "fire").unwrap();
        reg.register_with_shape(
            fire_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::CUTOUT.bits()
                    | StateFlags::EMISSIVE.bits(),
            ),
            crate::shape::BlockShape::cross(),
        );

        let nether_portal_id = Identifier::new("telos", "nether_portal").unwrap();
        reg.register_with_shape(
            nether_portal_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::TRANSLUCENT.bits()
                    | StateFlags::EMISSIVE.bits(),
            ),
            crate::shape::BlockShape::flat_plate(),
        );

        // Fluid reactions & mechanics (Phase 43)
        let cobblestone_id = Identifier::new("telos", "cobblestone").unwrap();
        reg.register(cobblestone_id, StateFlags::OPAQUE_CUBE);

        let obsidian_id = Identifier::new("telos", "obsidian").unwrap();
        reg.register(obsidian_id, StateFlags::OPAQUE_CUBE);

        // Flowing water decay levels 2..=7 and falling vertical column
        for lvl in 2..=7u8 {
            let id = Identifier::new("telos", format!("flowing_water_{lvl}")).unwrap();
            reg.register_with_shape(
                id,
                StateFlags::NON_EMPTY | StateFlags::TRANSLUCENT | StateFlags::FLUID,
                crate::shape::BlockShape::fluid(lvl, false),
            );
        }
        let falling_water_id = Identifier::new("telos", "falling_water").unwrap();
        reg.register_with_shape(
            falling_water_id,
            StateFlags::NON_EMPTY | StateFlags::TRANSLUCENT | StateFlags::FLUID,
            crate::shape::BlockShape::fluid(1, true),
        );

        // Flowing lava decay levels 2..=7 and falling vertical column
        for lvl in 2..=7u8 {
            let id = Identifier::new("telos", format!("flowing_lava_{lvl}")).unwrap();
            reg.register_with_shape(
                id,
                StateFlags::NON_EMPTY
                    | StateFlags::TRANSLUCENT
                    | StateFlags::EMISSIVE
                    | StateFlags::FLUID,
                crate::shape::BlockShape::fluid(lvl, false),
            );
        }
        let falling_lava_id = Identifier::new("telos", "falling_lava").unwrap();
        reg.register_with_shape(
            falling_lava_id,
            StateFlags::NON_EMPTY
                | StateFlags::TRANSLUCENT
                | StateFlags::EMISSIVE
                | StateFlags::FLUID,
            crate::shape::BlockShape::fluid(1, true),
        );

        // Procedural ore distribution & large sinuous ore veins (Phase 45)
        // Standard Ores
        let coal_ore_id = Identifier::new("telos", "coal_ore").unwrap();
        reg.register(coal_ore_id, StateFlags::OPAQUE_CUBE);

        let iron_ore_id = Identifier::new("telos", "iron_ore").unwrap();
        reg.register(iron_ore_id, StateFlags::OPAQUE_CUBE);

        let copper_ore_id = Identifier::new("telos", "copper_ore").unwrap();
        reg.register(copper_ore_id, StateFlags::OPAQUE_CUBE);

        let gold_ore_id = Identifier::new("telos", "gold_ore").unwrap();
        reg.register(gold_ore_id, StateFlags::OPAQUE_CUBE);

        let redstone_ore_id = Identifier::new("telos", "redstone_ore").unwrap();
        reg.register(redstone_ore_id, StateFlags::OPAQUE_CUBE);

        let lapis_ore_id = Identifier::new("telos", "lapis_ore").unwrap();
        reg.register(lapis_ore_id, StateFlags::OPAQUE_CUBE);

        let diamond_ore_id = Identifier::new("telos", "diamond_ore").unwrap();
        reg.register(diamond_ore_id, StateFlags::OPAQUE_CUBE);

        let emerald_ore_id = Identifier::new("telos", "emerald_ore").unwrap();
        reg.register(emerald_ore_id, StateFlags::OPAQUE_CUBE);

        // Deepslate & Deepslate Ores
        let deepslate_id = Identifier::new("telos", "deepslate").unwrap();
        reg.register(deepslate_id, StateFlags::OPAQUE_CUBE);

        let deepslate_coal_ore_id = Identifier::new("telos", "deepslate_coal_ore").unwrap();
        reg.register(deepslate_coal_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_iron_ore_id = Identifier::new("telos", "deepslate_iron_ore").unwrap();
        reg.register(deepslate_iron_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_copper_ore_id = Identifier::new("telos", "deepslate_copper_ore").unwrap();
        reg.register(deepslate_copper_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_gold_ore_id = Identifier::new("telos", "deepslate_gold_ore").unwrap();
        reg.register(deepslate_gold_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_redstone_ore_id = Identifier::new("telos", "deepslate_redstone_ore").unwrap();
        reg.register(deepslate_redstone_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_lapis_ore_id = Identifier::new("telos", "deepslate_lapis_ore").unwrap();
        reg.register(deepslate_lapis_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_diamond_ore_id = Identifier::new("telos", "deepslate_diamond_ore").unwrap();
        reg.register(deepslate_diamond_ore_id, StateFlags::OPAQUE_CUBE);

        let deepslate_emerald_ore_id = Identifier::new("telos", "deepslate_emerald_ore").unwrap();
        reg.register(deepslate_emerald_ore_id, StateFlags::OPAQUE_CUBE);

        // Host rock variants and filler rocks
        let granite_id = Identifier::new("telos", "granite").unwrap();
        reg.register(granite_id, StateFlags::OPAQUE_CUBE);

        let diorite_id = Identifier::new("telos", "diorite").unwrap();
        reg.register(diorite_id, StateFlags::OPAQUE_CUBE);

        let andesite_id = Identifier::new("telos", "andesite").unwrap();
        reg.register(andesite_id, StateFlags::OPAQUE_CUBE);

        let tuff_id = Identifier::new("telos", "tuff").unwrap();
        reg.register(tuff_id, StateFlags::OPAQUE_CUBE);

        // Raw Metal Blocks (vein core concentrates)
        let raw_iron_block_id = Identifier::new("telos", "raw_iron_block").unwrap();
        reg.register(raw_iron_block_id, StateFlags::OPAQUE_CUBE);

        let raw_copper_block_id = Identifier::new("telos", "raw_copper_block").unwrap();
        reg.register(raw_copper_block_id, StateFlags::OPAQUE_CUBE);

        // Procedural trees & foliage canopies (Phase 47)
        let oak_log_id = Identifier::new("telos", "oak_log").unwrap();
        reg.register(oak_log_id, StateFlags::OPAQUE_CUBE);

        let birch_log_id = Identifier::new("telos", "birch_log").unwrap();
        reg.register(birch_log_id, StateFlags::OPAQUE_CUBE);

        let spruce_log_id = Identifier::new("telos", "spruce_log").unwrap();
        reg.register(spruce_log_id, StateFlags::OPAQUE_CUBE);

        let birch_leaves_id = Identifier::new("telos", "birch_leaves").unwrap();
        reg.register(
            birch_leaves_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::CUTOUT.bits()
                    | StateFlags::LIGHT_BLOCKING.bits()
                    | StateFlags::TICKABLE.bits(),
            ),
        );

        let spruce_leaves_id = Identifier::new("telos", "spruce_leaves").unwrap();
        reg.register(
            spruce_leaves_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::CUTOUT.bits()
                    | StateFlags::LIGHT_BLOCKING.bits()
                    | StateFlags::TICKABLE.bits(),
            ),
        );

        // Procedural structures, dungeons & ruins (Phase 49)
        let mossy_cobble_id = Identifier::new("telos", "mossy_cobblestone").unwrap();
        reg.register(mossy_cobble_id, StateFlags::OPAQUE_CUBE);

        let monster_spawner_id = Identifier::new("telos", "monster_spawner").unwrap();
        reg.register(monster_spawner_id, StateFlags::OPAQUE_CUBE);

        let chest_id = Identifier::new("telos", "chest").unwrap();
        reg.register_with_shape(
            chest_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::LIGHT_BLOCKING.bits()
                    | StateFlags::HAS_BLOCK_ENTITY.bits(),
            ),
            crate::shape::BlockShape::chest(),
        );

        let furnace_id = Identifier::new("telos", "furnace").unwrap();
        reg.register(
            furnace_id,
            StateFlags::from_bits_truncate(
                StateFlags::OPAQUE_CUBE.bits()
                    | StateFlags::HAS_BLOCK_ENTITY.bits()
                    | StateFlags::TICKABLE.bits(),
            ),
        );

        let lit_furnace_id = Identifier::new("telos", "lit_furnace").unwrap();
        reg.register(
            lit_furnace_id,
            StateFlags::from_bits_truncate(
                StateFlags::OPAQUE_CUBE.bits()
                    | StateFlags::HAS_BLOCK_ENTITY.bits()
                    | StateFlags::EMISSIVE.bits()
                    | StateFlags::TICKABLE.bits(),
            ),
        );

        reg.freeze();
        reg
    }

    /// Returns `true` if the given block state represents standard stone rock.
    #[inline]
    #[must_use]
    pub fn is_stone(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "stone")
    }

    /// Returns `true` if the given block state represents deepslate rock.
    #[inline]
    #[must_use]
    pub fn is_deepslate(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "deepslate")
    }

    /// Returns `true` if the given block state is a replaceable host rock that can host ore blobs/veins
    /// (e.g. stone, deepslate, granite, diorite, andesite, tuff).
    #[inline]
    #[must_use]
    pub fn is_replaceable_rock(&self, id: BlockStateId) -> bool {
        self.identifier(id).is_some_and(|ident| {
            matches!(
                ident.path(),
                "stone" | "deepslate" | "granite" | "diorite" | "andesite" | "tuff"
            )
        })
    }

    /// Returns `true` if the given block state represents a tree wood log (oak, birch, spruce).
    #[inline]
    #[must_use]
    pub fn is_log(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| matches!(ident.path(), "oak_log" | "birch_log" | "spruce_log"))
    }

    /// Returns `true` if the given block state represents tree leaves (oak, birch, spruce).
    #[inline]
    #[must_use]
    pub fn is_leaves(&self, id: BlockStateId) -> bool {
        self.identifier(id).is_some_and(|ident| {
            matches!(
                ident.path(),
                "oak_leaves" | "birch_leaves" | "spruce_leaves"
            )
        })
    }

    /// Returns `true` if the given block state is a logic component.
    #[inline]
    #[must_use]
    pub fn is_logic_component(&self, id: BlockStateId) -> bool {
        self.flags(id).contains(StateFlags::LOGIC_COMPONENT)
    }

    /// Returns `true` if the given block state is actively powered.
    #[inline]
    #[must_use]
    pub fn is_logic_powered(&self, id: BlockStateId) -> bool {
        self.flags(id).contains(StateFlags::LOGIC_POWERED)
    }

    /// Returns `true` if the given block state represents a fluid (water or lava).
    #[inline]
    #[must_use]
    pub fn is_fluid(&self, id: BlockStateId) -> bool {
        self.flags(id).contains(StateFlags::FLUID)
            || matches!(self.shape(id), crate::shape::BlockShape::Fluid { .. })
    }

    /// Returns `true` if the given block state represents a monster spawner.
    #[inline]
    #[must_use]
    pub fn is_spawner(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "monster_spawner")
    }

    /// Returns `true` if the given block state represents a chest container.
    #[inline]
    #[must_use]
    pub fn is_chest(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "chest")
    }

    /// Returns `true` if the given block state represents a furnace (either unlit or lit).
    #[inline]
    #[must_use]
    pub fn is_furnace(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "furnace" || ident.path() == "lit_furnace")
    }

    /// Returns `true` if the given block state represents an actively lit furnace.
    #[inline]
    #[must_use]
    pub fn is_lit_furnace(&self, id: BlockStateId) -> bool {
        self.identifier(id)
            .is_some_and(|ident| ident.path() == "lit_furnace")
    }

    /// Returns `true` if the block state at `id` can be freely replaced by flowing fluids.
    #[inline]
    #[must_use]
    pub fn is_replaceable(&self, id: BlockStateId) -> bool {
        if id.is_air() {
            return true;
        }
        let flags = self.flags(id);
        let shape = self.shape(id);
        matches!(
            shape,
            crate::shape::BlockShape::Cross | crate::shape::BlockShape::Torch { .. }
        ) || (flags.contains(StateFlags::LOGIC_COMPONENT)
            && matches!(shape, crate::shape::BlockShape::Boxes(_)))
    }

    /// Evaluates the fluid properties (kind, decay level, falling flag) of a block state.
    #[must_use]
    pub fn fluid_state(&self, id: BlockStateId) -> Option<crate::fluid::FluidState> {
        let shape = self.shape(id);
        if let crate::shape::BlockShape::Fluid { level, falling } = *shape {
            let is_lava = self.flags(id).contains(StateFlags::EMISSIVE);
            let kind = if is_lava {
                crate::fluid::FluidKind::Lava
            } else {
                crate::fluid::FluidKind::Water
            };
            Some(crate::fluid::FluidState::new(kind, level, falling))
        } else {
            None
        }
    }

    /// Resolves the corresponding `BlockStateId` for a given fluid kind, level (0..=7), and falling state.
    #[must_use]
    pub fn fluid_state_id(
        &self,
        kind: crate::fluid::FluidKind,
        level: u8,
        falling: bool,
    ) -> BlockStateId {
        let name = match kind {
            crate::fluid::FluidKind::Water => {
                if level == 0 && !falling {
                    "water"
                } else if falling {
                    "falling_water"
                } else {
                    match level {
                        1 => "flowing_water",
                        2 => "flowing_water_2",
                        3 => "flowing_water_3",
                        4 => "flowing_water_4",
                        5 => "flowing_water_5",
                        6 => "flowing_water_6",
                        _ => "flowing_water_7",
                    }
                }
            }
            crate::fluid::FluidKind::Lava => {
                if level == 0 && !falling {
                    "lava"
                } else if falling {
                    "falling_lava"
                } else {
                    match level {
                        1 => "flowing_lava",
                        2 => "flowing_lava_2",
                        3 => "flowing_lava_3",
                        4 => "flowing_lava_4",
                        5 => "flowing_lava_5",
                        6 => "flowing_lava_6",
                        _ => "flowing_lava_7",
                    }
                }
            }
        };

        if let Ok(ident) = Identifier::new("telos", name)
            && let Some(&idx) = self.by_identifier.get(&ident)
        {
            self.blocks[idx].default_state()
        } else {
            match kind {
                crate::fluid::FluidKind::Water => BlockStateId::new(6),
                crate::fluid::FluidKind::Lava => BlockStateId::new(31),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_is_zero() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.total_states(), 1);
        let air = reg.get(&Identifier::new("telos", "air").unwrap()).unwrap();
        assert_eq!(air.default_state(), BlockStateId::AIR);
        assert_eq!(reg.flags(BlockStateId::AIR), StateFlags::AIR);
    }

    #[test]
    fn test_standard_registry_logic_blocks() {
        let reg = BlockRegistry::standard();
        assert!(reg.total_states() >= 31);
        let wire = reg
            .get(&Identifier::new("telos", "logic_wire").unwrap())
            .unwrap();
        assert!(reg.is_logic_component(wire.default_state()));
        assert!(!reg.is_logic_powered(wire.default_state()));
        let wire_powered = reg
            .get(&Identifier::new("telos", "logic_wire_powered").unwrap())
            .unwrap();
        assert!(reg.is_logic_component(wire_powered.default_state()));
        assert!(reg.is_logic_powered(wire_powered.default_state()));
    }

    #[test]
    fn test_standard_registry_fluid_blocks() {
        let reg = BlockRegistry::standard();
        let water = reg
            .get(&Identifier::new("telos", "water").unwrap())
            .unwrap()
            .default_state();
        assert!(reg.is_fluid(water));
        let state = reg.fluid_state(water).unwrap();
        assert_eq!(state.kind, crate::fluid::FluidKind::Water);
        assert_eq!(state.level, 0);
        assert!(!state.falling);
        assert!(state.is_source());

        let flowing_water_3 = reg.fluid_state_id(crate::fluid::FluidKind::Water, 3, false);
        let s3 = reg.fluid_state(flowing_water_3).unwrap();
        assert_eq!(s3.level, 3);
        assert!(!s3.falling);

        let falling_water = reg.fluid_state_id(crate::fluid::FluidKind::Water, 1, true);
        let sf = reg.fluid_state(falling_water).unwrap();
        assert!(sf.falling);

        let lava = reg.fluid_state_id(crate::fluid::FluidKind::Lava, 0, false);
        let sl = reg.fluid_state(lava).unwrap();
        assert_eq!(sl.kind, crate::fluid::FluidKind::Lava);
        assert_eq!(sl.level, 0);

        let cobblestone = reg
            .get(&Identifier::new("telos", "cobblestone").unwrap())
            .unwrap()
            .default_state();
        assert!(!reg.is_fluid(cobblestone));
        assert!(reg.flags(cobblestone).contains(StateFlags::OPAQUE_FULL));

        let obsidian = reg
            .get(&Identifier::new("telos", "obsidian").unwrap())
            .unwrap()
            .default_state();
        assert!(!reg.is_fluid(obsidian));
        assert!(reg.flags(obsidian).contains(StateFlags::OPAQUE_FULL));
    }

    #[test]
    fn test_standard_registry_ore_and_rock_blocks() {
        let reg = BlockRegistry::standard();

        let stone = reg
            .get(&Identifier::new("telos", "stone").unwrap())
            .unwrap()
            .default_state();
        let deepslate = reg
            .get(&Identifier::new("telos", "deepslate").unwrap())
            .unwrap()
            .default_state();
        let granite = reg
            .get(&Identifier::new("telos", "granite").unwrap())
            .unwrap()
            .default_state();
        let iron_ore = reg
            .get(&Identifier::new("telos", "iron_ore").unwrap())
            .unwrap()
            .default_state();
        let deepslate_diamond = reg
            .get(&Identifier::new("telos", "deepslate_diamond_ore").unwrap())
            .unwrap()
            .default_state();
        let raw_iron = reg
            .get(&Identifier::new("telos", "raw_iron_block").unwrap())
            .unwrap()
            .default_state();

        assert!(reg.is_stone(stone));
        assert!(!reg.is_stone(deepslate));
        assert!(reg.is_deepslate(deepslate));
        assert!(!reg.is_deepslate(stone));

        assert!(reg.is_replaceable_rock(stone));
        assert!(reg.is_replaceable_rock(deepslate));
        assert!(reg.is_replaceable_rock(granite));
        assert!(!reg.is_replaceable_rock(iron_ore));
        assert!(!reg.is_replaceable_rock(raw_iron));

        assert!(reg.flags(iron_ore).contains(StateFlags::OPAQUE_FULL));
        assert!(
            reg.flags(deepslate_diamond)
                .contains(StateFlags::OPAQUE_FULL)
        );
        assert!(reg.flags(raw_iron).contains(StateFlags::OPAQUE_FULL));
    }

    #[test]
    fn test_standard_registry_tree_blocks() {
        let reg = BlockRegistry::standard();

        let oak_log = reg
            .get(&Identifier::new("telos", "oak_log").unwrap())
            .unwrap()
            .default_state();
        let birch_log = reg
            .get(&Identifier::new("telos", "birch_log").unwrap())
            .unwrap()
            .default_state();
        let spruce_log = reg
            .get(&Identifier::new("telos", "spruce_log").unwrap())
            .unwrap()
            .default_state();

        let oak_leaves = reg
            .get(&Identifier::new("telos", "oak_leaves").unwrap())
            .unwrap()
            .default_state();
        let birch_leaves = reg
            .get(&Identifier::new("telos", "birch_leaves").unwrap())
            .unwrap()
            .default_state();
        let spruce_leaves = reg
            .get(&Identifier::new("telos", "spruce_leaves").unwrap())
            .unwrap()
            .default_state();

        assert!(reg.is_log(oak_log));
        assert!(reg.is_log(birch_log));
        assert!(reg.is_log(spruce_log));
        assert!(!reg.is_log(oak_leaves));

        assert!(reg.is_leaves(oak_leaves));
        assert!(reg.is_leaves(birch_leaves));
        assert!(reg.is_leaves(spruce_leaves));
        assert!(!reg.is_leaves(oak_log));

        assert!(reg.flags(oak_log).contains(StateFlags::OPAQUE_FULL));
        assert!(reg.flags(oak_leaves).contains(StateFlags::TICKABLE));
        assert!(reg.flags(birch_leaves).contains(StateFlags::TICKABLE));
        assert!(reg.flags(spruce_leaves).contains(StateFlags::TICKABLE));
    }

    #[test]
    fn test_furnace_registration_and_predicates() {
        let reg = BlockRegistry::standard();

        let furnace = reg
            .get(&Identifier::new("telos", "furnace").unwrap())
            .unwrap()
            .default_state();
        let lit_furnace = reg
            .get(&Identifier::new("telos", "lit_furnace").unwrap())
            .unwrap()
            .default_state();
        let chest = reg
            .get(&Identifier::new("telos", "chest").unwrap())
            .unwrap()
            .default_state();

        assert!(reg.is_furnace(furnace));
        assert!(reg.is_furnace(lit_furnace));
        assert!(!reg.is_furnace(chest));

        assert!(!reg.is_lit_furnace(furnace));
        assert!(reg.is_lit_furnace(lit_furnace));
        assert!(!reg.is_lit_furnace(chest));

        assert!(reg.flags(furnace).contains(StateFlags::HAS_BLOCK_ENTITY));
        assert!(
            reg.flags(lit_furnace)
                .contains(StateFlags::HAS_BLOCK_ENTITY)
        );
        assert!(reg.flags(lit_furnace).contains(StateFlags::EMISSIVE));
        assert!(!reg.flags(furnace).contains(StateFlags::EMISSIVE));
    }
}
