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
                StateFlags::NON_EMPTY.bits() | StateFlags::TRANSLUCENT.bits(),
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
                    | StateFlags::LIGHT_BLOCKING.bits(),
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
            StateFlags::NON_EMPTY | StateFlags::TRANSLUCENT,
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

        reg.freeze();
        reg
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
}
