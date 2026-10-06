//! Block and block-state registry providing stable namespaced ID mapping.

use hashbrown::HashMap;
use vx_core::ident::Identifier;

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
        let air_id = Identifier::new("voxel", "air").expect("Valid identifier");
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
    /// Panics if the registry is already frozen or the identifier is registered.
    pub fn register(&mut self, identifier: Identifier, flags: StateFlags) -> BlockStateId {
        let default_shape = if flags.contains(StateFlags::TRANSLUCENT) {
            crate::shape::BlockShape::Fluid { level: 0 }
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
        face: vx_core::coords::Face,
    ) -> crate::shape::FaceOcclusionMask {
        self.shape(id).occlusion_mask(face)
    }

    /// Total number of registered block states.
    #[must_use]
    pub fn total_states(&self) -> usize {
        self.state_to_flags.len()
    }

    /// Creates a standard registry populated with baseline voxel blocks (stone, dirt, grass, etc.).
    #[allow(clippy::similar_names)]
    #[must_use]
    pub fn standard() -> Self {
        let mut reg = Self::new();

        let stone_id = Identifier::new("voxel", "stone").unwrap();
        reg.register(stone_id, StateFlags::OPAQUE_CUBE);

        let dirt_id = Identifier::new("voxel", "dirt").unwrap();
        reg.register(dirt_id, StateFlags::OPAQUE_CUBE);

        let grass_id = Identifier::new("voxel", "grass_block").unwrap();
        reg.register(grass_id, StateFlags::OPAQUE_CUBE);

        let bedrock_id = Identifier::new("voxel", "bedrock").unwrap();
        reg.register(bedrock_id, StateFlags::OPAQUE_CUBE);

        let sand_id = Identifier::new("voxel", "sand").unwrap();
        reg.register(sand_id, StateFlags::OPAQUE_CUBE);

        let water_id = Identifier::new("voxel", "water").unwrap();
        reg.register(
            water_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits() | StateFlags::TRANSLUCENT.bits(),
            ),
        );

        let planks_id = Identifier::new("voxel", "oak_planks").unwrap();
        reg.register(planks_id, StateFlags::OPAQUE_CUBE);

        let leaves_id = Identifier::new("voxel", "oak_leaves").unwrap();
        reg.register(
            leaves_id,
            StateFlags::from_bits_truncate(
                StateFlags::NON_EMPTY.bits()
                    | StateFlags::CUTOUT.bits()
                    | StateFlags::LIGHT_BLOCKING.bits(),
            ),
        );

        let glass_id = Identifier::new("voxel", "glass").unwrap();
        reg.register(glass_id, StateFlags::CUTOUT_CUBE);

        let slab_id = Identifier::new("voxel", "stone_slab").unwrap();
        reg.register_with_shape(
            slab_id,
            StateFlags::NON_EMPTY,
            crate::shape::BlockShape::bottom_slab(),
        );

        let stairs_id = Identifier::new("voxel", "oak_stairs").unwrap();
        reg.register_with_shape(
            stairs_id,
            StateFlags::NON_EMPTY,
            crate::shape::BlockShape::stairs(vx_core::coords::Face::North, false),
        );

        reg.freeze();
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_is_zero() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.total_states(), 1);
        let air = reg.get(&Identifier::new("voxel", "air").unwrap()).unwrap();
        assert_eq!(air.default_state(), BlockStateId::AIR);
        assert_eq!(reg.flags(BlockStateId::AIR), StateFlags::AIR);
    }

    #[test]
    fn test_standard_registry() {
        let reg = BlockRegistry::standard();
        assert!(reg.is_frozen());
        let stone = reg
            .get(&Identifier::new("voxel", "stone").unwrap())
            .unwrap();
        assert_eq!(stone.default_state(), BlockStateId::new(1));
        assert!(
            reg.flags(stone.default_state())
                .contains(StateFlags::OPAQUE_FULL)
        );
    }
}
