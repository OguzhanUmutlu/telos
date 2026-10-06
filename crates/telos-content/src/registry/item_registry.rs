//! Central registry mapping namespaced item identifiers to dense numeric item IDs.

use hashbrown::HashMap;
use telos_core::ident::Identifier;

use crate::schema::item::{ArmorSlotDef, ItemDef, ItemTypeDef};

/// Central registry mapping namespaced item identifiers to dense numeric `u32` item IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemRegistry {
    items: Vec<ItemDef>,
    by_identifier: HashMap<Identifier, u32>,
    by_id: Vec<Identifier>,
    is_frozen: bool,
}

impl Default for ItemRegistry {
    fn default() -> Self {
        let mut reg = Self {
            items: Vec::new(),
            by_identifier: HashMap::new(),
            by_id: Vec::new(),
            is_frozen: false,
        };

        // ID 0 is strictly Air / Empty
        let air_id = Identifier::new("telos", "air").expect("Valid air identifier");
        reg.register(air_id, ItemDef::new_generic("Air"));

        reg
    }
}

impl ItemRegistry {
    /// Creates a new `ItemRegistry` pre-populated with Air at ID 0.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new item with the registry.
    ///
    /// # Panics
    /// Panics if the registry is already frozen or the identifier is already registered.
    pub fn register(&mut self, identifier: Identifier, def: ItemDef) -> u32 {
        assert!(!self.is_frozen, "Cannot register item: registry is frozen");
        assert!(
            !self.by_identifier.contains_key(&identifier),
            "Item identifier '{identifier}' is already registered"
        );

        let id = self.items.len() as u32;
        self.by_identifier.insert(identifier.clone(), id);
        self.by_id.push(identifier);
        self.items.push(def);

        id
    }

    /// Freezes the item registry, preventing further registrations.
    pub fn freeze(&mut self) {
        self.is_frozen = true;
    }

    /// Returns `true` if the item registry is frozen.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Returns the total number of registered items (including Air at 0).
    #[must_use]
    pub fn total_items(&self) -> usize {
        self.items.len()
    }

    /// Looks up an item definition by numeric ID.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&ItemDef> {
        self.items.get(id as usize)
    }

    /// Looks up an item definition by numeric ID (alias for `get`).
    #[must_use]
    pub fn get_by_id(&self, id: u32) -> Option<&ItemDef> {
        self.get(id)
    }

    /// Looks up numeric ID by namespaced identifier.
    #[must_use]
    pub fn get_by_ident(&self, identifier: &Identifier) -> Option<u32> {
        self.by_identifier.get(identifier).copied()
    }

    /// Looks up namespaced identifier by numeric ID.
    #[must_use]
    pub fn get_ident(&self, id: u32) -> Option<&Identifier> {
        self.by_id.get(id as usize)
    }

    /// Returns the user-facing name for an item ID.
    #[must_use]
    pub fn item_name(&self, id: u32) -> &str {
        self.get(id).map_or("Unknown Item", |item| &item.name)
    }

    /// Returns the maximum stack size for an item ID.
    #[must_use]
    pub fn max_stack_size(&self, id: u32) -> u16 {
        self.get(id).map_or(64, |item| item.max_stack_size)
    }

    /// Returns whether the item is equipable armor.
    #[must_use]
    pub fn is_armor(&self, id: u32) -> bool {
        self.get(id).is_some_and(ItemDef::is_armor)
    }

    /// Returns the target armor slot if this item is armor.
    #[must_use]
    pub fn armor_slot(&self, id: u32) -> Option<ArmorSlotDef> {
        self.get(id).and_then(ItemDef::armor_slot)
    }

    /// Returns the placed block identifier if this item is a block item.
    #[must_use]
    pub fn placed_block(&self, id: u32) -> Option<&str> {
        match self.get(id)?.item_type {
            ItemTypeDef::Block(ref b) => Some(b.as_str()),
            _ => None,
        }
    }

    /// Returns an iterator over all registered item IDs and their identifiers.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &Identifier, &ItemDef)> {
        self.items
            .iter()
            .enumerate()
            .map(move |(idx, def)| (idx as u32, &self.by_id[idx], def))
    }
}
