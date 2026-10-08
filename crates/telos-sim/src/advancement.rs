//! Server-authoritative advancement registry, criterion trigger evaluation,
//! and per-player progress tracking.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Frame appearance defining the border and presentation tier of an advancement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AdvancementFrame {
    /// Standard square/rounded frame for normal milestone advancements.
    Task = 0,
    /// Rounded bracket/circle frame for key gameplay goals.
    Goal = 1,
    /// Pointed star/spiky bracket frame for difficult challenges with distinct fanfares.
    Challenge = 2,
}

impl AdvancementFrame {
    /// Returns the wire numeric representation.
    #[must_use]
    pub const fn to_wire(self) -> u8 {
        self as u8
    }

    /// Converts from wire numeric representation.
    #[must_use]
    pub const fn from_wire(val: u8) -> Self {
        match val {
            1 => Self::Goal,
            2 => Self::Challenge,
            _ => Self::Task,
        }
    }
}

/// Category / tab grouping in the advancements menu tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdvancementCategory {
    /// Core gameplay progression: wood, stone, tools, iron, armor, etc.
    Story,
    /// Exploration, combat, mob hunting, archery, and navigation.
    Adventure,
    /// Nether subterranean dimension progression.
    Nether,
    /// Farming, animal breeding, cooking, and domestication.
    Husbandry,
    /// End dimension progression.
    End,
}

impl AdvancementCategory {
    /// Returns the tab title text.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Story => "Story",
            Self::Adventure => "Adventure",
            Self::Nether => "Nether",
            Self::Husbandry => "Husbandry",
            Self::End => "The End",
        }
    }
}

/// Criterion rule evaluated against gameplay simulation triggers.
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancementCriterion {
    /// Player inventory contains at least `min_count` of `item`.
    InventoryChanged {
        /// Item ID to detect.
        item: u32,
        /// Minimum quantity required.
        min_count: u32,
    },
    /// Player crafted an item with ID `item`.
    ItemCrafted {
        /// Item ID crafted.
        item: u32,
    },
    /// Player broke a voxel block matching `block`.
    BlockBroken {
        /// Block state or base block ID.
        block: u32,
    },
    /// Player placed a voxel block matching `block`.
    BlockPlaced {
        /// Block state or base block ID.
        block: u32,
    },
    /// Player killed an entity of type `entity_type`.
    EntityKilled {
        /// Entity type ID (e.g. 1 = Zombie).
        entity_type: u32,
    },
    /// Player struck an entity with a projectile from at least `min_distance` blocks away.
    ArcheryHit {
        /// Minimum distance in blocks.
        min_distance: f32,
    },
    /// Arbitrary named criterion granted directly by game systems or scripts.
    Arbitrary(&'static str),
}

/// Gameplay trigger event passed into the criterion evaluation pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum CriterionTrigger<'a> {
    /// Inventory holds or acquired an item with count.
    Inventory {
        /// Item ID present in inventory.
        item: u32,
        /// Total count present.
        count: u32,
    },
    /// Item was crafted.
    Craft {
        /// Item ID crafted.
        item: u32,
    },
    /// Block was broken by player in survival mode.
    BlockBreak {
        /// Block state or base block ID.
        block: u32,
    },
    /// Block was placed by player.
    BlockPlace {
        /// Block state or base block ID.
        block: u32,
    },
    /// Living entity was killed by player.
    MobKill {
        /// Entity type ID.
        entity_type: u32,
    },
    /// Projectile hit an entity at distance.
    ArcheryHit {
        /// Euclidean distance in blocks.
        distance: f32,
    },
    /// Custom trigger ID.
    Arbitrary(&'a str),
}

impl AdvancementCriterion {
    /// Evaluates whether this criterion matches a given gameplay trigger.
    #[must_use]
    pub fn matches(&self, trigger: &CriterionTrigger) -> bool {
        match (self, trigger) {
            (
                Self::InventoryChanged { item, min_count },
                CriterionTrigger::Inventory {
                    item: trig_item,
                    count,
                },
            ) => item == trig_item && count >= min_count,
            (Self::ItemCrafted { item }, CriterionTrigger::Craft { item: trig_item }) => {
                item == trig_item
            }
            (Self::BlockBroken { block }, CriterionTrigger::BlockBreak { block: trig_block })
            | (Self::BlockPlaced { block }, CriterionTrigger::BlockPlace { block: trig_block }) => {
                block == trig_block
            }
            (
                Self::EntityKilled { entity_type },
                CriterionTrigger::MobKill {
                    entity_type: trig_type,
                },
            ) => entity_type == trig_type,
            (Self::ArcheryHit { min_distance }, CriterionTrigger::ArcheryHit { distance }) => {
                distance >= min_distance
            }
            (Self::Arbitrary(name), CriterionTrigger::Arbitrary(trig_name)) => name == trig_name,
            _ => false,
        }
    }
}

/// Static definition of an advancement in the game.
#[derive(Debug, Clone, PartialEq)]
pub struct Advancement {
    /// Unique namespaced identifier (e.g. `"telos:story/root"`).
    pub id: &'static str,
    /// Optional parent advancement ID required to unlock or connect in the tree.
    pub parent: Option<&'static str>,
    /// Category / tab grouping.
    pub category: AdvancementCategory,
    /// Localized display title.
    pub title: &'static str,
    /// Localized descriptive tooltip text.
    pub description: &'static str,
    /// Item ID used for UI icon display.
    pub icon_item: u32,
    /// Frame visual style.
    pub frame: AdvancementFrame,
    /// Whether a top-right toast popup should display upon completion.
    pub display_toast: bool,
    /// Whether an announcement should be broadcast to the server chat.
    pub announce_chat: bool,
    /// Visual 2D coordinate on the tree canvas (X: columns, Y: rows).
    pub x: f32,
    /// Visual 2D coordinate on the tree canvas.
    pub y: f32,
    /// Criteria required for completion: slice of `(criterion_id, criterion)`.
    pub criteria: &'static [(&'static str, AdvancementCriterion)],
    /// Optional custom requirements (AND of ORs). If empty, all criteria are required.
    pub requirements: &'static [&'static [&'static str]],
}

impl Advancement {
    /// Returns true if this advancement is a root node for its category.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent.is_none()
    }
}

/// Thread-safe registry containing all advancement definitions.
#[derive(Debug, Clone)]
pub struct AdvancementRegistry {
    advancements: Vec<Advancement>,
    by_id: BTreeMap<&'static str, usize>,
}

impl Default for AdvancementRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl AdvancementRegistry {
    /// Creates an empty advancement registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            advancements: Vec::new(),
            by_id: BTreeMap::new(),
        }
    }

    /// Registers a new advancement.
    pub fn register(&mut self, adv: Advancement) {
        let idx = self.advancements.len();
        self.by_id.insert(adv.id, idx);
        self.advancements.push(adv);
    }

    /// Looks up an advancement by its namespaced ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Advancement> {
        self.by_id.get(id).map(|&idx| &self.advancements[idx])
    }

    /// Returns all registered advancements.
    #[must_use]
    pub fn all(&self) -> &[Advancement] {
        &self.advancements
    }

    /// Returns the root advancement for a given category, if any.
    #[must_use]
    pub fn root_for_category(&self, cat: AdvancementCategory) -> Option<&Advancement> {
        self.advancements
            .iter()
            .find(|a| a.category == cat && a.is_root())
    }

    /// Returns all direct child advancements of a given parent ID.
    #[must_use]
    pub fn children_of<'a>(&'a self, parent_id: &str) -> Vec<&'a Advancement> {
        self.advancements
            .iter()
            .filter(|a| a.parent == Some(parent_id))
            .collect()
    }

    /// Builds the standard pre-configured survival advancements registry.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn standard() -> Self {
        let mut reg = Self::new();

        // -------------------------------------------------------------
        // Story Progression Tree
        // -------------------------------------------------------------
        reg.register(Advancement {
            id: "telos:story/root",
            parent: None,
            category: AdvancementCategory::Story,
            title: "Telos",
            description: "The heart and story of the world",
            icon_item: crate::inventory::ITEM_CRAFTING_TABLE,
            frame: AdvancementFrame::Task,
            display_toast: false,
            announce_chat: false,
            x: 0.0,
            y: 0.0,
            criteria: &[("auto", AdvancementCriterion::Arbitrary("auto"))],
            requirements: &[],
        });

        reg.register(Advancement {
            id: "telos:story/mine_wood",
            parent: Some("telos:story/root"),
            category: AdvancementCategory::Story,
            title: "Getting Wood",
            description: "Attack a tree until a block of wood pops out",
            icon_item: 5, // Oak Log
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 1.5,
            y: 0.0,
            criteria: &[
                (
                    "oak_log",
                    AdvancementCriterion::InventoryChanged {
                        item: 5,
                        min_count: 1,
                    },
                ),
                (
                    "birch_log",
                    AdvancementCriterion::InventoryChanged {
                        item: 10,
                        min_count: 1,
                    },
                ),
                (
                    "spruce_log",
                    AdvancementCriterion::InventoryChanged {
                        item: 11,
                        min_count: 1,
                    },
                ),
            ],
            requirements: &[&["oak_log", "birch_log", "spruce_log"]],
        });

        reg.register(Advancement {
            id: "telos:story/craft_workbench",
            parent: Some("telos:story/mine_wood"),
            category: AdvancementCategory::Story,
            title: "Benchmarking",
            description: "Craft a workbench with four wooden planks",
            icon_item: crate::inventory::ITEM_CRAFTING_TABLE,
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 3.0,
            y: 0.0,
            criteria: &[
                (
                    "crafting_table",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_CRAFTING_TABLE,
                        min_count: 1,
                    },
                ),
                (
                    "craft_table",
                    AdvancementCriterion::ItemCrafted {
                        item: crate::inventory::ITEM_CRAFTING_TABLE,
                    },
                ),
            ],
            requirements: &[&["crafting_table", "craft_table"]],
        });

        reg.register(Advancement {
            id: "telos:story/craft_pickaxe",
            parent: Some("telos:story/craft_workbench"),
            category: AdvancementCategory::Story,
            title: "Time to Mine!",
            description: "Use planks and sticks to make a pickaxe",
            icon_item: crate::inventory::ITEM_WOODEN_PICKAXE,
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 4.5,
            y: 0.0,
            criteria: &[
                (
                    "wooden_pickaxe",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_WOODEN_PICKAXE,
                        min_count: 1,
                    },
                ),
                (
                    "stone_pickaxe",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_STONE_PICKAXE,
                        min_count: 1,
                    },
                ),
                (
                    "iron_pickaxe",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_IRON_PICKAXE,
                        min_count: 1,
                    },
                ),
            ],
            requirements: &[&["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"]],
        });

        reg.register(Advancement {
            id: "telos:story/mine_stone",
            parent: Some("telos:story/craft_pickaxe"),
            category: AdvancementCategory::Story,
            title: "Stone Age",
            description: "Mine stone with your new pickaxe",
            icon_item: 4, // Cobblestone
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 6.0,
            y: 0.0,
            criteria: &[
                (
                    "cobblestone",
                    AdvancementCriterion::InventoryChanged {
                        item: 4,
                        min_count: 1,
                    },
                ),
                (
                    "break_stone",
                    AdvancementCriterion::BlockBroken { block: 1 },
                ),
            ],
            requirements: &[&["cobblestone", "break_stone"]],
        });

        reg.register(Advancement {
            id: "telos:story/craft_furnace",
            parent: Some("telos:story/mine_stone"),
            category: AdvancementCategory::Story,
            title: "Hot Topic",
            description: "Construct a furnace out of eight cobblestone blocks",
            icon_item: crate::inventory::ITEM_FURNACE,
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 7.5,
            y: -0.8,
            criteria: &[
                (
                    "furnace",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_FURNACE,
                        min_count: 1,
                    },
                ),
                (
                    "craft_furnace",
                    AdvancementCriterion::ItemCrafted {
                        item: crate::inventory::ITEM_FURNACE,
                    },
                ),
            ],
            requirements: &[&["furnace", "craft_furnace"]],
        });

        reg.register(Advancement {
            id: "telos:story/acquire_iron",
            parent: Some("telos:story/craft_furnace"),
            category: AdvancementCategory::Story,
            title: "Acquire Hardware",
            description: "Smelt an iron ingot in your furnace",
            icon_item: 52, // Iron Ingot
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 9.0,
            y: -0.8,
            criteria: &[(
                "iron_ingot",
                AdvancementCriterion::InventoryChanged {
                    item: 52,
                    min_count: 1,
                },
            )],
            requirements: &[],
        });

        reg.register(Advancement {
            id: "telos:story/suit_up",
            parent: Some("telos:story/craft_workbench"),
            category: AdvancementCategory::Story,
            title: "Suit Up",
            description: "Protect yourself with a piece of armor",
            icon_item: 17, // Iron Chestplate
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 4.5,
            y: 1.2,
            criteria: &[
                (
                    "helmet",
                    AdvancementCriterion::InventoryChanged {
                        item: 16,
                        min_count: 1,
                    },
                ),
                (
                    "chestplate",
                    AdvancementCriterion::InventoryChanged {
                        item: 17,
                        min_count: 1,
                    },
                ),
                (
                    "leggings",
                    AdvancementCriterion::InventoryChanged {
                        item: 18,
                        min_count: 1,
                    },
                ),
                (
                    "boots",
                    AdvancementCriterion::InventoryChanged {
                        item: 19,
                        min_count: 1,
                    },
                ),
                (
                    "leather_chest",
                    AdvancementCriterion::InventoryChanged {
                        item: crate::inventory::ITEM_LEATHER_CHESTPLATE,
                        min_count: 1,
                    },
                ),
            ],
            requirements: &[&["helmet", "chestplate", "leggings", "boots", "leather_chest"]],
        });

        // -------------------------------------------------------------
        // Adventure Progression Tree
        // -------------------------------------------------------------
        reg.register(Advancement {
            id: "telos:adventure/root",
            parent: None,
            category: AdvancementCategory::Adventure,
            title: "Adventure",
            description: "Adventure, exploration and combat",
            icon_item: crate::inventory::ITEM_IRON_SWORD,
            frame: AdvancementFrame::Task,
            display_toast: false,
            announce_chat: false,
            x: 0.0,
            y: 0.0,
            criteria: &[("auto", AdvancementCriterion::Arbitrary("auto"))],
            requirements: &[],
        });

        reg.register(Advancement {
            id: "telos:adventure/kill_mob",
            parent: Some("telos:adventure/root"),
            category: AdvancementCategory::Adventure,
            title: "Monster Hunter",
            description: "Attack and destroy a dangerous monster",
            icon_item: crate::inventory::ITEM_IRON_SWORD,
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 1.5,
            y: 0.0,
            criteria: &[(
                "kill_zombie",
                AdvancementCriterion::EntityKilled {
                    entity_type: 1, // Zombie
                },
            )],
            requirements: &[],
        });

        reg.register(Advancement {
            id: "telos:adventure/shoot_arrow",
            parent: Some("telos:adventure/kill_mob"),
            category: AdvancementCategory::Adventure,
            title: "Take Aim",
            description: "Shoot a target with a bow and arrow",
            icon_item: crate::inventory::ITEM_BOW,
            frame: AdvancementFrame::Task,
            display_toast: true,
            announce_chat: true,
            x: 3.0,
            y: 0.0,
            criteria: &[(
                "hit",
                AdvancementCriterion::ArcheryHit { min_distance: 0.0 },
            )],
            requirements: &[],
        });

        reg.register(Advancement {
            id: "telos:adventure/sniper_duel",
            parent: Some("telos:adventure/shoot_arrow"),
            category: AdvancementCategory::Adventure,
            title: "Sniper Duel",
            description: "Hit a monster with an arrow from more than 25 meters away",
            icon_item: crate::inventory::ITEM_ARROW,
            frame: AdvancementFrame::Challenge,
            display_toast: true,
            announce_chat: true,
            x: 4.5,
            y: 0.0,
            criteria: &[(
                "distance_hit",
                AdvancementCriterion::ArcheryHit { min_distance: 25.0 },
            )],
            requirements: &[],
        });

        reg
    }
}

/// Persistent player advancement progress state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerAdvancements {
    /// Completed advancements mapped to completion unix timestamp (seconds).
    pub completed: BTreeMap<String, u64>,
    /// Fulfilled criterion names mapped per advancement ID.
    pub criteria_progress: BTreeMap<String, BTreeSet<String>>,
}

impl PlayerAdvancements {
    /// Creates empty advancement progress.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the advancement with `id` has been completed.
    #[must_use]
    pub fn is_completed(&self, id: &str) -> bool {
        self.completed.contains_key(id)
    }

    /// Grants an advancement directly with a completion timestamp.
    /// Returns true if it was newly granted.
    pub fn grant(&mut self, id: &str, timestamp: u64) -> bool {
        if self.completed.contains_key(id) {
            return false;
        }
        self.completed.insert(id.to_string(), timestamp);
        true
    }

    /// Revokes an advancement.
    /// Returns true if it was previously completed.
    pub fn revoke(&mut self, id: &str) -> bool {
        let removed = self.completed.remove(id).is_some();
        self.criteria_progress.remove(id);
        removed
    }

    /// Evaluates a gameplay trigger against all pending advancements in the registry.
    /// Returns a list of advancement IDs that were newly unlocked by this trigger.
    pub fn evaluate_trigger(
        &mut self,
        registry: &AdvancementRegistry,
        trigger: &CriterionTrigger,
        timestamp: u64,
    ) -> Vec<String> {
        let mut newly_completed = Vec::new();

        for adv in registry.all() {
            // Skip already completed advancements
            if self.is_completed(adv.id) {
                continue;
            }

            // Verify parent dependency is satisfied (if any)
            if let Some(parent_id) = adv.parent
                && !self.is_completed(parent_id)
            {
                continue;
            }

            let mut progress_changed = false;
            let set = self
                .criteria_progress
                .entry(adv.id.to_string())
                .or_default();

            for (crit_id, criterion) in adv.criteria {
                if !set.contains(*crit_id) && criterion.matches(trigger) {
                    set.insert((*crit_id).to_string());
                    progress_changed = true;
                }
            }

            if progress_changed && self.check_advancement_complete(adv) {
                self.completed.insert(adv.id.to_string(), timestamp);
                newly_completed.push(adv.id.to_string());
            }
        }

        newly_completed
    }

    /// Checks if all requirements of an advancement are met.
    #[must_use]
    pub fn check_advancement_complete(&self, adv: &Advancement) -> bool {
        let Some(set) = self.criteria_progress.get(adv.id) else {
            return false;
        };

        if adv.requirements.is_empty() {
            // Default: all criteria must be completed
            adv.criteria.iter().all(|(id, _)| set.contains(*id))
        } else {
            // Custom requirements: AND of OR groups
            adv.requirements
                .iter()
                .all(|group| group.iter().any(|req| set.contains(*req)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advancement_registry_structure() {
        let reg = AdvancementRegistry::standard();
        assert!(reg.all().len() >= 10);

        let root = reg.get("telos:story/root").expect("story root exists");
        assert!(root.is_root());
        assert_eq!(root.category, AdvancementCategory::Story);

        let wood = reg.get("telos:story/mine_wood").expect("mine wood exists");
        assert_eq!(wood.parent, Some("telos:story/root"));

        let children = reg.children_of("telos:story/root");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "telos:story/mine_wood");
    }

    #[test]
    fn test_advancement_trigger_evaluation_and_grant() {
        let reg = AdvancementRegistry::standard();
        let mut player = PlayerAdvancements::new();

        // 1. Initial state
        assert!(!player.is_completed("telos:story/root"));

        // 2. Grant root
        assert!(player.grant("telos:story/root", 100));
        assert!(player.is_completed("telos:story/root"));

        // 3. Trigger wood pickup
        let unlocked = player.evaluate_trigger(
            &reg,
            &CriterionTrigger::Inventory {
                item: 5, // Oak log
                count: 1,
            },
            105,
        );
        assert_eq!(unlocked, vec!["telos:story/mine_wood"]);
        assert!(player.is_completed("telos:story/mine_wood"));

        // 4. Trigger craft workbench
        let unlocked = player.evaluate_trigger(
            &reg,
            &CriterionTrigger::Craft {
                item: crate::inventory::ITEM_CRAFTING_TABLE,
            },
            110,
        );
        assert_eq!(unlocked, vec!["telos:story/craft_workbench"]);
        assert!(player.is_completed("telos:story/craft_workbench"));
    }

    #[test]
    fn test_archery_distance_criterion() {
        let reg = AdvancementRegistry::standard();
        let mut player = PlayerAdvancements::new();
        player.grant("telos:adventure/root", 1);
        player.grant("telos:adventure/kill_mob", 2);
        player.grant("telos:adventure/shoot_arrow", 3);

        // Near shot (10m): sniper duel shouldn't trigger
        let unlocked =
            player.evaluate_trigger(&reg, &CriterionTrigger::ArcheryHit { distance: 10.0 }, 4);
        assert!(unlocked.is_empty());

        // Far shot (30m): sniper duel triggers!
        let unlocked =
            player.evaluate_trigger(&reg, &CriterionTrigger::ArcheryHit { distance: 30.0 }, 5);
        assert_eq!(unlocked, vec!["telos:adventure/sniper_duel"]);
        assert!(player.is_completed("telos:adventure/sniper_duel"));
    }

    #[test]
    fn test_player_advancements_serde_roundtrip() {
        let mut player = PlayerAdvancements::new();
        player.grant("telos:story/root", 1000);
        player.grant("telos:story/mine_wood", 1050);
        player
            .criteria_progress
            .entry("telos:story/mine_wood".to_string())
            .or_default()
            .insert("oak_log".to_string());

        let json = serde_json::to_string(&player).expect("serialize");
        let decoded: PlayerAdvancements = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(player, decoded);
    }
}
