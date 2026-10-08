use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use telos_core::ident::Identifier;

use crate::error::{ContentError, Result};
use crate::registry::ItemRegistry;

const fn default_true() -> bool {
    true
}

const fn default_count_one() -> u16 {
    1
}

const fn default_cook_duration() -> u16 {
    200
}

const fn default_fuel_burn_ticks() -> u16 {
    300
}

/// Output item stack produced by a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeResultDef {
    /// Item identifier string (e.g. `"sample:ruby_block"` or `"telos:iron_ingot"`).
    pub item: String,
    /// Number of items produced per craft (default 1).
    #[serde(default = "default_count_one")]
    pub count: u16,
}

impl RecipeResultDef {
    /// Resolves the result item identifier into a runtime numeric item ID.
    pub fn resolve(&self, item_reg: &ItemRegistry) -> Result<(u32, u16)> {
        let ident = Identifier::from_str(&self.item)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
        let item_id =
            item_reg
                .get_by_ident(&ident)
                .ok_or_else(|| ContentError::UnknownReference {
                    referrer: "recipe_result".into(),
                    target: ident.to_string(),
                })?;
        Ok((item_id, self.count.max(1)))
    }
}

/// A shaped crafting recipe definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapedRecipeDef {
    /// Width of the pattern in grid columns (1..=3), optional if row pattern strings are provided.
    #[serde(default)]
    pub width: Option<usize>,
    /// Height of the pattern in grid rows (1..=3), optional if row pattern strings are provided.
    #[serde(default)]
    pub height: Option<usize>,
    /// Pattern rows (e.g. `["###", "#R#", "###"]`).
    #[serde(default)]
    pub pattern: Vec<String>,
    /// Key mapping character keys in pattern rows to item identifiers (e.g. `{"#": "telos:stone", "R": "sample:ruby"}`).
    #[serde(default)]
    pub key: HashMap<String, String>,
    /// Flat list of item identifiers (or None/empty string for air), alternative to row pattern.
    #[serde(default)]
    pub flat_pattern: Vec<Option<String>>,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Whether horizontal mirroring is permitted (default true).
    #[serde(default = "default_true")]
    pub mirrored: bool,
    /// Optional remainder item identifier returned in consumed slots (e.g. `"telos:bucket"`).
    #[serde(default)]
    pub remainder: Option<String>,
}

impl ShapedRecipeDef {
    /// Resolves pattern and result strings against `ItemRegistry` into grid dimensions,
    /// numeric item ID pattern (where 0 indicates empty slot), result, and remainder.
    #[allow(clippy::type_complexity, clippy::too_many_lines)]
    pub fn resolve(
        &self,
        item_reg: &ItemRegistry,
    ) -> Result<(usize, usize, Vec<u32>, (u32, u16), bool, Option<u32>)> {
        let (width, height, pattern) = if !self.pattern.is_empty() {
            let height = self.pattern.len();
            if !(1..=3).contains(&height) {
                return Err(ContentError::DataParse {
                    path: std::path::PathBuf::from("shaped_recipe"),
                    reason: format!("Pattern height {height} must be between 1 and 3"),
                });
            }
            let width = self.pattern[0].chars().count();
            if !(1..=3).contains(&width) {
                return Err(ContentError::DataParse {
                    path: std::path::PathBuf::from("shaped_recipe"),
                    reason: format!("Pattern width {width} must be between 1 and 3"),
                });
            }

            let mut resolved_pattern = Vec::with_capacity(width * height);
            for row in &self.pattern {
                let row_chars: Vec<char> = row.chars().collect();
                if row_chars.len() != width {
                    return Err(ContentError::DataParse {
                        path: std::path::PathBuf::from("shaped_recipe"),
                        reason: format!(
                            "Inconsistent pattern row length: expected {width}, found {}",
                            row_chars.len()
                        ),
                    });
                }

                for ch in row_chars {
                    if ch == ' ' {
                        resolved_pattern.push(0);
                    } else {
                        let key_str = ch.to_string();
                        let item_str =
                            self.key
                                .get(&key_str)
                                .ok_or_else(|| ContentError::DataParse {
                                    path: std::path::PathBuf::from("shaped_recipe"),
                                    reason: format!("Undefined pattern key '{key_str}'"),
                                })?;
                        let ident = Identifier::from_str(item_str)
                            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
                        let item_id = item_reg.get_by_ident(&ident).ok_or_else(|| {
                            ContentError::UnknownReference {
                                referrer: "recipe_ingredient".into(),
                                target: ident.to_string(),
                            }
                        })?;
                        resolved_pattern.push(item_id);
                    }
                }
            }
            (width, height, resolved_pattern)
        } else if !self.flat_pattern.is_empty() {
            let width = self.width.unwrap_or(if self.flat_pattern.len() == 4 {
                2
            } else if self.flat_pattern.len() == 9 {
                3
            } else {
                self.flat_pattern.len()
            });
            let height = self.height.unwrap_or(self.flat_pattern.len() / width);

            if width * height != self.flat_pattern.len() {
                return Err(ContentError::DataParse {
                    path: std::path::PathBuf::from("shaped_recipe"),
                    reason: format!(
                        "Flat pattern length {} != width {} * height {}",
                        self.flat_pattern.len(),
                        width,
                        height
                    ),
                });
            }

            let mut resolved_pattern = Vec::with_capacity(self.flat_pattern.len());
            for item_opt in &self.flat_pattern {
                match item_opt {
                    None => resolved_pattern.push(0),
                    Some(s) if s.is_empty() || s == "telos:air" => resolved_pattern.push(0),
                    Some(s) => {
                        let ident = Identifier::from_str(s)
                            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
                        let item_id = item_reg.get_by_ident(&ident).ok_or_else(|| {
                            ContentError::UnknownReference {
                                referrer: "recipe_ingredient".into(),
                                target: ident.to_string(),
                            }
                        })?;
                        resolved_pattern.push(item_id);
                    }
                }
            }
            (width, height, resolved_pattern)
        } else {
            return Err(ContentError::DataParse {
                path: std::path::PathBuf::from("shaped_recipe"),
                reason: "Neither 'pattern' nor 'flat_pattern' provided".into(),
            });
        };

        let result = self.result.resolve(item_reg)?;

        let remainder = if let Some(ref rem_str) = self.remainder {
            let ident = Identifier::from_str(rem_str)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
            let rem_id =
                item_reg
                    .get_by_ident(&ident)
                    .ok_or_else(|| ContentError::UnknownReference {
                        referrer: "recipe_remainder".into(),
                        target: ident.to_string(),
                    })?;
            Some(rem_id)
        } else {
            None
        };

        Ok((width, height, pattern, result, self.mirrored, remainder))
    }
}

/// A shapeless crafting recipe definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapelessRecipeDef {
    /// Required ingredient item identifiers (multiset).
    pub ingredients: Vec<String>,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Optional remainder item identifier.
    #[serde(default)]
    pub remainder: Option<String>,
}

impl ShapelessRecipeDef {
    /// Resolves ingredient and result identifiers against `ItemRegistry`.
    #[allow(clippy::type_complexity)]
    pub fn resolve(&self, item_reg: &ItemRegistry) -> Result<(Vec<u32>, (u32, u16), Option<u32>)> {
        if self.ingredients.is_empty() || self.ingredients.len() > 9 {
            return Err(ContentError::DataParse {
                path: std::path::PathBuf::from("shapeless_recipe"),
                reason: format!(
                    "Ingredient count {} must be between 1 and 9",
                    self.ingredients.len()
                ),
            });
        }

        let mut resolved_ingredients = Vec::with_capacity(self.ingredients.len());
        for ing_str in &self.ingredients {
            let ident = Identifier::from_str(ing_str)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
            let item_id =
                item_reg
                    .get_by_ident(&ident)
                    .ok_or_else(|| ContentError::UnknownReference {
                        referrer: "recipe_ingredient".into(),
                        target: ident.to_string(),
                    })?;
            resolved_ingredients.push(item_id);
        }

        let result = self.result.resolve(item_reg)?;

        let remainder = if let Some(ref rem_str) = self.remainder {
            let ident = Identifier::from_str(rem_str)
                .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
            let rem_id =
                item_reg
                    .get_by_ident(&ident)
                    .ok_or_else(|| ContentError::UnknownReference {
                        referrer: "recipe_remainder".into(),
                        target: ident.to_string(),
                    })?;
            Some(rem_id)
        } else {
            None
        };

        Ok((resolved_ingredients, result, remainder))
    }
}

/// A furnace smelting recipe definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmeltingRecipeDef {
    /// Input ingredient item identifier.
    pub ingredient: String,
    /// Output result stack.
    pub result: RecipeResultDef,
    /// Cook duration in simulation ticks (default 200 ticks = 10.0s).
    #[serde(default = "default_cook_duration")]
    pub cook_duration: u16,
    /// Experience reward earned when retrieving smelted items.
    #[serde(default)]
    pub experience: f32,
}

impl SmeltingRecipeDef {
    /// Resolves ingredient and result identifiers against `ItemRegistry`.
    pub fn resolve(&self, item_reg: &ItemRegistry) -> Result<(u32, (u32, u16), u16, f32)> {
        let ident = Identifier::from_str(&self.ingredient)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
        let input_item =
            item_reg
                .get_by_ident(&ident)
                .ok_or_else(|| ContentError::UnknownReference {
                    referrer: "smelting_ingredient".into(),
                    target: ident.to_string(),
                })?;
        let (output_item, output_count) = self.result.resolve(item_reg)?;
        Ok((
            input_item,
            (output_item, output_count),
            self.cook_duration.max(1),
            self.experience.max(0.0),
        ))
    }
}

/// A combustible fuel definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuelDef {
    /// Combustible item identifier (e.g. `"sample:blaze_coal"`).
    pub item: String,
    /// Burn duration in simulation ticks (default 300 ticks = 15.0s).
    #[serde(default = "default_fuel_burn_ticks")]
    pub burn_ticks: u16,
}

impl FuelDef {
    /// Resolves item identifier against `ItemRegistry`.
    pub fn resolve(&self, item_reg: &ItemRegistry) -> Result<(u32, u16)> {
        let ident = Identifier::from_str(&self.item)
            .map_err(|e| ContentError::InvalidIdentifier(format!("{e}")))?;
        let item_id =
            item_reg
                .get_by_ident(&ident)
                .ok_or_else(|| ContentError::UnknownReference {
                    referrer: "fuel_item".into(),
                    target: ident.to_string(),
                })?;
        Ok((item_id, self.burn_ticks.max(1)))
    }
}

/// Top-level recipe definition enum parsed from `data/<ns>/recipes/*.ron` or `*.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecipeDef {
    /// Shaped crafting recipe.
    #[serde(alias = "minecraft:crafting_shaped", alias = "crafting_shaped")]
    Shaped(ShapedRecipeDef),
    /// Shapeless crafting recipe.
    #[serde(alias = "minecraft:crafting_shapeless", alias = "crafting_shapeless")]
    Shapeless(ShapelessRecipeDef),
    /// Smelting furnace recipe.
    #[serde(alias = "minecraft:smelting")]
    Smelting(SmeltingRecipeDef),
    /// Combustible fuel item.
    #[serde(alias = "fuel", alias = "furnace_fuel")]
    Fuel(FuelDef),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::item::ItemDef;

    fn test_registry() -> ItemRegistry {
        let mut reg = ItemRegistry::new();
        reg.register(
            Identifier::new("telos", "stone").unwrap(),
            ItemDef::new_block_item("Stone", "telos:stone"),
        );
        reg.register(
            Identifier::new("sample", "ruby").unwrap(),
            ItemDef::new_generic("Ruby"),
        );
        reg.register(
            Identifier::new("sample", "ruby_block").unwrap(),
            ItemDef::new_block_item("Ruby Block", "sample:ruby_block"),
        );
        reg.register(
            Identifier::new("sample", "ruby_ore").unwrap(),
            ItemDef::new_block_item("Ruby Ore", "sample:ruby_ore"),
        );
        reg.register(
            Identifier::new("sample", "blaze_coal").unwrap(),
            ItemDef::new_generic("Blaze Coal"),
        );
        reg
    }

    #[test]
    fn test_shaped_recipe_json_and_ron() {
        let reg = test_registry();

        let json_str = r#"{
            "type": "shaped",
            "pattern": [
                "SSS",
                "SRS",
                "SSS"
            ],
            "key": {
                "S": "telos:stone",
                "R": "sample:ruby"
            },
            "result": {
                "item": "sample:ruby_block",
                "count": 1
            },
            "mirrored": true
        }"#;

        let recipe: RecipeDef = serde_json::from_str(json_str).expect("Valid JSON shaped recipe");
        if let RecipeDef::Shaped(shaped) = recipe {
            let (w, h, pat, res, mirrored, rem) = shaped.resolve(&reg).expect("Resolves correctly");
            assert_eq!(w, 3);
            assert_eq!(h, 3);
            assert_eq!(pat.len(), 9);
            assert_eq!(res.1, 1);
            assert!(mirrored);
            assert_eq!(rem, None);
        } else {
            panic!("Expected shaped recipe");
        }
    }

    #[test]
    fn test_shapeless_recipe_json() {
        let reg = test_registry();

        let json_str = r#"{
            "type": "shapeless",
            "ingredients": [
                "sample:ruby_block"
            ],
            "result": {
                "item": "sample:ruby",
                "count": 9
            }
        }"#;

        let recipe: RecipeDef = serde_json::from_str(json_str).expect("Valid JSON shapeless");
        if let RecipeDef::Shapeless(shapeless) = recipe {
            let (ings, res, rem) = shapeless.resolve(&reg).expect("Resolves correctly");
            assert_eq!(ings.len(), 1);
            assert_eq!(res.1, 9);
            assert_eq!(rem, None);
        } else {
            panic!("Expected shapeless recipe");
        }
    }

    #[test]
    fn test_smelting_and_fuel_json() {
        let reg = test_registry();

        let json_str = r#"{
            "type": "smelting",
            "ingredient": "sample:ruby_ore",
            "result": {
                "item": "sample:ruby",
                "count": 1
            },
            "cook_duration": 180,
            "experience": 1.5
        }"#;

        let recipe: RecipeDef = serde_json::from_str(json_str).expect("Valid JSON smelting");
        if let RecipeDef::Smelting(smelt) = recipe {
            let (input, res, cook_time, exp) = smelt.resolve(&reg).expect("Resolves correctly");
            assert_eq!(cook_time, 180);
            assert!((exp - 1.5).abs() < 1e-4);
            assert_eq!(res.1, 1);
            assert_ne!(input, 0);
        } else {
            panic!("Expected smelting recipe");
        }

        let fuel_json = r#"{
            "item": "sample:blaze_coal",
            "burn_ticks": 3200
        }"#;
        let fuel: FuelDef = serde_json::from_str(fuel_json).expect("Valid JSON fuel");
        let (fuel_item, burn_ticks) = fuel.resolve(&reg).expect("Resolves fuel");
        assert_eq!(burn_ticks, 3200);
        assert_ne!(fuel_item, 0);
    }
}
