//! Target selector parsing for `@p`, `@a`, `@r`, `@s`, `@e`, and player names.

use crate::command::reader::{CommandSyntaxError, StringReader};

/// Selector base target type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectorType {
    /// `@p` — Nearest player to execution position.
    NearestPlayer,
    /// `@a` — All connected players.
    AllPlayers,
    /// `@r` — Random player.
    RandomPlayer,
    /// `@e` — All entities in the world (mobs, players, items).
    AllEntities,
    /// `@s` — The entity currently executing the command.
    SelfEntity,
    /// An explicit player username (e.g. `Alex` or `Steve`).
    Named(String),
}

/// A distance filter range (e.g. `..10`, `10..20`, `5..`, or `10`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceRange {
    /// Minimum distance (inclusive).
    pub min: Option<f32>,
    /// Maximum distance (inclusive).
    pub max: Option<f32>,
}

impl DistanceRange {
    /// Returns true if a given distance falls within this range.
    #[must_use]
    pub fn matches(&self, d: f32) -> bool {
        if matches!(self.min, Some(min) if d < min) {
            return false;
        }
        if matches!(self.max, Some(max) if d > max) {
            return false;
        }
        true
    }
}

/// Selector filter constraints specified inside brackets `[...]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SelectorFilters {
    /// Entity type filter (e.g. `zombie`, `cow`).
    pub entity_type: Option<String>,
    /// Whether entity type match is inverted (`!`).
    pub type_negated: bool,
    /// Distance filter range.
    pub distance: Option<DistanceRange>,
    /// Maximum number of entities to select (`limit=N`).
    pub limit: Option<usize>,
    /// Sorting order (`sort=nearest|furthest|random|arbitrary`).
    pub sort: Option<String>,
}

/// A fully parsed entity target selector.
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySelector {
    /// Base selector type.
    pub selector_type: SelectorType,
    /// Optional filter arguments.
    pub filters: SelectorFilters,
}

impl EntitySelector {
    /// Creates a selector for a specific player name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            selector_type: SelectorType::Named(name.into()),
            filters: SelectorFilters::default(),
        }
    }

    /// Parses an entity selector or player name from the reader.
    pub fn parse(reader: &mut StringReader) -> Result<Self, CommandSyntaxError> {
        reader.skip_whitespace();
        let start = reader.cursor();

        if let Some('@') = reader.peek() {
            reader.read_char();
            let kind_char = reader
                .read_char()
                .ok_or(CommandSyntaxError::UnexpectedEof {
                    cursor: reader.cursor(),
                })?;

            let selector_type = match kind_char {
                'p' => SelectorType::NearestPlayer,
                'a' => SelectorType::AllPlayers,
                'r' => SelectorType::RandomPlayer,
                'e' => SelectorType::AllEntities,
                's' => SelectorType::SelfEntity,
                _ => {
                    return Err(CommandSyntaxError::Custom {
                        cursor: start,
                        message: format!(
                            "Unknown selector '@{kind_char}' (expected p, a, r, e, or s)"
                        ),
                    });
                }
            };

            let mut filters = SelectorFilters::default();

            // Default limit for @p and @r is 1 unless overridden
            if matches!(
                selector_type,
                SelectorType::NearestPlayer | SelectorType::RandomPlayer
            ) {
                filters.limit = Some(1);
            }

            // Check if brackets follow immediately
            if let Some('[') = reader.peek() {
                reader.read_char();
                filters = Self::parse_filters(reader, filters)?;
            }

            Ok(Self {
                selector_type,
                filters,
            })
        } else {
            // Unquoted player username token
            let name = reader.read_unquoted_string();
            if name.is_empty() {
                return Err(CommandSyntaxError::Custom {
                    cursor: start,
                    message: "Expected entity selector or player name".to_string(),
                });
            }
            Ok(Self {
                selector_type: SelectorType::Named(name.to_string()),
                filters: SelectorFilters::default(),
            })
        }
    }

    fn parse_filters(
        reader: &mut StringReader,
        mut filters: SelectorFilters,
    ) -> Result<SelectorFilters, CommandSyntaxError> {
        loop {
            reader.skip_whitespace();
            if let Some(']') = reader.peek() {
                reader.read_char();
                break;
            }

            let key_start = reader.cursor();
            // Read key until '='
            while let Some(c) = reader.peek() {
                if c == '=' || c == ']' || c.is_whitespace() {
                    break;
                }
                reader.read_char();
            }
            let key = reader.input()[key_start..reader.cursor()].trim();
            reader.skip_whitespace();
            reader.expect('=')?;
            reader.skip_whitespace();

            match key {
                "type" => {
                    let mut negated = false;
                    if let Some('!') = reader.peek() {
                        reader.read_char();
                        negated = true;
                    }
                    let val = reader.read_unquoted_string().to_string();
                    filters.entity_type = Some(val);
                    filters.type_negated = negated;
                }
                "limit" => {
                    let limit = reader.read_i32()?;
                    filters.limit = Some(limit.max(0) as usize);
                }
                "sort" => {
                    filters.sort = Some(reader.read_unquoted_string().to_string());
                }
                "distance" => {
                    let dist_start = reader.cursor();
                    // Distance can be: "N..M", "..M", "N..", "N"
                    let token = reader.read_unquoted_string();
                    if let Some((min_str, max_str)) = token.split_once("..") {
                        let min = if min_str.is_empty() {
                            None
                        } else {
                            Some(min_str.parse::<f32>().map_err(|_| {
                                CommandSyntaxError::InvalidFloat {
                                    cursor: dist_start,
                                    found: min_str.to_string(),
                                    reason: "invalid distance min".to_string(),
                                }
                            })?)
                        };
                        let max = if max_str.is_empty() {
                            None
                        } else {
                            Some(max_str.parse::<f32>().map_err(|_| {
                                CommandSyntaxError::InvalidFloat {
                                    cursor: dist_start,
                                    found: max_str.to_string(),
                                    reason: "invalid distance max".to_string(),
                                }
                            })?)
                        };
                        filters.distance = Some(DistanceRange { min, max });
                    } else {
                        let d =
                            token
                                .parse::<f32>()
                                .map_err(|_| CommandSyntaxError::InvalidFloat {
                                    cursor: dist_start,
                                    found: token.to_string(),
                                    reason: "invalid distance".to_string(),
                                })?;
                        filters.distance = Some(DistanceRange {
                            min: Some(d),
                            max: Some(d),
                        });
                    }
                }
                _ => {
                    // Unknown or unsupported filter parameter, consume value
                    let _ = reader.read_unquoted_string();
                }
            }

            reader.skip_whitespace();
            if let Some(',') = reader.peek() {
                reader.read_char();
            } else if let Some(']') = reader.peek() {
                reader.read_char();
                break;
            } else {
                return Err(CommandSyntaxError::Custom {
                    cursor: reader.cursor(),
                    message: "Expected ',' or ']' in selector filters".to_string(),
                });
            }
        }

        Ok(filters)
    }
}
