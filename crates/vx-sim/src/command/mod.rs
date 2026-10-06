//! Brigadier-style command subsystem: lexing, syntax trees, target selectors, and execution.

pub mod builtin;
pub mod coords;
pub mod reader;
pub mod selector;
pub mod tree;

pub use builtin::register_builtins;
pub use coords::{CoordinateArg, Vec3Arg};
pub use reader::{CommandSyntaxError, StringReader};
pub use selector::{DistanceRange, EntitySelector, SelectorFilters, SelectorType};
pub use tree::{
    ArgumentType, CommandContext, CommandDispatcher, CommandNode, CommandNodeType, CommandOutput,
    CommandSuggestions, ParsedValue, Suggestion,
};
