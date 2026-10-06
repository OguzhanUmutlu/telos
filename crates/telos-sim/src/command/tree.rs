//! Brigadier-style command tree, execution contexts, and auto-completion suggestion graph.

use crate::command::coords::Vec3Arg;
use crate::command::reader::{CommandSyntaxError, StringReader};
use crate::command::selector::EntitySelector;
use glam::{Vec2, Vec3};
use hashbrown::HashMap;
use std::sync::Arc;

/// A parsed argument value stored inside `CommandContext`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedValue {
    /// 32-bit signed integer.
    Int(i32),
    /// 32-bit float.
    Float(f32),
    /// Boolean flag.
    Bool(bool),
    /// String token or greedy message string.
    String(String),
    /// 3D coordinate argument vector.
    Vec3(Vec3Arg),
    /// Target selector or player username.
    Selector(EntitySelector),
    /// Namespaced resource identifier (`namespace:path`).
    Identifier(String),
}

/// Execution output returned by a command handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// Whether execution succeeded.
    pub success: bool,
    /// Human-readable feedback message or error explanation.
    pub message: String,
}

impl CommandOutput {
    /// Creates a successful command output.
    #[must_use]
    pub fn success(msg: impl Into<String>) -> Self {
        Self {
            success: true,
            message: msg.into(),
        }
    }

    /// Creates an error command output.
    #[must_use]
    pub fn failure(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            message: msg.into(),
        }
    }
}

/// Context provided to command execution handlers.
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Session/entity ID of the executor (`None` if console/server).
    pub executor_id: Option<u64>,
    /// Executor world-space position.
    pub executor_pos: Vec3,
    /// Executor view rotation (yaw, pitch in degrees).
    pub executor_rot: Vec2,
    /// Display name of the executor.
    pub executor_name: String,
    /// Parsed argument values mapped by argument name.
    pub args: HashMap<String, ParsedValue>,
}

impl CommandContext {
    /// Creates a default context for testing or console execution.
    #[must_use]
    pub fn console() -> Self {
        Self {
            executor_id: None,
            executor_pos: Vec3::ZERO,
            executor_rot: Vec2::ZERO,
            executor_name: "Server".to_string(),
            args: HashMap::new(),
        }
    }

    /// Retrieves an integer argument by name.
    #[must_use]
    pub fn get_int(&self, name: &str) -> Option<i32> {
        match self.args.get(name) {
            Some(ParsedValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// Retrieves a float argument by name.
    #[must_use]
    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.args.get(name) {
            Some(ParsedValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Retrieves a bool argument by name.
    #[must_use]
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.args.get(name) {
            Some(ParsedValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Retrieves a string argument by name.
    #[must_use]
    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.args.get(name) {
            Some(ParsedValue::String(s) | ParsedValue::Identifier(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Retrieves a namespaced identifier argument by name.
    #[must_use]
    pub fn get_identifier(&self, name: &str) -> Option<&str> {
        match self.args.get(name) {
            Some(ParsedValue::Identifier(s) | ParsedValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Retrieves a Vec3 argument by name.
    #[must_use]
    pub fn get_vec3(&self, name: &str) -> Option<Vec3Arg> {
        match self.args.get(name) {
            Some(ParsedValue::Vec3(v)) => Some(*v),
            _ => None,
        }
    }

    /// Retrieves an entity selector argument by name.
    #[must_use]
    pub fn get_selector(&self, name: &str) -> Option<&EntitySelector> {
        match self.args.get(name) {
            Some(ParsedValue::Selector(s)) => Some(s),
            _ => None,
        }
    }
}

/// Supported argument types for syntax nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentType {
    /// Single unquoted or quoted word.
    Word,
    /// Greedy string consuming all remaining input.
    GreedyString,
    /// 32-bit integer with optional bounds.
    Integer {
        /// Optional minimum value (inclusive).
        min: Option<i32>,
        /// Optional maximum value (inclusive).
        max: Option<i32>,
    },
    /// 32-bit float with optional bounds.
    Float {
        /// Optional minimum value (inclusive).
        min: Option<f32>,
        /// Optional maximum value (inclusive).
        max: Option<f32>,
    },
    /// Boolean flag (`true` or `false`).
    Bool,
    /// 3D coordinate vector (`X Y Z`).
    Vec3,
    /// Entity target selector (`@p`, `@a`, `@s`, `@e[...]`, player name).
    Entity,
    /// Namespaced identifier (`namespace:path`).
    Identifier,
}

impl ArgumentType {
    /// Parses this argument type from a `StringReader`.
    pub fn parse(&self, reader: &mut StringReader<'_>) -> Result<ParsedValue, CommandSyntaxError> {
        match self {
            Self::Word => {
                let s = reader.read_string()?;
                if s.is_empty() {
                    Err(CommandSyntaxError::UnexpectedEof {
                        cursor: reader.cursor(),
                    })
                } else {
                    Ok(ParsedValue::String(s))
                }
            }
            Self::GreedyString => {
                let remaining = reader.remaining().trim();
                reader.set_cursor(reader.total_len());
                Ok(ParsedValue::String(remaining.to_string()))
            }
            Self::Integer { min, max } => {
                let val = reader.read_i32()?;
                if matches!(min, Some(m) if val < *m) {
                    return Err(CommandSyntaxError::Custom {
                        cursor: reader.cursor(),
                        message: format!("Integer {val} must not be less than {}", min.unwrap()),
                    });
                }
                if matches!(max, Some(m) if val > *m) {
                    return Err(CommandSyntaxError::Custom {
                        cursor: reader.cursor(),
                        message: format!("Integer {val} must not be greater than {}", max.unwrap()),
                    });
                }
                Ok(ParsedValue::Int(val))
            }
            Self::Float { min, max } => {
                let val = reader.read_f32()?;
                if matches!(min, Some(m) if val < *m) {
                    return Err(CommandSyntaxError::Custom {
                        cursor: reader.cursor(),
                        message: format!("Float {val} must not be less than {}", min.unwrap()),
                    });
                }
                if matches!(max, Some(m) if val > *m) {
                    return Err(CommandSyntaxError::Custom {
                        cursor: reader.cursor(),
                        message: format!("Float {val} must not be greater than {}", max.unwrap()),
                    });
                }
                Ok(ParsedValue::Float(val))
            }
            Self::Bool => {
                let val = reader.read_bool()?;
                Ok(ParsedValue::Bool(val))
            }
            Self::Vec3 => {
                let vec = Vec3Arg::parse(reader)?;
                Ok(ParsedValue::Vec3(vec))
            }
            Self::Entity => {
                let sel = EntitySelector::parse(reader)?;
                Ok(ParsedValue::Selector(sel))
            }
            Self::Identifier => {
                let id = reader.read_unquoted_string().to_string();
                if id.is_empty() {
                    Err(CommandSyntaxError::UnexpectedEof {
                        cursor: reader.cursor(),
                    })
                } else {
                    Ok(ParsedValue::Identifier(id))
                }
            }
        }
    }
}

/// The kind and matching criteria of a `CommandNode`.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandNodeType {
    /// Root dispatcher node.
    Root,
    /// Exact literal match (e.g. `time`, `weather`, `set`).
    Literal {
        /// Literal text to match.
        literal: String,
    },
    /// Typed argument parameter.
    Argument {
        /// Parameter variable name.
        name: String,
        /// Type and parser for argument.
        arg_type: ArgumentType,
    },
}

/// Command execution closure type.
pub type CommandHandler = Arc<dyn Fn(&CommandContext) -> CommandOutput + Send + Sync>;

/// A node in the Brigadier-style syntax and execution graph.
#[derive(Clone)]
pub struct CommandNode {
    /// Node discriminator.
    pub node_type: CommandNodeType,
    /// Child command nodes.
    pub children: Vec<CommandNode>,
    /// Optional execution handler when parsing terminates at this node.
    pub handler: Option<CommandHandler>,
    /// Tooltip or description shown during auto-completion.
    pub tooltip: Option<String>,
}

impl CommandNode {
    /// Creates a literal command node.
    #[must_use]
    pub fn literal(literal: impl Into<String>) -> Self {
        Self {
            node_type: CommandNodeType::Literal {
                literal: literal.into(),
            },
            children: Vec::new(),
            handler: None,
            tooltip: None,
        }
    }

    /// Creates an argument command node.
    #[must_use]
    pub fn argument(name: impl Into<String>, arg_type: ArgumentType) -> Self {
        Self {
            node_type: CommandNodeType::Argument {
                name: name.into(),
                arg_type,
            },
            children: Vec::new(),
            handler: None,
            tooltip: None,
        }
    }

    /// Sets the execution handler for this node.
    #[must_use]
    pub fn executes<F>(mut self, handler: F) -> Self
    where
        F: Fn(&CommandContext) -> CommandOutput + Send + Sync + 'static,
    {
        self.handler = Some(Arc::new(handler));
        self
    }

    /// Attaches a child command node.
    #[must_use]
    pub fn then(mut self, child: CommandNode) -> Self {
        self.children.push(child);
        self
    }

    /// Sets a descriptive tooltip for auto-completion.
    #[must_use]
    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = Some(tooltip.into());
        self
    }
}

/// A candidate suggestion for auto-completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    /// Suggested replacement text.
    pub value: String,
    /// Optional description or parameter tooltip.
    pub tooltip: Option<String>,
}

/// Auto-completion suggestions response for a given cursor position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSuggestions {
    /// Start character index in the input string to replace.
    pub start: usize,
    /// Length of the range to replace.
    pub length: usize,
    /// List of candidate suggestions.
    pub candidates: Vec<Suggestion>,
}

/// Master command dispatcher and syntax tree executor.
#[derive(Clone)]
pub struct CommandDispatcher {
    root: CommandNode,
}

impl Default for CommandDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandDispatcher {
    /// Creates a new empty `CommandDispatcher`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: CommandNode {
                node_type: CommandNodeType::Root,
                children: Vec::new(),
                handler: None,
                tooltip: None,
            },
        }
    }

    /// Registers a command node under the root.
    pub fn register(&mut self, node: CommandNode) {
        self.root.children.push(node);
    }

    /// Returns a reference to the root node.
    #[must_use]
    pub const fn root(&self) -> &CommandNode {
        &self.root
    }

    /// Parses and executes a command string against the registered syntax tree.
    pub fn execute(&self, input: &str, ctx: &mut CommandContext) -> CommandOutput {
        let clean_input = input.trim();
        let stripped = clean_input.strip_prefix('/').unwrap_or(clean_input);

        if stripped.is_empty() {
            return CommandOutput::failure("Empty command");
        }

        let mut reader = StringReader::new(stripped);
        match Self::parse_node(&self.root, &mut reader, ctx) {
            Ok(Some(handler)) => handler(ctx),
            Ok(None) => CommandOutput::failure("Incomplete command"),
            Err(err) => CommandOutput::failure(err.to_string()),
        }
    }

    fn parse_node(
        current: &CommandNode,
        reader: &mut StringReader<'_>,
        ctx: &mut CommandContext,
    ) -> Result<Option<CommandHandler>, CommandSyntaxError> {
        reader.skip_whitespace();

        if !reader.can_read() {
            return Ok(current.handler.clone());
        }

        for child in &current.children {
            let bookmark = reader.cursor();
            let mut matched = false;

            match &child.node_type {
                CommandNodeType::Literal { literal } => {
                    let word = reader.read_unquoted_string();
                    if word.eq_ignore_ascii_case(literal) {
                        matched = true;
                    }
                }
                CommandNodeType::Argument { name, arg_type } => {
                    if let Ok(val) = arg_type.parse(reader) {
                        ctx.args.insert(name.clone(), val);
                        matched = true;
                    }
                }
                CommandNodeType::Root => {}
            }

            if matched {
                // If there's more input, there MUST be whitespace after a matched token
                if reader.can_read() && !reader.peek().unwrap().is_whitespace() {
                    reader.set_cursor(bookmark);
                    continue;
                }

                if let Ok(res) = Self::parse_node(child, reader, ctx) {
                    let complete = res.is_some() || !reader.can_read();
                    if complete {
                        return Ok(res);
                    }
                }
            }

            // Backtrack if branch did not match or execute
            reader.set_cursor(bookmark);
        }

        if current.handler.is_some() && !reader.can_read() {
            Ok(current.handler.clone())
        } else {
            Err(CommandSyntaxError::Custom {
                cursor: reader.cursor(),
                message: format!("Unknown argument or token at '{}'", reader.remaining()),
            })
        }
    }

    /// Generates tab-completion suggestions for a command input at the given cursor position.
    #[must_use]
    pub fn suggest(&self, input: &str, cursor: usize) -> CommandSuggestions {
        let cursor_clamped = cursor.min(input.len());
        let effective_input = &input[..cursor_clamped];
        let has_slash = effective_input.starts_with('/');
        let slice = if has_slash {
            &effective_input[1..]
        } else {
            effective_input
        };

        let mut candidates = Vec::new();
        let (token_start, partial_token, parent_node) = self.find_suggest_node(slice);

        for child in &parent_node.children {
            match &child.node_type {
                CommandNodeType::Literal { literal } => {
                    if literal
                        .to_ascii_lowercase()
                        .starts_with(&partial_token.to_ascii_lowercase())
                    {
                        candidates.push(Suggestion {
                            value: literal.clone(),
                            tooltip: child.tooltip.clone(),
                        });
                    }
                }
                CommandNodeType::Argument { name, arg_type } => {
                    match arg_type {
                        ArgumentType::Bool => {
                            for b in ["true", "false"] {
                                if b.starts_with(&partial_token.to_ascii_lowercase()) {
                                    candidates.push(Suggestion {
                                        value: b.to_string(),
                                        tooltip: Some(name.clone()),
                                    });
                                }
                            }
                        }
                        ArgumentType::Entity => {
                            for sel in ["@p", "@a", "@s", "@r", "@e"] {
                                if sel.starts_with(partial_token) {
                                    candidates.push(Suggestion {
                                        value: sel.to_string(),
                                        tooltip: Some(name.clone()),
                                    });
                                }
                            }
                        }
                        ArgumentType::Vec3 => {
                            if "~".starts_with(partial_token) || partial_token.is_empty() {
                                candidates.push(Suggestion {
                                    value: "~".to_string(),
                                    tooltip: Some(format!("<{name}: x y z>")),
                                });
                            }
                        }
                        _ => {
                            // Suggest the argument placeholder
                            candidates.push(Suggestion {
                                value: format!("<{name}>"),
                                tooltip: child.tooltip.clone(),
                            });
                        }
                    }
                }
                CommandNodeType::Root => {}
            }
        }

        let abs_start = usize::from(has_slash) + token_start;
        CommandSuggestions {
            start: abs_start,
            length: partial_token.len(),
            candidates,
        }
    }

    fn find_suggest_node<'a>(&'a self, input: &'a str) -> (usize, &'a str, &'a CommandNode) {
        let mut reader = StringReader::new(input);
        let mut current_node = &self.root;

        loop {
            reader.skip_whitespace();
            if !reader.can_read() {
                return (reader.cursor(), "", current_node);
            }

            let token_start = reader.cursor();
            let word = reader.read_unquoted_string();

            if !reader.can_read() {
                // We are currently typing this token
                return (token_start, word, current_node);
            }

            // Find matching child to advance
            let mut matched_child = None;
            for child in &current_node.children {
                match &child.node_type {
                    CommandNodeType::Literal { literal } => {
                        if word.eq_ignore_ascii_case(literal) {
                            matched_child = Some(child);
                            break;
                        }
                    }
                    CommandNodeType::Argument { .. } => {
                        matched_child = Some(child);
                        break;
                    }
                    CommandNodeType::Root => {}
                }
            }

            if let Some(child) = matched_child {
                current_node = child;
            } else {
                return (token_start, word, current_node);
            }
        }
    }
}
