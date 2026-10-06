//! Low-level character cursor and lexer for Brigadier-style command parsing.

use thiserror::Error;

/// Syntax errors that can occur during command string reading and parsing.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommandSyntaxError {
    /// Unexpected end of input while expecting additional tokens.
    #[error("Unexpected end of input at position {cursor}")]
    UnexpectedEof {
        /// Byte offset where unexpected EOF was reached.
        cursor: usize,
    },

    /// Expected whitespace separator between command tokens.
    #[error("Expected whitespace at position {cursor}")]
    ExpectedWhitespace {
        /// Byte offset where whitespace was expected.
        cursor: usize,
    },

    /// Invalid integer literal.
    #[error("Invalid integer '{found}' at position {cursor}: {reason}")]
    InvalidInteger {
        /// Byte offset where invalid integer started.
        cursor: usize,
        /// Invalid literal slice found.
        found: String,
        /// Parsing error message.
        reason: String,
    },

    /// Invalid floating point literal.
    #[error("Invalid float '{found}' at position {cursor}: {reason}")]
    InvalidFloat {
        /// Byte offset where invalid float started.
        cursor: usize,
        /// Invalid literal slice found.
        found: String,
        /// Parsing error message.
        reason: String,
    },

    /// Invalid boolean literal (expected 'true' or 'false').
    #[error("Invalid boolean '{found}' at position {cursor}")]
    InvalidBool {
        /// Byte offset where invalid boolean started.
        cursor: usize,
        /// Invalid literal slice found.
        found: String,
    },

    /// Unterminated quoted string literal.
    #[error("Expected closing quote '{quote}' at position {cursor}")]
    UnterminatedQuote {
        /// Byte offset where unclosed quote started.
        cursor: usize,
        /// Quote character (' or ").
        quote: char,
    },

    /// Invalid escape sequence in quoted string.
    #[error("Invalid escape sequence '\\{escape}' at position {cursor}")]
    InvalidEscape {
        /// Byte offset of the invalid escape sequence.
        cursor: usize,
        /// Escaped character that was not recognized.
        escape: char,
    },

    /// Expected a specific character.
    #[error("Expected '{expected}' at position {cursor}, found '{found}'")]
    ExpectedChar {
        /// Byte offset where expected character was missing.
        cursor: usize,
        /// Expected character.
        expected: char,
        /// String token or character actually found.
        found: String,
    },

    /// Unknown command or subcommand literal.
    #[error("Unknown command '{name}' at position {cursor}")]
    UnknownCommand {
        /// Byte offset of the unknown command.
        cursor: usize,
        /// Command name that was not recognized.
        name: String,
    },

    /// Generic syntax or validation error.
    #[error("Syntax error at position {cursor}: {message}")]
    Custom {
        /// Byte offset where syntax error occurred.
        cursor: usize,
        /// Detailed human-readable error description.
        message: String,
    },
}

impl CommandSyntaxError {
    /// Returns the character position where the error occurred.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        match *self {
            Self::UnexpectedEof { cursor }
            | Self::ExpectedWhitespace { cursor }
            | Self::InvalidInteger { cursor, .. }
            | Self::InvalidFloat { cursor, .. }
            | Self::InvalidBool { cursor, .. }
            | Self::UnterminatedQuote { cursor, .. }
            | Self::InvalidEscape { cursor, .. }
            | Self::ExpectedChar { cursor, .. }
            | Self::UnknownCommand { cursor, .. }
            | Self::Custom { cursor, .. } => cursor,
        }
    }
}

/// A stateful string reader and cursor used by the command parser.
#[derive(Debug, Clone)]
pub struct StringReader<'a> {
    input: &'a str,
    cursor: usize,
}

impl<'a> StringReader<'a> {
    /// Creates a new `StringReader` at index 0.
    #[must_use]
    pub const fn new(input: &'a str) -> Self {
        Self { input, cursor: 0 }
    }

    /// Returns the full underlying command string.
    #[must_use]
    pub const fn input(&self) -> &'a str {
        self.input
    }

    /// Current cursor index in UTF-8 bytes.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Sets the cursor to a specific byte index.
    pub fn set_cursor(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.input.len());
    }

    /// Remaining slice of string from cursor to end.
    #[must_use]
    pub fn remaining(&self) -> &'a str {
        if self.cursor >= self.input.len() {
            ""
        } else {
            &self.input[self.cursor..]
        }
    }

    /// Total byte length of input.
    #[must_use]
    pub const fn total_len(&self) -> usize {
        self.input.len()
    }

    /// Returns true if more characters remain to be read.
    #[must_use]
    pub const fn can_read(&self) -> bool {
        self.cursor < self.input.len()
    }

    /// Peeks at the next character without advancing the cursor.
    #[must_use]
    pub fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    /// Peeks at a character at a byte offset from the current cursor.
    #[must_use]
    pub fn peek_offset(&self, offset: usize) -> Option<char> {
        let start = self.cursor.saturating_add(offset);
        if start >= self.input.len() {
            None
        } else {
            self.input[start..].chars().next()
        }
    }

    /// Reads the next character and advances the cursor.
    pub fn read_char(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.cursor += c.len_utf8();
        Some(c)
    }

    /// Skips any leading ASCII whitespace characters (` `, `\t`, `\r`, `\n`).
    pub fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.cursor += c.len_utf8();
            } else {
                break;
            }
        }
    }

    /// Returns true if `c` is allowed in an unquoted string token.
    #[must_use]
    pub const fn is_allowed_in_unquoted(c: char) -> bool {
        c.is_ascii_alphanumeric()
            || c == '_'
            || c == '-'
            || c == '.'
            || c == '+'
            || c == ':'
            || c == '/'
    }

    /// Reads an unquoted token containing only valid identifier characters.
    pub fn read_unquoted_string(&mut self) -> &'a str {
        let start = self.cursor;
        while let Some(c) = self.peek() {
            if Self::is_allowed_in_unquoted(c) {
                self.cursor += c.len_utf8();
            } else {
                break;
            }
        }
        &self.input[start..self.cursor]
    }

    /// Reads a quoted string literal enclosed in `'` or `"`, resolving escape characters.
    pub fn read_quoted_string(&mut self) -> Result<String, CommandSyntaxError> {
        if !self.can_read() {
            return Err(CommandSyntaxError::UnexpectedEof {
                cursor: self.cursor,
            });
        }

        let quote = self.read_char().ok_or(CommandSyntaxError::UnexpectedEof {
            cursor: self.cursor,
        })?;

        if quote != '"' && quote != '\'' {
            return Err(CommandSyntaxError::ExpectedChar {
                cursor: self.cursor.saturating_sub(quote.len_utf8()),
                expected: '"',
                found: quote.to_string(),
            });
        }

        let mut result = String::new();
        let mut escaped = false;

        while let Some(c) = self.read_char() {
            if escaped {
                match c {
                    '\\' | '"' | '\'' => result.push(c),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    _ => {
                        return Err(CommandSyntaxError::InvalidEscape {
                            cursor: self.cursor.saturating_sub(c.len_utf8()),
                            escape: c,
                        });
                    }
                }
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == quote {
                return Ok(result);
            } else {
                result.push(c);
            }
        }

        Err(CommandSyntaxError::UnterminatedQuote {
            cursor: self.cursor,
            quote,
        })
    }

    /// Reads either a quoted string or an unquoted string token.
    pub fn read_string(&mut self) -> Result<String, CommandSyntaxError> {
        if matches!(self.peek(), Some('"' | '\'')) {
            return self.read_quoted_string();
        }
        Ok(self.read_unquoted_string().to_string())
    }

    /// Reads an i32 integer.
    pub fn read_i32(&mut self) -> Result<i32, CommandSyntaxError> {
        let start = self.cursor;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || (c == '-' && self.cursor == start) {
                self.cursor += c.len_utf8();
            } else {
                break;
            }
        }

        let text = &self.input[start..self.cursor];
        if text.is_empty() || text == "-" {
            return Err(CommandSyntaxError::InvalidInteger {
                cursor: start,
                found: text.to_string(),
                reason: "expected digit".to_string(),
            });
        }

        text.parse::<i32>()
            .map_err(|e| CommandSyntaxError::InvalidInteger {
                cursor: start,
                found: text.to_string(),
                reason: e.to_string(),
            })
    }

    /// Reads an f32 float.
    pub fn read_f32(&mut self) -> Result<f32, CommandSyntaxError> {
        let start = self.cursor;
        let mut has_dot = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || (c == '-' && self.cursor == start) {
                self.cursor += c.len_utf8();
            } else if c == '.' && !has_dot {
                has_dot = true;
                self.cursor += c.len_utf8();
            } else {
                break;
            }
        }

        let text = &self.input[start..self.cursor];
        if text.is_empty() || text == "-" || text == "." {
            return Err(CommandSyntaxError::InvalidFloat {
                cursor: start,
                found: text.to_string(),
                reason: "expected number".to_string(),
            });
        }

        text.parse::<f32>()
            .map_err(|e| CommandSyntaxError::InvalidFloat {
                cursor: start,
                found: text.to_string(),
                reason: e.to_string(),
            })
    }

    /// Reads a boolean value (`true` or `false`).
    pub fn read_bool(&mut self) -> Result<bool, CommandSyntaxError> {
        let start = self.cursor;
        let text = self.read_unquoted_string();
        match text {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(CommandSyntaxError::InvalidBool {
                cursor: start,
                found: text.to_string(),
            }),
        }
    }

    /// Expects a specific character at the current cursor and consumes it.
    pub fn expect(&mut self, expected: char) -> Result<(), CommandSyntaxError> {
        if !self.can_read() {
            return Err(CommandSyntaxError::UnexpectedEof {
                cursor: self.cursor,
            });
        }
        let c = self.peek().unwrap();
        if c == expected {
            self.cursor += c.len_utf8();
            Ok(())
        } else {
            Err(CommandSyntaxError::ExpectedChar {
                cursor: self.cursor,
                expected,
                found: c.to_string(),
            })
        }
    }
}
