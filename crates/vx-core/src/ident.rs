//! Namespaced identifiers for blocks, items, entities, and registry entries.

use std::{fmt, str::FromStr};
use thiserror::Error;

/// The default namespace used when none is explicitly specified.
pub const DEFAULT_NAMESPACE: &str = "voxel";

/// An error that occurs when parsing an [`Identifier`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseIdentError {
    /// Identifier contains no colon and no path.
    #[error("empty identifier")]
    Empty,
    /// Invalid character found in namespace.
    #[error("invalid character '{character}' in namespace '{namespace}'")]
    InvalidNamespace {
        /// The invalid namespace string.
        namespace: String,
        /// The offending character.
        character: char,
    },
    /// Invalid character found in path.
    #[error("invalid character '{character}' in path '{path}'")]
    InvalidPath {
        /// The invalid path string.
        path: String,
        /// The offending character.
        character: char,
    },
    /// Namespace is empty before the colon.
    #[error("namespace cannot be empty")]
    EmptyNamespace,
    /// Path is empty after the colon.
    #[error("path cannot be empty")]
    EmptyPath,
}

/// An immutable namespaced identifier representing a resource, block, or registry entry.
///
/// Syntax follows `namespace:path`, where:
/// - `namespace` contains only `[a-z0-9_.-]`.
/// - `path` contains only `[a-z0-9_./-]`.
///
/// If parsed from a string without a colon, the namespace defaults to `"voxel"`.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Identifier {
    namespace: Box<str>,
    path: Box<str>,
}

impl Identifier {
    /// Creates a new `Identifier` from namespace and path strings.
    ///
    /// Validates both components according to identifier rules.
    pub fn new(namespace: impl AsRef<str>, path: impl AsRef<str>) -> Result<Self, ParseIdentError> {
        let ns = namespace.as_ref();
        let p = path.as_ref();

        validate_namespace(ns)?;
        validate_path(p)?;

        Ok(Self {
            namespace: ns.into(),
            path: p.into(),
        })
    }

    /// Creates an identifier in the default `"voxel"` namespace.
    pub fn voxel(path: impl AsRef<str>) -> Result<Self, ParseIdentError> {
        Self::new(DEFAULT_NAMESPACE, path)
    }

    /// Creates an identifier in the `"classic"` namespace for compatibility.
    pub fn classic(path: impl AsRef<str>) -> Result<Self, ParseIdentError> {
        Self::new("classic", path)
    }

    /// Returns the namespace part.
    #[inline]
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the path part.
    #[inline]
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

fn validate_namespace(ns: &str) -> Result<(), ParseIdentError> {
    if ns.is_empty() {
        return Err(ParseIdentError::EmptyNamespace);
    }
    for ch in ns.chars() {
        if !matches!(ch, 'a'..='z' | '0'..='9' | '_' | '-' | '.') {
            return Err(ParseIdentError::InvalidNamespace {
                namespace: ns.to_string(),
                character: ch,
            });
        }
    }
    Ok(())
}

fn validate_path(p: &str) -> Result<(), ParseIdentError> {
    if p.is_empty() {
        return Err(ParseIdentError::EmptyPath);
    }
    for ch in p.chars() {
        if !matches!(ch, 'a'..='z' | '0'..='9' | '_' | '-' | '.' | '/') {
            return Err(ParseIdentError::InvalidPath {
                path: p.to_string(),
                character: ch,
            });
        }
    }
    Ok(())
}

impl FromStr for Identifier {
    type Err = ParseIdentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(ParseIdentError::Empty);
        }
        match s.split_once(':') {
            Some((ns, p)) => Self::new(ns, p),
            None => Self::voxel(s),
        }
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.path)
    }
}

impl fmt::Debug for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Identifier(\"{}:{}\")", self.namespace, self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identifier_parsing() {
        let id: Identifier = "stone".parse().unwrap();
        assert_eq!(id.namespace(), "voxel");
        assert_eq!(id.path(), "stone");
        assert_eq!(id.to_string(), "voxel:stone");

        let mc: Identifier = "classic:oak_stairs".parse().unwrap();
        assert_eq!(mc.namespace(), "classic");
        assert_eq!(mc.path(), "oak_stairs");
        assert_eq!(mc.to_string(), "classic:oak_stairs");

        let deep: Identifier = "voxel:block/textures/stone".parse().unwrap();
        assert_eq!(deep.path(), "block/textures/stone");
    }

    #[test]
    fn test_invalid_identifiers() {
        assert!("".parse::<Identifier>().is_err());
        assert!("Voxel:Stone".parse::<Identifier>().is_err());
        assert!("voxel:".parse::<Identifier>().is_err());
        assert!(":stone".parse::<Identifier>().is_err());
        assert!("voxel:stone with space".parse::<Identifier>().is_err());
    }
}
