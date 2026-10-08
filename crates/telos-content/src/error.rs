//! Error types for data-pack loading, manifest validation, and registry operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result alias for content operations.
pub type Result<T> = std::result::Result<T, ContentError>;

/// Errors that can occur during content pack discovery, resolution, parsing, or registration.
#[derive(Debug, Error)]
pub enum ContentError {
    /// Filesystem IO error.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// File path where the error occurred.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// Manifest parsing error.
    #[error("failed to parse manifest at {path}: {source}")]
    ManifestParse {
        /// Manifest path.
        path: PathBuf,
        /// TOML deserialization error.
        #[source]
        source: toml::de::Error,
    },

    /// Data file parsing error.
    #[error("failed to parse data definition at {path}: {reason}")]
    DataParse {
        /// Data file path.
        path: PathBuf,
        /// Error details.
        reason: String,
    },

    /// Multiple packs with the same mod ID found.
    #[error("duplicate mod ID '{id}' discovered at '{first}' and '{second}'")]
    DuplicateModId {
        /// Mod identifier.
        id: String,
        /// First discovered path.
        first: PathBuf,
        /// Second discovered path.
        second: PathBuf,
    },

    /// Required dependency is missing or version mismatch.
    #[error(
        "mod '{mod_id}' requires dependency '{dep_id}' matching '{req}', but it was not satisfied"
    )]
    MissingDependency {
        /// Dependent mod ID.
        mod_id: String,
        /// Missing dependency ID.
        dep_id: String,
        /// Semver requirement string.
        req: String,
    },

    /// Incompatible mod detected.
    #[error("mod '{mod_id}' is incompatible with discovered mod '{conflict_id}' (rule: {rule})")]
    IncompatibleMod {
        /// Dependent mod ID.
        mod_id: String,
        /// Conflicting mod ID.
        conflict_id: String,
        /// Rule description.
        rule: String,
    },

    /// Dependency cycle detected in load order.
    #[error("circular dependency detected in mod load order: {}", cycle.join(" -> "))]
    DependencyCycle {
        /// Sequence of mod IDs forming the cycle.
        cycle: Vec<String>,
    },

    /// Invalid namespaced identifier.
    #[error("invalid namespaced identifier '{0}'")]
    InvalidIdentifier(String),

    /// Missing or invalid reference to another block/item/tag.
    #[error("invalid reference: '{referrer}' references unknown identifier '{target}'")]
    UnknownReference {
        /// File or definition holding the reference.
        referrer: String,
        /// Missing target identifier.
        target: String,
    },

    /// Operation attempted on a frozen registry.
    #[error("cannot register content: registry is frozen")]
    RegistryFrozen,
}

impl From<telos_sdk::manifest::ManifestError> for ContentError {
    fn from(err: telos_sdk::manifest::ManifestError) -> Self {
        match err {
            telos_sdk::manifest::ManifestError::Io(e) => Self::Io {
                path: PathBuf::from("manifest"),
                source: e,
            },
            telos_sdk::manifest::ManifestError::Toml(e) => Self::ManifestParse {
                path: PathBuf::from("manifest"),
                source: e,
            },
            telos_sdk::manifest::ManifestError::Validation(msg) => Self::DataParse {
                path: PathBuf::from("manifest"),
                reason: msg,
            },
        }
    }
}
