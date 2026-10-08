//! Mod manifest schema (`mod.toml`) and dependency specification.

use semver::{Version, VersionReq};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Error type for SDK manifest operations.
#[derive(Debug)]
pub enum ManifestError {
    /// IO error reading manifest file.
    Io(std::io::Error),
    /// TOML parsing error.
    Toml(toml::de::Error),
    /// Manifest validation failure.
    Validation(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Toml(e) => write!(f, "TOML error: {e}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Toml(e) => Some(e),
            Self::Validation(_) => None,
        }
    }
}

/// Execution environment side for a mod or dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModSide {
    /// Server-only mod or dependency.
    Server,
    /// Client-only mod or dependency.
    Client,
    /// Required or supported on both server and client.
    #[default]
    Both,
}

/// Primary mod metadata section (`[mod]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModInfo {
    /// Primary mod identifier and namespace (lowercase alphanumeric with `_`).
    pub id: String,
    /// Semantic version of the mod.
    pub version: Version,
    /// Human-readable mod display name.
    #[serde(default)]
    pub name: String,
    /// Short description of the mod.
    #[serde(default)]
    pub description: String,
    /// Mod authors.
    #[serde(default)]
    pub authors: Vec<String>,
    /// SPDX license identifier.
    #[serde(default = "default_license")]
    pub license: String,
    /// Side where the mod runs.
    #[serde(default)]
    pub side: ModSide,
    /// Additional namespaces declared and owned by this mod.
    #[serde(default)]
    pub namespaces: Vec<String>,
    /// Foreign namespaces whose data this mod replaces.
    #[serde(default)]
    pub overrides: Vec<String>,
}

fn default_license() -> String {
    "MIT".to_string()
}

/// Dependency specification supporting both a simple version string or detailed table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DependencySpec {
    /// Required semver version requirement.
    pub version: VersionReq,
    /// Side on which dependency is required.
    pub side: ModSide,
    /// Whether the dependency is optional (soft).
    pub optional: bool,
}

impl<'de> Deserialize<'de> for DependencySpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Simple(String),
            Detailed {
                version: String,
                #[serde(default)]
                side: ModSide,
                #[serde(default)]
                optional: bool,
            },
        }

        match Helper::deserialize(deserializer)? {
            Helper::Simple(v) => {
                let req = VersionReq::parse(&v).map_err(serde::de::Error::custom)?;
                Ok(Self {
                    version: req,
                    side: ModSide::Both,
                    optional: false,
                })
            }
            Helper::Detailed {
                version,
                side,
                optional,
            } => {
                let req = VersionReq::parse(&version).map_err(serde::de::Error::custom)?;
                Ok(Self {
                    version: req,
                    side,
                    optional,
                })
            }
        }
    }
}

/// Soft ordering and conflict constraints (`[ordering]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModOrdering {
    /// Mod IDs that must load before this mod, if present.
    #[serde(default)]
    pub after: Vec<String>,
    /// Mod IDs that must load after this mod, if present.
    #[serde(default)]
    pub before: Vec<String>,
    /// Mod IDs and version rules that conflict with this mod.
    #[serde(default)]
    pub incompatible: HashMap<String, String>,
}

/// Optional API version requirements (`[api]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModApi {
    /// Server WIT API version requirement.
    #[serde(default)]
    pub server: Option<String>,
    /// Client WIT API version requirement.
    #[serde(default)]
    pub client: Option<String>,
    /// Data-pack schema version (e.g. "^1").
    #[serde(default)]
    pub data: Option<String>,
}

/// Permissions and capabilities requested (`[permissions]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModPermissions {
    /// Server capabilities requested.
    #[serde(default)]
    pub server: Vec<String>,
    /// Client capabilities requested.
    #[serde(default)]
    pub client: Vec<String>,
}

/// Root mod manifest parsed from `mod.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModManifest {
    /// Primary mod metadata.
    #[serde(rename = "mod")]
    pub info: ModInfo,
    /// API requirements.
    #[serde(default)]
    pub api: ModApi,
    /// Dependencies by mod ID.
    #[serde(default)]
    pub dependencies: HashMap<String, DependencySpec>,
    /// Load order and conflict rules.
    #[serde(default)]
    pub ordering: ModOrdering,
    /// Requested capabilities.
    #[serde(default)]
    pub permissions: ModPermissions,
}

impl ModManifest {
    /// Creates a minimal new mod manifest.
    #[must_use]
    pub fn new(id: impl Into<String>, version: Version, name: impl Into<String>) -> Self {
        Self {
            info: ModInfo {
                id: id.into(),
                version,
                name: name.into(),
                description: String::new(),
                authors: Vec::new(),
                license: default_license(),
                side: ModSide::Both,
                namespaces: Vec::new(),
                overrides: Vec::new(),
            },
            api: ModApi::default(),
            dependencies: HashMap::new(),
            ordering: ModOrdering::default(),
            permissions: ModPermissions::default(),
        }
    }

    /// Parses a manifest from a TOML string.
    pub fn from_toml_str(s: &str) -> std::result::Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Serializes the manifest into a formatted TOML string.
    pub fn to_toml_string(&self) -> std::result::Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Loads a manifest from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> std::result::Result<Self, ManifestError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(ManifestError::Io)?;
        Self::from_toml_str(&content).map_err(ManifestError::Toml)
    }

    /// Validates basic integrity of the manifest fields.
    pub fn validate(&self) -> std::result::Result<(), ManifestError> {
        let id = &self.info.id;
        if id.is_empty() {
            return Err(ManifestError::Validation(
                "Mod ID cannot be empty".to_string(),
            ));
        }
        for ch in id.chars() {
            if !matches!(ch, 'a'..='z' | '0'..='9' | '_') {
                return Err(ManifestError::Validation(format!(
                    "Invalid character '{ch}' in mod ID '{id}'. Mod IDs must be lowercase [a-z0-9_]."
                )));
            }
        }

        // Validate extra namespaces
        for ns in &self.info.namespaces {
            for ch in ns.chars() {
                if !matches!(ch, 'a'..='z' | '0'..='9' | '_') {
                    return Err(ManifestError::Validation(format!(
                        "Invalid character '{ch}' in declared namespace '{ns}'."
                    )));
                }
            }
        }

        // Validate permissions
        let known_server_perms = [
            "world.read",
            "world.write",
            "command.register",
            "events.listen",
            "storage.kv",
            "net.mod-channel",
        ];
        for perm in &self.permissions.server {
            if !known_server_perms.contains(&perm.as_str()) {
                return Err(ManifestError::Validation(format!(
                    "Unknown server permission '{perm}'. Known: {known_server_perms:?}"
                )));
            }
        }

        Ok(())
    }
}
