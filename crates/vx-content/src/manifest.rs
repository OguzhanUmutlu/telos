//! Mod manifest schema (`mod.toml`) and dependency specification.

use semver::{Version, VersionReq};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{ContentError, Result};

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
    /// Parses a manifest from a TOML string.
    pub fn from_toml_str(s: &str) -> std::result::Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Loads a manifest from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| ContentError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_toml_str(&content).map_err(|e| ContentError::ManifestParse {
            path: path.to_path_buf(),
            source: e,
        })
    }

    /// Validates basic integrity of the manifest fields.
    pub fn validate(&self) -> Result<()> {
        let id = &self.info.id;
        if id.is_empty() {
            return Err(ContentError::InvalidIdentifier(
                "Mod ID cannot be empty".into(),
            ));
        }
        for ch in id.chars() {
            if !matches!(ch, 'a'..='z' | '0'..='9' | '_') {
                return Err(ContentError::InvalidIdentifier(format!(
                    "Invalid character '{ch}' in mod ID '{id}'. Mod IDs must be lowercase [a-z0-9_]."
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_manifest() {
        let toml = r#"
            [mod]
            id = "test_mod"
            version = "0.1.0"
            name = "Test Mod"
        "#;
        let manifest = ModManifest::from_toml_str(toml).expect("Valid minimal manifest");
        assert_eq!(manifest.info.id, "test_mod");
        assert_eq!(manifest.info.version, Version::parse("0.1.0").unwrap());
        assert_eq!(manifest.info.side, ModSide::Both);
        assert!(manifest.dependencies.is_empty());
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_parse_full_manifest() {
        let toml = r#"
            [mod]
            id = "hello_block"
            version = "1.2.3"
            name = "Hello Block"
            description = "Adds a glowing block."
            authors = ["Alice", "Bob"]
            license = "MIT"
            side = "server"
            namespaces = ["hello_block", "extra"]
            overrides = ["voxel"]

            [api]
            server = "vx:server@^0.1"
            data = "^1"

            [dependencies]
            voxel = ">=0.1.0, <0.2.0"
            some_lib = { version = "^2.0", side = "server", optional = true }

            [ordering]
            after = ["some_lib"]
            before = ["other_mod"]
            incompatible = { broken_mod = "<1.0" }

            [permissions]
            server = ["world.read", "world.write"]
        "#;
        let manifest = ModManifest::from_toml_str(toml).expect("Valid full manifest");
        assert_eq!(manifest.info.id, "hello_block");
        assert_eq!(manifest.info.authors, vec!["Alice", "Bob"]);
        assert_eq!(manifest.info.side, ModSide::Server);
        assert_eq!(manifest.dependencies.len(), 2);
        assert!(
            manifest.dependencies["voxel"]
                .version
                .matches(&Version::parse("0.1.5").unwrap())
        );
        assert!(manifest.dependencies["some_lib"].optional);
        assert_eq!(manifest.ordering.after, vec!["some_lib"]);
        assert_eq!(manifest.ordering.before, vec!["other_mod"]);
        assert_eq!(manifest.ordering.incompatible["broken_mod"], "<1.0");
        assert_eq!(
            manifest.permissions.server,
            vec!["world.read", "world.write"]
        );
    }
}
