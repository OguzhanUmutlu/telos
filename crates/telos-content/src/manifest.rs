//! Mod manifest schema (`mod.toml`) and dependency specification.
//!
//! Re-exports canonical definitions from `telos-sdk::manifest`.

pub use telos_sdk::manifest::{
    DependencySpec, ManifestError, ModApi, ModInfo, ModManifest, ModOrdering, ModPermissions,
    ModSide,
};

#[cfg(test)]
mod tests {
    use super::*;
    use semver::Version;

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

            [api]
            server = "telos:server@^0.1"
            data = "^1"

            [dependencies]
            telos = ">=0.1.0, <0.2.0"
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
            manifest.dependencies["telos"]
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
