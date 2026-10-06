//! Classic Voxel-compatible tag definitions (`data/<ns>/tags/<registry>/<name>.json`).

use serde::{Deserialize, Deserializer, Serialize};

/// An entry in a tag's values array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TagValueEntry {
    /// Identifier or tag reference (prefixed with `#`).
    pub id: String,
    /// Whether this entry is required to exist.
    pub required: bool,
}

impl<'de> Deserialize<'de> for TagValueEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Simple(String),
            Detailed {
                id: String,
                #[serde(default = "default_true")]
                required: bool,
            },
        }

        match Helper::deserialize(deserializer)? {
            Helper::Simple(id) => Ok(Self { id, required: true }),
            Helper::Detailed { id, required } => Ok(Self { id, required }),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Classic Voxel-compatible JSON tag schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TagDef {
    /// If `true`, replaces existing values instead of appending.
    #[serde(default)]
    pub replace: bool,
    /// List of identifiers or nested tag references.
    #[serde(default)]
    pub values: Vec<TagValueEntry>,
}

impl TagDef {
    /// Creates a tag definition from values.
    pub fn new(values: Vec<TagValueEntry>, replace: bool) -> Self {
        Self { replace, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_json_deserialization() {
        let json_str = r##"
            {
                "replace": false,
                "values": [
                    "voxel:stone",
                    "#voxel:base_stone",
                    { "id": "sample:ruby_block", "required": false }
                ]
            }
        "##;
        let tag: TagDef = serde_json::from_str(json_str).expect("Valid JSON tag");
        assert!(!tag.replace);
        assert_eq!(tag.values.len(), 3);
        assert_eq!(tag.values[0].id, "voxel:stone");
        assert!(tag.values[0].required);
        assert_eq!(tag.values[1].id, "#voxel:base_stone");
        assert_eq!(tag.values[2].id, "sample:ruby_block");
        assert!(!tag.values[2].required);
    }
}
