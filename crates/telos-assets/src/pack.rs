//! Resource pack stack loader and asset resolution.

use std::path::{Path, PathBuf};

use crate::{error::AssetError, image_buf::RgbaImage, mcmeta::AnimationDef};

/// Ordered stack of resource packs resolving assets from highest to lowest priority.
#[derive(Debug, Clone, Default)]
pub struct ResourcePackStack {
    roots: Vec<PathBuf>,
}

impl ResourcePackStack {
    /// Creates an empty resource pack stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a resource pack root directory to the stack.
    ///
    /// Packs added first take precedence over packs added later.
    pub fn add_root(&mut self, root: impl AsRef<Path>) {
        let path = root.as_ref().to_path_buf();
        if path.exists() {
            self.roots.push(path);
        }
    }

    /// Finds a block texture by name (e.g. `"stone"`, `"grass_block_top"`).
    ///
    /// Searches each mounted pack in order under `textures/block/` across any namespace.
    #[must_use]
    pub fn find_block_texture(&self, name: &str) -> Option<PathBuf> {
        let rel_path = format!("textures/block/{name}.png");
        self.find_texture(&rel_path)
    }

    /// Finds the `.png.mcmeta` animation metadata file for a block texture by name.
    #[must_use]
    pub fn find_block_mcmeta(&self, name: &str) -> Option<PathBuf> {
        let rel_path = format!("textures/block/{name}.png.mcmeta");
        self.find_texture(&rel_path)
    }

    /// Loads and parses the animation metadata for a block texture if present.
    pub fn load_block_mcmeta(&self, name: &str) -> Result<AnimationDef, AssetError> {
        let path = self
            .find_block_mcmeta(name)
            .ok_or_else(|| AssetError::MissingTexture(format!("{name}.png.mcmeta")))?;

        let content = std::fs::read_to_string(&path).map_err(|source| AssetError::Io {
            path: path.clone(),
            source,
        })?;

        AnimationDef::from_json_str(&content).map_err(|err| AssetError::Parse {
            path,
            message: err.to_string(),
        })
    }

    /// Loads and decodes a block texture from the mounted packs.
    pub fn load_block_texture(&self, name: &str) -> Result<RgbaImage, AssetError> {
        let path = self
            .find_block_texture(name)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::from_file(&path)
    }

    /// Finds a namespaced block texture (e.g. `ns = "sample"`, `name = "ruby_block"` -> `assets/sample/textures/block/ruby_block.png`).
    #[must_use]
    pub fn find_namespaced_block_texture(&self, ns: &str, name: &str) -> Option<PathBuf> {
        let rel_path = format!("assets/{ns}/textures/block/{name}.png");
        for root in &self.roots {
            let candidate = root.join(&rel_path);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        // Fallback: search without explicit namespace prefix
        self.find_block_texture(name)
    }

    /// Loads and decodes a namespaced block texture from the mounted packs.
    pub fn load_namespaced_block_texture(
        &self,
        ns: &str,
        name: &str,
    ) -> Result<RgbaImage, AssetError> {
        let path = self
            .find_namespaced_block_texture(ns, name)
            .ok_or_else(|| AssetError::MissingTexture(format!("{ns}:{name}")))?;

        RgbaImage::from_file(&path)
    }

    /// Loads and decodes all frames of an animated block texture strip from the mounted packs.
    pub fn load_animated_block_texture(&self, name: &str) -> Result<Vec<RgbaImage>, AssetError> {
        let path = self
            .find_block_texture(name)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::frames_from_file(&path)
    }

    /// Loads and decodes all frames of an animated block texture along with its animation metadata.
    pub fn load_block_texture_with_animation(
        &self,
        name: &str,
    ) -> Result<(Vec<RgbaImage>, Option<AnimationDef>), AssetError> {
        let frames = self.load_animated_block_texture(name)?;
        let anim_def = self.load_block_mcmeta(name).ok();
        Ok((frames, anim_def))
    }

    /// Finds a generic asset texture across mounted packs and namespace directories.
    #[must_use]
    pub fn find_texture(&self, rel_path: &str) -> Option<PathBuf> {
        let rel = Path::new(rel_path);
        for root in &self.roots {
            // Direct path under root
            let direct = root.join(rel);
            if direct.is_file() {
                return Some(direct);
            }

            // Path under assets/<namespace>/<rel_path>
            let assets_dir = root.join("assets");
            if let Ok(entries) = std::fs::read_dir(&assets_dir) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join(rel);
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }

        None
    }

    /// Loads and decodes a GUI sprite by relative sprite path (e.g. `"hud/hotbar"`).
    pub fn load_gui_sprite(&self, name: &str) -> Result<RgbaImage, AssetError> {
        let rel_path = format!("textures/gui/sprites/{name}.png");
        let path = self
            .find_texture(&rel_path)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::from_file_exact(&path)
    }

    /// Loads and decodes a font texture sheet by name (e.g. `"ascii"`).
    pub fn load_font_texture(&self, name: &str) -> Result<RgbaImage, AssetError> {
        let rel_path = format!("textures/font/{name}.png");
        let path = self
            .find_texture(&rel_path)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::from_file_exact(&path)
    }

    /// Scans all mounted resource packs for `lang/*.json` files and merges them into `catalog`.
    ///
    /// Packs earlier in the stack take precedence over packs later in the stack.
    pub fn populate_language_catalog(&self, catalog: &mut telos_core::i18n::LanguageCatalog) {
        // Iterate in reverse so higher priority packs override lower priority packs
        for root in self.roots.iter().rev() {
            // Direct lang/ directory
            let direct_lang = root.join("lang");
            let _ = catalog.load_from_dir(&direct_lang);

            // assets/<namespace>/lang/ directories
            let assets_dir = root.join("assets");
            if let Ok(entries) = std::fs::read_dir(&assets_dir) {
                for entry in entries.flatten() {
                    let ns_lang = entry.path().join("lang");
                    let _ = catalog.load_from_dir(&ns_lang);
                }
            }
        }
    }

    /// Scans all mounted resource packs for JavaScript files (`scripts/*.js` or `assets/<namespace>/scripts/*.js`).
    ///
    /// Returns a list of `(script_name, script_content)` pairs.
    #[must_use]
    pub fn load_scripts(&self) -> Vec<(String, String)> {
        let mut scripts = Vec::new();
        for root in &self.roots {
            // Direct scripts/ directory
            let direct_scripts = root.join("scripts");
            if let Ok(entries) = std::fs::read_dir(&direct_scripts) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("js")
                        && let Ok(content) = std::fs::read_to_string(&path)
                    {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("script.js")
                            .to_string();
                        scripts.push((name, content));
                    }
                }
            }

            // assets/<namespace>/scripts/ directories
            let assets_dir = root.join("assets");
            if let Ok(entries) = std::fs::read_dir(&assets_dir) {
                for entry in entries.flatten() {
                    let ns_scripts = entry.path().join("scripts");
                    if let Ok(js_entries) = std::fs::read_dir(&ns_scripts) {
                        for js_entry in js_entries.flatten() {
                            let path = js_entry.path();
                            if path.extension().and_then(|s| s.to_str()) == Some("js")
                                && let Ok(content) = std::fs::read_to_string(&path)
                            {
                                let name = path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("script.js")
                                    .to_string();
                                scripts.push((name, content));
                            }
                        }
                    }
                }
            }
        }
        scripts
    }
}
