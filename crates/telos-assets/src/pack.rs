//! Resource pack stack loader and asset resolution.

use std::path::{Path, PathBuf};

use crate::{
    error::AssetError,
    gui::{ContainerLayoutDef, GuiStyleSheet, NineSliceBorderDef},
    image_buf::RgbaImage,
    mcmeta::{AnimationDef, GuiMetaDef, GuiScaling},
};

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

    /// Finds a GUI sprite metadata file by name (e.g. `"container/slot"`, `"widget/button"`).
    #[must_use]
    pub fn find_gui_sprite_mcmeta(&self, name: &str) -> Option<PathBuf> {
        let rel_path = format!("textures/gui/sprites/{name}.png.mcmeta");
        self.find_texture(&rel_path)
    }

    /// Loads GUI metadata for a sprite from its `.mcmeta` file if present.
    #[must_use]
    pub fn load_gui_sprite_mcmeta(&self, name: &str) -> Option<GuiMetaDef> {
        let path = self.find_gui_sprite_mcmeta(name)?;
        let content = std::fs::read_to_string(&path).ok()?;
        GuiMetaDef::from_json_str(&content).ok()
    }

    /// Loads a container layout definition by name from mounted resource packs.
    ///
    /// Searches `assets/<namespace>/gui/containers/<name>.json` and `.ron`.
    #[must_use]
    pub fn load_container_layout(&self, name: &str) -> Option<ContainerLayoutDef> {
        for root in &self.roots {
            let assets_dir = root.join("assets");
            if let Ok(entries) = std::fs::read_dir(&assets_dir) {
                for entry in entries.flatten() {
                    let containers_dir = entry.path().join("gui").join("containers");
                    let json_file = containers_dir.join(format!("{name}.json"));
                    if json_file.is_file()
                        && let Ok(content) = std::fs::read_to_string(&json_file)
                        && let Ok(def) = ContainerLayoutDef::from_json_str(&content)
                    {
                        return Some(def);
                    }
                    let ron_file = containers_dir.join(format!("{name}.ron"));
                    if ron_file.is_file()
                        && let Ok(content) = std::fs::read_to_string(&ron_file)
                        && let Ok(def) = ContainerLayoutDef::from_ron_str(&content)
                    {
                        return Some(def);
                    }
                }
            }
        }
        None
    }

    fn scan_sprite_nine_slices(
        root: &Path,
        nine_slices: &mut std::collections::HashMap<String, NineSliceBorderDef>,
    ) {
        let mut candidates = Vec::new();
        candidates.push(root.join("textures").join("gui").join("sprites"));

        let assets_dir = root.join("assets");
        if let Ok(entries) = std::fs::read_dir(&assets_dir) {
            for entry in entries.flatten() {
                candidates.push(entry.path().join("textures").join("gui").join("sprites"));
            }
        }

        for base_dir in candidates {
            if base_dir.is_dir() {
                Self::walk_and_collect_mcmeta(&base_dir, &base_dir, nine_slices);
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn walk_and_collect_mcmeta(
        current_dir: &Path,
        base_dir: &Path,
        nine_slices: &mut std::collections::HashMap<String, NineSliceBorderDef>,
    ) {
        if let Ok(entries) = std::fs::read_dir(current_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::walk_and_collect_mcmeta(&path, base_dir, nine_slices);
                } else if path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.ends_with(".png.mcmeta"))
                    && let Ok(content) = std::fs::read_to_string(&path)
                    && let Ok(gui_meta) = GuiMetaDef::from_json_str(&content)
                    && let GuiScaling::NineSlice {
                        border,
                        stretch_inner,
                        ..
                    } = gui_meta.scaling
                    && let Ok(rel) = path.strip_prefix(base_dir)
                {
                    let rel_str = rel.to_string_lossy();
                    if let Some(sprite_name) = rel_str.strip_suffix(".png.mcmeta") {
                        nine_slices.insert(
                            sprite_name.to_string(),
                            NineSliceBorderDef {
                                left: border.left() as u8,
                                top: border.top() as u8,
                                right: border.right() as u8,
                                bottom: border.bottom() as u8,
                                stretch_inner,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Loads the combined GUI style sheet by merging mounted resource packs over default styles.
    #[must_use]
    pub fn load_gui_style_sheet(&self) -> GuiStyleSheet {
        let mut stylesheet = GuiStyleSheet::default();

        // Iterate in reverse root priority so higher-priority packs override lower-priority packs
        for root in self.roots.iter().rev() {
            // 1. Scan sprite nine-slice borders
            Self::scan_sprite_nine_slices(root, &mut stylesheet.nine_slices);

            // 2. Scan assets/<namespace>/gui/style.{json,ron}
            let assets_dir = root.join("assets");
            if let Ok(entries) = std::fs::read_dir(&assets_dir) {
                for entry in entries.flatten() {
                    let gui_dir = entry.path().join("gui");
                    let style_json = gui_dir.join("style.json");
                    if style_json.is_file()
                        && let Ok(content) = std::fs::read_to_string(&style_json)
                        && let Ok(sheet_override) = GuiStyleSheet::from_json_str(&content)
                    {
                        stylesheet.merge(sheet_override);
                    }
                    let style_ron = gui_dir.join("style.ron");
                    if style_ron.is_file()
                        && let Ok(content) = std::fs::read_to_string(&style_ron)
                        && let Ok(sheet_override) = GuiStyleSheet::from_ron_str(&content)
                    {
                        stylesheet.merge(sheet_override);
                    }

                    // 3. Scan assets/<namespace>/gui/containers/*.{json,ron}
                    let containers_dir = gui_dir.join("containers");
                    if let Ok(c_entries) = std::fs::read_dir(&containers_dir) {
                        for c_entry in c_entries.flatten() {
                            let path = c_entry.path();
                            let stem = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or_default();
                            let ext = path.extension().and_then(|s| s.to_str());
                            if ext == Some("json")
                                && let Ok(content) = std::fs::read_to_string(&path)
                                && let Ok(layout) = ContainerLayoutDef::from_json_str(&content)
                            {
                                stylesheet.override_container(stem, layout);
                            } else if ext == Some("ron")
                                && let Ok(content) = std::fs::read_to_string(&path)
                                && let Ok(layout) = ContainerLayoutDef::from_ron_str(&content)
                            {
                                stylesheet.override_container(stem, layout);
                            }
                        }
                    }
                }
            }
        }

        stylesheet
    }
}
