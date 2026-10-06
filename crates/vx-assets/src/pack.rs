//! Resource pack stack loader and asset resolution.

use std::path::{Path, PathBuf};

use crate::{error::AssetError, image_buf::RgbaImage};

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
    /// Searches each mounted pack in order under `assets/classic/textures/block/`
    /// and `assets/voxel/textures/block/`.
    #[must_use]
    pub fn find_block_texture(&self, name: &str) -> Option<PathBuf> {
        let rel_paths = [
            format!("assets/classic/textures/block/{name}.png"),
            format!("assets/voxel/textures/block/{name}.png"),
            format!("textures/block/{name}.png"),
        ];

        for root in &self.roots {
            for rel in &rel_paths {
                let candidate = root.join(rel);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }

        None
    }

    /// Loads and decodes a block texture from the mounted packs.
    pub fn load_block_texture(&self, name: &str) -> Result<RgbaImage, AssetError> {
        let path = self
            .find_block_texture(name)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::from_file(&path)
    }

    /// Loads and decodes all frames of an animated block texture strip from the mounted packs.
    pub fn load_animated_block_texture(&self, name: &str) -> Result<Vec<RgbaImage>, AssetError> {
        let path = self
            .find_block_texture(name)
            .ok_or_else(|| AssetError::MissingTexture(name.to_string()))?;

        RgbaImage::frames_from_file(&path)
    }

    /// Finds a generic asset texture across standard Classic Voxel and Voxel asset directories.
    #[must_use]
    pub fn find_texture(&self, rel_path: &str) -> Option<PathBuf> {
        let rel_paths = [
            format!("assets/classic/{rel_path}"),
            format!("assets/voxel/{rel_path}"),
            rel_path.to_string(),
        ];

        for root in &self.roots {
            for rel in &rel_paths {
                let candidate = root.join(rel);
                if candidate.is_file() {
                    return Some(candidate);
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
}
