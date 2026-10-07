//! 2D texture array baking and mipmap chain layout.

use std::collections::HashMap;

use crate::{image_buf::RgbaImage, mcmeta::AnimationDef};

/// Specification for copying a single mip slice from staging buffer to `VkImage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipCopyRegion {
    /// Byte offset into the staging buffer.
    pub buffer_offset: u64,
    /// Texture array layer index (slice).
    pub layer: u32,
    /// Mipmap level index (0 = full resolution).
    pub mip_level: u32,
    /// Width of this mip level in pixels.
    pub width: u32,
    /// Height of this mip level in pixels.
    pub height: u32,
    /// Total bytes of pixel data for this mip level (`width * height * 4`).
    pub byte_len: usize,
}

/// Contiguous pixel bytes and upload regions for a Vulkan 2D texture array.
#[derive(Debug, Clone)]
pub struct BakedTextureArray {
    /// Base resolution width and height of all layers.
    pub resolution: u32,
    /// Total number of 2D layers (slices) in the array.
    pub layer_count: u32,
    /// Total number of mip levels generated down to $1\times 1$.
    pub mip_levels: u32,
    /// Packed RGBA8 pixel bytes for all layers and all mips.
    pub pixel_data: Vec<u8>,
    /// Buffer copy regions ready for `vkCmdCopyBufferToImage`.
    pub copy_regions: Vec<MipCopyRegion>,
}

/// Metadata describing an animated sequence of consecutive texture layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimatedTextureInfo {
    /// First texture layer index in the baked array.
    pub base_layer: u32,
    /// Total number of animation frames.
    pub frame_count: u32,
    /// Duration of each frame in game ticks.
    pub frame_time: u32,
}

/// Builder that collects textures, enforces uniform resolution, and bakes mipmap chains.
#[derive(Debug, Clone)]
pub struct TextureArrayBuilder {
    resolution: u32,
    textures: Vec<(String, RgbaImage)>,
    name_to_index: HashMap<String, u32>,
    animations: HashMap<String, AnimatedTextureInfo>,
}

impl TextureArrayBuilder {
    /// Creates a new builder with the target base layer resolution (e.g. 16 or 32).
    #[must_use]
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            textures: Vec::new(),
            name_to_index: HashMap::new(),
            animations: HashMap::new(),
        }
    }

    /// Registers a texture into the array.
    ///
    /// If the texture was already inserted, returns the existing layer index.
    /// Rescales the image to `(resolution, resolution)` if dimensions differ.
    pub fn insert(&mut self, name: &str, image: RgbaImage) -> u32 {
        if let Some(&idx) = self.name_to_index.get(name) {
            return idx;
        }

        let scaled = if image.width != self.resolution || image.height != self.resolution {
            image.rescale(self.resolution, self.resolution)
        } else {
            image
        };

        #[allow(clippy::cast_possible_truncation)]
        let idx = self.textures.len() as u32;
        self.name_to_index.insert(name.to_string(), idx);
        self.textures.push((name.to_string(), scaled));
        idx
    }

    /// Registers an animated texture sequence, resolving custom frame ordering and frametime from `anim`.
    pub fn insert_animated_with_meta(
        &mut self,
        name: &str,
        physical_frames: &[RgbaImage],
        anim: Option<&AnimationDef>,
    ) -> AnimatedTextureInfo {
        if let Some(&info) = self.animations.get(name) {
            return info;
        }

        if physical_frames.is_empty() {
            let info = AnimatedTextureInfo {
                base_layer: 0,
                frame_count: 0,
                frame_time: 1,
            };
            self.animations.insert(name.to_string(), info);
            return info;
        }

        let frame_time = anim.map_or(2, |a| a.frametime.max(1));
        #[allow(clippy::cast_possible_truncation)]
        let frame_indices = anim.map_or_else(
            || (0..physical_frames.len() as u32).collect(),
            |a| a.resolve_frames(physical_frames.len() as u32),
        );

        #[allow(clippy::cast_possible_truncation)]
        let frame_count = frame_indices.len() as u32;
        let mut base_layer = 0;

        for (seq_idx, &phys_idx) in frame_indices.iter().enumerate() {
            let frame = &physical_frames[(phys_idx as usize) % physical_frames.len()];
            let scaled = if frame.width != self.resolution || frame.height != self.resolution {
                frame.rescale(self.resolution, self.resolution)
            } else {
                frame.clone()
            };

            #[allow(clippy::cast_possible_truncation)]
            let layer_idx = self.textures.len() as u32;
            if seq_idx == 0 {
                base_layer = layer_idx;
                self.name_to_index.insert(name.to_string(), base_layer);
            }
            self.textures.push((format!("{name}_{seq_idx}"), scaled));
        }

        let info = AnimatedTextureInfo {
            base_layer,
            frame_count,
            frame_time,
        };
        self.animations.insert(name.to_string(), info);
        info
    }

    /// Registers an animated texture sequence as consecutive layers in the array.
    pub fn insert_animated(
        &mut self,
        name: &str,
        frames: &[RgbaImage],
        frame_time: u32,
    ) -> AnimatedTextureInfo {
        let anim = AnimationDef {
            frametime: frame_time,
            interpolate: false,
            frames: None,
        };
        self.insert_animated_with_meta(name, frames, Some(&anim))
    }

    /// Number of registered texture layers.
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        (self.textures.len() as u32)
    }

    /// Layer index of a registered texture, if present.
    #[must_use]
    pub fn get_index(&self, name: &str) -> Option<u32> {
        self.name_to_index.get(name).copied()
    }

    /// Animation metadata for an animated texture, if registered.
    #[must_use]
    pub fn get_animation(&self, name: &str) -> Option<AnimatedTextureInfo> {
        self.animations.get(name).copied()
    }

    /// Bakes all texture layers into a contiguous memory layout with full mipmap chains.
    #[must_use]
    pub fn bake(&self) -> BakedTextureArray {
        let layer_count = self.layer_count().max(1);

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let mip_levels = (self.resolution as f32).log2().floor() as u32 + 1;

        let mut pixel_data = Vec::new();
        let mut copy_regions = Vec::new();

        // If no textures were inserted, provide 1 blank magenta/black checker placeholder
        let placeholder = RgbaImage::new(self.resolution, self.resolution);

        for layer in 0..layer_count {
            let img = if self.textures.is_empty() {
                &placeholder
            } else {
                &self.textures[layer as usize].1
            };

            let mips = img.generate_mips();

            for (mip_level, mip_img) in mips.into_iter().enumerate() {
                let buffer_offset = pixel_data.len() as u64;
                let byte_len = mip_img.data.len();

                copy_regions.push(MipCopyRegion {
                    buffer_offset,
                    layer,
                    #[allow(clippy::cast_possible_truncation)]
                    mip_level: mip_level as u32,
                    width: mip_img.width,
                    height: mip_img.height,
                    byte_len,
                });

                pixel_data.extend_from_slice(&mip_img.data);
            }
        }

        BakedTextureArray {
            resolution: self.resolution,
            layer_count,
            mip_levels,
            pixel_data,
            copy_regions,
        }
    }
}
