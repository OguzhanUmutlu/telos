//! CPU RGBA8 image buffer with PNG decoding and linear-light mipmap downsampling.

use std::{fs, path::Path};

use crate::error::AssetError;

/// Simple CPU-side 8-bit RGBA pixel buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Width of the image in pixels.
    pub width: u32,
    /// Height of the image in pixels.
    pub height: u32,
    /// Raw RGBA8 pixel bytes (length is always `width * height * 4`).
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Creates a new blank image filled with transparent black.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width * height * 4) as usize],
        }
    }

    /// Creates an image filled with a solid RGBA color.
    #[must_use]
    pub fn from_color(width: u32, height: u32, color: [u8; 4]) -> Self {
        let pixel_count = (width * height) as usize;
        let mut data = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            data.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Generates a procedural fallback texture with a beveled border and subtle pseudo-noise.
    ///
    /// This ensures custom blocks without PNG assets render distinctly and legibly.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn procedural_pattern(width: u32, height: u32, base_color: [u8; 4], seed: u32) -> Self {
        let mut img = Self::new(width, height);
        let [r, g, b, a] = base_color;

        for y in 0..height {
            for x in 0..width {
                let is_top_left = x == 0 || y == 0;
                let is_bottom_right = x == width - 1 || y == height - 1;
                let is_border = is_top_left || is_bottom_right;

                // Fast hash-based pseudo-noise
                let h = ((x.wrapping_mul(374_761_393)
                    ^ y.wrapping_mul(668_265_263)
                    ^ seed.wrapping_mul(1_274_126_177))
                    >> 16)
                    & 0x1F;
                let noise_offset = (h as i32) - 16; // -16..=15

                let factor = if is_top_left {
                    24
                } else if is_bottom_right {
                    -24
                } else if is_border {
                    -12
                } else {
                    noise_offset
                };

                let pr = (i32::from(r) + factor).clamp(0, 255) as u8;
                let pg = (i32::from(g) + factor).clamp(0, 255) as u8;
                let pb = (i32::from(b) + factor).clamp(0, 255) as u8;

                let idx = ((y * width + x) * 4) as usize;
                img.data[idx] = pr;
                img.data[idx + 1] = pg;
                img.data[idx + 2] = pb;
                img.data[idx + 3] = a;
            }
        }
        img
    }

    /// Loads and decodes a PNG file from disk into an RGBA8 buffer.
    ///
    /// If the texture is an animated strip where `height > width`, the first square frame
    /// `width * width` is extracted as the static placeholder frame.
    pub fn from_file(path: &Path) -> Result<Self, AssetError> {
        let bytes = fs::read(path).map_err(|source| AssetError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        Self::from_png_bytes(&bytes, path)
    }

    /// Loads and decodes a PNG file without any animation strip cropping.
    pub fn from_file_exact(path: &Path) -> Result<Self, AssetError> {
        let bytes = fs::read(path).map_err(|source| AssetError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let img = image::load_from_memory(&bytes).map_err(|e| AssetError::Decode {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let rgba = img.to_rgba8();
        Ok(Self {
            width: rgba.width(),
            height: rgba.height(),
            data: rgba.into_raw(),
        })
    }

    /// Loads an animated PNG strip from disk and returns all constituent square frames.
    pub fn frames_from_file(path: &Path) -> Result<Vec<Self>, AssetError> {
        let bytes = fs::read(path).map_err(|source| AssetError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        Self::frames_from_png_bytes(&bytes, path)
    }

    /// Decodes all constituent square frames from an animated PNG byte buffer.
    pub fn frames_from_png_bytes(bytes: &[u8], path: &Path) -> Result<Vec<Self>, AssetError> {
        let img = image::load_from_memory(bytes).map_err(|e| AssetError::Decode {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

        let rgba = img.to_rgba8();
        let width = rgba.width();
        let full_height = rgba.height();
        let raw_pixels = rgba.into_raw();

        if full_height >= width && full_height % width == 0 {
            let frame_count = full_height / width;
            let frame_byte_len = (width * width * 4) as usize;
            let mut frames = Vec::with_capacity(frame_count as usize);

            for f in 0..frame_count {
                let start = (f as usize) * frame_byte_len;
                let end = start + frame_byte_len;
                frames.push(Self {
                    width,
                    height: width,
                    data: raw_pixels[start..end].to_vec(),
                });
            }
            Ok(frames)
        } else {
            Ok(vec![Self {
                width,
                height: full_height,
                data: raw_pixels,
            }])
        }
    }

    /// Decodes PNG byte data into an RGBA8 buffer.
    pub fn from_png_bytes(bytes: &[u8], path: &Path) -> Result<Self, AssetError> {
        let img = image::load_from_memory(bytes).map_err(|e| AssetError::Decode {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

        let rgba = img.to_rgba8();
        let width = rgba.width();
        let full_height = rgba.height();

        // If texture is an animated strip (e.g. 16x32 or 16x512), use the first square frame
        let height = if full_height >= width && full_height % width == 0 {
            width
        } else {
            full_height
        };

        let raw_pixels = rgba.into_raw();
        let frame_byte_len = (width * height * 4) as usize;
        let data = raw_pixels[..frame_byte_len].to_vec();

        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Resamples the image to `(target_width, target_height)` using bilinear interpolation.
    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn rescale(&self, target_width: u32, target_height: u32) -> Self {
        if self.width == target_width && self.height == target_height {
            return self.clone();
        }

        let mut output = Self::new(target_width, target_height);

        for dy in 0..target_height {
            for dx in 0..target_width {
                #[allow(clippy::cast_precision_loss)]
                let src_x = (dx as f32 + 0.5) * (self.width as f32 / target_width as f32) - 0.5;
                #[allow(clippy::cast_precision_loss)]
                let src_y = (dy as f32 + 0.5) * (self.height as f32 / target_height as f32) - 0.5;

                let x0 = (src_x.floor() as i32).clamp(0, self.width as i32 - 1) as u32;
                let y0 = (src_y.floor() as i32).clamp(0, self.height as i32 - 1) as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);

                let fx = src_x - src_x.floor();
                let fy = src_y - src_y.floor();

                let p00 = self.get_pixel(x0, y0);
                let p10 = self.get_pixel(x1, y0);
                let p01 = self.get_pixel(x0, y1);
                let p11 = self.get_pixel(x1, y1);

                let mut out_pixel = [0u8; 4];
                for c in 0..4 {
                    #[allow(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss
                    )]
                    let val = (1.0 - fx) * (1.0 - fy) * f32::from(p00[c])
                        + fx * (1.0 - fy) * f32::from(p10[c])
                        + (1.0 - fx) * fy * f32::from(p01[c])
                        + fx * fy * f32::from(p11[c]);
                    out_pixel[c] = val.round().clamp(0.0, 255.0) as u8;
                }

                output.set_pixel(dx, dy, out_pixel);
            }
        }

        output
    }

    /// Generates a full mipmap chain from level 0 (the image itself) down to $1\times 1$.
    ///
    /// Filtering is performed in linear space with color-bleed protection for alpha cutouts.
    #[must_use]
    pub fn generate_mips(&self) -> Vec<Self> {
        let max_dim = self.width.max(self.height);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let num_mips = (max_dim as f32).log2().floor() as u32 + 1;

        let mut mips = Vec::with_capacity(num_mips as usize);
        mips.push(self.clone());

        let mut current = self.clone();

        for _ in 1..num_mips {
            let next_w = (current.width / 2).max(1);
            let next_h = (current.height / 2).max(1);

            let mut next = Self::new(next_w, next_h);

            for ny in 0..next_h {
                for nx in 0..next_w {
                    let sx0 = (nx * 2).min(current.width - 1);
                    let sy0 = (ny * 2).min(current.height - 1);
                    let sx1 = (sx0 + 1).min(current.width - 1);
                    let sy1 = (sy0 + 1).min(current.height - 1);

                    let samples = [
                        current.get_pixel(sx0, sy0),
                        current.get_pixel(sx1, sy0),
                        current.get_pixel(sx0, sy1),
                        current.get_pixel(sx1, sy1),
                    ];

                    let mut total_alpha = 0.0f32;
                    let mut r_acc = 0.0f32;
                    let mut g_acc = 0.0f32;
                    let mut b_acc = 0.0f32;

                    for s in &samples {
                        let a = f32::from(s[3]) / 255.0;
                        total_alpha += a;
                        // Linear light conversion (gamma 2.2 approximation)
                        let r = (f32::from(s[0]) / 255.0).powf(2.2) * a;
                        let g = (f32::from(s[1]) / 255.0).powf(2.2) * a;
                        let b = (f32::from(s[2]) / 255.0).powf(2.2) * a;
                        r_acc += r;
                        g_acc += g;
                        b_acc += b;
                    }

                    let avg_alpha = total_alpha / 4.0;
                    let (out_r, out_g, out_b) = if total_alpha > 0.0 {
                        let r_lin = (r_acc / total_alpha).powf(1.0 / 2.2);
                        let g_lin = (g_acc / total_alpha).powf(1.0 / 2.2);
                        let b_lin = (b_acc / total_alpha).powf(1.0 / 2.2);
                        (
                            (r_lin * 255.0).round().clamp(0.0, 255.0) as u8,
                            (g_lin * 255.0).round().clamp(0.0, 255.0) as u8,
                            (b_lin * 255.0).round().clamp(0.0, 255.0) as u8,
                        )
                    } else {
                        (0, 0, 0)
                    };

                    let out_a = (avg_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
                    next.set_pixel(nx, ny, [out_r, out_g, out_b, out_a]);
                }
            }

            mips.push(next.clone());
            current = next;
        }

        mips
    }

    #[inline]
    fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) * 4) as usize;
        [
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ]
    }

    #[inline]
    fn set_pixel(&mut self, x: u32, y: u32, pixel: [u8; 4]) {
        let idx = ((y * self.width + x) * 4) as usize;
        self.data[idx..idx + 4].copy_from_slice(&pixel);
    }
}
