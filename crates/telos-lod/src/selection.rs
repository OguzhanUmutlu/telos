//! Clipmap LOD selection, distance thresholds, and node ring generation.

use glam::Vec3;
use hashbrown::HashSet;

use crate::coords::LodNodeKey;

/// Configuration for projected-voxel-size LOD selection and clipmap rings.
#[derive(Debug, Clone)]
pub struct LodClipmapConfig {
    /// Base switching radius in world blocks for Level 1 (default: 512.0 = 16 chunks).
    pub r0: f32,
    /// Maximum LOD level in the clipmap hierarchy (default: 6 = 32,768 blocks).
    pub max_level: u8,
    /// Hysteresis fraction to prevent border oscillation thrashing (default: 0.12).
    pub hysteresis: f32,
    /// Vertical node half-span above and below player level (default: 8 nodes).
    pub vertical_span: i32,
}

impl Default for LodClipmapConfig {
    fn default() -> Self {
        Self {
            r0: 512.0,
            max_level: 6,
            hysteresis: 0.12,
            vertical_span: 8,
        }
    }
}

/// Clipmap selector calculating desired LOD nodes and distance thresholds.
#[derive(Debug, Clone)]
pub struct LodClipmap {
    config: LodClipmapConfig,
}

impl LodClipmap {
    /// Creates a new `LodClipmap` with the specified configuration.
    #[must_use]
    pub const fn new(config: LodClipmapConfig) -> Self {
        Self { config }
    }

    /// Switching distance threshold in world blocks for level `L >= 1`.
    ///
    /// `T_L = R_0 * 2^(L - 1)`.
    #[inline]
    #[must_use]
    pub fn transition_distance(&self, level: u8) -> f32 {
        if level == 0 {
            0.0
        } else {
            self.config.r0 * (1 << (level - 1)) as f32
        }
    }

    /// Computes the set of active far-field LOD node keys ($L \ge 1$) covering the horizon around the camera.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn compute_desired_nodes(&self, camera_pos: Vec3) -> HashSet<LodNodeKey> {
        let mut result = HashSet::new();

        for level in 1..=self.config.max_level {
            let shift = 5 + i32::from(level);
            let node_size = 1 << shift;

            let cam_nx = (camera_pos.x as i32) >> shift;
            let cam_ny = (camera_pos.y as i32) >> shift;
            let cam_nz = (camera_pos.z as i32) >> shift;

            // In level-L node units, the ring extends from R_inner (R0 / 64) to R_outer (2 * R0 / 64)
            #[allow(clippy::cast_possible_truncation)]
            let r_inner = ((self.config.r0 / (node_size as f32 * 2.0)).floor() as i32).max(1);
            #[allow(clippy::cast_possible_truncation)]
            let r_outer =
                ((self.config.r0 * 2.0 / (node_size as f32)).ceil() as i32).max(r_inner + 1);

            let r_inner_sq = r_inner * r_inner;
            let r_outer_sq = r_outer * r_outer;

            for dz in -r_outer..=r_outer {
                for dx in -r_outer..=r_outer {
                    let dist_sq = dx * dx + dz * dz;
                    // Annulus ring selection
                    if dist_sq < r_inner_sq || dist_sq > r_outer_sq {
                        continue;
                    }

                    for dy in -self.config.vertical_span..=self.config.vertical_span {
                        let node_key =
                            LodNodeKey::new(level, cam_nx + dx, cam_ny + dy, cam_nz + dz);
                        result.insert(node_key);
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_transition_distances() {
        let clipmap = LodClipmap::new(LodClipmapConfig {
            r0: 512.0,
            ..Default::default()
        });

        assert_eq!(clipmap.transition_distance(0), 0.0);
        assert_eq!(clipmap.transition_distance(1), 512.0);
        assert_eq!(clipmap.transition_distance(2), 1024.0);
        assert_eq!(clipmap.transition_distance(3), 2048.0);
        assert_eq!(clipmap.transition_distance(4), 4096.0);
    }

    #[test]
    fn test_desired_nodes_generation() {
        let clipmap = LodClipmap::new(LodClipmapConfig {
            r0: 512.0,
            max_level: 2,
            hysteresis: 0.12,
            vertical_span: 1,
        });

        let nodes = clipmap.compute_desired_nodes(Vec3::new(0.0, 64.0, 0.0));
        assert!(!nodes.is_empty());
        assert!(nodes.iter().any(|k| k.level == 1));
        assert!(nodes.iter().any(|k| k.level == 2));
    }
}
