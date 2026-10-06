//! Seed of Andromeda double-queue BFS flood-fill lighting propagation.

use crate::light::heightmap::ColumnHeights;
use crate::light::layer::LightLayer;
use std::collections::VecDeque;

/// Offset applied to coordinate components to ensure non-negative representation in `[-32, 63]`.
const COORD_OFFSET: i32 = 32;

/// Packs `(x, y, z)` where coordinates are in `[-32, 63]`.
#[inline]
#[must_use]
pub const fn pack_coord(x: i32, y: i32, z: i32) -> u32 {
    ((x + COORD_OFFSET) as u32 & 0x7F)
        | (((y + COORD_OFFSET) as u32 & 0x7F) << 7)
        | (((z + COORD_OFFSET) as u32 & 0x7F) << 14)
}

/// Unpacks `(x, y, z)` from a packed coordinate.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub const fn unpack_coord(packed: u32) -> (i32, i32, i32) {
    let x = (packed & 0x7F) as i32 - COORD_OFFSET;
    let y = ((packed >> 7) & 0x7F) as i32 - COORD_OFFSET;
    let z = ((packed >> 14) & 0x7F) as i32 - COORD_OFFSET;
    (x, y, z)
}

/// Packs `(x, y, z, val)` for remove queue.
#[inline]
#[must_use]
pub const fn pack_remove_node(x: i32, y: i32, z: i32, val: u8) -> u32 {
    pack_coord(x, y, z) | ((val as u32 & 0x0F) << 21)
}

/// Unpacks `(x, y, z, val)` from a remove queue node.
#[inline]
#[must_use]
pub const fn unpack_remove_node(packed: u32) -> (i32, i32, i32, u8) {
    let (x, y, z) = unpack_coord(packed);
    let val = ((packed >> 21) & 0x0F) as u8;
    (x, y, z, val)
}

const NEIGHBORS_6: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Reusable scratch queues for BFS light propagation without per-call heap allocation.
#[derive(Debug, Default)]
pub struct LightBfs {
    add_queue: VecDeque<u32>,
    remove_queue: VecDeque<u32>,
}

impl LightBfs {
    /// Creates a new `LightBfs` with initial queue capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            add_queue: VecDeque::with_capacity(512),
            remove_queue: VecDeque::with_capacity(512),
        }
    }

    /// Clears both queues.
    pub fn clear(&mut self) {
        self.add_queue.clear();
        self.remove_queue.clear();
    }

    /// Propagates block light additions within a single chunk.
    ///
    /// - `light`: Mutable reference to the chunk's block light layer.
    /// - `get_opacity`: Closure returning opacity (0 = transparent air, 15 = fully opaque solid).
    #[allow(clippy::manual_range_contains)]
    pub fn propagate_block_add<F>(&mut self, light: &mut LightLayer, mut get_opacity: F)
    where
        F: FnMut(usize) -> u8,
    {
        while let Some(packed) = self.add_queue.pop_front() {
            let (x, y, z) = unpack_coord(packed);
            if x < 0 || x >= 32 || y < 0 || y >= 32 || z < 0 || z >= 32 {
                continue;
            }
            let idx = ((y << 10) | (z << 5) | x) as usize;
            let current_level = light.get(idx);
            if current_level == 0 {
                continue;
            }

            for &(dx, dy, dz) in &NEIGHBORS_6 {
                let nx = x + dx;
                let ny = y + dy;
                let nz = z + dz;
                if nx < 0 || nx >= 32 || ny < 0 || ny >= 32 || nz < 0 || nz >= 32 {
                    continue;
                }
                let n_idx = ((ny << 10) | (nz << 5) | nx) as usize;
                let opacity = get_opacity(n_idx);
                if opacity >= 15 {
                    continue;
                }
                let step_cost = opacity.max(1);
                if current_level > step_cost {
                    let new_level = current_level - step_cost;
                    if light.get(n_idx) < new_level {
                        light.set(n_idx, new_level);
                        self.add_queue.push_back(pack_coord(nx, ny, nz));
                    }
                }
            }
        }
    }

    /// Enqueues a light source addition at `(x, y, z)` with initial `emission`.
    #[allow(clippy::cast_possible_wrap)]
    pub fn add_source(&mut self, light: &mut LightLayer, x: u32, y: u32, z: u32, emission: u8) {
        let emission = emission & 0x0F;
        let idx = ((y << 10) | (z << 5) | x) as usize;
        light.set(idx, emission);
        self.add_queue
            .push_back(pack_coord(x as i32, y as i32, z as i32));
    }

    /// Propagates block light removal (e.g. torch broken) using Seed of Andromeda double-queue BFS.
    #[allow(clippy::manual_range_contains)]
    pub fn propagate_block_remove<F>(&mut self, light: &mut LightLayer, get_opacity: F)
    where
        F: FnMut(usize) -> u8,
    {
        while let Some(packed) = self.remove_queue.pop_front() {
            let (x, y, z, old_level) = unpack_remove_node(packed);
            if x < 0 || x >= 32 || y < 0 || y >= 32 || z < 0 || z >= 32 {
                continue;
            }

            for &(dx, dy, dz) in &NEIGHBORS_6 {
                let nx = x + dx;
                let ny = y + dy;
                let nz = z + dz;
                if nx < 0 || nx >= 32 || ny < 0 || ny >= 32 || nz < 0 || nz >= 32 {
                    continue;
                }
                let n_idx = ((ny << 10) | (nz << 5) | nx) as usize;
                let neighbor_level = light.get(n_idx);

                if neighbor_level != 0 && neighbor_level < old_level {
                    light.set(n_idx, 0);
                    self.remove_queue
                        .push_back(pack_remove_node(nx, ny, nz, neighbor_level));
                } else if neighbor_level >= old_level {
                    // Another source or brighter neighbor survived: re-propagate from here!
                    self.add_queue.push_back(pack_coord(nx, ny, nz));
                }
            }
        }

        // Now run the add propagation queue to re-fill light from surviving sources
        self.propagate_block_add(light, get_opacity);
    }

    /// Enqueues removal of a light source at `(x, y, z)`.
    #[allow(clippy::cast_possible_wrap)]
    pub fn remove_source(&mut self, light: &mut LightLayer, x: u32, y: u32, z: u32) {
        let idx = ((y << 10) | (z << 5) | x) as usize;
        let old_level = light.get(idx);
        if old_level > 0 {
            light.set(idx, 0);
            self.remove_queue
                .push_back(pack_remove_node(x as i32, y as i32, z as i32, old_level));
        }
    }

    /// Computes and propagates initial sky lighting for a chunk given column heightmaps.
    #[allow(clippy::cast_possible_wrap)]
    pub fn compute_initial_sky_light<F>(
        &mut self,
        chunk_cy: i32,
        heights: &ColumnHeights,
        light: &mut LightLayer,
        mut get_opacity: F,
    ) where
        F: FnMut(usize) -> u8,
    {
        let chunk_base_y = chunk_cy << 5;

        // 1. Mark all voxels above heightmap with 15 (direct open sky)
        // Also find boundary voxels that can diffuse light into caves / overhangs
        for z in 0..32 {
            for x in 0..32 {
                let top_y = heights.get_top(x, z);
                for y in (0..32).rev() {
                    let world_y = chunk_base_y + y as i32;
                    let idx = ((y << 10) | (z << 5) | x) as usize;
                    let opacity = get_opacity(idx);

                    if world_y > i32::from(top_y) {
                        light.set(idx, 15);
                    } else if opacity < 15 {
                        // Below heightmap: if it's transparent, it might receive sunlight
                        // from adjacent open voxels (or straight down if top was higher)
                        light.set(idx, 0);
                    } else {
                        // Opaque solid
                        light.set(idx, 0);
                    }
                }
            }
        }

        // 2. Seed BFS queue from open-sky voxels adjacent to shaded voxels
        for z in 0..32i32 {
            for x in 0..32i32 {
                for y in 0..32i32 {
                    let idx = ((y << 10) | (z << 5) | x) as usize;
                    if light.get(idx) == 15 {
                        // Check if any horizontal or downward neighbor is < 15 and transparent
                        let mut has_shaded_neighbor = false;
                        for &(dx, dy, dz) in &NEIGHBORS_6 {
                            let nx = x + dx;
                            let ny = y + dy;
                            let nz = z + dz;
                            if (0..32).contains(&nx)
                                && (0..32).contains(&ny)
                                && (0..32).contains(&nz)
                            {
                                let n_idx = ((ny << 10) | (nz << 5) | nx) as usize;
                                if light.get(n_idx) < 14 && get_opacity(n_idx) < 15 {
                                    has_shaded_neighbor = true;
                                    break;
                                }
                            }
                        }
                        if has_shaded_neighbor {
                            self.add_queue.push_back(pack_coord(x, y, z));
                        }
                    }
                }
            }
        }

        // 3. Propagate diffusion through transparent overhangs
        self.propagate_block_add(light, get_opacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_attenuation() {
        let mut light = LightLayer::zero();
        let mut bfs = LightBfs::new();

        // Air everywhere (opacity = 0)
        let get_opacity = |_idx| 0u8;

        // Place torch at center (16, 16, 16) with level 15
        bfs.add_source(&mut light, 16, 16, 16, 15);
        bfs.propagate_block_add(&mut light, get_opacity);

        // Center must be 15
        let center_idx = crate::coords::LocalIdx::from_coords_unchecked(16, 16, 16).as_usize();
        assert_eq!(light.get(center_idx), 15);

        // 1 block away = 14
        let neighbor_idx = crate::coords::LocalIdx::from_coords_unchecked(17, 16, 16).as_usize();
        assert_eq!(light.get(neighbor_idx), 14);

        // 2 blocks away = 13
        let n2_idx = crate::coords::LocalIdx::from_coords_unchecked(18, 16, 16).as_usize();
        assert_eq!(light.get(n2_idx), 13);

        // 14 blocks away = 1
        let n14_idx = crate::coords::LocalIdx::from_coords_unchecked(30, 16, 16).as_usize();
        assert_eq!(light.get(n14_idx), 1);

        // 15 blocks away = 0
        let n15_idx = crate::coords::LocalIdx::from_coords_unchecked(31, 16, 16).as_usize();
        assert_eq!(light.get(n15_idx), 0);
    }

    #[test]
    fn test_light_removal_and_reflood() {
        let mut light = LightLayer::zero();
        let mut bfs = LightBfs::new();
        let get_opacity = |_idx| 0u8;

        // Place torch A at (10, 16, 16) level 15
        bfs.add_source(&mut light, 10, 16, 16, 15);
        // Place torch B at (14, 16, 16) level 15
        bfs.add_source(&mut light, 14, 16, 16, 15);
        bfs.propagate_block_add(&mut light, get_opacity);

        // Point between them (12, 16, 16): distance 2 from both -> level 13
        let mid_idx = crate::coords::LocalIdx::from_coords_unchecked(12, 16, 16).as_usize();
        assert_eq!(light.get(mid_idx), 13);

        // Remove torch A
        bfs.remove_source(&mut light, 10, 16, 16);
        bfs.propagate_block_remove(&mut light, get_opacity);

        // Torch A position is now lit by Torch B!
        // Distance from Torch B (14) to Torch A position (10) is 4 blocks -> level = 15 - 4 = 11!
        let torch_a_idx = crate::coords::LocalIdx::from_coords_unchecked(10, 16, 16).as_usize();
        assert_eq!(light.get(torch_a_idx), 11);
    }
}
