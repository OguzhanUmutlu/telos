//! Material to texture array layer mapping table.

use std::collections::HashMap;

/// Maps a `(material_id, face_dir)` pair to a 2D texture array layer index.
#[derive(Debug, Clone, Default)]
pub struct MaterialTextureMap {
    entries: HashMap<(u16, u8), u32>,
}

impl MaterialTextureMap {
    /// Creates an empty mapping.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the texture array layer for a given material and face direction (0..=5).
    pub fn set(&mut self, material_id: u16, dir: u8, layer: u32) {
        self.entries.insert((material_id, dir), layer);
    }

    /// Sets the same texture layer for all 6 faces of a material.
    pub fn set_uniform(&mut self, material_id: u16, layer: u32) {
        for dir in 0..6 {
            self.set(material_id, dir, layer);
        }
    }

    /// Queries the texture array layer for a given material and face direction.
    ///
    /// Falls back to 0 if not explicitly registered.
    #[must_use]
    pub fn get(&self, material_id: u16, dir: u8) -> u32 {
        self.entries.get(&(material_id, dir)).copied().unwrap_or(0)
    }

    /// Exports the mapping as a flat array of `u32` suitable for uniform buffers or push constants.
    ///
    /// Layout: `table[material_id * 6 + dir] = layer`.
    #[must_use]
    pub fn to_flat_table(&self, max_material_id: u16) -> Vec<u32> {
        let size = (max_material_id as usize + 1) * 6;
        let mut table = vec![0u32; size];
        for material in 0..=max_material_id {
            for dir in 0..6 {
                let idx = (material as usize) * 6 + (dir as usize);
                table[idx] = self.get(material, dir);
            }
        }
        table
    }
}
