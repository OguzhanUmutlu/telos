//! High-performance lighting engine: 4-bit sky + 4-bit block lighting, heightmaps, and BFS flood-fill.

pub mod floodfill;
pub mod heightmap;
pub mod layer;

pub use floodfill::{LightBfs, pack_coord, pack_remove_node, unpack_coord, unpack_remove_node};
pub use heightmap::{COLUMN_AREA, ChunkHeightmap, ColumnHeights};
pub use layer::{ChunkLight, LIGHT_LAYER_BYTES, LightLayer};
