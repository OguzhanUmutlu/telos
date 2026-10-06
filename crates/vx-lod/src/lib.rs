//! Far-field terrain Level of Detail (LOD) clipmap hierarchy, downsampling, and meshing.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod color;
pub mod coords;
pub mod downsample;
pub mod exterior;
pub mod mesher;
pub mod node;
pub mod pyramid;
pub mod quad;
pub mod selection;

pub use color::{LodColor, LodColorTable};
pub use coords::LodNodeKey;
pub use downsample::downsample_octants;
pub use exterior::exterior_flood_fill;
pub use mesher::{LodMesh, mesh_lod_node};
pub use node::{LodNode, LodVoxelSource};
pub use pyramid::LodPyramid;
pub use quad::LodQuad;
pub use selection::{LodClipmap, LodClipmapConfig};
