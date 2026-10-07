//! 3D Voxel navigation mesh, walkability evaluation, and hierarchical A* pathfinding.

pub mod astar;
pub mod path;
pub mod portal;
pub mod profile;
pub mod reader;
pub mod walkability;

pub use astar::{LocalAStar, octile_heuristic, update_mob_navigation_paths};
pub use path::NavPath;
pub use portal::{ChunkPortalGraph, ChunkPortals};
pub use profile::PathProfile;
pub use reader::NavWorldReader;
pub use walkability::{is_hazard, is_passable, is_solid_ground, is_walkable_node};
