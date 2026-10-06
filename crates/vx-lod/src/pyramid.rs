//! Hierarchical in-memory LOD pyramid storage and downsampling coordination.

use hashbrown::HashMap;
use std::sync::Arc;

use crate::coords::LodNodeKey;
use crate::downsample::downsample_octants;
use crate::exterior::exterior_flood_fill;
use crate::node::{LodNode, LodVoxelSource};

/// Hierarchical multi-level pyramid storage for downsampled LOD nodes.
#[derive(Debug, Default)]
pub struct LodPyramid {
    nodes: HashMap<LodNodeKey, Arc<LodNode>>,
}

impl LodPyramid {
    /// Creates an empty `LodPyramid`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Total number of cached LOD nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Retrieves an LOD node by its key.
    #[must_use]
    pub fn get(&self, key: &LodNodeKey) -> Option<Arc<LodNode>> {
        self.nodes.get(key).cloned()
    }

    /// Inserts or replaces an LOD node in the pyramid.
    pub fn insert(&mut self, node: Arc<LodNode>) {
        self.nodes.insert(node.key, node);
    }

    /// Removes an LOD node by key.
    pub fn remove(&mut self, key: &LodNodeKey) -> Option<Arc<LodNode>> {
        self.nodes.remove(key)
    }

    /// Downsamples 8 children, runs exterior cave culling, and stores the resulting node in the pyramid.
    pub fn downsample_and_store<S: LodVoxelSource>(
        &mut self,
        key: LodNodeKey,
        children: &[Option<&S>; 8],
    ) -> Arc<LodNode> {
        let mut node = downsample_octants(key, children);
        exterior_flood_fill(&mut node);
        let arc = Arc::new(node);
        self.nodes.insert(key, arc.clone());
        arc
    }
}
