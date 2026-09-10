//! What a cached graph is keyed by, and what is stored under the key.

/// A key for identifying computation graph patterns.
///
/// This key is designed to efficiently identify when two computation graphs
/// have the same structure (topology, operations, and shapes).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ComputeGraphKey {
    /// Fingerprint of the computation graph structure.
    fingerprint: u64,
    /// Generation ID for invalidation (incremented on cache config changes).
    pub(super) generation: u32,
}

impl ComputeGraphKey {
    /// Create a new cache key with the given fingerprint and generation.
    pub fn new(fingerprint: u64, generation: u32) -> Self {
        Self {
            fingerprint,
            generation,
        }
    }

    /// Get the fingerprint.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Get the generation ID.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

/// Cached computation graph information.
#[derive(Clone, Debug)]
pub(super) struct CachedGraph {
    /// Serialized graph metadata used for verification and accounting.
    pub(super) graph_info: GraphInfo,
    /// Accounted memory used by this entry, including heap-backed fields.
    pub(super) memory_bytes: usize,
    /// Access count for LRU tracking.
    pub(super) access_count: u64,
}

/// Information about a computation graph structure.
#[derive(Clone, Debug)]
pub struct GraphInfo {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Number of leaf variables.
    pub leaf_count: usize,
    /// Maximum depth of the graph.
    pub max_depth: usize,
    /// Operation names in traversal order (for verification).
    pub op_sequence: Vec<String>,
}
