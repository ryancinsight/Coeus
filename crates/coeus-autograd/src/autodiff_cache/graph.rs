//! Computation-graph keys, metadata, and the type-erased topology-plan
//! interface backing [`super::ComputeGraphCache`].

use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::any::Any;
use std::sync::{Arc, Weak};

use super::snapshot::TopologyPlanSnapshot;

/// A key for identifying computation graph patterns.
///
/// This key is designed to efficiently identify when two computation graphs
/// have the same structure (topology, operations, and shapes).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ComputeGraphKey {
    /// Fingerprint of the computation graph structure.
    pub(super) fingerprint: u64,
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
pub(crate) struct CachedGraph {
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

/// Type-erased interface for a graph-local topology plan.
pub(crate) trait ErasedPlan: Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn root_is_alive(&self) -> bool;
    fn lru_access_tick(&self) -> u64;
    fn memory_bytes(&self) -> usize;
    fn snapshot(&self, root_id: usize, now: u64) -> TopologyPlanSnapshot;
    fn record_access(&mut self, access_tick: u64);
}

/// A topology plan retaining live nodes only while its graph root is alive.
pub(crate) struct TypedPlan<T: Scalar, B: ComputeBackend + Default> {
    pub(super) root: Weak<dyn BackwardNode<T, B>>,
    pub(super) fingerprint: u64,
    pub(super) graph_info: Arc<GraphInfo>,
    pub(super) order: Vec<Weak<dyn BackwardNode<T, B>>>,
    /// Number of times this plan has been used, including its insertion.
    pub(super) access_count: u64,
    /// Global topology-cache tick used only for LRU ordering.
    pub(super) last_access_tick: u64,
    pub(super) resident_since: u64,
    pub(super) memory_bytes: usize,
}

impl<T: Scalar + 'static, B: ComputeBackend + Default + 'static> ErasedPlan for TypedPlan<T, B> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn root_is_alive(&self) -> bool {
        self.root.strong_count() != 0
    }

    fn lru_access_tick(&self) -> u64 {
        self.last_access_tick
    }

    fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn snapshot(&self, root_id: usize, now: u64) -> TopologyPlanSnapshot {
        TopologyPlanSnapshot {
            root_id,
            fingerprint: self.fingerprint,
            node_count: self.graph_info.node_count,
            memory_bytes: self.memory_bytes,
            access_count: self.access_count,
            residency_age: now.saturating_sub(self.resident_since),
        }
    }

    fn record_access(&mut self, access_tick: u64) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_access_tick = access_tick;
    }
}

/// A topology-plan hit for the current live graph instance.
pub(crate) struct TopologyPlanHit<T: Scalar, B: ComputeBackend + Default> {
    /// Fingerprint recorded when the plan was built.
    pub fingerprint: u64,
    /// Metadata used if the generation-scoped metadata cache was invalidated.
    pub graph_info: Arc<GraphInfo>,
    /// Live post-order nodes for the current graph.
    pub order: Vec<Arc<dyn BackwardNode<T, B>>>,
}
