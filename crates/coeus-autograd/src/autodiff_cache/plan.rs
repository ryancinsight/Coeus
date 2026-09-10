//! Topology plans: the type-erased slot, its typed body, and a hit.
//!
//! A plan holds weak root references, so a dropped graph is never kept
//! alive by its cached topology.

use super::key::GraphInfo;
use super::stats::TopologyPlanSnapshot;
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::any::Any;
use std::sync::{Arc, Weak};

/// Type-erased interface for a graph-local topology plan.
pub(super) trait ErasedPlan: Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn root_is_alive(&self) -> bool;
    fn lru_access_tick(&self) -> u64;
    fn memory_bytes(&self) -> usize;
    fn snapshot(&self, root_id: usize, now: u64) -> TopologyPlanSnapshot;
    fn record_access(&mut self, access_tick: u64);
}

/// A topology plan retaining live nodes only while its graph root is alive.
pub(super) struct TypedPlan<T: Scalar, B: ComputeBackend + Default> {
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
