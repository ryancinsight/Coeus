//! Read-only cache observability snapshots: per-category memory, statistics,
//! and per-plan residency.

/// Per-category memory residency captured at snapshot time.
///
/// `metadata_bytes + plan_bytes == total_bytes` always holds; the breakdown is
/// captured in one consistent read so callers never recompute the split.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryBreakdown {
    /// Approximate memory retained by resident metadata entries (bytes).
    pub metadata_bytes: usize,
    /// Approximate memory retained by resident topology plans (bytes).
    pub plan_bytes: usize,
    /// Total resident cache memory (metadata + plans, bytes).
    pub total_bytes: usize,
}

/// A read-only view of cache residency at one point in time.
#[derive(Clone, Debug, Default)]
pub struct CacheSnapshot {
    /// Aggregate cache statistics captured with this snapshot.
    pub stats: CacheStats,
    /// Number of resident metadata entries.
    pub metadata_entries: usize,
    /// Per-category memory residency (metadata vs topology plans vs total).
    pub memory: MemoryBreakdown,
    /// Resident topology plans and their per-plan metadata.
    pub plans: Vec<TopologyPlanSnapshot>,
    /// Topology-plan reuse rate as a percentage at snapshot time.
    pub plan_hit_rate: f64,
    /// Total number of topology-plan lookups at snapshot time.
    pub total_plan_ops: u64,
}

/// Read-only residency information for one topology plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlanSnapshot {
    /// Opaque allocation identity of the graph root.
    pub root_id: usize,
    /// Structural fingerprint shared by equivalent graph patterns.
    pub fingerprint: u64,
    /// Number of nodes in the planned graph.
    pub node_count: usize,
    /// Approximate memory retained by this topology plan.
    pub memory_bytes: usize,
    /// Number of plan-cache accesses recorded for this plan.
    pub access_count: u64,
    /// Number of plan-cache access ticks since this plan was inserted.
    pub residency_age: u64,
}

/// Statistics for cache performance monitoring.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of cache misses requiring computation.
    pub misses: u64,
    /// Number of resident metadata entries.
    pub metadata_entries: usize,
    /// Number of cache invalidations due to generation change.
    pub invalidations: u64,
    /// Number of metadata entries evicted by the metadata LRU.
    pub metadata_evictions: u64,
    /// Approximate memory used by all resident metadata and topology plans (bytes).
    pub memory_bytes: usize,
    /// Number of topology plans reused for the same live graph instance.
    pub plan_hits: u64,
    /// Number of topology plan lookups that required graph traversal.
    pub plan_misses: u64,
    /// Number of topology plans currently resident.
    pub plan_entries: usize,
    /// Approximate memory retained by resident topology plans (bytes).
    pub plan_memory_bytes: usize,
    /// Highest number of resident topology plans observed since the last reset.
    ///
    /// Monotonic high-water mark: survives `clear()` and only resets via
    /// [`super::ComputeGraphCache::reset_stats`], so monitoring can see how close the
    /// plan table has come to `max_cache_entries` even after evictions and
    /// expired-root reclamation bring current residency back down.
    pub peak_plan_entries: usize,
    /// Highest topology-plan memory residency observed since the last reset.
    ///
    /// Monotonic high-water mark for `plan_memory_bytes`, useful for sizing
    /// `max_plan_memory` against real workloads.
    pub peak_plan_memory_bytes: usize,
    /// Number of topology plans evicted by the plan LRU.
    pub plan_evictions: u64,
    /// Number of topology plans reclaimed after their graph roots were dropped.
    pub plan_expirations: u64,
}

impl CacheStats {
    /// Get the cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get the total number of cache operations.
    pub fn total_ops(&self) -> u64 {
        self.hits + self.misses
    }

    /// Get the topology-plan reuse rate as a percentage.
    ///
    /// Measures how often a live graph instance reused its retained topology
    /// plan instead of being traversed again. A value below 100% indicates
    /// graphs that are rebuilt per iteration or plan-evicted graphs.
    pub fn plan_hit_rate(&self) -> f64 {
        let total = self.plan_hits + self.plan_misses;
        if total == 0 {
            0.0
        } else {
            (self.plan_hits as f64 / total as f64) * 100.0
        }
    }

    /// Get the total number of topology-plan lookups.
    pub fn total_plan_ops(&self) -> u64 {
        self.plan_hits + self.plan_misses
    }
}
