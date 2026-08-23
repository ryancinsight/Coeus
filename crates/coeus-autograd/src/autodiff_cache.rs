//! Computation graph caching for autodiff.
//!
//! This module provides a memoization cache for computation graphs to reduce
//! autodiff compilation overhead in iterative solvers by caching and reusing
//! computation graph topologies for repeated operation patterns.
//!
//! # Design
//!
//! The cache works by:
//! 1. Computing a fingerprint of the computation graph structure
//! 2. Keying on input shapes, operation sequence, and backend type
//! 3. Storing graph metadata for repeated-pattern accounting
//! 4. Using LRU eviction with generation-based validation
//!
//! # Performance
//!
//! - Cache hits: reuse graph metadata and, for the same live root, its topology plan
//! - Cache misses: compute graph metadata and cache it for future iterations
//! - Topology plans retain weak node references, so dropped graphs are not kept alive
//! - Structurally equivalent but separately allocated graphs still build separate plans

use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};

/// Default plan-table purge period: once the table is large enough to bother
/// deferring, full expired-plan scans run every Nth operation instead of on
/// every lookup. Override via [`CacheConfig::plan_purge_interval`].
const PLAN_PURGE_INTERVAL: u64 = 64;

/// Minimum plan-table size at which deferred (amortized) purging engages.
///
/// Below this, scans are cheap and run on every operation so residency
/// accounting stays exact. Exposed so benchmarks and callers can reference the
/// same threshold the cache uses instead of re-deriving it.
pub const PLAN_PURGE_MIN_TABLE_SIZE: usize = 64;

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
    /// [`ComputeGraphCache::reset_stats`], so monitoring can see how close the
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

/// A key for identifying computation graph patterns.
///
/// This key is designed to efficiently identify when two computation graphs
/// have the same structure (topology, operations, and shapes).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ComputeGraphKey {
    /// Fingerprint of the computation graph structure.
    fingerprint: u64,
    /// Generation ID for invalidation (incremented on cache config changes).
    generation: u32,
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
struct CachedGraph {
    /// Serialized graph metadata used for verification and accounting.
    graph_info: GraphInfo,
    /// Accounted memory used by this entry, including heap-backed fields.
    memory_bytes: usize,
    /// Access count for LRU tracking.
    access_count: u64,
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
trait ErasedPlan: Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn root_is_alive(&self) -> bool;
    fn lru_access_tick(&self) -> u64;
    fn memory_bytes(&self) -> usize;
    fn snapshot(&self, root_id: usize, now: u64) -> TopologyPlanSnapshot;
    fn record_access(&mut self, access_tick: u64);
}

/// A topology plan retaining live nodes only while its graph root is alive.
struct TypedPlan<T: Scalar, B: ComputeBackend + Default> {
    root: Weak<dyn BackwardNode<T, B>>,
    fingerprint: u64,
    graph_info: Arc<GraphInfo>,
    order: Vec<Weak<dyn BackwardNode<T, B>>>,
    /// Number of times this plan has been used, including its insertion.
    access_count: u64,
    /// Global topology-cache tick used only for LRU ordering.
    last_access_tick: u64,
    resident_since: u64,
    memory_bytes: usize,
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

/// Configuration trait for cache behavior.
///
/// Allows solver-specific customization of cache parameters.
pub trait CacheConfig: Send + Sync {
    /// Maximum number of cached graphs.
    fn max_cache_entries(&self) -> usize {
        1024
    }

    /// Maximum memory per cached entry (bytes).
    fn max_entry_memory(&self) -> usize {
        1024 * 1024 // 1MB default
    }

    /// Maximum aggregate memory for cached metadata entries (bytes).
    ///
    /// The default is unlimited so existing configurations retain the previous
    /// per-entry-only behavior. Override this to bound metadata residency.
    fn max_metadata_memory(&self) -> usize {
        usize::MAX
    }

    /// Maximum aggregate memory for resident topology plans (bytes).
    ///
    /// The default is unlimited so existing configurations retain the previous
    /// per-entry-only behavior. Override this to bound topology-plan residency.
    fn max_plan_memory(&self) -> usize {
        usize::MAX
    }

    /// Amortized expired-plan purge period (plan operations).
    ///
    /// Small plan tables (below the fixed 64-entry minimum) are purged exactly
    /// on every operation. Once the table is at or above that size, the full
    /// expired-plan scan is deferred to every Nth operation, keeping repeated
    /// lookups amortized O(1). A value of `0` disables deferral entirely (purge
    /// on every operation, so residency counters are always exact). Tune lower
    /// for fresher residency accounting, higher for cheaper lookups on large
    /// plan tables.
    fn plan_purge_interval(&self) -> u64 {
        PLAN_PURGE_INTERVAL
    }

    /// Whether to enable caching.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Cache generation ID (invalidates all entries when incremented).
    fn generation(&self) -> u32 {
        0
    }
}

/// Default cache configuration.
#[derive(Clone, Debug, Default)]
pub struct DefaultCacheConfig;

impl CacheConfig for DefaultCacheConfig {
    fn max_cache_entries(&self) -> usize {
        1024
    }

    fn max_entry_memory(&self) -> usize {
        1024 * 1024
    }

    fn is_enabled(&self) -> bool {
        true
    }

    fn generation(&self) -> u32 {
        0
    }
}

/// Thread-safe computation graph cache.
///
/// Every field is `Arc`-shared and lock-protected, and erased topology plans
/// are `Send + Sync`, so a single instance may be shared across threads — for
/// example data-parallel training loops where each worker trains its own graph
/// against one cache. Clone the cache per worker (clones share the same
/// underlying state) or pass it through an `Arc`, then drive it with
/// [`topological_sort_with_cache`](crate::backward_cache::topological_sort_with_cache).
/// The default [`Var::backward`](crate::var::Var::backward) path instead uses a
/// per-thread cache; cross-thread sharing is exercised by the integration test
/// `test_shared_cache_across_threads`.
pub struct ComputeGraphCache {
    /// The actual cache storage.
    cache: Arc<RwLock<HashMap<ComputeGraphKey, CachedGraph>>>,
    /// Live-graph topology plans keyed by root allocation identity.
    plans: Arc<RwLock<HashMap<usize, Box<dyn ErasedPlan>>>>,
    /// Monotonic access counter for topology-plan LRU tracking.
    plan_access_counter: Arc<RwLock<u64>>,
    /// Plan operations since the last deferred expired-plan scan.
    plan_purge_ops: Arc<AtomicU64>,
    /// Amortized purge interval captured from the cache configuration.
    plan_purge_interval: u64,
    /// Cache statistics.
    stats: Arc<RwLock<CacheStats>>,
    /// Configuration.
    config: Arc<dyn CacheConfig>,
    /// Global access counter for LRU tracking.
    access_counter: Arc<RwLock<u64>>,
    /// Last generation whose entries are resident in the cache.
    active_generation: Arc<AtomicU32>,
}

impl ComputeGraphCache {
    /// Create a new computation graph cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(Arc::new(DefaultCacheConfig))
    }

    /// Create a new computation graph cache with a custom configuration.
    pub fn with_config(config: Arc<dyn CacheConfig>) -> Self {
        let generation = config.generation();
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            plans: Arc::new(RwLock::new(HashMap::new())),
            plan_access_counter: Arc::new(RwLock::new(0)),
            plan_purge_ops: Arc::new(AtomicU64::new(0)),
            plan_purge_interval: config.plan_purge_interval(),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            config,
            access_counter: Arc::new(RwLock::new(0)),
            active_generation: Arc::new(AtomicU32::new(generation)),
        }
    }

    /// Look up a topology plan for the exact live graph rooted at `root`.
    pub(crate) fn lookup_plan<T: Scalar + 'static, B: ComputeBackend + Default + 'static>(
        &self,
        root: &Arc<dyn BackwardNode<T, B>>,
    ) -> Option<TopologyPlanHit<T, B>> {
        if !self.config.is_enabled() || self.config.max_cache_entries() == 0 {
            let mut stats = self.stats.write().expect("stats lock poisoned");
            stats.plan_misses = stats.plan_misses.saturating_add(1);
            return None;
        }

        let key = Arc::as_ptr(root) as *const () as usize;
        let mut plans = self.plans.write().expect("plan lock poisoned");
        self.purge_expired_plans_amortized(&mut plans);

        let hit = plans.get_mut(&key).and_then(|entry| {
            let plan = entry.as_any_mut().downcast_mut::<TypedPlan<T, B>>()?;
            let live_root = plan.root.upgrade()?;
            if !Arc::ptr_eq(&live_root, root) {
                return None;
            }

            let order = plan
                .order
                .iter()
                .map(Weak::upgrade)
                .collect::<Option<Vec<_>>>()?;
            let mut counter = self
                .plan_access_counter
                .write()
                .expect("plan counter lock poisoned");
            *counter = counter.saturating_add(1);
            plan.record_access(*counter);

            Some(TopologyPlanHit {
                fingerprint: plan.fingerprint,
                graph_info: Arc::clone(&plan.graph_info),
                order,
            })
        });

        let mut stats = self.stats.write().expect("stats lock poisoned");
        if hit.is_some() {
            stats.plan_hits = stats.plan_hits.saturating_add(1);
        } else {
            stats.plan_misses = stats.plan_misses.saturating_add(1);
        }
        hit
    }

    /// Insert a topology plan for one live graph instance.
    pub(crate) fn insert_plan<T: Scalar + 'static, B: ComputeBackend + Default + 'static>(
        &self,
        root: &Arc<dyn BackwardNode<T, B>>,
        fingerprint: u64,
        graph_info: GraphInfo,
        order: Vec<Arc<dyn BackwardNode<T, B>>>,
    ) {
        if !self.config.is_enabled() || self.config.max_cache_entries() == 0 {
            return;
        }

        let plan_memory_bytes = Self::plan_memory_bytes::<T, B>(&graph_info, &order);
        if plan_memory_bytes > self.config.max_entry_memory()
            || plan_memory_bytes > self.config.max_plan_memory()
        {
            return;
        }

        let key = Arc::as_ptr(root) as *const () as usize;
        let mut plans = self.plans.write().expect("plan lock poisoned");
        self.purge_expired_plans_amortized(&mut plans);
        let replacing_memory = plans.get(&key).map(|plan| plan.memory_bytes()).unwrap_or(0);

        if replacing_memory != 0 {
            let resident = self.stats().plan_memory_bytes;
            if resident
                .saturating_sub(replacing_memory)
                .saturating_add(plan_memory_bytes)
                > self.config.max_plan_memory()
            {
                return;
            }
        } else {
            while plans.len() >= self.config.max_cache_entries()
                || self
                    .stats()
                    .plan_memory_bytes
                    .saturating_add(plan_memory_bytes)
                    > self.config.max_plan_memory()
            {
                // An amortized purge may have left expired entries inflating
                // the counters the budget check reads; reclaim them before
                // evicting a live plan.
                self.purge_expired_plans(&mut plans);
                if !self.evict_plan_lru(&mut plans) {
                    return;
                }
            }
        }

        let mut counter = self
            .plan_access_counter
            .write()
            .expect("plan counter lock poisoned");
        *counter = counter.saturating_add(1);
        let replaced = plans.insert(
            key,
            Box::new(TypedPlan {
                root: Arc::downgrade(root),
                fingerprint,
                graph_info: Arc::new(graph_info),
                order: order
                    .into_iter()
                    .map(|node| Arc::downgrade(&node))
                    .collect(),
                access_count: 1,
                last_access_tick: *counter,
                resident_since: *counter,
                memory_bytes: plan_memory_bytes,
            }),
        );
        let mut stats = self.stats.write().expect("stats lock poisoned");
        if let Some(previous) = replaced {
            stats.memory_bytes = stats
                .memory_bytes
                .saturating_sub(previous.memory_bytes())
                .saturating_add(plan_memory_bytes);
            stats.plan_memory_bytes = stats
                .plan_memory_bytes
                .saturating_sub(previous.memory_bytes())
                .saturating_add(plan_memory_bytes);
        } else {
            stats.memory_bytes = stats.memory_bytes.saturating_add(plan_memory_bytes);
            stats.plan_entries = stats.plan_entries.saturating_add(1);
            stats.plan_memory_bytes = stats.plan_memory_bytes.saturating_add(plan_memory_bytes);
        }
        // Track the plan table's high-water mark. Only insertions can raise
        // residency, so this is the single update point for the watermark.
        stats.peak_plan_entries = stats.peak_plan_entries.max(stats.plan_entries);
        stats.peak_plan_memory_bytes = stats.peak_plan_memory_bytes.max(stats.plan_memory_bytes);
    }

    /// Look up or compute a cache key for a computation graph.
    pub fn lookup(&self, fingerprint: u64) -> Option<GraphInfo> {
        if !self.config.is_enabled() {
            return None;
        }

        let gen = self.config.generation();
        let key = ComputeGraphKey::new(fingerprint, gen);
        let mut cache = self.cache.write().expect("cache lock poisoned");
        self.purge_stale_generations(&mut cache, gen);

        if self.record_lookup_locked(&mut cache, &key) {
            Some(
                cache
                    .get(&key)
                    .expect("cache entry present after successful lookup")
                    .graph_info
                    .clone(),
            )
        } else {
            None
        }
    }

    /// Record a cache lookup without cloning the cached graph metadata.
    pub(crate) fn record_lookup(&self, fingerprint: u64) -> bool {
        if !self.config.is_enabled() {
            return false;
        }

        let gen = self.config.generation();
        let key = ComputeGraphKey::new(fingerprint, gen);
        let mut cache = self.cache.write().expect("cache lock poisoned");
        self.purge_stale_generations(&mut cache, gen);
        self.record_lookup_locked(&mut cache, &key)
    }

    /// Record a lookup while the cache write lock is held.
    fn record_lookup_locked(
        &self,
        cache: &mut HashMap<ComputeGraphKey, CachedGraph>,
        key: &ComputeGraphKey,
    ) -> bool {
        if let Some(entry) = cache.get_mut(key) {
            let mut stats = self.stats.write().expect("stats lock poisoned");
            stats.hits += 1;

            let mut counter = self.access_counter.write().expect("counter lock poisoned");
            *counter += 1;
            entry.access_count = *counter;
            true
        } else {
            let mut stats = self.stats.write().expect("stats lock poisoned");
            stats.misses += 1;
            false
        }
    }

    /// Insert a new computation graph into the cache.
    pub fn insert(&self, fingerprint: u64, graph_info: GraphInfo) {
        if !self.config.is_enabled() {
            return;
        }

        let key = ComputeGraphKey::new(fingerprint, self.config.generation());

        // Include the heap-backed Vec<String> buffer and each String buffer in
        // the estimate. `size_of::<CachedGraph>()` accounts for the inline Vec
        // and String headers; the capacity terms account for their heap-backed
        // allocations, including spare capacity retained by the cache.
        let memory_bytes = Self::entry_memory_bytes(&graph_info);
        if memory_bytes > self.config.max_entry_memory()
            || memory_bytes > self.config.max_metadata_memory()
            || self.config.max_cache_entries() == 0
        {
            return;
        }

        let mut cache = self.cache.write().expect("cache lock poisoned");
        self.purge_stale_generations(&mut cache, key.generation());
        let replacing_memory = cache.get(&key).map(|entry| entry.memory_bytes).unwrap_or(0);
        if replacing_memory != 0 {
            let resident = self.metadata_memory_bytes();
            if resident
                .saturating_sub(replacing_memory)
                .saturating_add(memory_bytes)
                > self.config.max_metadata_memory()
            {
                return;
            }
        } else {
            // Evict before inserting so the new entry cannot be selected as the
            // least-recently-used item and so both the count and memory budgets
            // are satisfied independently of topology-plan residency.
            while cache.len() >= self.config.max_cache_entries()
                || self.metadata_memory_bytes().saturating_add(memory_bytes)
                    > self.config.max_metadata_memory()
            {
                self.evict_lru(&mut cache);
                if cache.is_empty() && memory_bytes > self.config.max_metadata_memory() {
                    return;
                }
            }
        }

        let mut counter = self.access_counter.write().expect("counter lock poisoned");
        *counter = counter.saturating_add(1);

        let entry = CachedGraph {
            graph_info,
            memory_bytes,
            access_count: *counter,
        };

        // Replacing an existing key must not grow the cache or double-count its
        // old allocation. This is common for repeated graph fingerprints when a
        // caller refreshes metadata after a generation change.
        let replaced = cache.insert(key, entry);
        let mut stats = self.stats.write().expect("stats lock poisoned");
        if let Some(previous) = replaced {
            stats.memory_bytes = stats
                .memory_bytes
                .saturating_sub(previous.memory_bytes)
                .saturating_add(memory_bytes);
        } else {
            stats.memory_bytes = stats.memory_bytes.saturating_add(memory_bytes);
            stats.metadata_entries = stats.metadata_entries.saturating_add(1);
        }
    }

    /// Estimate heap-backed storage owned by graph metadata.
    #[inline]
    fn graph_info_heap_bytes(graph_info: &GraphInfo) -> usize {
        graph_info
            .op_sequence
            .capacity()
            .saturating_mul(std::mem::size_of::<String>())
            .saturating_add(
                graph_info
                    .op_sequence
                    .iter()
                    .fold(0usize, |total, op| total.saturating_add(op.capacity())),
            )
    }

    /// Estimate the retained size of one metadata cache entry.
    #[inline]
    fn entry_memory_bytes(graph_info: &GraphInfo) -> usize {
        std::mem::size_of::<CachedGraph>().saturating_add(Self::graph_info_heap_bytes(graph_info))
    }

    /// Estimate the retained size of one weak topology plan.
    #[inline]
    fn plan_memory_bytes<T: Scalar, B: ComputeBackend + Default>(
        graph_info: &GraphInfo,
        order: &Vec<Arc<dyn BackwardNode<T, B>>>,
    ) -> usize {
        std::mem::size_of::<TypedPlan<T, B>>()
            .saturating_add(
                order
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Weak<dyn BackwardNode<T, B>>>()),
            )
            .saturating_add(std::mem::size_of::<GraphInfo>())
            .saturating_add(Self::graph_info_heap_bytes(graph_info))
    }

    /// Reclaim plans whose graph roots and weak node references are gone.
    fn purge_expired_plans(&self, plans: &mut HashMap<usize, Box<dyn ErasedPlan>>) {
        let mut reclaimed = 0usize;
        let mut removed = 0usize;
        plans.retain(|_, plan| {
            if plan.root_is_alive() {
                true
            } else {
                reclaimed = reclaimed.saturating_add(plan.memory_bytes());
                removed = removed.saturating_add(1);
                false
            }
        });

        if removed != 0 {
            let mut stats = self.stats.write().expect("stats lock poisoned");
            stats.plan_expirations = stats.plan_expirations.saturating_add(removed as u64);
            stats.plan_entries = stats.plan_entries.saturating_sub(removed);
            stats.plan_memory_bytes = stats.plan_memory_bytes.saturating_sub(reclaimed);
            stats.memory_bytes = stats.memory_bytes.saturating_sub(reclaimed);
        }
    }

    /// Purge expired plans on an amortized schedule.
    ///
    /// Small tables (below `PLAN_PURGE_INTERVAL` entries) purge on every
    /// operation so residency accounting stays exact at negligible cost. Once
    /// the table is large, the full scan is deferred to every
    /// `PLAN_PURGE_INTERVAL`-th operation, keeping the per-operation cost
    /// amortized O(1) for repeated lookups. `snapshot()` always performs an
    /// exact purge before reporting residency.
    fn purge_expired_plans_amortized(&self, plans: &mut HashMap<usize, Box<dyn ErasedPlan>>) {
        let interval = self.plan_purge_interval;
        // Interval 0 disables deferral: purge on every operation so residency
        // accounting is always exact. `0` must be handled before any modulo.
        if interval == 0 || plans.len() < PLAN_PURGE_MIN_TABLE_SIZE {
            self.purge_expired_plans(plans);
            return;
        }
        let ops = self.plan_purge_ops.fetch_add(1, Ordering::Relaxed);
        if ops % interval == 0 {
            self.purge_expired_plans(plans);
        }
    }

    /// Record one plan-LRU eviction and reclaim its accounted memory.
    fn record_plan_eviction(&self, memory_bytes: usize) {
        let mut stats = self.stats.write().expect("stats lock poisoned");
        stats.plan_evictions = stats.plan_evictions.saturating_add(1);
        stats.plan_entries = stats.plan_entries.saturating_sub(1);
        stats.plan_memory_bytes = stats.plan_memory_bytes.saturating_sub(memory_bytes);
        stats.memory_bytes = stats.memory_bytes.saturating_sub(memory_bytes);
    }

    /// Evict the least recently used topology plan.
    fn evict_plan_lru(&self, plans: &mut HashMap<usize, Box<dyn ErasedPlan>>) -> bool {
        let lru_key = plans
            .iter()
            .min_by_key(|(_, plan)| plan.lru_access_tick())
            .map(|(key, _)| *key);
        let Some(lru_key) = lru_key else {
            return false;
        };

        let Some(previous) = plans.remove(&lru_key) else {
            return false;
        };
        self.record_plan_eviction(previous.memory_bytes());
        true
    }

    /// Return aggregate memory currently used by metadata entries.
    fn metadata_memory_bytes(&self) -> usize {
        let stats = self.stats.read().expect("stats lock poisoned");
        stats.memory_bytes.saturating_sub(stats.plan_memory_bytes)
    }

    /// Remove entries from generations that are no longer addressable.
    fn purge_stale_generations(
        &self,
        cache: &mut HashMap<ComputeGraphKey, CachedGraph>,
        generation: u32,
    ) {
        if self.active_generation.load(Ordering::Relaxed) == generation {
            return;
        }

        let mut reclaimed = 0usize;
        let mut removed = 0u64;
        cache.retain(|key, entry| {
            if key.generation == generation {
                true
            } else {
                reclaimed = reclaimed.saturating_add(entry.memory_bytes);
                removed = removed.saturating_add(1);
                false
            }
        });

        if removed != 0 {
            let mut stats = self.stats.write().expect("stats lock poisoned");
            stats.invalidations = stats.invalidations.saturating_add(removed);
            stats.metadata_entries = stats.metadata_entries.saturating_sub(removed as usize);
            stats.memory_bytes = stats.memory_bytes.saturating_sub(reclaimed);
        }
        self.active_generation.store(generation, Ordering::Relaxed);
    }

    /// Evict the least recently used entry from the cache.
    fn evict_lru(&self, cache: &mut HashMap<ComputeGraphKey, CachedGraph>) {
        if cache.is_empty() {
            return;
        }

        // Find the entry with the lowest access count
        let min_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(k, _)| k.clone());

        if let Some(key) = min_key {
            if let Some(entry) = cache.remove(&key) {
                let mut stats = self.stats.write().expect("stats lock poisoned");
                stats.metadata_evictions = stats.metadata_evictions.saturating_add(1);
                stats.metadata_entries = stats.metadata_entries.saturating_sub(1);
                stats.memory_bytes = stats.memory_bytes.saturating_sub(entry.memory_bytes);
            }
        }
    }

    /// Clear all cached metadata and graph-local topology plans.
    pub fn clear(&self) {
        let mut cache = self.cache.write().expect("cache lock poisoned");
        cache.clear();
        self.plans.write().expect("plan lock poisoned").clear();
        *self
            .plan_access_counter
            .write()
            .expect("plan counter lock poisoned") = 0;
        self.plan_purge_ops.store(0, Ordering::Relaxed);

        let mut stats = self.stats.write().expect("stats lock poisoned");
        stats.memory_bytes = 0;
        stats.metadata_entries = 0;
        stats.plan_entries = 0;
        stats.plan_memory_bytes = 0;

        // Reset the logical clock with the entries. This avoids eventual
        // saturation after a long-lived workload repeatedly clears the cache.
        let mut counter = self.access_counter.write().expect("counter lock poisoned");
        *counter = 0;
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.read().expect("stats lock poisoned").clone()
    }

    /// Capture a consistent, read-only view of metadata and topology-plan residency.
    ///
    /// Expired weak-root plans and stale metadata generations are reclaimed before
    /// the snapshot is assembled. Plan `residency_age` values are measured in
    /// topology-cache access ticks since insertion, not wall-clock time.
    pub fn snapshot(&self) -> CacheSnapshot {
        let generation = self.config.generation();
        let mut cache = self.cache.write().expect("cache lock poisoned");
        self.purge_stale_generations(&mut cache, generation);

        let mut plans = self.plans.write().expect("plan lock poisoned");
        self.purge_expired_plans(&mut plans);
        let now = *self
            .plan_access_counter
            .read()
            .expect("plan counter lock poisoned");
        let mut plan_snapshots: Vec<_> = plans
            .iter()
            .map(|(root_id, plan)| plan.snapshot(*root_id, now))
            .collect();
        plan_snapshots.sort_by_key(|plan| (plan.fingerprint, plan.root_id));

        let stats = self.stats();
        let plan_bytes = stats.plan_memory_bytes;
        let total_bytes = stats.memory_bytes;
        let metadata_bytes = total_bytes.saturating_sub(plan_bytes);
        let plan_hit_rate = stats.plan_hit_rate();
        let total_plan_ops = stats.total_plan_ops();
        CacheSnapshot {
            metadata_entries: cache.len(),
            memory: MemoryBreakdown {
                metadata_bytes,
                plan_bytes,
                total_bytes,
            },
            stats,
            plans: plan_snapshots,
            plan_hit_rate,
            total_plan_ops,
        }
    }

    /// Get the number of cached entries.
    pub fn size(&self) -> usize {
        self.cache.read().expect("cache lock poisoned").len()
    }

    /// The amortized plan-purge interval captured from this cache's config at
    /// construction.
    ///
    /// `0` disables deferral (every plan operation runs the exact expired-plan
    /// scan); any other value defers the scan to every Nth operation once the
    /// plan table reaches [`PLAN_PURGE_MIN_TABLE_SIZE`] entries. This is the
    /// value actually driving purge behavior — the config default only matters
    /// until a cache is constructed, after which it is fixed here.
    pub fn plan_purge_interval(&self) -> u64 {
        self.plan_purge_interval
    }

    /// Reset event counters while preserving live memory and residency values.
    pub fn reset_stats(&self) {
        let current = self.stats.read().expect("stats lock poisoned").clone();
        let mut stats = self.stats.write().expect("stats lock poisoned");
        *stats = CacheStats {
            memory_bytes: current.memory_bytes,
            metadata_entries: current.metadata_entries,
            plan_entries: current.plan_entries,
            plan_memory_bytes: current.plan_memory_bytes,
            // Watermarks are event state: reset them so a fresh benchmark or
            // monitoring window starts from zero.
            ..CacheStats::default()
        };
    }
}

impl Default for ComputeGraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ComputeGraphCache {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
            plans: Arc::clone(&self.plans),
            plan_access_counter: Arc::clone(&self.plan_access_counter),
            plan_purge_ops: Arc::clone(&self.plan_purge_ops),
            plan_purge_interval: self.plan_purge_interval,
            stats: Arc::clone(&self.stats),
            config: Arc::clone(&self.config),
            access_counter: Arc::clone(&self.access_counter),
            active_generation: Arc::clone(&self.active_generation),
        }
    }
}

/// Compute a fingerprint for a computation graph.
///
/// This function computes a hash of the graph structure based on:
/// - Input tensor shapes
/// - Operation sequence
/// - Backend type identifier
///
/// The fingerprint can be used as a cache key to identify repeated patterns.
pub fn compute_graph_fingerprint(
    op_names: &[&str],
    input_shapes: &[&[usize]],
    backend_id: u32,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash operation sequence
    for op in op_names {
        op.hash(&mut hasher);
    }

    // Hash input shapes
    for shape in input_shapes {
        for dim in *shape {
            dim.hash(&mut hasher);
        }
        // Add shape length to distinguish [2,3] from [2],[3]
        shape.len().hash(&mut hasher);
    }

    // Hash backend type
    backend_id.hash(&mut hasher);

    hasher.finish()
}
