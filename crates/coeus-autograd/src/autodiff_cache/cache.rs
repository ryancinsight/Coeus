//! The cache itself: the shared tables and every operation over them.

use super::config::{CacheConfig, DefaultCacheConfig};
use super::key::{CachedGraph, ComputeGraphKey, GraphInfo};
use super::plan::ErasedPlan;
use super::stats::{CacheSnapshot, CacheStats, MemoryBreakdown};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
    pub(super) cache: Arc<RwLock<HashMap<ComputeGraphKey, CachedGraph>>>,
    /// Live-graph topology plans keyed by root allocation identity.
    pub(super) plans: Arc<RwLock<HashMap<usize, Box<dyn ErasedPlan>>>>,
    /// Monotonic access counter for topology-plan LRU tracking.
    pub(super) plan_access_counter: Arc<RwLock<u64>>,
    /// Plan operations since the last deferred expired-plan scan.
    pub(super) plan_purge_ops: Arc<AtomicU64>,
    /// Amortized purge interval captured from the cache configuration.
    pub(super) plan_purge_interval: u64,
    /// Cache statistics.
    pub(super) stats: Arc<RwLock<CacheStats>>,
    /// Configuration.
    pub(super) config: Arc<dyn CacheConfig>,
    /// Global access counter for LRU tracking.
    pub(super) access_counter: Arc<RwLock<u64>>,
    /// Last generation whose entries are resident in the cache.
    pub(super) active_generation: Arc<AtomicU32>,
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
    pub(super) fn record_lookup_locked(
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
    pub(super) fn graph_info_heap_bytes(graph_info: &GraphInfo) -> usize {
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
    pub(super) fn entry_memory_bytes(graph_info: &GraphInfo) -> usize {
        std::mem::size_of::<CachedGraph>().saturating_add(Self::graph_info_heap_bytes(graph_info))
    }

    /// Return aggregate memory currently used by metadata entries.
    pub(super) fn metadata_memory_bytes(&self) -> usize {
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
    /// plan table reaches [`super::PLAN_PURGE_MIN_TABLE_SIZE`] entries. This is
    /// the value actually driving purge behavior — the config default only
    /// matters until a cache is constructed, after which it is fixed here.
    pub fn plan_purge_interval(&self) -> u64 {
        self.plan_purge_interval
    }

    /// Reset event counters while preserving live memory and residency values.
    ///
    /// The residency values are read from the same guard that writes them: a
    /// read lock released before taking the write lock leaves a window in
    /// which an insertion, eviction, purge or `clear()` moves residency, and
    /// the reset would then write back the pre-window figures. Budget checks
    /// read those counters, so an undercount lets the cache exceed its
    /// configured memory.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().expect("stats lock poisoned");
        let current = stats.clone();
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
