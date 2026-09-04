//! LRU eviction and expired-plan/root reclamation behind
//! [`super::ComputeGraphCache`].

use super::graph::{CachedGraph, ComputeGraphKey, ErasedPlan};
use super::{ComputeGraphCache, PLAN_PURGE_MIN_TABLE_SIZE};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

impl ComputeGraphCache {
    /// Reclaim plans whose graph roots and weak node references are gone.
    pub(super) fn purge_expired_plans(&self, plans: &mut HashMap<usize, Box<dyn ErasedPlan>>) {
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
    pub(super) fn purge_expired_plans_amortized(
        &self,
        plans: &mut HashMap<usize, Box<dyn ErasedPlan>>,
    ) {
        let interval = self.plan_purge_interval;
        // Interval 0 disables deferral: purge on every operation so residency
        // accounting is always exact. `0` must be handled before any modulo.
        if interval == 0 || plans.len() < PLAN_PURGE_MIN_TABLE_SIZE {
            self.purge_expired_plans(plans);
            return;
        }
        let ops = self.plan_purge_ops.fetch_add(1, Ordering::Relaxed);
        if ops.is_multiple_of(interval) {
            self.purge_expired_plans(plans);
        }
    }

    /// Record one plan-LRU eviction and reclaim its accounted memory.
    pub(super) fn record_plan_eviction(&self, memory_bytes: usize) {
        let mut stats = self.stats.write().expect("stats lock poisoned");
        stats.plan_evictions = stats.plan_evictions.saturating_add(1);
        stats.plan_entries = stats.plan_entries.saturating_sub(1);
        stats.plan_memory_bytes = stats.plan_memory_bytes.saturating_sub(memory_bytes);
        stats.memory_bytes = stats.memory_bytes.saturating_sub(memory_bytes);
    }

    /// Evict the least recently used topology plan.
    pub(super) fn evict_plan_lru(&self, plans: &mut HashMap<usize, Box<dyn ErasedPlan>>) -> bool {
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
    pub(super) fn metadata_memory_bytes(&self) -> usize {
        let stats = self.stats.read().expect("stats lock poisoned");
        stats.memory_bytes.saturating_sub(stats.plan_memory_bytes)
    }

    /// Remove entries from generations that are no longer addressable.
    pub(super) fn purge_stale_generations(
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
    pub(super) fn evict_lru(&self, cache: &mut HashMap<ComputeGraphKey, CachedGraph>) {
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
}
