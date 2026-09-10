//! The topology-plan table: lookup, insertion, purging and eviction.
//!
//! Plans key on the live root's address and hold weak references, so this
//! half of the cache is the one whose residency decays on its own; the
//! metadata half in [`super::cache`] is keyed by fingerprint and does not.

use super::cache::ComputeGraphCache;
use super::key::GraphInfo;
use super::plan::{ErasedPlan, TopologyPlanHit, TypedPlan};
use super::stats::PLAN_PURGE_MIN_TABLE_SIZE;
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

impl ComputeGraphCache {
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

    /// Retained size of one topology plan, header plus its weak node vector.
    pub(super) fn plan_memory_bytes<T: Scalar, B: ComputeBackend + Default>(
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
}
