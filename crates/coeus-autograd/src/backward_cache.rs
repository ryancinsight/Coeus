//! Caching integration for backward pass compilation.
//!
//! This module provides utilities to integrate the computation graph cache
//! into the automatic differentiation backward pass. The helper retains a
//! bounded topology plan for the exact live graph instance, while the metadata
//! cache continues to identify repeated graph patterns across graph instances.

use crate::autodiff_cache::{ComputeGraphCache, GraphInfo};
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// Compute a fingerprint of a computation graph rooted at a given node.
///
/// This function traverses the computation graph and produces a deterministic
/// fingerprint based on:
/// - Operation sequence (pre-order traversal)
/// - Input tensor shapes
/// - Node connectivity patterns
///
/// The fingerprint is used as a cache key for identifying repeated patterns.
pub fn compute_graph_structure_fingerprint<T: Scalar, B: ComputeBackend + Default>(
    root_node: &Arc<dyn BackwardNode<T, B>>,
) -> (u64, GraphInfo) {
    let (fingerprint, graph_info, _) = collect_graph(root_node, false);
    (fingerprint, graph_info)
}

/// A graph's structural hash, its metadata, and the post-order the caller
/// asked for -- `None` when it did not.
type CollectedGraph<T, B> = (u64, GraphInfo, Option<Vec<Arc<dyn BackwardNode<T, B>>>>);

/// Collect graph metadata and a live post-order in one traversal.
fn collect_graph<T: Scalar, B: ComputeBackend + Default>(
    root_node: &Arc<dyn BackwardNode<T, B>>,
    collect_order: bool,
) -> CollectedGraph<T, B> {
    let mut traversal = Traversal {
        visited: HashSet::new(),
        hasher: DefaultHasher::new(),
        op_sequence: Vec::new(),
        order: collect_order.then(Vec::new),
        node_count: 0,
        leaf_count: 0,
        max_depth: 0,
    };
    traversal.visit(root_node, 0);
    traversal.finish()
}

/// Everything one traversal accumulates.
///
/// These were eight `&mut` parameters threaded through a recursive function,
/// which meant every call site restated the whole set in order and a new
/// statistic meant touching all of them.
struct Traversal<T: Scalar, B: ComputeBackend + Default> {
    visited: HashSet<*const ()>,
    hasher: DefaultHasher,
    op_sequence: Vec<String>,
    order: Option<Vec<Arc<dyn BackwardNode<T, B>>>>,
    node_count: usize,
    leaf_count: usize,
    max_depth: usize,
}

impl<T: Scalar, B: ComputeBackend + Default> Traversal<T, B> {
    /// Visit `node` and everything reachable from it, once each.
    fn visit(&mut self, node: &Arc<dyn BackwardNode<T, B>>, depth: usize) {
        use std::hash::Hash;

        let ptr = Arc::as_ptr(node) as *const ();
        if !self.visited.insert(ptr) {
            return;
        }

        self.node_count += 1;
        self.max_depth = self.max_depth.max(depth);

        let op_name = node.op_name();
        op_name.hash(&mut self.hasher);
        self.op_sequence.push(op_name.to_string());

        let inputs = node.inputs();
        inputs.len().hash(&mut self.hasher);
        for input in inputs {
            input.tensor.shape().hash(&mut self.hasher);
            if input.creator.is_none() {
                self.leaf_count += 1;
            }

            if let Some(ref creator) = input.creator {
                self.visit(creator, depth + 1);
            }
        }

        // Post-order is the order required by reverse-mode propagation.
        if let Some(order) = &mut self.order {
            order.push(node.clone());
        }
    }

    /// Consume the traversal into its fingerprint, metadata and post-order.
    fn finish(self) -> CollectedGraph<T, B> {
        let graph_info = GraphInfo {
            node_count: self.node_count,
            leaf_count: self.leaf_count,
            max_depth: self.max_depth,
            op_sequence: self.op_sequence,
        };
        (self.hasher.finish(), graph_info, self.order)
    }
}

/// Perform topological sort using the cache when available.
///
/// The first call traverses the graph once to compute metadata and build a
/// live post-order. Later calls for the same live root reuse that post-order;
/// separately constructed graphs still traverse independently, avoiding unsafe
/// pointer reuse across graphs.
pub fn topological_sort_with_cache<T: Scalar, B: ComputeBackend + Default>(
    root_node: Option<&Arc<dyn BackwardNode<T, B>>>,
    cache: &ComputeGraphCache,
) -> Vec<Arc<dyn BackwardNode<T, B>>> {
    let Some(root) = root_node else {
        return Vec::new();
    };

    if let Some(plan) = cache.lookup_plan(root) {
        if !cache.record_lookup(plan.fingerprint) {
            cache.insert(plan.fingerprint, (*plan.graph_info).clone());
        }
        return plan.order;
    }

    let (fingerprint, graph_info, order) = collect_graph(root, true);
    if !cache.record_lookup(fingerprint) {
        cache.insert(fingerprint, graph_info.clone());
    }

    let order = order.expect("order collection is enabled for topological sorting");
    cache.insert_plan(root, fingerprint, graph_info, order.clone());
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff_cache::CacheConfig;
    use crate::grad_buffer::GradBuffer;
    use crate::var::Var;
    use coeus_core::{BackendError, MoiraiBackend};
    use coeus_tensor::Tensor;

    struct TestNode {
        name: &'static str,
        output_grad: Arc<GradBuffer<f32, MoiraiBackend>>,
        inputs: Vec<Var<f32, MoiraiBackend>>,
    }

    struct BudgetConfig {
        metadata_memory: usize,
        plan_memory: usize,
        max_entries: usize,
    }

    impl CacheConfig for BudgetConfig {
        fn max_cache_entries(&self) -> usize {
            self.max_entries
        }

        fn max_metadata_memory(&self) -> usize {
            self.metadata_memory
        }

        fn max_plan_memory(&self) -> usize {
            self.plan_memory
        }
    }

    struct GenerationConfig {
        generation: std::sync::atomic::AtomicU32,
    }

    impl CacheConfig for GenerationConfig {
        fn generation(&self) -> u32 {
            self.generation.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    struct PurgeIntervalConfig {
        interval: u64,
    }

    impl CacheConfig for PurgeIntervalConfig {
        fn plan_purge_interval(&self) -> u64 {
            self.interval
        }
    }

    impl BackwardNode<f32, MoiraiBackend> for TestNode {
        fn op_name(&self) -> &'static str {
            self.name
        }

        fn output_grad(&self) -> &Arc<GradBuffer<f32, MoiraiBackend>> {
            &self.output_grad
        }

        fn inputs(&self) -> &[Var<f32, MoiraiBackend>] {
            &self.inputs
        }

        fn backward(
            &self,
            _grad_out: &Tensor<f32, MoiraiBackend>,
            _input_grads: &[Option<Arc<GradBuffer<f32, MoiraiBackend>>>],
        ) -> Result<(), BackendError> {
            Ok(())
        }
    }

    fn test_graph() -> Arc<dyn BackwardNode<f32, MoiraiBackend>> {
        let leaf = Var::new(Tensor::zeros([1]), true);
        let child: Arc<dyn BackwardNode<f32, MoiraiBackend>> = Arc::new(TestNode {
            name: "child",
            output_grad: Arc::new(GradBuffer::new(Tensor::zeros([1]))),
            inputs: vec![leaf],
        });
        let child_output = Var::with_creator(
            Tensor::zeros([1]),
            Some(Arc::clone(child.output_grad())),
            child,
        );

        Arc::new(TestNode {
            name: "root",
            output_grad: Arc::new(GradBuffer::new(Tensor::zeros([1]))),
            inputs: vec![child_output],
        })
    }

    #[test]
    fn topological_sort_is_single_pass_and_cache_aware() {
        let cache = ComputeGraphCache::new();
        let root = test_graph();
        let (_, _, metadata_order) = collect_graph(&root, false);
        assert!(metadata_order.is_none());

        let first = topological_sort_with_cache(Some(&root), &cache);
        let first_names: Vec<_> = first.iter().map(|node| node.op_name()).collect();
        assert_eq!(first_names, ["child", "root"]);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        let second = topological_sort_with_cache(Some(&root), &cache);
        let second_names: Vec<_> = second.iter().map(|node| node.op_name()).collect();
        assert_eq!(second_names, first_names);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.plan_misses, 1);
        assert_eq!(stats.plan_hits, 1);
        assert_eq!(stats.plan_entries, 1);
        assert!(stats.plan_memory_bytes > 0);
        assert!(stats.memory_bytes >= stats.plan_memory_bytes);

        let snapshot = cache.snapshot();
        assert_eq!(snapshot.metadata_entries, 1);
        assert_eq!(stats.metadata_entries, 1);
        assert_eq!(stats.metadata_evictions, 0);
        assert!(snapshot.memory.metadata_bytes > 0);
        assert_eq!(
            snapshot.memory.metadata_bytes + snapshot.memory.plan_bytes,
            snapshot.memory.total_bytes
        );
        assert_eq!(snapshot.memory.total_bytes, stats.memory_bytes);
        assert_eq!(snapshot.plans.len(), 1);
        assert_eq!(snapshot.plans[0].node_count, 2);
        assert_eq!(snapshot.plans[0].access_count, 2);
        assert_eq!(snapshot.plans[0].residency_age, 1);
        assert_eq!(snapshot.plans[0].memory_bytes, stats.plan_memory_bytes);
        assert_eq!(snapshot.stats, stats);
    }

    #[test]
    fn reset_stats_preserves_live_plan_residency() {
        let cache = ComputeGraphCache::new();
        let root = test_graph();

        let _ = topological_sort_with_cache(Some(&root), &cache);
        let before = cache.stats();
        cache.reset_stats();
        let after = cache.stats();

        assert_eq!(after.hits, 0);
        assert_eq!(after.misses, 0);
        assert_eq!(after.plan_hits, 0);
        assert_eq!(after.plan_misses, 0);
        assert_eq!(after.plan_entries, 1);
        assert_eq!(after.plan_memory_bytes, before.plan_memory_bytes);
        assert_eq!(after.memory_bytes, before.memory_bytes);
    }

    #[test]
    fn topology_plans_are_scoped_to_the_live_root() {
        let cache = ComputeGraphCache::new();
        let first_root = test_graph();
        let second_root = test_graph();

        let _ = topological_sort_with_cache(Some(&first_root), &cache);
        assert_eq!(Arc::strong_count(&first_root), 1);
        let _ = topological_sort_with_cache(Some(&second_root), &cache);
        let _ = topological_sort_with_cache(Some(&first_root), &cache);

        let stats = cache.stats();
        assert_eq!(stats.plan_misses, 2);
        assert_eq!(stats.plan_hits, 1);
        // The two graphs have the same structure, so metadata is shared even
        // though their live node orders must remain separate.
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 2);

        let snapshot = cache.snapshot();
        let first_root_id = Arc::as_ptr(&first_root) as *const () as usize;
        let second_root_id = Arc::as_ptr(&second_root) as *const () as usize;
        let first_plan = snapshot
            .plans
            .iter()
            .find(|plan| plan.root_id == first_root_id)
            .expect("first root plan should be resident");
        let second_plan = snapshot
            .plans
            .iter()
            .find(|plan| plan.root_id == second_root_id)
            .expect("second root plan should be resident");
        assert_eq!(first_plan.access_count, 2);
        assert_eq!(second_plan.access_count, 1);
        assert_eq!(first_plan.residency_age, 2);
        assert_eq!(second_plan.residency_age, 1);
    }

    #[test]
    fn snapshot_reports_plan_reuse_rate_and_lookup_total() {
        let cache = ComputeGraphCache::new();
        let root = test_graph();

        let empty = cache.snapshot();
        assert_eq!(empty.plan_hit_rate, 0.0);
        assert_eq!(empty.total_plan_ops, 0);
        assert_eq!(empty.plan_hit_rate, empty.stats.plan_hit_rate());
        assert_eq!(empty.total_plan_ops, empty.stats.total_plan_ops());

        let _ = topological_sort_with_cache(Some(&root), &cache);
        let after_miss = cache.snapshot();
        assert_eq!(after_miss.total_plan_ops, 1);
        assert_eq!(after_miss.plan_hit_rate, 0.0);

        let _ = topological_sort_with_cache(Some(&root), &cache);
        let after_hit = cache.snapshot();
        assert_eq!(after_hit.total_plan_ops, 2);
        assert_eq!(after_hit.plan_hit_rate, 50.0);
        assert_eq!(after_hit.plan_hit_rate, after_hit.stats.plan_hit_rate());
    }

    #[test]
    fn failed_plan_upgrades_do_not_advance_the_access_clock() {
        let cache = ComputeGraphCache::new();
        let stats_before = cache.stats();
        assert_eq!(stats_before.plan_hit_rate(), 0.0);
        assert_eq!(stats_before.total_plan_ops(), 0);

        let snapshot = cache.snapshot();
        assert_eq!(snapshot.stats.plan_hit_rate(), 0.0);
        assert_eq!(snapshot.stats.total_plan_ops(), 0);

        {
            let root = test_graph();
            let _ = topological_sort_with_cache(Some(&root), &cache);
            let snapshot = cache.snapshot();
            assert_eq!(snapshot.plans[0].residency_age, 0);
            assert_eq!(snapshot.plans[0].access_count, 1);
        }

        // A lookup against the dead root must not count as a plan access or
        // advance the access clock.
        let _ = topological_sort_with_cache::<f32, MoiraiBackend>(None, &cache);
        let stats = cache.stats();
        assert_eq!(stats.plan_misses, 1);
        assert_eq!(stats.plan_hits, 0);
        assert_eq!(stats.total_plan_ops(), 1);

        // A replacement graph whose root allocation differs is a fresh miss.
        let replacement = test_graph();
        let _ = topological_sort_with_cache(Some(&replacement), &cache);
        let stats = cache.stats();
        assert_eq!(stats.plan_misses, 2);
        assert_eq!(stats.plan_hits, 0);
        assert_eq!(stats.plan_hit_rate(), 0.0);
        assert_eq!(stats.total_plan_ops(), 2);

        // Reuse the replacement root: one genuine plan hit.
        let _ = topological_sort_with_cache(Some(&replacement), &cache);
        let stats = cache.stats();
        assert_eq!(stats.plan_hits, 1);
        assert_eq!(stats.plan_misses, 2);
        assert!((stats.plan_hit_rate() - 100.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn clearing_cache_invalidates_live_topology_plans() {
        let cache = ComputeGraphCache::new();
        let root = test_graph();

        let _ = topological_sort_with_cache(Some(&root), &cache);
        assert!(cache.stats().plan_memory_bytes > 0);
        cache.clear();
        let cleared = cache.stats();
        assert_eq!(cleared.plan_entries, 0);
        assert_eq!(cleared.plan_memory_bytes, 0);
        assert_eq!(cleared.memory_bytes, 0);
        let _ = topological_sort_with_cache(Some(&root), &cache);

        let stats = cache.stats();
        assert_eq!(stats.plan_misses, 2);
        assert_eq!(stats.plan_hits, 0);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn expired_topology_plans_reclaim_resident_memory() {
        let cache = ComputeGraphCache::new();
        {
            let root = test_graph();
            let _ = topological_sort_with_cache(Some(&root), &cache);
            assert_eq!(cache.stats().plan_entries, 1);
        }

        let replacement = test_graph();
        let _ = topological_sort_with_cache(Some(&replacement), &cache);
        let stats = cache.stats();

        assert_eq!(stats.plan_entries, 1);
        assert!(stats.plan_memory_bytes > 0);
        assert!(stats.plan_expirations >= 1);
        assert!(stats.memory_bytes >= stats.plan_memory_bytes);
    }

    #[test]
    fn large_plan_tables_purge_on_an_amortized_schedule() {
        let cache = ComputeGraphCache::new();

        // Fill the plan table past the purge-interval threshold.
        let mut roots: Vec<_> = (0..80).map(|_| test_graph()).collect();
        for root in &roots {
            let _ = topological_sort_with_cache(Some(root), &cache);
        }
        assert_eq!(cache.stats().plan_entries, 80);

        // Keep one live root; the other 79 plans expire when their roots drop.
        let survivor = roots.pop().expect("roots populated");
        drop(roots);
        // No scan has run since expiry: residency is deliberately stale.
        assert_eq!(cache.stats().plan_entries, 80);

        // Repeated hits on the survivor advance the amortization counter; the
        // first several must not rescan the 80-entry table.
        for _ in 0..5 {
            let _ = topological_sort_with_cache(Some(&survivor), &cache);
        }
        assert_eq!(cache.stats().plan_entries, 80);

        // Crossing the interval triggers one deferred scan that reclaims all
        // expired plans in a single pass.
        for _ in 0..80 {
            let _ = topological_sort_with_cache(Some(&survivor), &cache);
        }
        let stats = cache.stats();
        assert_eq!(stats.plan_entries, 1);
        assert!(stats.plan_expirations >= 79);
    }

    #[test]
    fn custom_purge_interval_tunes_reclamation_frequency() {
        // Interval 0 disables deferral: one operation reclaims expired plans
        // immediately, keeping residency counters exact on every lookup.
        let exact = ComputeGraphCache::with_config(Arc::new(PurgeIntervalConfig { interval: 0 }));
        {
            let mut roots: Vec<_> = (0..80).map(|_| test_graph()).collect();
            for root in &roots {
                let _ = topological_sort_with_cache(Some(root), &exact);
            }
            let survivor = roots.pop().expect("roots populated");
            drop(roots);
            let _ = topological_sort_with_cache(Some(&survivor), &exact);
            assert_eq!(exact.stats().plan_entries, 1);
            assert!(exact.stats().plan_expirations >= 79);
        }

        // A huge interval defers the scan indefinitely: repeated lookups never
        // reclaim, and only an explicit snapshot does.
        let deferred =
            ComputeGraphCache::with_config(Arc::new(PurgeIntervalConfig { interval: u64::MAX }));
        // The cache must expose the interval it captured at construction.
        assert_eq!(deferred.plan_purge_interval(), u64::MAX);
        let mut roots: Vec<_> = (0..80).map(|_| test_graph()).collect();
        for root in &roots {
            let _ = topological_sort_with_cache(Some(root), &deferred);
        }
        let survivor = roots.pop().expect("roots populated");
        drop(roots);
        for _ in 0..70 {
            let _ = topological_sort_with_cache(Some(&survivor), &deferred);
        }
        assert_eq!(deferred.stats().plan_entries, 80);
        let snapshot = deferred.snapshot();
        assert_eq!(snapshot.plans.len(), 1);
        assert_eq!(snapshot.stats.plan_entries, 1);
    }

    #[test]
    fn peak_residency_tracks_the_plan_tables_high_water_mark() {
        let cache = ComputeGraphCache::new();
        let roots: Vec<_> = (0..5).map(|_| test_graph()).collect();
        for root in &roots {
            let _ = topological_sort_with_cache(Some(root), &cache);
        }
        let peak = cache.stats();
        assert_eq!(peak.peak_plan_entries, 5);
        assert_eq!(peak.peak_plan_memory_bytes, peak.plan_memory_bytes);

        // Expiration brings current residency down; the watermark holds.
        drop(roots);
        let _ = cache.snapshot(); // purges expired plans exactly
        let after_expiry = cache.stats();
        assert_eq!(after_expiry.plan_entries, 0);
        assert_eq!(after_expiry.plan_memory_bytes, 0);
        assert_eq!(after_expiry.peak_plan_entries, 5);
        assert_eq!(
            after_expiry.peak_plan_memory_bytes,
            peak.peak_plan_memory_bytes
        );

        // A smaller re-fill does not raise the watermark.
        let small = test_graph();
        let _ = topological_sort_with_cache(Some(&small), &cache);
        let refilled = cache.stats();
        assert_eq!(refilled.plan_entries, 1);
        assert_eq!(refilled.peak_plan_entries, 5);

        // clear() keeps the watermark; reset_stats() clears it.
        cache.clear();
        let cleared = cache.stats();
        assert_eq!(cleared.plan_entries, 0);
        assert_eq!(cleared.peak_plan_entries, 5);
        cache.reset_stats();
        let reset = cache.stats();
        assert_eq!(reset.peak_plan_entries, 0);
        assert_eq!(reset.peak_plan_memory_bytes, 0);
    }

    #[test]
    fn snapshot_reclaims_expired_plans_before_reporting_residency() {
        let cache = ComputeGraphCache::new();
        {
            let root = test_graph();
            let _ = topological_sort_with_cache(Some(&root), &cache);
            let snapshot = cache.snapshot();
            assert_eq!(snapshot.plans.len(), 1);
            assert_eq!(snapshot.plans[0].node_count, 2);
            assert_eq!(snapshot.plans[0].residency_age, 0);
        }

        let snapshot = cache.snapshot();
        assert!(snapshot.plans.is_empty());
        assert_eq!(snapshot.stats.plan_entries, 0);
        assert_eq!(snapshot.stats.plan_memory_bytes, 0);
        assert!(snapshot.stats.plan_expirations >= 1);
    }

    #[test]
    fn metadata_and_plan_budgets_are_independent() {
        let metadata_only = ComputeGraphCache::with_config(Arc::new(BudgetConfig {
            metadata_memory: usize::MAX,
            plan_memory: 0,
            max_entries: 16,
        }));
        let metadata_root = test_graph();
        let _ = topological_sort_with_cache(Some(&metadata_root), &metadata_only);
        let metadata_stats = metadata_only.stats();
        assert!(metadata_stats.memory_bytes > 0);
        assert_eq!(metadata_stats.plan_entries, 0);
        assert_eq!(metadata_stats.plan_memory_bytes, 0);

        let probe = ComputeGraphCache::new();
        let plan_root = test_graph();
        let _ = topological_sort_with_cache(Some(&plan_root), &probe);
        let plan_budget = probe.stats().plan_memory_bytes;
        assert!(plan_budget > 0);

        let plan_only = ComputeGraphCache::with_config(Arc::new(BudgetConfig {
            metadata_memory: 0,
            plan_memory: plan_budget,
            max_entries: 16,
        }));
        let plan_root = test_graph();
        let _ = topological_sort_with_cache(Some(&plan_root), &plan_only);
        let plan_stats = plan_only.stats();
        assert_eq!(plan_stats.memory_bytes, plan_stats.plan_memory_bytes);
        assert_eq!(plan_stats.plan_entries, 1);
        assert!(plan_stats.plan_memory_bytes > 0);

        let _ = topological_sort_with_cache(Some(&plan_root), &plan_only);
        assert_eq!(plan_only.stats().plan_hits, 1);
    }

    #[test]
    fn metadata_budget_evicts_metadata_without_affecting_plans() {
        let info = GraphInfo {
            node_count: 1,
            leaf_count: 1,
            max_depth: 0,
            op_sequence: vec!["leaf".to_owned()],
        };
        let probe = ComputeGraphCache::new();
        probe.insert(1, info.clone());
        let budget = probe.stats().memory_bytes;
        assert!(budget > 0);

        let cache = ComputeGraphCache::with_config(Arc::new(BudgetConfig {
            metadata_memory: budget,
            plan_memory: 0,
            max_entries: 16,
        }));
        cache.insert(1, info.clone());
        cache.insert(2, info);

        let stats = cache.stats();
        assert_eq!(cache.size(), 1);
        assert!(stats.memory_bytes <= budget);
        assert!(stats.metadata_evictions >= 1);
        // LRU pressure is tracked separately from generation invalidations.
        assert_eq!(stats.invalidations, 0);
        assert_eq!(stats.plan_entries, 0);
    }

    #[test]
    fn generation_invalidations_are_distinct_from_lru_evictions() {
        let info = GraphInfo {
            node_count: 1,
            leaf_count: 1,
            max_depth: 0,
            op_sequence: vec!["leaf".to_owned()],
        };

        let generation = Arc::new(GenerationConfig {
            generation: std::sync::atomic::AtomicU32::new(0),
        });
        let cache = ComputeGraphCache::with_config(generation.clone());
        cache.insert(1, info.clone());
        let seeded = cache.stats();
        assert_eq!(seeded.metadata_entries, 1);
        assert_eq!(seeded.invalidations, 0);
        assert_eq!(seeded.metadata_evictions, 0);

        // A generation bump invalidates stale-generation entries on the same
        // cache; this is not an LRU eviction and must not advance the eviction
        // counter.
        generation
            .generation
            .store(1, std::sync::atomic::Ordering::Relaxed);
        assert!(cache.lookup(1).is_none());
        let after_bump = cache.stats();
        assert_eq!(after_bump.metadata_entries, 0);
        assert_eq!(after_bump.invalidations, 1);
        assert_eq!(after_bump.metadata_evictions, 0);

        // Budget-driven eviction advances only the eviction counter.
        let probe = ComputeGraphCache::new();
        probe.insert(1, info.clone());
        let budget = probe.stats().memory_bytes;
        let tight = ComputeGraphCache::with_config(Arc::new(BudgetConfig {
            metadata_memory: budget,
            plan_memory: 0,
            max_entries: 16,
        }));
        tight.insert(1, info.clone());
        tight.insert(2, info);
        let after_eviction = tight.stats();
        assert_eq!(after_eviction.metadata_entries, 1);
        assert_eq!(after_eviction.invalidations, 0);
        assert_eq!(after_eviction.metadata_evictions, 1);
    }

    #[test]
    fn plan_budget_evicts_plans_without_affecting_metadata() {
        let probe = ComputeGraphCache::new();
        let probe_root = test_graph();
        let _ = topological_sort_with_cache(Some(&probe_root), &probe);
        let budget = probe.stats().plan_memory_bytes;

        let cache = ComputeGraphCache::with_config(Arc::new(BudgetConfig {
            metadata_memory: usize::MAX,
            plan_memory: budget,
            max_entries: 16,
        }));
        let first_root = test_graph();
        let second_root = test_graph();
        let _ = topological_sort_with_cache(Some(&first_root), &cache);
        let _ = topological_sort_with_cache(Some(&second_root), &cache);

        let stats = cache.stats();
        assert_eq!(stats.plan_entries, 1);
        assert!(stats.plan_memory_bytes <= budget);
        assert_eq!(stats.plan_evictions, 1);
        assert!(stats.memory_bytes > stats.plan_memory_bytes);
    }
}
