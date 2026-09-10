//! The metadata and plan budgets evict independently of each other.

use super::*;

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
