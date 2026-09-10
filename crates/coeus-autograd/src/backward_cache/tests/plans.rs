//! Plan reuse, root scoping, and the residency the purge schedule reclaims.

use super::*;

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
