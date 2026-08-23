//! Integration test for autodiff computation graph caching.
//!
//! This test verifies that:
//! 1. Cache integration doesn't break existing autodiff functionality
//! 2. Cache correctly tracks hits and misses
//! 3. Gradients remain accurate with caching enabled

#[cfg(test)]
mod cache_integration {
    use coeus_autograd::{add, get_backward_cache, mul, reset_backward_cache_stats, sub, sum, Var};
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    /// Test that basic gradients work with cache enabled.
    #[test]
    fn test_basic_gradient_with_cache() {
        reset_backward_cache_stats();

        // y = x^2 at x=3
        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([1], &[3.0]), true);
        let y = mul(&x, &x);
        y.backward().expect("backward failed");

        let grad = x.grad().expect("grad is None");
        let grad_val = grad.as_slice()[0];

        // dy/dx = 2*x = 6.0
        assert!(
            (grad_val - 6.0).abs() < 1e-5,
            "expected gradient 6.0, got {}",
            grad_val
        );
    }

    /// Test that repeated iterations show cache hits.
    #[test]
    fn test_cache_hits_repeated_iteration() {
        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        const ITERATIONS: usize = 5;
        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);

        for iter in 0..ITERATIONS {
            let y = add(&x, &x);
            let z = mul(&y, &x);
            let loss = sum(&z);
            loss.backward().expect("backward failed");
            x.zero_grad();

            let stats = cache.stats();
            if iter > 0 {
                assert!(
                    stats.hits > 0,
                    "Expected cache hits after iteration 0, got: hits={}, misses={}",
                    stats.hits,
                    stats.misses
                );
            }
        }

        let final_stats = cache.stats();
        println!(
            "Cache stats: hits={}, misses={}, hit_rate={:.1}%",
            final_stats.hits,
            final_stats.misses,
            final_stats.hit_rate()
        );
    }

    /// Test that different shapes don't incorrectly hit cache.
    #[test]
    fn test_cache_shape_discrimination() {
        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        // Shape 1: [2]
        let x1 = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);
        let y1 = mul(&x1, &x1);
        let loss1 = sum(&y1);
        loss1.backward().expect("backward 1 failed");
        x1.zero_grad();

        let stats_after_first = cache.stats();
        assert_eq!(
            stats_after_first.misses, 1,
            "Expected 1 miss after first iteration"
        );

        // Shape 2: [3]
        let x2 = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
        let y2 = mul(&x2, &x2);
        let loss2 = sum(&y2);
        loss2.backward().expect("backward 2 failed");
        x2.zero_grad();

        let stats_after_second = cache.stats();
        assert_eq!(
            stats_after_second.misses, 2,
            "Expected 2 misses (different shapes), got misses={}",
            stats_after_second.misses
        );

        // Back to shape 1: should hit cache
        let x3 = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[3.0, 4.0]), true);
        let y3 = mul(&x3, &x3);
        let loss3 = sum(&y3);
        loss3.backward().expect("backward 3 failed");
        x3.zero_grad();

        let stats_after_third = cache.stats();
        assert!(
            stats_after_third.hits > 0,
            "Expected cache hit for repeated shape [2], got hits={}",
            stats_after_third.hits
        );
    }

    /// Test complex computation graph with correct gradients.
    #[test]
    fn test_complex_graph_gradient_correctness() {
        reset_backward_cache_stats();

        // Test: y = (x^2 + x) * (x - 1)
        // dy/dx = (2x + 1)(x - 1) + (x^2 + x) = 3x^2 - 1
        // At x=2: dy/dx = 3(4) - 1 = 11

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([1], &[2.0]), true);

        let x2 = mul(&x, &x);
        let x2_plus_x = add(&x2, &x);
        let x_minus_1 = sub(&x, &Var::new(Tensor::from_slice([1], &[1.0]), false));
        let y = mul(&x2_plus_x, &x_minus_1);

        y.backward().expect("backward failed");

        let grad = x.grad().expect("grad is None");
        let grad_val = grad.as_slice()[0];

        assert!(
            (grad_val - 11.0).abs() < 1e-4,
            "expected gradient 11.0, got {}",
            grad_val
        );
    }

    /// Test that cache is thread-local and doesn't interfere with other threads.
    #[test]
    fn test_cache_thread_local() {
        use std::thread;

        reset_backward_cache_stats();

        let t1 = thread::spawn(|| {
            let cache1 = get_backward_cache();
            cache1.clear();

            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([1], &[1.0]), true);
            let y = mul(&x, &x);
            y.backward().expect("backward failed");

            let stats1 = cache1.stats();
            stats1.hits + stats1.misses
        });

        let t2 = thread::spawn(|| {
            let cache2 = get_backward_cache();
            cache2.clear();

            for _ in 0..10 {
                let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[2.0, 3.0]), true);
                let y = add(&x, &x);
                let loss = sum(&y);
                loss.backward().expect("backward failed");
                x.zero_grad();
            }

            let stats2 = cache2.stats();
            stats2.hits + stats2.misses
        });

        let ops1 = t1.join().expect("thread 1 panicked");
        let ops2 = t2.join().expect("thread 2 panicked");

        println!("Thread 1 ops: {}, Thread 2 ops: {}", ops1, ops2);
        assert!(ops1 > 0 && ops2 > 0, "Both threads should have operations");
    }

    /// Test that the public snapshot API reports live topology-plan details.
    #[test]
    fn test_public_cache_snapshot_reports_plan_details() {
        let cache = get_backward_cache();
        cache.clear();

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);
        let y = mul(&x, &x);
        let loss = sum(&y);
        loss.backward().expect("backward failed");

        let snapshot: coeus_autograd::CacheSnapshot = cache.snapshot();
        assert_eq!(snapshot.plans.len(), 1);
        assert_eq!(snapshot.metadata_entries, 1);
        assert_eq!(snapshot.plans[0].node_count, 2);
        assert_eq!(snapshot.plans[0].residency_age, 0);
        assert_eq!(snapshot.plans[0].access_count, 1);
        assert_eq!(
            snapshot.plans[0].memory_bytes,
            snapshot.stats.plan_memory_bytes
        );
        // The per-category breakdown is reported in one consistent read; the
        // split sums to the total without any caller-side recomputation.
        assert_eq!(
            snapshot.memory.metadata_bytes + snapshot.memory.plan_bytes,
            snapshot.memory.total_bytes
        );
        assert_eq!(snapshot.memory.total_bytes, snapshot.stats.memory_bytes);
        assert_eq!(snapshot.memory.plan_bytes, snapshot.stats.plan_memory_bytes);
    }

    /// Test that cache statistics are properly reported.
    #[test]
    fn test_cache_statistics() {
        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([1], &[1.0]), true);

        // Iteration 1: miss
        let y1 = mul(&x, &x);
        y1.backward().expect("backward 1 failed");
        x.zero_grad();

        let stats1 = cache.stats();
        assert_eq!(stats1.misses, 1, "Expected 1 miss");
        assert_eq!(stats1.hits, 0, "Expected 0 hits");
        assert_eq!(stats1.total_ops(), 1);

        // Iteration 2: hit
        let y2 = mul(&x, &x);
        y2.backward().expect("backward 2 failed");
        x.zero_grad();

        let stats2 = cache.stats();
        assert!(stats2.hits > 0, "Expected cache hit");
        assert!(stats2.hit_rate() > 0.0, "Hit rate should be > 0%");

        println!("Final stats: {:#?}", stats2);
    }

    /// Test that one `ComputeGraphCache` instance can be shared across threads.
    ///
    /// The default `Var::backward` path uses a per-thread cache, but a solver
    /// may share a single instance across workers (e.g. data-parallel training
    /// loops) — every field is `Arc`-shared and lock-protected, and erased
    /// plans are `Send + Sync`. This test validates that claim under real
    /// contention: `THREADS` workers each look up their own graph's plan
    /// (distinct root allocation, identical structure) against one shared
    /// cache.
    ///
    /// Roots are retained by the test for the whole run — that is the
    /// training-loop shape (a model outlives an iteration). Weak-root scoping
    /// would otherwise expire each worker's plan the moment that worker's
    /// thread ends, which is correct behavior but leaves nothing to observe.
    /// With live roots, plan counters are exact: one miss per worker on first
    /// use, then all hits, zero evictions and zero expirations. Metadata
    /// counters share one key across workers, so only their sum is asserted
    /// (the miss/hit split at the very first insertion is racy).
    #[test]
    fn test_shared_cache_across_threads() {
        use coeus_autograd::{topological_sort_with_cache, ComputeGraphCache};
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const ITERATIONS: usize = 100;

        let cache = ComputeGraphCache::new();
        // Build every worker's graph up front and retain the creator roots
        // here, so all plans stay resident for the whole run.
        let roots: Vec<Arc<dyn coeus_autograd::BackwardNode<f32, MoiraiBackend>>> = (0..THREADS)
            .map(|_| {
                let x = Var::<f32, MoiraiBackend>::new(
                    Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]),
                    true,
                );
                let y = mul(&x, &x);
                let loss = sum(&y);
                loss.creator
                    .as_ref()
                    .expect("loss must have a creator")
                    .clone()
            })
            .collect();

        let handles: Vec<_> = roots
            .iter()
            .map(|root| {
                let cache = cache.clone();
                let root = Arc::clone(root);
                thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        let order = topological_sort_with_cache(Some(&root), &cache);
                        assert!(
                            !order.is_empty(),
                            "shared-cache lookup must return a live order"
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("worker thread panicked");
        }

        let stats = cache.stats();
        let snapshot = cache.snapshot();

        // One plan miss per worker (first pass on its own root); every later
        // pass reuses that root's retained plan.
        assert_eq!(stats.plan_misses, THREADS as u64);
        assert_eq!(stats.plan_hits, (THREADS * (ITERATIONS - 1)) as u64);
        assert_eq!(stats.total_plan_ops(), (THREADS * ITERATIONS) as u64);
        assert_eq!(stats.plan_entries, THREADS);
        assert_eq!(stats.plan_evictions, 0);
        assert_eq!(stats.plan_expirations, 0);
        assert_eq!(snapshot.plans.len(), THREADS);
        // Each worker's plan saw exactly `ITERATIONS` accesses (insert + hits),
        // all serialized under the plan-table lock.
        for plan in &snapshot.plans {
            assert_eq!(plan.access_count, ITERATIONS as u64);
        }

        // Every pass records exactly one metadata lookup; the hit/miss split
        // for the single shared metadata key is racy, the total is not.
        assert_eq!(
            stats.hits + stats.misses,
            (THREADS * ITERATIONS) as u64,
            "every pass records exactly one metadata lookup"
        );
        assert_eq!(stats.invalidations, 0);
        assert_eq!(stats.metadata_evictions, 0);
    }

    /// Test that many threads hammering ONE shared root's plan stay consistent.
    ///
    /// Complements `test_shared_cache_across_threads`: there each worker owns
    /// its plan; here every worker contends on the same hot plan (the
    /// shared-model case). Plan access accounting is serialized under the
    /// plan-table write lock, but the very first insertion has a benign race:
    /// `lookup_plan` (miss) and `insert_plan` are separate lock acquisitions,
    /// so two workers can both miss before either inserts — the second insert
    /// then replaces the first. Hence `plan_misses` may be 2 (not 1) and the
    /// surviving plan's `access_count` reflects that replacement. The
    /// invariants that always hold: exactly one resident plan, every lookup
    /// recorded exactly once, and no deadlock or panic under contention.
    #[test]
    fn test_shared_cache_single_hot_root() {
        use coeus_autograd::{topological_sort_with_cache, ComputeGraphCache};
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const ITERATIONS: usize = 100;
        const LOOKUPS: usize = THREADS * ITERATIONS;

        let cache = ComputeGraphCache::new();
        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0; 4]), true);
        let y = mul(&x, &x);
        let loss = sum(&y);
        let creator = loss
            .creator
            .as_ref()
            .expect("loss must have a creator")
            .clone();

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let cache = cache.clone();
                let creator = Arc::clone(&creator);
                thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        let order = topological_sort_with_cache(Some(&creator), &cache);
                        assert!(
                            !order.is_empty(),
                            "hot-root lookup must return a live order"
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("worker thread panicked");
        }

        let stats = cache.stats();
        let snapshot = cache.snapshot();

        // Every lookup records exactly one plan operation.
        assert_eq!(stats.total_plan_ops(), LOOKUPS as u64);
        assert!(stats.plan_misses >= 1, "the first lookup must miss");
        assert_eq!(
            stats.plan_hits + stats.plan_misses,
            LOOKUPS as u64,
            "every lookup is exactly one plan hit or miss"
        );
        // One resident plan for the one live root, regardless of how many
        // racing first-insertions happened (later inserts replace).
        assert_eq!(stats.plan_entries, 1);
        assert_eq!(snapshot.plans.len(), 1);
        // The surviving plan's access count cannot exceed the total lookups;
        // a racing replacement may have lost earlier accumulated accesses.
        assert!(snapshot.plans[0].access_count <= LOOKUPS as u64);
        assert!(snapshot.plans[0].access_count >= 1);

        // Every pass records exactly one metadata lookup.
        assert_eq!(
            stats.hits + stats.misses,
            LOOKUPS as u64,
            "every pass records exactly one metadata lookup"
        );
    }
}
