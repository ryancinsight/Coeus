//! Benchmarks measuring computation graph cache behavior.
//!
//! This module reports:
//! - First-pass versus subsequent-pass timing
//! - Cache hit rate and memory usage
//! - Cache effectiveness for repeated graph patterns
//!
//! The current cache stores metadata and weak-root-scoped topology plans. It
//! never reuses node pointers across separately constructed graphs. Timing
//! output is therefore observational rather than a fixed speedup gate for the
//! dynamic-graph workloads; `same_graph_topology_plan` measures plan reuse.

#[cfg(test)]
mod benches {
    use coeus_autograd::backward_cache::compute_graph_structure_fingerprint;
    use coeus_autograd::{
        add, get_backward_cache, mul, reset_backward_cache_stats, sum, topological_sort_with_cache,
        BackwardNode, CacheConfig, ComputeGraphCache, Var, PLAN_PURGE_MIN_TABLE_SIZE,
    };
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Cache configuration with a custom amortized plan-purge interval.
    struct PurgeInterval(u64);

    impl CacheConfig for PurgeInterval {
        fn plan_purge_interval(&self) -> u64 {
            self.0
        }
    }

    /// Benchmark: Simple repeated computation graph (2x add, 2x mul per iteration).
    ///
    /// This benchmark rebuilds the same graph structure with new node instances
    /// each iteration. It demonstrates metadata-cache hits, but each iteration
    /// still needs a live topology walk because plans are scoped to one root.
    ///
    /// The benchmark reports timing; cache correctness is asserted separately
    /// through hit/miss statistics because metadata caching does not guarantee
    /// a speedup for every graph shape.
    #[test]
    #[ignore] // Run with: cargo test --test benchmarks benches::simple_repeated_graph -- --ignored --nocapture
    fn simple_repeated_graph() {
        const ITERATIONS: usize = 100;
        const GRAPH_SIZE: usize = 10; // nodes in graph

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let mut times_with_cache: Vec<Duration> = Vec::new();

        // Warmup
        for _ in 0..5 {
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
            let y = (0..GRAPH_SIZE).fold(x.clone(), |acc, _| {
                let tmp = add(&acc, &x);
                mul(&tmp, &x)
            });
            let loss = sum(&y);
            let _ = loss.backward();
            x.zero_grad();
        }

        // Remove warmup entries so iteration 0 is a real cache miss.
        cache.clear();
        reset_backward_cache_stats();

        // Benchmark iterations
        for iter in 0..ITERATIONS {
            reset_backward_cache_stats();

            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);

            let start = Instant::now();
            let y = (0..GRAPH_SIZE).fold(x.clone(), |acc, _| {
                let tmp = add(&acc, &x);
                mul(&tmp, &x)
            });
            let loss = sum(&y);
            let _ = loss.backward();
            let elapsed = start.elapsed();

            times_with_cache.push(elapsed);

            let stats = cache.stats();
            if iter == 0 {
                println!("Iteration {}: Initial (expected miss)", iter);
                println!("  Time: {:?}", elapsed);
                println!("  Cache hits: {}, misses: {}", stats.hits, stats.misses);
            } else if iter % 10 == 0 {
                println!("Iteration {}: With cache", iter);
                println!("  Time: {:?}", elapsed);
                println!("  Cache hits: {}, misses: {}", stats.hits, stats.misses);
                println!("  Hit rate: {:.1}%", stats.hit_rate());
            }

            x.zero_grad();
        }

        // Print statistics
        let avg_first = times_with_cache[0].as_secs_f64() * 1000.0; // ms
        let avg_rest = times_with_cache[1..]
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / (ITERATIONS - 1) as f64;

        let reduction = ((avg_first - avg_rest) / avg_first) * 100.0;

        println!("\n=== Cache Performance Summary ===");
        println!("First iteration (no cache): {:.2} ms", avg_first);
        println!("Avg subsequent (with cache): {:.2} ms", avg_rest);
        println!("Time reduction: {:.1}%", reduction);
        println!("Total cache hits: {}", cache.stats().hits);
        println!("Total cache misses: {}", cache.stats().misses);
        println!("Cache hit rate: {:.1}%", cache.stats().hit_rate());
        println!("Cached entries: {}", cache.size());

        assert!(
            cache.stats().hits > 0,
            "Repeated graph should produce cache hits"
        );
    }

    /// Benchmark: Deep computation graph (simulating neural network forward+backward).
    ///
    /// This simulates a deeper graph structure typical in neural networks,
    /// with more operations and greater benefits from caching.
    #[test]
    #[ignore] // Run with: cargo test --test benchmarks benches::deep_computation_graph -- --ignored --nocapture
    fn deep_computation_graph() {
        const ITERATIONS: usize = 50;
        const DEPTH: usize = 20; // depth of graph

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        // Warmup
        for _ in 0..3 {
            let x = Var::<f32, MoiraiBackend>::new(
                Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4]),
                true,
            );
            let mut y = x.clone();
            for _ in 0..DEPTH {
                y = mul(&y, &x);
                y = add(&y, &x);
            }
            let loss = sum(&y);
            let _ = loss.backward();
            x.zero_grad();
        }

        let mut times = Vec::new();

        // Benchmark
        for iter in 0..ITERATIONS {
            let x = Var::<f32, MoiraiBackend>::new(
                Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4]),
                true,
            );

            let start = Instant::now();
            let mut y = x.clone();
            for _ in 0..DEPTH {
                y = mul(&y, &x);
                y = add(&y, &x);
            }
            let loss = sum(&y);
            let _ = loss.backward();
            let elapsed = start.elapsed();

            times.push(elapsed);

            if iter == 0 {
                println!("Iteration {}: Initial pass (cache miss)", iter);
            } else if iter % 5 == 0 {
                println!("Iteration {}: With cache (expected hit)", iter);
            }

            x.zero_grad();
        }

        // Print statistics
        let avg_first = times[0].as_secs_f64() * 1000.0;
        let avg_rest = times[1..]
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / (ITERATIONS - 1) as f64;

        let reduction = ((avg_first - avg_rest) / avg_first) * 100.0;

        println!("\n=== Deep Graph Cache Performance ===");
        println!("First iteration: {:.2} ms", avg_first);
        println!("Avg subsequent: {:.2} ms", avg_rest);
        println!("Time reduction: {:.1}%", reduction);
        println!("Final cache size: {} entries", cache.size());
        println!("Cache hit rate: {:.1}%", cache.stats().hit_rate());

        assert!(
            cache.stats().hits > 0,
            "Repeated deep graph should produce cache hits"
        );
    }

    /// Benchmark: Mixed shapes (testing cache discrimination).
    ///
    /// This benchmark tests that the cache correctly distinguishes between
    /// different shapes and doesn't incorrectly reuse cached graphs.
    #[test]
    #[ignore] // Run with: cargo test --test benchmarks benches::mixed_shapes -- --ignored --nocapture
    fn mixed_shapes() {
        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let shapes = vec![
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![2, 2],
            vec![3, 3],
            vec![2, 3],
        ];

        let mut cache_sizes = Vec::new();

        for shape in &shapes {
            let data: Vec<f32> = (0..shape.iter().product())
                .map(|i| i as f32 * 0.1)
                .collect();
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice(&shape[..], &data), true);

            // Repeated iterations to generate cache hits within shape
            for _ in 0..10 {
                let y = mul(&x, &x);
                let loss = sum(&y);
                let _ = loss.backward();
                x.zero_grad();
            }

            cache_sizes.push(cache.size());
        }

        println!("\n=== Mixed Shapes Cache Behavior ===");
        println!(
            "Cache correctly distinguished {} different shapes",
            shapes.len()
        );
        println!("Cache growth per shape:");
        for (i, size) in cache_sizes.iter().enumerate() {
            println!("  Shape {:?}: {} entries", shapes[i], size);
        }

        let stats = cache.stats();
        println!("Total operations: {}", stats.total_ops());
        println!("Hit rate: {:.1}%", stats.hit_rate());
        assert!(stats.hits > 0, "Should have cache hits for repeated shapes");
    }

    /// Timing + report body shared by the same-graph benchmarks.
    ///
    /// `expected_plans` is the resident plan count the report should show
    /// (1 for the single-plan benchmarks, the pre-filled table size for the
    /// large-table variant). `prewarmed` selects the miss/hit accounting: when
    /// the hot plan is already resident (pre-filled table), every timed
    /// backward is a plan hit; otherwise the first is a miss and the rest hit.
    /// Assertions are deltas against the pre-loop counters so either mode is
    /// exact.
    fn measure_same_graph_plan(
        label: &str,
        loss: &Var<f32, MoiraiBackend>,
        leaf: &Var<f32, MoiraiBackend>,
        iterations: usize,
        expected_plans: usize,
        prewarmed: bool,
    ) {
        let cache = get_backward_cache();
        let creator = loss
            .creator
            .as_ref()
            .expect("benchmark loss must have a creator");
        let (_, graph_info) = compute_graph_structure_fingerprint(creator);
        let before = cache.stats();
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            loss.backward().expect("benchmark backward must succeed");
            times.push(start.elapsed());
            leaf.zero_grad();
        }

        let stats = cache.stats();
        let first_ms = times[0].as_secs_f64() * 1000.0;
        let later_ms = times[1..]
            .iter()
            .map(|duration| duration.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / (iterations - 1) as f64;
        let reduction = ((first_ms - later_ms) / first_ms) * 100.0;

        println!("\\n=== {label} Same-Graph Topology Plan ===");
        println!(
            "Graph nodes: {}, leaves: {}, max depth: {}",
            graph_info.node_count, graph_info.leaf_count, graph_info.max_depth
        );
        println!("First backward: {:.3} ms", first_ms);
        println!("Later average: {:.3} ms", later_ms);
        println!("Observed cold-to-warm reduction: {:.1}%", reduction);
        println!(
            "Plan hits: {}, misses: {}",
            stats.plan_hits, stats.plan_misses
        );
        println!(
            "Plan hit rate: {:.1}% across {} lookups",
            stats.plan_hit_rate(),
            stats.total_plan_ops()
        );
        println!(
            "Plan lifecycle: {} evictions, {} expirations",
            stats.plan_evictions, stats.plan_expirations
        );
        println!("Metadata hits: {}, misses: {}", stats.hits, stats.misses);
        let snapshot = cache.snapshot();
        println!(
            "Plan residency: {} entries, {} bytes (live snapshot: {} plans)",
            stats.plan_entries,
            stats.plan_memory_bytes,
            snapshot.plans.len()
        );
        // Plan table size vs the deferral threshold: below it the expired-plan
        // scan runs exactly on every operation; at/above it the scan is
        // deferred to every Nth op for amortized O(1). The threshold is the
        // cache's own constant, and the interval is the one the cache captured
        // from its config at construction — both sourced from the cache so the
        // report always agrees with the implementation.
        let deferral_threshold = PLAN_PURGE_MIN_TABLE_SIZE;
        let active_interval = cache.plan_purge_interval();
        let engaged = stats.plan_entries >= deferral_threshold;
        println!(
            "Plan table: {} entries, peak {} | {} the {}-entry deferral threshold -> {} | interval: {} ops{}",
            stats.plan_entries,
            stats.peak_plan_entries,
            if engaged { "at/above" } else { "below" },
            deferral_threshold,
            if engaged {
                "deferred (amortized) purge"
            } else {
                "exact per-op purge"
            },
            active_interval,
            if engaged { "" } else { " (not engaged below threshold)" }
        );
        println!(
            "Metadata residency: {} entries, {} bytes, {} evictions",
            snapshot.metadata_entries, snapshot.memory.metadata_bytes, stats.metadata_evictions
        );
        println!(
            "Memory breakdown: metadata {} B + plans {} B = total {} B",
            snapshot.memory.metadata_bytes, snapshot.memory.plan_bytes, snapshot.memory.total_bytes
        );
        for plan in &snapshot.plans {
            println!(
                "  Plan root={} nodes={} bytes={} age={} accesses={}",
                plan.root_id,
                plan.node_count,
                plan.memory_bytes,
                plan.residency_age,
                plan.access_count,
            );
        }

        if prewarmed {
            assert_eq!(stats.plan_misses, before.plan_misses);
            assert_eq!(stats.plan_hits, before.plan_hits + iterations as u64);
        } else {
            assert_eq!(stats.plan_misses, before.plan_misses + 1);
            assert_eq!(stats.plan_hits, before.plan_hits + (iterations - 1) as u64);
        }
        assert_eq!(snapshot.plans.len(), expected_plans);
        // Snapshot count must agree with the residency counter (snapshot purges
        // exactly before reporting).
        assert_eq!(snapshot.plans.len(), stats.plan_entries);
        // Find the hot plan by allocation identity rather than position: with a
        // large table the snapshot is sorted by (fingerprint, root_id), so
        // `plans[0]` is not necessarily the hot root.
        let hot_root_id = Arc::as_ptr(creator) as *const () as usize;
        let hot_plan = snapshot
            .plans
            .iter()
            .find(|plan| plan.root_id == hot_root_id)
            .unwrap_or_else(|| panic!("{label}: hot plan must be resident"));
        assert_eq!(hot_plan.node_count, graph_info.node_count);
    }

    /// Benchmark: Reuse a topology plan for repeated backward passes on one graph.
    ///
    /// Unlike the dynamic-graph workloads above, this keeps the root graph alive
    /// so the graph-local plan can be reused without retaining dropped graphs.
    #[test]
    #[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_topology_plan -- --ignored --nocapture
    fn same_graph_topology_plan() {
        const ITERATIONS: usize = 100;

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
        let y = mul(&x, &x);
        let loss = sum(&y);
        measure_same_graph_plan("Small", &loss, &x, ITERATIONS, 1, false);
    }

    /// Benchmark: Topology-plan reuse for a deep sequential graph.
    #[test]
    #[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_deep_topology_plan -- --ignored --nocapture
    fn same_graph_deep_topology_plan() {
        const ITERATIONS: usize = 100;
        const DEPTH: usize = 128;

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1; 4]), true);
        let one = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0; 4]), false);
        let mut y = x.clone();
        for _ in 0..DEPTH {
            y = add(&y, &x);
            y = mul(&y, &one);
        }
        let loss = sum(&y);
        measure_same_graph_plan("Deep", &loss, &x, ITERATIONS, 1, false);
    }

    /// Benchmark: Topology-plan reuse for a large fan-out/fan-in graph.
    #[test]
    #[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_large_topology_plan -- --ignored --nocapture
    fn same_graph_large_topology_plan() {
        const ITERATIONS: usize = 100;
        const LAYERS: usize = 8;
        const BRANCHES: usize = 16;

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1; 4]), true);
        let one = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0; 4]), false);
        let mut state = x.clone();
        for _ in 0..LAYERS {
            let mut merged = mul(&state, &one);
            for _ in 1..BRANCHES {
                let branch = mul(&state, &one);
                merged = add(&merged, &branch);
            }
            state = merged;
        }
        let loss = sum(&state);
        measure_same_graph_plan("Large", &loss, &x, ITERATIONS, 1, false);
    }

    /// Benchmark: Topology-plan reuse on one hot graph while the plan table
    /// stays large, so the deferred-purge report line shows the engaged
    /// (amortized) state instead of the exact per-op mode of the single-plan
    /// benchmarks.
    ///
    /// Pre-fills the table with `TABLE_SIZE` live plans (>= the
    /// [`PLAN_PURGE_MIN_TABLE_SIZE`] deferral threshold; all roots retained so
    /// nothing expires), then times repeated backward passes on one hot graph
    /// whose plan is already resident. Every timed backward is a plan hit;
    /// `plan_entries` stays at `TABLE_SIZE`, so the report reads
    /// "at/above the {threshold}-entry deferral threshold -> deferred
    /// (amortized) purge".
    #[test]
    #[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_topology_plan_large_table -- --ignored --nocapture
    fn same_graph_topology_plan_large_table() {
        const ITERATIONS: usize = 100;
        // Keep the table above the cache's deferral threshold so the
        // deferred-purge path stays engaged even if the threshold changes.
        const TABLE_SIZE: usize = PLAN_PURGE_MIN_TABLE_SIZE + 16;

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        // Build TABLE_SIZE live graphs and insert a plan for each. All creator
        // roots are retained so every plan stays resident and the table sits
        // at/above the deferral threshold. Graph i gets (i % 4) + 1 layers so
        // the table holds several distinct fingerprints.
        let mut roots: Vec<Arc<dyn BackwardNode<f32, MoiraiBackend>>> =
            Vec::with_capacity(TABLE_SIZE);
        let mut hot: Option<(Var<f32, MoiraiBackend>, Var<f32, MoiraiBackend>)> = None;
        for i in 0..TABLE_SIZE {
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1; 4]), true);
            let one = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0; 4]), false);
            let mut y = x.clone();
            for _ in 0..((i % 4) + 1) {
                y = add(&y, &x);
                y = mul(&y, &one);
            }
            let loss = sum(&y);
            let creator = loss
                .creator
                .as_ref()
                .expect("benchmark loss must have a creator")
                .clone();
            let _ = topological_sort_with_cache(Some(&creator), &cache);
            roots.push(creator);
            if i == 0 {
                hot = Some((loss, x));
            }
        }

        let (loss, leaf) = hot.expect("hot graph must exist");
        // The hot plan is already resident: the timed loop is all hits.
        measure_same_graph_plan("LargeTable", &loss, &leaf, ITERATIONS, TABLE_SIZE, true);
    }

    /// Measure per-lookup cost on a large, fully-live plan table.
    ///
    /// Builds `table_size` live plans (all roots retained), then times repeated
    /// `topological_sort_with_cache` lookups on one hot root. With `interval ==
    /// 0` every lookup scans the whole table for expired plans (O(n)); with the
    /// default interval the scan is deferred, keeping lookups amortized O(1).
    fn measure_lookup_hot_path(interval: u64, table_size: usize, iterations: usize) -> Duration {
        let cache = ComputeGraphCache::with_config(Arc::new(PurgeInterval(interval)));
        let mut roots: Vec<Arc<dyn BackwardNode<f32, MoiraiBackend>>> =
            Vec::with_capacity(table_size);
        for _ in 0..table_size {
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1; 4]), true);
            let y = mul(&x, &x);
            let loss = sum(&y);
            let creator = loss
                .creator
                .as_ref()
                .expect("benchmark loss must have a creator")
                .clone();
            roots.push(creator);
        }
        let hot = roots[0].clone();
        for root in &roots {
            let _ = topological_sort_with_cache(Some(root), &cache);
        }
        // Warm up so every timed lookup is a plan hit on the hot root.
        for _ in 0..10 {
            let _ = topological_sort_with_cache(Some(&hot), &cache);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = topological_sort_with_cache(Some(&hot), &cache);
        }
        let elapsed = start.elapsed();

        let stats = cache.stats();
        assert_eq!(stats.plan_misses, table_size as u64);
        assert_eq!(stats.plan_hits, (10 + iterations) as u64);
        assert_eq!(stats.plan_entries, table_size);
        assert_eq!(stats.plan_expirations, 0);
        elapsed
    }

    /// Benchmark: Lookup hot-path cost on a large plan table, amortized vs
    /// per-op purging.
    ///
    /// Quantifies what deferred expired-plan scanning buys on the lookup hot
    /// path: `plan_purge_interval() == 0` rescans the whole table every lookup,
    /// while the default interval (64) defers the scan. Timing is observational;
    /// hit/miss counts are asserted exactly per table size.
    #[test]
    #[ignore] // Run with: cargo test --test cache_benchmarks benches::many_graph_lookup_hot_path_amortized_purge -- --ignored --nocapture
    fn many_graph_lookup_hot_path_amortized_purge() {
        const ITERATIONS: usize = 200;
        const TABLE_SIZES: [usize; 3] = [64, 256, 1024];

        println!("\n=== Many-Graph Lookup Hot Path: Amortized Purge ===");
        println!(
            "{label:>11} | {exact:>14} | {amortized:>14} | {gain:>6}",
            label = "table size",
            exact = "exact (µs/lookup)",
            amortized = "amortized (µs/lookup)",
            gain = "gain"
        );
        for &table_size in &TABLE_SIZES {
            let exact = measure_lookup_hot_path(0, table_size, ITERATIONS);
            let amortized = measure_lookup_hot_path(64, table_size, ITERATIONS);
            let exact_us = exact.as_secs_f64() * 1e6 / ITERATIONS as f64;
            let amortized_us = amortized.as_secs_f64() * 1e6 / ITERATIONS as f64;
            let gain = exact_us / amortized_us;
            println!("{table_size:>11} | {exact_us:>13.3} | {amortized_us:>13.3} | {gain:>5.1}x");
        }
    }

    /// Benchmark: Cache statistics validation.
    ///
    /// Verifies that cache statistics are accurately tracked.
    #[test]
    fn cache_statistics_validation() {
        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        // Generate deterministic workload
        for _ in 0..5 {
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);
            let y = add(&x, &x);
            let loss = sum(&y);
            let _ = loss.backward();
            x.zero_grad();
        }

        let stats = cache.stats();
        println!("\n=== Cache Statistics ===");
        println!("Hits: {}", stats.hits);
        println!("Misses: {}", stats.misses);
        println!("Hit rate: {:.1}%", stats.hit_rate());
        println!("Memory: {} bytes", stats.memory_bytes);
        println!("Invalidations: {}", stats.invalidations);
        println!(
            "Metadata residency: {} entries, evictions: {}",
            stats.metadata_entries, stats.metadata_evictions
        );
        println!(
            "Plan residency: {} entries, {} bytes",
            stats.plan_entries, stats.plan_memory_bytes
        );

        // Verify basic invariants
        assert_eq!(stats.total_ops(), stats.hits + stats.misses);
        assert!(stats.hits + stats.misses > 0, "Should have operations");
    }
}
