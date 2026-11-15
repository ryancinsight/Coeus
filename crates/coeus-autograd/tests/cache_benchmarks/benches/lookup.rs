use super::PurgeInterval;
use coeus_autograd::{
    add, get_backward_cache, mul, reset_backward_cache_stats, sum, topological_sort_with_cache,
    BackwardNode, ComputeGraphCache, Var,
};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Measure per-lookup cost on a large, fully-live plan table.
///
/// Builds `table_size` live plans (all roots retained), then times repeated
/// `topological_sort_with_cache` lookups on one hot root. With `interval ==
/// 0` every lookup scans the whole table for expired plans (O(n)); with the
/// default interval the scan is deferred, keeping lookups amortized O(1).
fn measure_lookup_hot_path(interval: u64, table_size: usize, iterations: usize) -> Duration {
    let cache = ComputeGraphCache::with_config(Arc::new(PurgeInterval(interval)));
    let mut roots: Vec<Arc<dyn BackwardNode<f32, MoiraiBackend>>> = Vec::with_capacity(table_size);
    for _ in 0..table_size {
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([4], &[0.1; 4]).expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let y = mul(&x, &x).expect("invariant: test operation succeeds");
        let loss = sum(&y).expect("invariant: test operation succeeds");
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
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([2], &[1.0, 2.0])
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let y = add(&x, &x).expect("invariant: test operation succeeds");
        let loss = sum(&y).expect("invariant: test operation succeeds");
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
        x.zero_grad()
            .expect("invariant: test backend operation succeeds");
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
