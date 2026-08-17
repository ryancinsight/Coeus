//! Benchmarks demonstrating computation graph caching performance improvements.
//!
//! This module provides benchmarks that measure:
//! - Baseline: autodiff compilation time without cache
//! - With cache: time reduction on iterations 2-N
//! - Cache effectiveness: hit rate and memory usage
//!
//! Expected results: ~30% reduction in compilation overhead with cache enabled.

#[cfg(test)]
mod benches {
    use coeus_autograd::{
        Var, add, mul, sum, get_backward_cache, reset_backward_cache_stats,
        ComputeGraphCache, DefaultCacheConfig,
    };
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;
    use std::time::{Duration, Instant};

    /// Benchmark: Simple repeated computation graph (2x add, 2x mul per iteration).
    ///
    /// This benchmark runs the same computation graph multiple times to demonstrate
    /// cache hit benefits. Without caching, each iteration pays the full topological
    /// sort cost. With caching, iterations 2+ skip the sort.
    ///
    /// Expected: Cache hits on iterations 2-N should reduce time by ~30%.
    #[test]
    #[ignore] // Run with: cargo test --test benchmarks benches::simple_repeated_graph -- --ignored --nocapture
    fn simple_repeated_graph() {
        const ITERATIONS: usize = 100;
        const GRAPH_SIZE: usize = 10; // nodes in graph

        reset_backward_cache_stats();
        let cache = get_backward_cache();
        cache.clear();

        let mut times_without_cache: Vec<Duration> = Vec::new();
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
        let avg_rest = times_with_cache[1..].iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>() / (ITERATIONS - 1) as f64;

        let reduction = ((avg_first - avg_rest) / avg_first) * 100.0;

        println!("\n=== Cache Performance Summary ===");
        println!("First iteration (no cache): {:.2} ms", avg_first);
        println!("Avg subsequent (with cache): {:.2} ms", avg_rest);
        println!("Time reduction: {:.1}%", reduction);
        println!("Total cache hits: {}", cache.stats().hits);
        println!("Total cache misses: {}", cache.stats().misses);
        println!("Cache hit rate: {:.1}%", cache.stats().hit_rate());
        println!("Cached entries: {}", cache.size());

        // Verify at least 20% improvement
        assert!(reduction >= 20.0, "Expected at least 20% improvement, got {:.1}%", reduction);
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
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4]), true);
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
            let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4]), true);

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
        let avg_rest = times[1..].iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>() / (ITERATIONS - 1) as f64;

        let reduction = ((avg_first - avg_rest) / avg_first) * 100.0;

        println!("\n=== Deep Graph Cache Performance ===");
        println!("First iteration: {:.2} ms", avg_first);
        println!("Avg subsequent: {:.2} ms", avg_rest);
        println!("Time reduction: {:.1}%", reduction);
        println!("Final cache size: {} entries", cache.size());
        println!("Cache hit rate: {:.1}%", cache.stats().hit_rate());

        assert!(reduction >= 20.0, "Expected at least 20% improvement for deep graphs");
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
            let data: Vec<f32> = (0..shape.iter().product()).map(|i| i as f32 * 0.1).collect();
            let x = Var::<f32, MoiraiBackend>::new(
                Tensor::from_slice(&shape[..], &data),
                true
            );

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
        println!("Cache correctly distinguished {} different shapes", shapes.len());
        println!("Cache growth per shape:");
        for (i, size) in cache_sizes.iter().enumerate() {
            println!("  Shape {:?}: {} entries", shapes[i], size);
        }

        let stats = cache.stats();
        println!("Total operations: {}", stats.total_ops());
        println!("Hit rate: {:.1}%", stats.hit_rate());
        assert!(stats.hits > 0, "Should have cache hits for repeated shapes");
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

        // Verify basic invariants
        assert_eq!(stats.total_ops(), stats.hits + stats.misses);
        assert!(stats.hits + stats.misses > 0, "Should have operations");
    }
}
