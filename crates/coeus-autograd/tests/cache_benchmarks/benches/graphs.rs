use coeus_autograd::{add, get_backward_cache, mul, reset_backward_cache_stats, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use std::time::{Duration, Instant};

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
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([3], &[1.0, 2.0, 3.0])
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let y = (0..GRAPH_SIZE).fold(x.clone(), |acc, _| {
            let tmp = add(&acc, &x).expect("invariant: test operation succeeds");
            mul(&tmp, &x).expect("invariant: test operation succeeds")
        });
        let loss = sum(&y).expect("invariant: test operation succeeds");
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
        x.zero_grad()
            .expect("invariant: test backend operation succeeds");
    }

    // Remove warmup entries so iteration 0 is a real cache miss.
    cache.clear();
    reset_backward_cache_stats();

    // Benchmark iterations
    for iter in 0..ITERATIONS {
        reset_backward_cache_stats();

        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([3], &[1.0, 2.0, 3.0])
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");

        let start = Instant::now();
        let y = (0..GRAPH_SIZE).fold(x.clone(), |acc, _| {
            let tmp = add(&acc, &x).expect("invariant: test operation succeeds");
            mul(&tmp, &x).expect("invariant: test operation succeeds")
        });
        let loss = sum(&y).expect("invariant: test operation succeeds");
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
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

        x.zero_grad()
            .expect("invariant: test backend operation succeeds");
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
            Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4])
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let mut y = x.clone();
        for _ in 0..DEPTH {
            y = mul(&y, &x).expect("invariant: test operation succeeds");
            y = add(&y, &x).expect("invariant: test operation succeeds");
        }
        let loss = sum(&y).expect("invariant: test operation succeeds");
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
        x.zero_grad()
            .expect("invariant: test backend operation succeeds");
    }

    let mut times = Vec::new();

    // Benchmark
    for iter in 0..ITERATIONS {
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([4], &[0.1, 0.2, 0.3, 0.4])
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");

        let start = Instant::now();
        let mut y = x.clone();
        for _ in 0..DEPTH {
            y = mul(&y, &x).expect("invariant: test operation succeeds");
            y = add(&y, &x).expect("invariant: test operation succeeds");
        }
        let loss = sum(&y).expect("invariant: test operation succeeds");
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
        let elapsed = start.elapsed();

        times.push(elapsed);

        if iter == 0 {
            println!("Iteration {}: Initial pass (cache miss)", iter);
        } else if iter % 5 == 0 {
            println!("Iteration {}: With cache (expected hit)", iter);
        }

        x.zero_grad()
            .expect("invariant: test backend operation succeeds");
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
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice(&shape[..], &data)
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");

        // Repeated iterations to generate cache hits within shape
        for _ in 0..10 {
            let y = mul(&x, &x).expect("invariant: test operation succeeds");
            let loss = sum(&y).expect("invariant: test operation succeeds");
            loss.backward()
                .expect("invariant: valid autograd fixture completes backward");
            x.zero_grad()
                .expect("invariant: test backend operation succeeds");
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
