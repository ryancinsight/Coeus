//! Integration test for autodiff computation graph caching.
//!
//! This test verifies that:
//! 1. Cache integration doesn't break existing autodiff functionality
//! 2. Cache correctly tracks hits and misses
//! 3. Gradients remain accurate with caching enabled

#[cfg(test)]
mod cache_integration {
    use coeus_autograd::{
        Var, add, mul, sum, sub, div, get_backward_cache, reset_backward_cache_stats,
    };
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
        assert!((grad_val - 6.0).abs() < 1e-5, "expected gradient 6.0, got {}", grad_val);
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
        assert_eq!(stats_after_first.misses, 1, "Expected 1 miss after first iteration");

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
        // dy/dx = (2x + 1)(x - 1) + (x^2 + x) = 3x^2 - 2x - 1
        // At x=2: dy/dx = 3(4) - 2(2) - 1 = 12 - 4 - 1 = 7

        let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([1], &[2.0]), true);

        let x2 = mul(&x, &x);
        let x2_plus_x = add(&x2, &x);
        let x_minus_1 = sub(&x, &Var::new(Tensor::from_slice([1], &[1.0]), false));
        let y = mul(&x2_plus_x, &x_minus_1);

        y.backward().expect("backward failed");

        let grad = x.grad().expect("grad is None");
        let grad_val = grad.as_slice()[0];

        assert!(
            (grad_val - 7.0).abs() < 1e-4,
            "expected gradient 7.0, got {}",
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
}
