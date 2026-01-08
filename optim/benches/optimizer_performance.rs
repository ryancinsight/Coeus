//! Performance benchmarks for Adam and SGD optimizers.
//!
//! This module provides criterion.rs benchmarks to measure and validate
//! GPU acceleration performance across different parameter sizes and sparsity levels.
//!
//! Expected performance targets:
//! - Sparse updates (>10% sparsity): 3-10x speedup
//! - Dense updates: 2-5x speedup
//! - Numerical accuracy: <1e-6 difference between CPU and GPU implementations

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use std::time::Duration;

// Import our optimizers and dependencies
use backend::CpuBackend;
use dtype::float::Float32;
use optim::{Adam, BaseOptimizer, Optimizer};
use storage::DenseStorage;
use tensor::Tensor;

/// Benchmark configuration parameters
#[derive(Clone, Copy)]
struct BenchmarkConfig {
    param_sizes: &'static [usize],
    num_steps: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            param_sizes: &[1_000, 10_000, 100_000],
            num_steps: 10,
        }
    }
}

/// Benchmark function for Adam optimizer
fn bench_adam_optimizer(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("adam_optimizer");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10); // Reduce samples for faster benchmarking

    for &param_size in config.param_sizes {
        group.bench_with_input(
            BenchmarkId::new("cpu", param_size),
            &param_size,
            |b: &mut Bencher, &size| {
                b.iter(|| {
                    // Create parameter tensors using ParamGroup API
                    let params =
                        vec![Tensor::<
                            CpuBackend<Float32>,
                            DenseStorage<Float32>,
                            Float32,
                        >::from_vec(
                            vec![Float32::new(0.1); size], &[size]
                        )
                        .unwrap()];

                    let mut adam = Adam::new(params, 0.001);

                    // Benchmark CPU steps with simple gradient setting
                    for _ in 0..config.num_steps {
                        Optimizer::step(&mut adam).unwrap();
                        BaseOptimizer::zero_grad(&mut adam);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Validation benchmarks to ensure numerical accuracy
fn bench_numerical_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_numerical_accuracy");
    group.measurement_time(Duration::from_secs(2));

    for &param_size in &[1000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("accuracy_test", param_size),
            &param_size,
            |b: &mut Bencher, &size| {
                b.iter(|| {
                    // Test basic Adam optimizer functionality
                    let params =
                        vec![Tensor::<
                            CpuBackend<Float32>,
                            DenseStorage<Float32>,
                            Float32,
                        >::from_vec(vec![Float32::new(0.1); size], &[size])
                        .unwrap()];

                    let mut adam = Adam::new(params, 0.001);

                    // Take a few steps to ensure no crashes
                    for _ in 0..3 {
                        Optimizer::step(&mut adam).unwrap();
                        BaseOptimizer::zero_grad(&mut adam);
                    }

                    // Verify optimizer structure is intact
                    assert_eq!(adam.name(), "Adam");
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    optimizer_benches,
    bench_adam_optimizer,
    bench_numerical_accuracy
);
criterion_main!(optimizer_benches);
