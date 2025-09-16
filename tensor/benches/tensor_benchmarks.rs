//! Comprehensive performance benchmarks for Coeus tensor operations
//!
//! This module provides detailed performance analysis comparing Coeus tensor operations
//! against theoretical performance bounds and measuring memory efficiency.

use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Benchmark tensor creation performance
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                black_box(Tensor::from_vec(data, vec![size]));
            });
        });
    }
    group.finish();
}

/// Benchmark element-wise operations performance
fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 10000, 100000].iter() {
        // Addition benchmark
        group.bench_with_input(BenchmarkId::new("addition", size), size, |b, &size| {
            let a = Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
            let b_tensor =
                Tensor::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size]);
            b.iter(|| black_box(&a + &b_tensor));
        });

        // Multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("multiplication", size),
            size,
            |b, &size| {
                let a = Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
                let b_tensor =
                    Tensor::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size]);
                b.iter(|| black_box(&a * &b_tensor));
            },
        );

        // Exponential benchmark
        group.bench_with_input(BenchmarkId::new("exponential", size), size, |b, &size| {
            b.iter(|| {
                let _a = Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
                black_box(_a.exp())
            });
        });
    }
    group.finish();
}

/// Benchmark matrix multiplication performance
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    group.measurement_time(Duration::from_secs(15));

    for size in [32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let a_data: Vec<f32> = (0..(size * size)).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..(size * size)).map(|i| (i + 1) as f32).collect();

            let a = Tensor::from_vec(a_data, vec![size, size]);
            let b_tensor = Tensor::from_vec(b_data, vec![size, size]);

            b.iter(|| black_box(a.matmul(&b_tensor)));
        });
    }
    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                black_box(Tensor::from_vec(data, vec![size]));
            });
        });
    }
    group.finish();
}

/// Benchmark gradient computation performance
fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        let mut x = Tensor::from_vec((0..*size).map(|i| i as f32).collect(), vec![*size]);
        x.set_requires_grad(true);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let y = (&x * &x).unwrap().sum();
                let _ = black_box(y.backward());
            });
        });
    }
    group.finish();
}

/// Benchmark tensor reshaping operations
fn bench_tensor_reshaping(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_reshaping");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let tensor = Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);

            // Reshape to different dimensions
            let new_shape = if size >= 10000 {
                vec![100, size / 100]
            } else {
                vec![10, size / 10]
            };

            b.iter(|| black_box(tensor.reshape(new_shape.clone())));
        });
    }
    group.finish();
}

/// Benchmark reduction operations
fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 10000, 100000].iter() {
        let tensor = Tensor::from_vec((0..*size).map(|i| i as f32).collect(), vec![*size]);

        // Sum reduction
        group.bench_with_input(BenchmarkId::new("sum", size), size, |b, _| {
            b.iter(|| black_box(tensor.sum()));
        });

        // Mean reduction
        group.bench_with_input(BenchmarkId::new("mean", size), size, |b, _| {
            b.iter(|| black_box(tensor.mean()));
        });
    }
    group.finish();
}

/// Benchmark concurrent tensor operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(15));

    for num_threads in [2, 4, 8].iter() {
        let size = 10000;
        let tensors: Vec<_> = (0..*num_threads)
            .map(|i| {
                let data: Vec<f32> = (0..size).map(|j| (i * size + j) as f32).collect();
                Tensor::from_vec(data, vec![size])
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, _| {
                b.iter(|| {
                    let handles: Vec<_> = tensors
                        .iter()
                        .map(|tensor| {
                            let tensor_clone = tensor.clone();
                            std::thread::spawn(move || {
                                black_box(tensor_clone.exp());
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(5));

    // Test different allocation patterns
    for pattern in ["contiguous", "sparse", "gradient"].iter() {
        match *pattern {
            "contiguous" => {
                group.bench_function("contiguous_allocation", |b| {
                    b.iter(|| {
                        let size = 100000;
                        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                        black_box(Tensor::from_vec(data, vec![size]));
                    });
                });
            }
            "sparse" => {
                group.bench_function("sparse_like_allocation", |b| {
                    b.iter(|| {
                        let size = 1000;
                        // Create a sparse-like pattern
                        let data: Vec<f32> = (0..size)
                            .map(|i| if i % 100 == 0 { i as f32 } else { 0.0 })
                            .collect();
                        black_box(Tensor::from_vec(data, vec![size]));
                    });
                });
            }
            "gradient" => {
                group.bench_function("gradient_tensor_allocation", |b| {
                    b.iter(|| {
                        let size = 10000;
                        let mut tensor =
                            Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
                        tensor.set_requires_grad(true);
                        black_box(tensor);
                    });
                });
            }
            _ => {}
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_elementwise_ops,
    bench_matrix_multiplication,
    bench_memory_allocation,
    bench_gradient_computation,
    bench_tensor_reshaping,
    bench_reduction_operations,
    bench_concurrent_operations,
    bench_memory_usage
);
criterion_main!(benches);
