//! Comprehensive benchmarks for neural network operations
//!
//! This module provides performance benchmarks for critical neural network
//! operations to ensure competitive performance and identify regressions.

use backend::CpuBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dtype::float::Float32;
use nn::{Conv2D, Linear, Module, ReLU, Sequential};
use std::time::Duration;
use storage::DenseStorage;
use tensor::Tensor;

/// Type alias for our benchmark tensor type
type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Generate random tensor data for benchmarking
fn random_tensor(shape: &[usize]) -> TestTensor {
    let size = shape.iter().product();
    let data = (0..size)
        .map(|i| Float32::new((i % 100) as f32 / 100.0))
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Benchmark Linear layer forward pass
fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_forward");

    // Small linear layer
    group.bench_function("linear_small", |b| {
        let linear =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 32).unwrap();
        let input = random_tensor(&[8, 64]);

        b.iter(|| {
            let output = linear.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    // Medium linear layer
    group.bench_function("linear_medium", |b| {
        let linear =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(512, 256).unwrap();
        let input = random_tensor(&[16, 512]);

        b.iter(|| {
            let output = linear.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    // Large linear layer
    group.bench_function("linear_large", |b| {
        let linear =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2048, 1024).unwrap();
        let input = random_tensor(&[4, 2048]);

        b.iter(|| {
            let output = linear.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Conv2D layer forward pass
fn bench_conv2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_forward");

    // Small conv2d
    group.bench_function("conv2d_small", |b| {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            (3, 3),
            None,
            None,
            None,
        )
        .unwrap();
        let input = random_tensor(&[2, 3, 32, 32]);

        b.iter(|| {
            let output = conv.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    // Medium conv2d
    group.bench_function("conv2d_medium", |b| {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            64,
            128,
            (3, 3),
            Some((2, 2)),
            None,
            None,
        )
        .unwrap();
        let input = random_tensor(&[2, 64, 16, 16]);

        b.iter(|| {
            let output = conv.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Sequential model forward pass
fn bench_sequential_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_forward");

    // Simple MLP
    group.bench_function("mlp_simple", |b| {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("linear1".to_string(), Linear::new(784, 256).unwrap());
        seq.add_module("relu1".to_string(), ReLU::new());
        seq.add_module("linear2".to_string(), Linear::new(256, 128).unwrap());
        seq.add_module("relu2".to_string(), ReLU::new());
        seq.add_module("linear3".to_string(), Linear::new(128, 10).unwrap());

        let input = random_tensor(&[32, 784]);

        b.iter(|| {
            let output = seq.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark parameter operations
fn bench_parameter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_ops");

    let linear =
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1024, 512).unwrap();

    // Parameter access
    group.bench_function("parameters_access", |b| {
        b.iter(|| {
            let params = linear.parameters();
            black_box(&params);
        });
    });

    // Zero gradients
    group.bench_function("zero_grad", |b| {
        let mut linear_copy =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1024, 512).unwrap();
        b.iter(|| {
            linear_copy.zero_grad();
        });
    });

    group.finish();
}

/// Benchmark memory usage for different operations
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Large tensor creation
    group.bench_function("large_tensor_creation", |b| {
        b.iter(|| {
            let tensor = random_tensor(&[1, 2048, 2048]); // ~16MB tensor
            black_box(tensor);
        });
    });

    // Large linear layer
    group.bench_function("large_linear_creation", |b| {
        b.iter(|| {
            let linear =
                Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4096, 4096)
                    .unwrap();
            black_box(linear);
        });
    });

    group.finish();
}

/// Benchmark tensor operations that are building blocks for NN layers
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_ops");

    // Matrix multiplication (core of linear layers)
    group.bench_function("matmul_small", |b| {
        let a = random_tensor(&[64, 128]);
        let other = random_tensor(&[128, 64]);

        b.iter(|| {
            let result = a.matmul(&other).unwrap();
            black_box(result);
        });
    });

    // Element-wise operations (used in activations)
    group.bench_function("elementwise_add", |b| {
        let a = random_tensor(&[1024, 1024]);
        let other = random_tensor(&[1024, 1024]);

        b.iter(|| {
            let result = &a + &other;
            black_box(result);
        });
    });

    // Sum reduction (used in loss functions)
    group.bench_function("sum_reduction", |b| {
        let tensor = random_tensor(&[1024, 1024]);

        b.iter(|| {
            let result = tensor.sum_simd().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = bench_linear_forward, bench_conv2d_forward, bench_sequential_forward,
             bench_parameter_operations, bench_memory_usage, bench_tensor_operations
}

criterion_main!(benches);
