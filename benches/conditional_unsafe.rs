//! Benchmarks validating conditional unsafe optimizations in Sprint 2.7
//!
//! Tests performance benefits of unwrap_unchecked() in release builds vs expect() in debug builds.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensor::{Tensor, CpuBackend, DenseStorage};
use dtype::float::Float32;

fn bench_tensor_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_addition");

    // Create test tensors of various sizes
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let data1: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..size).map(|i| Float32::new((i + 1) as f32)).collect();

        let tensor1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data1, &[size]).unwrap();
        let tensor2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data2, &[size]).unwrap();

        group.bench_function(format!("add_{}", size), |b| {
            b.iter(|| {
                let result = black_box(&tensor1 + &tensor2);
                // Ensure result is used to prevent optimization
                assert_eq!(result.shape().dims(), &[size]);
            });
        });
    }

    group.finish();
}

fn bench_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let grad_data: Vec<Float32> = (0..size).map(|_| Float32::new(1.0)).collect();
        let grad_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(grad_data, &[size]).unwrap();

        group.bench_function(format!("accumulate_{}", size), |b| {
            b.iter(|| {
                // Simulate gradient accumulation pattern
                let accumulated = black_box(&grad_tensor + &grad_tensor);
                assert_eq!(accumulated.shape().dims(), &[size]);
            });
        });
    }

    group.finish();
}

fn bench_broadcasting_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    let sizes = [(10, 100), (100, 100), (100, 1000)];

    for &(dim1, dim2) in &sizes {
        let data1: Vec<Float32> = (0..dim1).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..dim2).map(|i| Float32::new((i + 1) as f32)).collect();

        let tensor1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data1, &[dim1]).unwrap();
        let tensor2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data2, &[dim2]).unwrap();

        group.bench_function(format!("broadcast_{}x{}", dim1, dim2), |b| {
            b.iter(|| {
                let result = black_box(&tensor1 + &tensor2);
                assert_eq!(result.shape().dims(), &[dim2.max(dim1)]);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_addition,
    bench_gradient_accumulation,
    bench_broadcasting_operations
);
criterion_main!(benches);

