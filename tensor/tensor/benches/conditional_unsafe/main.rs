//! Benchmarks validating conditional unsafe optimizations in Sprint 2.7
//!
//! Tests performance benefits of unwrap_unchecked() in release builds vs expect() in debug builds.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_tensor_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_addition");

    // Create test tensors of various sizes
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let data1: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..size).map(|i| Float32::new((i + 1) as f32)).collect();

        let tensor1 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data1, &[size]).unwrap();
        let tensor2 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data2, &[size]).unwrap();

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
        let grad_tensor =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(grad_data, &[size])
                .unwrap();

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

    // Compatible broadcasting shapes: (smaller_dim, larger_dim) where broadcasting rules apply
    let sizes = [(1, 100), (10, 10), (1, 1000)];

    for &(dim1, dim2) in &sizes {
        let data1: Vec<Float32> = (0..dim1).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..dim2).map(|i| Float32::new((i + 1) as f32)).collect();

        let tensor1 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data1, &[dim1]).unwrap();
        let tensor2 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data2, &[dim2]).unwrap();

        group.bench_function(format!("broadcast_{}x{}", dim1, dim2), |b| {
            b.iter(|| {
                let result = black_box(&tensor1 + &tensor2);
                assert_eq!(result.shape().dims(), &[dim2]);
            });
        });
    }

    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    let sizes = [(10, 10), (50, 50), (100, 100)];

    for &(rows, cols) in &sizes {
        let size = rows * cols;
        let data1: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..size).map(|i| Float32::new((i + 1) as f32)).collect();

        let matrix1 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data1, &[rows, cols]).unwrap();
        let matrix2 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data2, &[rows, cols]).unwrap();

        group.bench_function(format!("matmul_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let result = black_box(matrix1.matmul(&matrix2).unwrap());
                assert_eq!(result.shape().dims(), &[rows, cols]);
            });
        });
    }

    group.finish();
}

fn bench_elementwise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let data1: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let data2: Vec<Float32> = (0..size).map(|i| Float32::new((i + 1) as f32)).collect();

        let tensor1 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data1, &[size]).unwrap();
        let tensor2 =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data2, &[size]).unwrap();

        group.bench_function(format!("add_{}", size), |b| {
            b.iter(|| {
                let result = black_box(&tensor1 + &tensor2);
                assert_eq!(result.shape().dims(), &[size]);
            });
        });

        group.bench_function(format!("mul_{}", size), |b| {
            b.iter(|| {
                let result = black_box(&tensor1 * &tensor2);
                assert_eq!(result.shape().dims(), &[size]);
            });
        });

        group.bench_function(format!("exp_{}", size), |b| {
            b.iter(|| {
                let result = black_box(crate::ops::arithmetic::exp(&tensor1).unwrap());
                assert_eq!(result.shape().dims(), &[size]);
            });
        });
    }

    group.finish();
}

fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    let sizes = [(100, 10), (1000, 100), (10000, 1000)];

    for &(total_size, reduce_size) in &sizes {
        let data: Vec<Float32> = (0..total_size).map(|i| Float32::new(i as f32)).collect();
        let tensor =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[total_size]).unwrap();

        group.bench_function(format!("sum_{}", total_size), |b| {
            b.iter(|| {
                let result = black_box(tensor.sum_dims(None, false).unwrap());
                assert_eq!(result.shape().dims(), &[1]);
            });
        });

        group.bench_function(format!("mean_{}", total_size), |b| {
            b.iter(|| {
                let result = black_box(tensor.mean_dims(None, false).unwrap());
                assert_eq!(result.shape().dims(), &[1]);
            });
        });
    }

    group.finish();
}

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    let sizes = [1000, 10000, 100000];

    for &size in &sizes {
        group.bench_function(format!("create_tensor_{}", size), |b| {
            b.iter(|| {
                let data: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
                let tensor = black_box(
                    Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[size]).unwrap()
                );
                assert_eq!(tensor.shape().dims(), &[size]);
            });
        });

        group.bench_function(format!("clone_tensor_{}", size), |b| {
            let data: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
            let tensor =
                Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[size]).unwrap();

            b.iter(|| {
                let cloned = black_box(tensor.clone());
                assert_eq!(cloned.shape().dims(), &[size]);
            });
        });
    }

    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd");

    let sizes = [1000, 10000, 100000];

    for &size in &sizes {
        let data: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let tensor =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[size]).unwrap();

        group.bench_function(format!("add_simd_{}", size), |b| {
            b.iter(|| {
                let result = black_box(tensor.add_simd(&tensor));
                assert!(result.is_ok());
            });
        });

        group.bench_function(format!("relu_simd_{}", size), |b| {
            b.iter(|| {
                let result = black_box(tensor.relu_simd());
                assert!(result.is_ok());
            });
        });

        group.bench_function(format!("sum_simd_{}", size), |b| {
            b.iter(|| {
                let result = black_box(tensor.sum_simd());
                assert!(result.is_ok());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_addition,
    bench_gradient_accumulation,
    bench_broadcasting_operations,
    bench_matrix_operations,
    bench_elementwise_operations,
    bench_reduction_operations,
    bench_memory_operations,
    bench_simd_operations
);
criterion_main!(benches);
