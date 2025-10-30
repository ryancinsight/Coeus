//! Benchmark suite for sparse automatic differentiation performance
//!
//! Measures memory usage and computational efficiency of sparse gradients
//! compared to dense gradients for various sparsity levels.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use coeus_autograd::sparse_gradients::*;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::{CsrStorage, DenseStorage};
use coeus_tensor::Tensor;

/// Generate a sparse matrix with specified sparsity level
fn generate_sparse_matrix(
    rows: usize,
    cols: usize,
    sparsity: f64,
) -> CsrStorage<Float32> {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0; rows + 1];
    let nnz_target = ((1.0 - sparsity) * (rows * cols) as f64) as usize;

    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    // Create roughly uniform sparsity pattern
    for row in 0..rows {
        let mut row_nnz = 0;
        for col in 0..cols {
            if rng.gen::<f64>() > sparsity && row_nnz < nnz_target / rows + 1 {
                let val = Float32::new(rng.gen_range(0.1..1.0));
                data.push(val);
                indices.push(col);
                row_nnz += 1;
            }
        }
        indptr[row + 1] = data.len();
    }

    CsrStorage::new(data, indices, indptr, &[rows, cols]).unwrap()
}

/// Benchmark sparse matrix multiplication performance
fn bench_sparse_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[256, 512, 1024].iter() {
        for &sparsity in &[0.1, 0.5, 0.9].iter() {
            let matrix_a = generate_sparse_matrix(size, size, sparsity);
            let matrix_b_data: Vec<Float32> = (0..size * size)
                .map(|_| Float32::new(rand::random::<f32>()))
                .collect();

            group.bench_function(
                format!("spmm_{}x{}_sparsity_{:.1}", size, size, sparsity),
                |b| {
                    b.iter(|| {
                        let backend = CpuBackend::<Float32>::default();
                        let spmm = SparseMatMul::new(backend);

                        black_box(spmm.spmm(&matrix_a, &matrix_b_data, size).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark sparse gradient accumulation memory efficiency
fn bench_sparse_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_gradient_accumulation");

    for &size in &[512, 1024, 2048].iter() {
        for &sparsity in &[0.1, 0.5, 0.9].iter() {
            let matrix = generate_sparse_matrix(size, size, sparsity);

            group.bench_function(
                format!("grad_accum_{}x{}_sparsity_{:.1}", size, size, sparsity),
                |b| {
                    b.iter(|| {
                        let mut accumulator =
                            SparseGradientAccumulator::<CpuBackend<Float32>, Float32>::new();

                        // Create a mock tensor (simplified for benchmarking)
                        let storage = DenseStorage::from_vec(
                            vec![Float32::new(1.0); size * size],
                            &[size, size],
                        ).unwrap();

                        let backend = CpuBackend::<Float32>::default();
                        let tensor = Tensor::from_storage(storage, backend).requires_grad_(true).unwrap();

                        // Accumulate gradients
                        black_box(accumulator.accumulate_sparse(&tensor, &tensor).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory usage comparison between dense and sparse storage
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for &size in &[256, 512, 1024].iter() {
        for &sparsity in &[0.1, 0.5, 0.9].iter() {
            let sparse_matrix = generate_sparse_matrix(size, size, sparsity);

            group.bench_function(
                format!("memory_{}x{}_sparsity_{:.1}", size, size, sparsity),
                |b| {
                    b.iter(|| {
                        // Measure memory usage of different storage formats
                        let csr_size = sparse_matrix.nnz() * std::mem::size_of::<Float32>() +
                                     sparse_matrix.nnz() * std::mem::size_of::<usize>() + // indices
                                     ((size + 1) * std::mem::size_of::<usize>()); // indptr

                        let dense_size = size * size * std::mem::size_of::<Float32>();

                        // Calculate compression ratio
                        let ratio = csr_size as f64 / dense_size as f64;

                        black_box((csr_size, dense_size, ratio));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark sparse utilities for format selection and optimization
fn bench_sparse_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_utilities");

    for &size in &[512, 1024].iter() {
        // Create test data with known sparsity patterns
        let dense_data: Vec<Float32> = (0..size * size)
            .map(|i| {
                if i % 10 == 0 { // 10% density pattern
                    Float32::new(rand::random::<f32>() * 10.0)
                } else {
                    Float32::new(0.0)
                }
            })
            .collect();

        group.bench_function(format!("format_selection_{}x{}", size, size), |b| {
            b.iter(|| {
                let nnz = dense_data.iter().filter(|&x| !x.is_zero()).count();
                let total = dense_data.len();
                let should_sparse = sparse_utils::should_use_sparse_format(nnz, total);
                black_box(should_sparse);
            });
        });

        group.bench_function(format!("memory_savings_calc_{}x{}", size, size), |b| {
            b.iter(|| {
                let nnz = dense_data.iter().filter(|&x| !x.is_zero()).count();
                let savings = sparse_utils::estimate_memory_savings(nnz, dense_data.len());
                black_box(savings);
            });
        });

        group.bench_function(format!("optimize_storage_{}x{}", size, size), |b| {
            b.iter(|| {
                let result = sparse_utils::optimize_gradient_storage(&dense_data, &[size, size]);
                black_box(result.unwrap());
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_sparse_matmul, bench_sparse_gradient_accumulation, bench_memory_usage, bench_sparse_utilities
}
criterion_main!(benches);
