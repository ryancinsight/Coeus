//! Performance benchmarks for sparse matrix operations
//!
//! Tests CSR-dense matrix multiplication, sparse-sparse multiplication,
//! and vector operations with various sparsity patterns.

use coeus_dtype::float::F32;
use coeus_storage::{CsrStorage, SparseFormat, SparseMatMul, SparseReduce};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Create a sparse CSR matrix with given dimensions and sparsity
fn create_sparse_csr(rows: usize, cols: usize, sparsity: f64) -> CsrStorage<F32> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    let nnz = ((rows * cols) as f64 * (1.0 - sparsity)) as usize;
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = vec![0; rows + 1];

    let mut current_nnz = 0;

    for row in 0..rows {
        let row_nnz = if row < rows - 1 {
            ((nnz - current_nnz) as f64 / (rows - row) as f64).round() as usize
        } else {
            nnz - current_nnz
        };

        indptr[row] = current_nnz;

        for _ in 0..row_nnz {
            let col = rng.gen_range(0..cols);
            data.push(F32::new(rng.gen()));
            indices.push(col);
            current_nnz += 1;
        }
    }
    indptr[rows] = current_nnz;

    CsrStorage::new(data, indices, indptr, &[rows, cols]).unwrap()
}

/// Create a dense matrix for testing
fn create_dense_matrix(rows: usize, cols: usize) -> Vec<F32> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    (0..rows * cols).map(|_| F32::new(rng.gen())).collect()
}

fn bench_csr_dense_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_dense_multiplication");

    // Test different matrix sizes
    let test_cases = vec![
        (100, 100, 0.1),  // Small, 10% sparsity
        (500, 500, 0.1),  // Medium, 10% sparsity
        (1000, 100, 0.1), // Large, 10% sparsity
        (500, 500, 0.5),  // Medium, 50% sparsity
        (500, 500, 0.9),  // Medium, 90% sparsity
    ];

    for (rows, cols, sparsity) in test_cases {
        let csr = create_sparse_csr(rows, cols, sparsity);
        let dense_cols = 50; // Fixed dense matrix width
        let dense = create_dense_matrix(cols, dense_cols);

        let bench_name = format!("{}x{}x{}_sparsity_{:.1}", rows, cols, dense_cols, sparsity);
        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                let _result = black_box(csr.matmul_dense(&dense, cols, dense_cols).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_csr_vector_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_vector_multiplication");

    let test_cases = vec![
        (1000, 0.1),  // Small, 10% sparsity
        (5000, 0.1),  // Medium, 10% sparsity
        (10000, 0.1), // Large, 10% sparsity
        (5000, 0.5),  // Medium, 50% sparsity
        (5000, 0.9),  // Medium, 90% sparsity
    ];

    for (size, sparsity) in test_cases {
        let csr = create_sparse_csr(size, size, sparsity);
        let vector: Vec<F32> = (0..size).map(|i| F32::new(i as f32)).collect();

        let bench_name = format!("{}x{}_sparsity_{:.1}", size, size, sparsity);
        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                let _result = black_box(csr.matvec_mul(&vector).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_csr_sparse_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_sparse_multiplication");

    let test_cases = vec![
        (200, 0.1), // Small, 10% sparsity
        (500, 0.1), // Medium, 10% sparsity
        (200, 0.5), // Small, 50% sparsity
        (200, 0.9), // Small, 90% sparsity
    ];

    for (size, sparsity) in test_cases {
        let csr_a = create_sparse_csr(size, size, sparsity);
        let csr_b = create_sparse_csr(size, size, sparsity);

        let bench_name = format!("{}x{}_sparsity_{:.1}", size, size, sparsity);
        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                let _result = black_box(csr_a.matmul_sparse(&csr_b, SparseFormat::Csr).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_sparse_format_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_format_conversions");

    let test_cases = vec![
        (500, 500, 0.1),   // Medium, 10% sparsity
        (1000, 1000, 0.1), // Large, 10% sparsity
        (500, 500, 0.5),   // Medium, 50% sparsity
    ];

    for (rows, cols, sparsity) in test_cases {
        let csr = create_sparse_csr(rows, cols, sparsity);

        group.bench_function(
            format!("csr_to_coo_{}x{}_sparsity_{:.1}", rows, cols, sparsity),
            |b| {
                b.iter(|| {
                    let _coo = black_box(csr.to_coo());
                });
            },
        );

        group.bench_function(
            format!("csr_to_csc_{}x{}_sparsity_{:.1}", rows, cols, sparsity),
            |b| {
                b.iter(|| {
                    let _csc = black_box(csr.to_csc());
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_reductions");

    let test_cases = vec![
        (1000, 1000, 0.1), // Large, 10% sparsity
        (2000, 2000, 0.1), // Very large, 10% sparsity
        (1000, 1000, 0.5), // Large, 50% sparsity
    ];

    for (rows, cols, sparsity) in test_cases {
        let csr = create_sparse_csr(rows, cols, sparsity);

        group.bench_function(
            format!("sum_nz_{}x{}_sparsity_{:.1}", rows, cols, sparsity),
            |b| {
                b.iter(|| {
                    let _sum = black_box(csr.sum_nz());
                });
            },
        );

        group.bench_function(
            format!("nnz_count_{}x{}_sparsity_{:.1}", rows, cols, sparsity),
            |b| {
                b.iter(|| {
                    let _nnz = black_box(csr.nnz());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_csr_dense_multiplication,
    bench_csr_vector_multiplication,
    bench_csr_sparse_multiplication,
    bench_sparse_format_conversions,
    bench_sparse_reductions
);
criterion_main!(benches);
