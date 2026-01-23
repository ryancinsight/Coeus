#[cfg(feature = "gpu")]
use backend::GpuBackend;
use backend::{Backend, CpuBackend};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dtype::float::Float32;
use std::time::Duration;
use storage::DenseStorage;
use storage::Storage;

fn create_sparse_matrix(
    m: usize,
    n: usize,
    sparsity: f32,
) -> (Vec<Float32>, Vec<usize>, Vec<usize>) {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];

    for _ in 0..m {
        for j in 0..n {
            if rand::random::<f32>() > sparsity {
                data.push(Float32::new(rand::random::<f32>()));
                indices.push(j);
            }
        }
        indptr.push(data.len());
    }

    (data, indices, indptr)
}

fn create_dense_matrix(rows: usize, cols: usize) -> DenseStorage<Float32> {
    let data = (0..rows * cols)
        .map(|_| Float32::new(rand::random::<f32>()))
        .collect();
    DenseStorage::from_vec(data, &[rows, cols]).unwrap()
}

fn bench_sparse_matvec_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matvec_cpu");

    // Small matrix (1000x1000, 90% sparse)
    let (data, indices, indptr) = create_sparse_matrix(1000, 1000, 0.9);

    group.bench_function("small_90pct_sparse", |b| {
        b.iter(|| {
            let backend = CpuBackend::new();
            let vector_data: Vec<Float32> = data.iter().take(1000).cloned().collect();
            let result = backend.spmv_csr(
                black_box(&data),
                black_box(&indices),
                black_box(&indptr),
                black_box(&vector_data),
                1000,
                1000,
            );
            black_box(result.unwrap());
        })
    });

    // Medium matrix (5000x5000, 95% sparse)
    let (data, indices, indptr) = create_sparse_matrix(5000, 5000, 0.95);

    group.bench_function("medium_95pct_sparse", |b| {
        b.iter(|| {
            let backend = CpuBackend::new();
            let vector_data: Vec<Float32> = data.iter().take(5000).cloned().collect();
            let result = backend.spmv_csr(
                black_box(&data),
                black_box(&indices),
                black_box(&indptr),
                black_box(&vector_data),
                5000,
                5000,
            );
            black_box(result.unwrap());
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_sparse_matvec_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matvec_gpu");

    // Small matrix (1000x1000, 90% sparse)
    let (data, indices, indptr) = create_sparse_matrix(1000, 1000, 0.9);
    let vector = create_dense_matrix(1000, 1)
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<f32>>();

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("small_90pct_sparse", |b| {
        b.iter(|| {
            rt.block_on(async {
                if let Ok(backend) = GpuBackend::new().await {
                    let result = backend.spmv_csr_float32(
                        black_box(&data),
                        black_box(&indices),
                        black_box(&indptr),
                        black_box(&vector),
                        1000,
                        1000,
                    );
                    black_box(result.unwrap());
                }
            });
        })
    });

    // Medium matrix (5000x5000, 95% sparse)
    let (data, indices, indptr) = create_sparse_matrix(5000, 5000, 0.95);
    let vector = create_dense_matrix(5000, 1)
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<f32>>();

    group.bench_function("medium_95pct_sparse", |b| {
        b.iter(|| {
            rt.block_on(async {
                if let Ok(backend) = GpuBackend::new().await {
                    let result = backend.spmv_csr_float32(
                        black_box(&data),
                        black_box(&indices),
                        black_box(&indptr),
                        black_box(&vector),
                        5000,
                        5000,
                    );
                    black_box(result.unwrap());
                }
            });
        })
    });

    group.finish();
}

fn bench_sparse_matmul_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul_cpu");

    // Test SPMM: sparse × dense matrices
    // Small: 500x500 sparse * 500x100 dense
    let (sparse_data, sparse_indices, sparse_indptr) = create_sparse_matrix(500, 500, 0.9);
    let dense_matrix = create_dense_matrix(500, 100);

    group.bench_function("small_spmm_90pct_sparse", |b| {
        b.iter(|| {
            let backend = CpuBackend::new();
            let result = backend.spmm_csr(
                black_box(&sparse_data),
                black_box(&sparse_indices),
                black_box(&sparse_indptr),
                black_box(&dense_matrix),
                500,
                500,
            );
            black_box(result.unwrap());
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_sparse_matmul_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul_gpu");

    // Test SPMM: sparse × dense matrices
    // Small: 500x500 sparse * 500x100 dense
    let (sparse_data, sparse_indices, sparse_indptr) = create_sparse_matrix(500, 500, 0.9);
    let dense_matrix = create_dense_matrix(500, 100);

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("small_spmm_90pct_sparse", |b| {
        b.iter(|| {
            rt.block_on(async {
                if let Ok(backend) = GpuBackend::new().await {
                    let result = backend
                        .spmm_dense_float32(
                            black_box(&sparse_data),
                            black_box(&sparse_indices),
                            black_box(&sparse_indptr),
                            black_box(dense_matrix.as_slice()),
                            500,
                            500,
                            100,
                        )
                        .await;
                    black_box(result.unwrap());
                }
            });
        })
    });

    group.finish();
}

fn bench_sparse_efficiency_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_efficiency_comparison");

    // Test: Dense 1000x1000 matmul vs equivalent sparse matmul
    // Create a sparse matrix with 90% sparsity (only 10% non-zeros)
    let (sparse_data, sparse_indices, sparse_indptr) = create_sparse_matrix(1000, 1000, 0.9);
    let dense_matrix = create_dense_matrix(1000, 1000);

    // Compare dense matmul with equivalent sparse operation
    group.bench_function("dense_matmul_baseline", |b| {
        b.iter(|| {
            let dense_a: Vec<f32> = (0..1000 * 1000).map(|_| rand::random::<f32>()).collect();
            let dense_b: Vec<f32> = (0..1000 * 1000).map(|_| rand::random::<f32>()).collect();

            // Simulate result storage
            let mut result = vec![0.0f32; 1000 * 1000];
            for i in 0..1000 {
                for j in 0..1000 {
                    for k in 0..1000 {
                        result[i * 1000 + j] += dense_a[i * 1000 + k] * dense_b[k * 1000 + j];
                    }
                }
            }
            black_box(result);
        })
    });

    group.bench_function("sparse_matmul_equivalent", |b| {
        b.iter(|| {
            // For each row in sparse matrix, multiply by dense columns
            let mut result = vec![0.0f32; 1000 * 1000];
            for row in 0..1000 {
                let start = sparse_indptr[row];
                let end = sparse_indptr[row + 1];

                for pos in start..end {
                    let col = sparse_indices[pos];
                    let val = sparse_data[pos].get();

                    // Multiply this sparse element by entire dense matrix row
                    for dense_col in 0..1000 {
                        let dense_idx = col * 1000 + dense_col;
                        let result_idx = row * 1000 + dense_col;
                        result[result_idx] += val * dense_matrix.as_slice()[dense_idx].get();
                    }
                }
            }
            black_box(result);
        })
    });

    group.finish();
}

#[cfg(not(feature = "gpu"))]
criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = bench_sparse_matvec_cpu, bench_sparse_matmul_cpu, bench_sparse_efficiency_comparison
}

#[cfg(feature = "gpu")]
criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = bench_sparse_matvec_cpu, bench_sparse_matvec_gpu, bench_sparse_matmul_cpu, bench_sparse_matmul_gpu, bench_sparse_efficiency_comparison
}

criterion_main!(benches);
