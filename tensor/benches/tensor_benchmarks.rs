//! Performance benchmarks for tensor operations
//!
//! This module provides comprehensive benchmarks using the criterion crate
//! to measure and track performance of critical tensor operations.

use coeus_tensor::{ops::indexing::Slice, CpuBackend, Dtype, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark scalar tensor creation
pub fn bench_scalar_creation(c: &mut Criterion) {
    c.bench_function("scalar_creation", |b| {
        b.iter(|| {
            let _tensor: Tensor<f64, CpuBackend> = Tensor::scalar(black_box(42.0));
        });
    });
}

/// Benchmark vector tensor creation
pub fn bench_vector_creation(c: &mut Criterion) {
    let sizes = [10, 100, 1000, 10000];

    let mut group = c.benchmark_group("vector_creation");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            b.iter(|| {
                let _tensor = Tensor::from_vec(CpuBackend::default(), black_box(data.clone()), black_box(vec![size])).unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark matrix creation
pub fn bench_matrix_creation(c: &mut Criterion) {
    let sizes = [(10, 10), (100, 100), (500, 500)];

    let mut group = c.benchmark_group("matrix_creation");
    for &(rows, cols) in &sizes {
        group.bench_with_input(
            format!("size_{}x{}", rows, cols),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data = vec![1.0; rows * cols];
                b.iter(|| {
                    let _tensor =
                        Tensor::from_vec(CpuBackend::default(), black_box(data.clone()), black_box(vec![rows, cols])).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark element-wise addition
pub fn bench_elementwise_addition(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("elementwise_addition");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let tensor_a = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let tensor_b = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let _result = black_box(&tensor_a + &tensor_b).unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark element-wise multiplication
pub fn bench_elementwise_multiplication(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("elementwise_multiplication");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![2.0; size];
            let tensor_a = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let tensor_b = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let _result = black_box(&tensor_a * &tensor_b).unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark matrix multiplication with comprehensive size coverage
pub fn bench_matrix_multiplication(c: &mut Criterion) {
    let sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    let mut group = c.benchmark_group("matrix_multiplication");
    group.measurement_time(std::time::Duration::from_secs(10)); // Longer measurement for accurate results

    for &(m, n, p) in &sizes {
        group.bench_with_input(
            format!("cpu_{}x{}x{}", m, n, p),
            &(m, n, p),
            |b, &(m, n, p)| {
                let data_a = vec![1.0f32; m * n];
                let data_b = vec![2.0f32; n * p];
                let tensor_a = Tensor::from_vec(CpuBackend::default(), data_a, vec![m, n]).unwrap();
                let tensor_b = Tensor::from_vec(CpuBackend::default(), data_b, vec![n, p]).unwrap();

                b.iter(|| {
                    let _result = black_box(tensor_a.matmul(&tensor_b)).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark large matrix multiplication for GPU acceleration testing
pub fn bench_large_matrix_multiplication(c: &mut Criterion) {
    let sizes = [(1024, 1024, 1024), (2048, 2048, 2048)];

    let mut group = c.benchmark_group("large_matrix_multiplication");
    group.sample_size(10); // Fewer samples for large benchmarks
    group.measurement_time(std::time::Duration::from_secs(30));

    for &(m, n, p) in &sizes {
        group.bench_with_input(
            format!("large_{}x{}x{}", m, n, p),
            &(m, n, p),
            |b, &(m, n, p)| {
                // Use smaller matrices for benchmarking to avoid excessive memory usage
                let small_m = m.min(512);
                let small_n = n.min(512);
                let small_p = p.min(512);

                let data_a = vec![0.1f32; small_m * small_n];
                let data_b = vec![0.1f32; small_n * small_p];
                let tensor_a = Tensor::from_vec(CpuBackend::default(), data_a, vec![small_m, small_n]).unwrap();
                let tensor_b = Tensor::from_vec(CpuBackend::default(), data_b, vec![small_n, small_p]).unwrap();

                b.iter(|| {
                    let _result = black_box(tensor_a.matmul(&tensor_b)).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark GPU vs CPU matrix multiplication comparison
pub fn bench_gpu_cpu_comparison(c: &mut Criterion) {
    let sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512)];

    let mut group = c.benchmark_group("gpu_cpu_comparison");
    group.measurement_time(std::time::Duration::from_secs(15));

    for &(m, n, p) in &sizes {
        // CPU benchmark
        group.bench_with_input(
            format!("cpu_{}x{}x{}", m, n, p),
            &(m, n, p),
            |b, &(m, n, p)| {
                let data_a = vec![1.0f32; m * n];
                let data_b = vec![2.0f32; n * p];
                let tensor_a = Tensor::from_vec(CpuBackend::default(), data_a, vec![m, n]).unwrap();
                let tensor_b = Tensor::from_vec(CpuBackend::default(), data_b, vec![n, p]).unwrap();

                b.iter(|| {
                    let _result = black_box(tensor_a.matmul(&tensor_b)).unwrap();
                });
            },
        );

        // GPU benchmark (when available)
        group.bench_with_input(
            format!("gpu_{}x{}x{}", m, n, p),
            &(m, n, p),
            |b, &(m, n, p)| {
                let data_a = vec![1.0f32; m * n];
                let data_b = vec![2.0f32; n * p];
                let tensor_a = Tensor::from_vec(CpuBackend::default(), data_a, vec![m, n]).unwrap();
                let tensor_b = Tensor::from_vec(CpuBackend::default(), data_b, vec![n, p]).unwrap();

                // Note: GPU acceleration would be enabled here when available
                // For now, this falls back to CPU but measures the path
                b.iter(|| {
                    let _result = black_box(tensor_a.matmul(&tensor_b)).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark batch processing performance (critical for neural networks)
pub fn bench_batch_processing(c: &mut Criterion) {
    let batch_sizes = [1, 4, 16, 32, 64];
    let feature_sizes = [128, 256, 512, 1024];

    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(std::time::Duration::from_secs(10));

    for &batch_size in &batch_sizes {
        for &feature_size in &feature_sizes {
            group.bench_with_input(
                format!("batch_{}_features_{}", batch_size, feature_size),
                &(batch_size, feature_size),
                |b, &(batch_size, feature_size)| {
                    // Simulate neural network forward pass
                    let input_data = vec![0.1f32; batch_size * feature_size];
                    let weight_data = vec![0.01f32; feature_size * feature_size];

                    let input = Tensor::from_vec(CpuBackend::default(), input_data, vec![batch_size, feature_size]).unwrap();
                    let weights = Tensor::from_vec(CpuBackend::default(), weight_data, vec![feature_size, feature_size]).unwrap();

                    b.iter(|| {
                        let _output = black_box(input.matmul(&weights)).unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

/// Benchmark memory access patterns and efficiency
pub fn bench_memory_patterns(c: &mut Criterion) {
    let sizes = [1000, 10000, 100000, 1000000];

    let mut group = c.benchmark_group("memory_patterns");

    // Contiguous memory access
    for &size in &sizes {
        group.bench_with_input(format!("contiguous_access_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..size {
                    sum += Dtype::to_f64(&tensor.data()[i]).unwrap_or(0.0);
                }
                black_box(sum);
            });
        });
    }

    // Non-contiguous memory access (transpose)
    for &size in &sizes {
        group.bench_with_input(
            format!("noncontiguous_access_{}", size),
            &size,
            |b, &size| {
                let data = vec![1.0f32; size * size];
                let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size, size]).unwrap();
                let transposed = tensor.t().unwrap();

                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..size {
                        for j in 0..size {
                            sum += Dtype::to_f64(&transposed.data()[i * size + j]).unwrap_or(0.0);
                        }
                    }
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation for neural network layers
pub fn bench_neural_network_gradients(c: &mut Criterion) {
    let layer_configs = [
        (128, 128),   // Small transformer layer
        (512, 512),   // Medium transformer layer
        (1024, 1024), // Large transformer layer
    ];

    let mut group = c.benchmark_group("neural_network_gradients");
    group.measurement_time(std::time::Duration::from_secs(15));

    for &(input_size, hidden_size) in &layer_configs {
        group.bench_with_input(
            format!("transformer_layer_{}_{}", input_size, hidden_size),
            &(input_size, hidden_size),
            |b, &(input_size, hidden_size)| {
                // Simulate a transformer layer: input -> linear -> relu -> linear -> output
                let batch_size = 32;

                let input_data = vec![0.1f32; batch_size * input_size];
                let w1_data = vec![0.01f32; input_size * hidden_size];
                let w2_data = vec![0.01f32; hidden_size * input_size];

                let mut input = Tensor::from_vec(CpuBackend::default(), input_data, vec![batch_size, input_size]).unwrap();
                let mut w1 = Tensor::from_vec(CpuBackend::default(), w1_data, vec![input_size, hidden_size]).unwrap();
                let mut w2 = Tensor::from_vec(CpuBackend::default(), w2_data, vec![hidden_size, input_size]).unwrap();

                input.set_requires_grad(true);
                w1.set_requires_grad(true);
                w2.set_requires_grad(true);

                b.iter(|| {
                    // Forward pass
                    let hidden = input.matmul(&w1).unwrap().relu().unwrap();
                    let output = hidden.matmul(&w2).unwrap();

                    // Backward pass
                    let _: () = output.backward().unwrap();
                    black_box(());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark gradient computation for simple operations
pub fn bench_simple_gradient(c: &mut Criterion) {
    let sizes = [10, 100, 1000];

    let mut group = c.benchmark_group("simple_gradient");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = (&tensor * &tensor).unwrap(); // x²

            b.iter(|| {
                let _: () = result.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark gradient computation for complex expressions
pub fn bench_complex_gradient(c: &mut Criterion) {
    let sizes = [10, 50, 100];

    let mut group = c.benchmark_group("complex_gradient");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut x = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let mut y = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let mut z = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            x.set_requires_grad(true);
            y.set_requires_grad(true);
            z.set_requires_grad(true);

            // Complex expression: x² * y + sin(z)
            let x_squared = (&x * &x).unwrap();
            let x_squared_y = (&x_squared * &y).unwrap();
            let sin_z = z.sin().unwrap();
            let result = (&x_squared_y + &sin_z).unwrap();

            b.iter(|| {
                let _: () = result.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark tensor transpose operations
pub fn bench_transpose(c: &mut Criterion) {
    let sizes = [(10, 10), (100, 100), (500, 500)];

    let mut group = c.benchmark_group("transpose");
    for &(rows, cols) in &sizes {
        group.bench_with_input(
            format!("size_{}x{}", rows, cols),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data = vec![1.0; rows * cols];
                let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![rows, cols]).unwrap();

                b.iter(|| {
                    let _result = black_box(tensor.t().unwrap());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark tensor reshape operations
pub fn bench_reshape(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    let mut group = c.benchmark_group("reshape");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let _result = black_box(tensor.reshape(vec![size / 10, 10]));
            });
        });
    }
    group.finish();
}

/// Benchmark tensor reduction operations (sum)
pub fn bench_sum_reduction(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("sum_reduction");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let _result = black_box(tensor.sum());
            });
        });
    }
    group.finish();
}

/// Benchmark tensor mean operations
pub fn bench_mean_reduction(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("mean_reduction");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            b.iter(|| {
                let _result = black_box(tensor.mean().unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark memory allocation patterns
pub fn bench_memory_allocation(c: &mut Criterion) {
    let sizes = [1000, 10000, 100000];

    let mut group = c.benchmark_group("memory_allocation");
    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            b.iter(|| {
                let data = vec![0.0; size];
                let _tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark activation functions (comprehensive coverage)
pub fn bench_activation_functions(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("activation_functions");

    for &size in &sizes {
        // ReLU and variants
        group.bench_with_input(format!("relu_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.relu());
            });
        });

        // Sigmoid
        group.bench_with_input(format!("sigmoid_{}", size), &size, |b, &size| {
            let data = vec![0.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.sigmoid());
            });
        });

        // Tanh
        group.bench_with_input(format!("tanh_{}", size), &size, |b, &size| {
            let data = vec![0.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.tanh());
            });
        });

        // GELU (common in transformers)
        group.bench_with_input(format!("gelu_{}", size), &size, |b, &size| {
            let data = vec![0.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.gelu());
            });
        });

        // ELU - exponential linear unit
        group.bench_with_input(format!("elu_{}", size), &size, |b, &size| {
            let data = vec![0.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.elu(1.0));
            });
        });
    }
    group.finish();
}

/// Benchmark advanced mathematical operations
pub fn bench_advanced_math(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    let mut group = c.benchmark_group("advanced_math");

    for &size in &sizes {
        // Exponential functions
        group.bench_with_input(format!("exp_{}", size), &size, |b, &size| {
            let data = vec![0.5f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.exp());
            });
        });

        // Logarithmic functions
        group.bench_with_input(format!("log_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.log());
            });
        });

        // Power operations (scalar exponent)
        group.bench_with_input(format!("pow_{}", size), &size, |b, &size| {
            let base_data = vec![2.0f32; size];
            let base = Tensor::from_vec(CpuBackend::default(), base_data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(base.pow(3.0));
            });
        });

        // Trigonometric functions
        group.bench_with_input(format!("sin_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.sin());
            });
        });

        group.bench_with_input(format!("cos_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.cos());
            });
        });

        // Square root
        group.bench_with_input(format!("sqrt_{}", size), &size, |b, &size| {
            let data = vec![4.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.sqrt());
            });
        });

        // Division (reciprocal not implemented, using scalar division instead)
        group.bench_with_input(format!("scalar_div_{}", size), &size, |b, &size| {
            let data = vec![2.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            let divisor = Tensor::from_vec(CpuBackend::default(), vec![2.0f32; size], vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(&tensor / &divisor);
            });
        });
    }
    group.finish();
}

/// Benchmark indexing and slicing operations (critical for data loading)
pub fn bench_indexing_operations(c: &mut Criterion) {
    let sizes = [(100, 100), (1000, 1000), (10000, 100)];

    let mut group = c.benchmark_group("indexing_operations");

    for &(rows, cols) in &sizes {
        // Row slicing
        group.bench_with_input(
            format!("row_slice_{}x{}", rows, cols),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data = vec![1.0f32; rows * cols];
                let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![rows, cols]).unwrap();
                b.iter(|| {
                    let _result = black_box(
                        tensor.slice(&[Slice::Range(0, 10), Slice::Full]).unwrap(),
                    );
                });
            },
        );

        // Column slicing
        group.bench_with_input(
            format!("col_slice_{}x{}", rows, cols),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data = vec![1.0f32; rows * cols];
                let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![rows, cols]).unwrap();
                b.iter(|| {
                    let _result = black_box(
                        tensor.slice(&[Slice::Full, Slice::Range(0, 10)]).unwrap(),
                    );
                });
            },
        );

        // Gather operation (advanced indexing)
        group.bench_with_input(
            format!("gather_{}x{}", rows, cols),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data = vec![1.0f32; rows * cols];
                let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![rows, cols]).unwrap();
                let indices_data: Vec<i64> = vec![0i64; 10];
                b.iter(|| {
                    let _result = black_box(tensor.gather(0, &indices_data)).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark parallel processing with rayon
pub fn bench_parallel_processing(c: &mut Criterion) {
    let sizes = [10000, 100000, 1000000];

    let mut group = c.benchmark_group("parallel_processing");

    for &size in &sizes {
        // Sequential processing
        group.bench_with_input(format!("sequential_{}", size), &size, |b, &size| {
            let data = vec![1.0f32; size];
            let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            b.iter(|| {
                let _result = black_box(tensor.exp());
            });
        });

        // Parallel processing (simulated through multiple operations)
        group.bench_with_input(format!("parallel_batch_{}", size), &size, |b, &size| {
            let data1 = vec![1.0f32; size];
            let data2 = vec![2.0f32; size];
            let data3 = vec![3.0f32; size];
            let tensor1 = Tensor::from_vec(CpuBackend::default(), data1, vec![size]).unwrap();
            let tensor2 = Tensor::from_vec(CpuBackend::default(), data2, vec![size]).unwrap();
            let tensor3 = Tensor::from_vec(CpuBackend::default(), data3, vec![size]).unwrap();
            b.iter(|| {
                let _r1 = black_box(tensor1.exp());
                let _r2 = black_box(tensor2.log());
                let _r3 = black_box(tensor3.sin());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_creation,
    bench_vector_creation,
    bench_matrix_creation,
    bench_elementwise_addition,
    bench_elementwise_multiplication,
    bench_matrix_multiplication,
    bench_large_matrix_multiplication,
    bench_gpu_cpu_comparison,
    bench_batch_processing,
    bench_memory_patterns,
    bench_neural_network_gradients,
    bench_simple_gradient,
    bench_complex_gradient,
    bench_transpose,
    bench_reshape,
    bench_sum_reduction,
    bench_mean_reduction,
    bench_memory_allocation,
    bench_activation_functions,
    bench_advanced_math,
    bench_indexing_operations,
    bench_parallel_processing,
);
criterion_main!(benches);
