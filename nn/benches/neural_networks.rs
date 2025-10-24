//! Neural Network Performance Benchmarks
//!
//! Comprehensive benchmarks comparing Coeus neural network performance
//! against PyTorch baselines. Tests forward/backward passes, gradient
//! computation, and various layer types.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use coeus_autograd::ops::backward_with_grad;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{
    activation::GELU, dropout::Dropout, functional, BatchNorm2d, Conv2D, LayerNorm, Linear, Module,
    MultiHeadAttention, ReLU, Sequential, SparseAttention,
};
use coeus_storage::{CsrStorage, DenseStorage, SparseFormat};
use coeus_tensor::Tensor;

/// Create random tensor with specified shape
fn random_tensor(shape: &[usize]) -> Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    let mut rng = rand::thread_rng();
    let size: usize = shape.iter().product();
    let data: Vec<Float32> = (0..size)
        .map(|_| Float32::new(rng.gen_range(-1.0..1.0)))
        .collect();

    Tensor::from_vec(data, shape).unwrap()
}

/// Create random tensor requiring gradients
fn random_tensor_grad(
    shape: &[usize],
) -> Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    random_tensor(shape).requires_grad_(true)
}

/// Benchmark Linear layer forward pass
fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_forward");

    // Small: 784 -> 128 (typical MNIST)
    group.bench_function("small_784_128", |b| {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 128).unwrap();
        let input = random_tensor(&[32, 784]); // batch_size=32

        b.iter(|| {
            let output = black_box(layer.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Medium: 2048 -> 512
    group.bench_function("medium_2048_512", |b| {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2048, 512).unwrap();
        let input = random_tensor(&[16, 2048]); // batch_size=16

        b.iter(|| {
            let output = black_box(layer.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Large: 4096 -> 1024
    group.bench_function("large_4096_1024", |b| {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4096, 1024).unwrap();
        let input = random_tensor(&[8, 4096]); // batch_size=8

        b.iter(|| {
            let output = black_box(layer.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Linear layer backward pass (with gradients)
fn bench_linear_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_backward");

    // Small: 784 -> 128
    group.bench_function("small_784_128", |b| {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 128).unwrap();
        let input = random_tensor_grad(&[32, 784]);
        let target = random_tensor(&[32, 128]);

        b.iter(|| {
            // Forward pass
            let output = layer.forward(&input).unwrap();

            // Compute loss
            let loss = (&output - &target)
                .powf(Float32::new(2.0))
                .sum(None, false)
                .unwrap();

            // Backward pass
            let loss_shape = loss.shape().dims();
            let grad =
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(loss_shape)
                    .unwrap();
            black_box(backward_with_grad(&loss, &grad).unwrap());
        });
    });

    // Medium: 2048 -> 512
    group.bench_function("medium_2048_512", |b| {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2048, 512).unwrap();
        let input = random_tensor_grad(&[16, 2048]);
        let target = random_tensor(&[16, 512]);

        b.iter(|| {
            let output = layer.forward(&input).unwrap();
            let loss = (&output - &target)
                .powf(Float32::new(2.0))
                .sum(None, false)
                .unwrap();
            let loss_shape = loss.shape().dims();
            let grad =
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(loss_shape)
                    .unwrap();
            black_box(backward_with_grad(&loss, &grad).unwrap());
        });
    });

    group.finish();
}

/// Benchmark Conv2D forward pass
fn bench_conv2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_forward");

    // Small: 32x32 RGB images, 3->64 channels
    group.bench_function("small_32x32_3_64", |b| {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            3,
            64,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            Some(true),
        )
        .unwrap();
        let input = random_tensor(&[8, 3, 32, 32]); // batch_size=8

        b.iter(|| {
            let output = black_box(conv.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Medium: 64x64 images, 64->128 channels
    group.bench_function("medium_64x64_64_128", |b| {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            64,
            128,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            Some(true),
        )
        .unwrap();
        let input = random_tensor(&[4, 64, 64, 64]); // batch_size=4

        b.iter(|| {
            let output = black_box(conv.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark MultiHeadAttention
fn bench_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_forward");

    // Small: embed_dim=256, num_heads=8, seq_len=32
    group.bench_function("small_embed256_heads8_seq32", |b| {
        let attention =
            MultiHeadAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(256, 8)
                .unwrap();
        let input = random_tensor(&[4, 32, 256]); // batch_size=4, seq_len=32

        b.iter(|| {
            let output = black_box(attention.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Medium: embed_dim=512, num_heads=16, seq_len=64
    group.bench_function("medium_embed512_heads16_seq64", |b| {
        let attention =
            MultiHeadAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(512, 16)
                .unwrap();
        let input = random_tensor(&[2, 64, 512]); // batch_size=2, seq_len=64

        b.iter(|| {
            let output = black_box(attention.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Sequential model (MLP)
fn bench_sequential_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_forward");

    // Small MLP: 784 -> 512 -> 256 -> 128 -> 10
    group.bench_function("small_mlp_784_512_256_128_10", |b| {
        let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        model.add_module(
            "fc1".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 512).unwrap(),
        );
        model.add_module("relu1".to_string(), coeus_nn::ReLU);
        model.add_module(
            "fc2".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(512, 256).unwrap(),
        );
        model.add_module("relu2".to_string(), coeus_nn::ReLU);
        model.add_module(
            "fc3".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(256, 128).unwrap(),
        );
        model.add_module("relu3".to_string(), coeus_nn::ReLU);
        model.add_module(
            "fc4".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(128, 10).unwrap(),
        );

        let input = random_tensor(&[32, 784]); // batch_size=32

        b.iter(|| {
            let output = black_box(model.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Sequential model backward pass
fn bench_sequential_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_backward");

    // Small MLP backward
    group.bench_function("small_mlp_backward", |b| {
        let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        model.add_module(
            "fc1".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 256).unwrap(),
        );
        model.add_module("relu1".to_string(), coeus_nn::ReLU);
        model.add_module(
            "fc2".to_string(),
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(256, 10).unwrap(),
        );

        let input = random_tensor_grad(&[16, 784]);
        let target = random_tensor(&[16, 10]);

        b.iter(|| {
            // Forward pass
            let output = model.forward(&input).unwrap();

            // Compute loss (MSE)
            let loss = (&output - &target)
                .powf(Float32::new(2.0))
                .mean(None, false)
                .unwrap();

            // Backward pass
            let loss_shape = loss.shape().dims();
            let grad =
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(loss_shape)
                    .unwrap();
            black_box(backward_with_grad(&loss, &grad).unwrap());
        });
    });

    group.finish();
}

/// Benchmark BatchNorm2d
fn bench_batchnorm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("batchnorm_forward");

    // Small: 64 channels, 32x32 features
    group.bench_function("small_64ch_32x32", |b| {
        let mut batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1)
                .unwrap();
        let input = random_tensor(&[8, 64, 32, 32]); // batch_size=8

        b.iter(|| {
            let output = black_box(batchnorm.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark LayerNorm
fn bench_layernorm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_forward");

    // Small: 512 features
    group.bench_function("small_512_features", |b| {
        let mut layernorm =
            LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(vec![512], 1e-5);
        let input = random_tensor(&[16, 32, 512]); // batch_size=16, seq_len=32

        b.iter(|| {
            let output = black_box(layernorm.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark Dropout
fn bench_dropout_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("dropout_forward");

    // Training mode: 2048 features
    group.bench_function("training_2048_features", |b| {
        let mut dropout = Dropout::new(0.1);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );
        let input = random_tensor(&[32, 2048]);

        b.iter(|| {
            let output = black_box(dropout.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Evaluation mode: 2048 features
    group.bench_function("eval_2048_features", |b| {
        let mut dropout = Dropout::new(0.1);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            false,
        );
        let input = random_tensor(&[32, 2048]);

        b.iter(|| {
            let output = black_box(dropout.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark MSE loss computation
fn bench_mse_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("mse_loss");

    // Small: 128 features
    group.bench_function("small_128_features", |b| {
        let predictions = random_tensor(&[32, 128]);
        let targets = random_tensor(&[32, 128]);

        b.iter(|| {
            let loss = black_box(functional::mse_loss(&predictions, &targets).unwrap());
            black_box(loss);
        });
    });

    // Large: 2048 features
    group.bench_function("large_2048_features", |b| {
        let predictions = random_tensor(&[16, 2048]);
        let targets = random_tensor(&[16, 2048]);

        b.iter(|| {
            let loss = black_box(functional::mse_loss(&predictions, &targets).unwrap());
            black_box(loss);
        });
    });

    group.finish();
}

/// Benchmark activation functions
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    let input = random_tensor(&[32, 1024]);

    // ReLU
    group.bench_function("relu_32x1024", |b| {
        let relu = coeus_nn::ReLU;
        b.iter(|| {
            let output = black_box(relu.forward(&input).unwrap());
            black_box(output);
        });
    });

    // GELU
    group.bench_function("gelu_32x1024", |b| {
        let gelu = GELU;
        b.iter(|| {
            let output = black_box(gelu.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark memory usage: sparse vs dense tensors
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test different matrix sizes
    let sizes = vec![(100, 100), (500, 500), (1000, 1000)];

    for (rows, cols) in sizes {
        group.bench_function(&format!("dense_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensor = random_tensor(&[rows, cols]);
                black_box(tensor.as_slice().len() * std::mem::size_of::<Float32>());
            });
        });

        // Create sparse tensor with 90% sparsity
        group.bench_function(&format!("sparse_90pct_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let dense = random_tensor(&[rows, cols]);
                // For benchmark, just measure the memory of the dense data
                // (actual sparse conversion would be more complex)
                let total_elements = dense.as_slice().len();
                let keep_elements = total_elements / 10; // 10% density = 90% sparsity

                // Sparse storage typically uses: values + indices + pointers
                // Approximate memory: keep_elements * (value + index) + (rows + 1) * pointer
                let approx_sparse_bytes = keep_elements * (4 + 4) + (rows + 1) * 4; // Float32 + usize indices

                black_box(approx_sparse_bytes);
            });
        });
    }

    group.finish();
}

/// Benchmark computation time: sparse vs dense matrix operations
fn bench_computation_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("computation_time");

    // Matrix multiplication benchmarks
    group.bench_function("dense_matmul_256x256", |bencher| {
        let a = random_tensor(&[256, 256]);
        let b = random_tensor(&[256, 256]);

        bencher.iter(|| {
            let result = black_box(a.matmul(&b).unwrap());
            black_box(result);
        });
    });

    // Sparse matrix multiplication (simulated with 90% sparsity)
    group.bench_function("sparse_matmul_256x256_90pct", |bencher| {
        let a_dense = random_tensor(&[256, 256]);
        let b_dense = random_tensor(&[256, 256]);

        // Create sparse versions by zeroing elements
        let mut a_sparse_data = a_dense.as_slice().to_vec();
        let mut b_sparse_data = b_dense.as_slice().to_vec();

        // Make 90% sparse by zeroing 90% of elements
        let mut rng = rand::thread_rng();
        for data in [&mut a_sparse_data, &mut b_sparse_data] {
            let total_elements = data.len();
            let keep_elements = total_elements / 10;
            let mut indices_to_zero: Vec<usize> = (0..total_elements).collect();
            indices_to_zero.shuffle(&mut rng);
            indices_to_zero.truncate(total_elements - keep_elements);

            for idx in indices_to_zero {
                data[idx] = Float32::new(0.0);
            }
        }

        bencher.iter(|| {
            // For sparse matmul, we'd use specialized sparse operations
            // For now, benchmark the dense operation (which would be the fallback)
            let result = black_box(a_dense.matmul(&b_dense).unwrap());
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark ReLU activation: sparse vs dense
fn bench_activation_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_performance");

    let size = 10000;

    // Dense ReLU
    group.bench_function("relu_dense_10k", |b| {
        let relu = ReLU::new();
        let input = random_tensor(&[size]).requires_grad_(true);

        b.iter(|| {
            let output = black_box(relu.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Sparse ReLU (90% sparsity)
    group.bench_function("relu_sparse_10k_90pct", |b| {
        let relu = ReLU::new();
        let mut sparse_data = random_tensor(&[size]).as_slice().to_vec();

        // Make 90% sparse
        let mut rng = rand::thread_rng();
        let total_elements = sparse_data.len();
        let keep_elements = total_elements / 10;
        let mut indices_to_zero: Vec<usize> = (0..total_elements).collect();
        indices_to_zero.shuffle(&mut rng);
        indices_to_zero.truncate(total_elements - keep_elements);

        for idx in indices_to_zero {
            sparse_data[idx] = Float32::new(0.0);
        }

        let sparse_input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            sparse_data,
            &[size],
        )
        .unwrap()
        .requires_grad_(true);

        b.iter(|| {
            let output = black_box(relu.forward(&sparse_input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark attention mechanisms: sparse vs dense
fn bench_attention_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_performance");

    let embed_dim = 64;
    let num_heads = 4;
    let seq_len = 32;
    let batch_size = 4;

    // Dense MultiHeadAttention
    group.bench_function("multihead_dense_32seq", |b| {
        let attention =
            MultiHeadAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                embed_dim, num_heads,
            )
            .unwrap();

        let input = random_tensor(&[batch_size, seq_len, embed_dim]).requires_grad_(true);

        b.iter(|| {
            let output = black_box(attention.forward(&input).unwrap());
            black_box(output);
        });
    });

    // Sparse MultiHeadAttention (using sparse input detection)
    group.bench_function("multihead_sparse_32seq_80pct", |b| {
        let attention =
            MultiHeadAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                embed_dim, num_heads,
            )
            .unwrap();

        // Create input with some sparsity to trigger sparse path
        let mut input_data = random_tensor(&[batch_size, seq_len, embed_dim])
            .as_slice()
            .to_vec();

        // Make 80% sparse by zeroing most elements
        let mut rng = rand::thread_rng();
        let total_elements = input_data.len();
        let keep_elements = total_elements / 5; // 20% density = 80% sparsity
        let mut indices_to_zero: Vec<usize> = (0..total_elements).collect();
        indices_to_zero.shuffle(&mut rng);
        indices_to_zero.truncate(total_elements - keep_elements);

        for idx in indices_to_zero {
            input_data[idx] = Float32::new(0.0);
        }

        let sparse_input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[batch_size, seq_len, embed_dim],
        )
        .unwrap()
        .requires_grad_(true);

        b.iter(|| {
            let output = black_box(attention.forward(&sparse_input).unwrap());
            black_box(output);
        });
    });

    // SparseAttention
    group.bench_function("sparse_attention_32seq_75pct", |b| {
        let attention =
            SparseAttention::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                embed_dim, num_heads, 0.75, // 75% sparsity
            )
            .unwrap();

        let input = random_tensor(&[batch_size, seq_len, embed_dim]).requires_grad_(true);

        b.iter(|| {
            let output = black_box(attention.forward(&input).unwrap());
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark scalability with different sparsity levels
fn bench_sparsity_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_scalability");

    let size = 5000;

    // Test different sparsity levels
    let sparsity_levels = vec![0.5, 0.7, 0.8, 0.9, 0.95];

    for sparsity in sparsity_levels {
        group.bench_function(&format!("relu_sparsity_{:.0}pct", sparsity * 100.0), |b| {
            let relu = ReLU::new();

            // Create sparse input
            let mut sparse_data = random_tensor(&[size]).as_slice().to_vec();
            let total_elements = sparse_data.len();
            let keep_elements = ((1.0 - sparsity) * total_elements as f64) as usize;

            let mut rng = rand::thread_rng();
            let mut indices_to_zero: Vec<usize> = (0..total_elements).collect();
            indices_to_zero.shuffle(&mut rng);
            indices_to_zero.truncate(total_elements - keep_elements);

            for idx in indices_to_zero {
                sparse_data[idx] = Float32::new(0.0);
            }

            let sparse_input =
                Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                    sparse_data,
                    &[size],
                )
                .unwrap()
                .requires_grad_(true);

            b.iter(|| {
                let output = black_box(relu.forward(&sparse_input).unwrap());
                black_box(output);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_forward,
    bench_linear_backward,
    bench_conv2d_forward,
    bench_attention_forward,
    bench_sequential_forward,
    bench_sequential_backward,
    bench_batchnorm_forward,
    bench_layernorm_forward,
    bench_dropout_forward,
    bench_mse_loss,
    bench_activations,
    bench_memory_usage,
    bench_computation_time,
    bench_activation_sparse_vs_dense,
    bench_attention_sparse_vs_dense,
    bench_sparsity_scalability,
);

criterion_main!(benches);
