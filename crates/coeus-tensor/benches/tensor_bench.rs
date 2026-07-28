//! Provider-owned Coeus tensor benchmarks.
//!
//! Run all: `cargo bench -p coeus-tensor --bench tensor_bench`.
//! Run one group: `cargo bench -p coeus-tensor --bench tensor_bench -- GELU`.
//!
//! The suite measures one operation through Coeus's Sequential and Moirai
//! execution policies, plus the direct Leto and Coeus-Leto dispatch paths
//! where the operation exposes them. Legacy tensor providers are not benchmark
//! dependencies or comparison oracles.
//!
//! For every retained operation, a fixed input `x` is evaluated as `P_f(x)`
//! by each provider-owned path. The benchmark group owns the shape and input
//! construction, so rows compare the same operation contract without a second
//! reference implementation redefining the layout or tolerance policy.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leto::Array;

#[path = "tensor_bench/provider/mod.rs"]
mod provider;

fn bench_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let shape = vec![size, size];

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone()).expect("construct tensor");
    let b_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone()).expect("construct tensor");

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone()).expect("construct tensor");
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone()).expect("construct tensor");

    let mut group = c.benchmark_group("Elementwise Add (1024x1024)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ).expect("benchmark addition"));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ).expect("benchmark addition"));
        })
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    // 256x256 matmul keeps the benchmark bounded while exercising tiled paths.
    let m = 256;
    let k = 256;
    let n = 256;

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(vec![m, k]).expect("construct tensor");
    let b_seq = Tensor::<f32, SequentialBackend>::ones(vec![k, n]).expect("construct tensor");

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![m, k]).expect("construct tensor");
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![k, n]).expect("construct tensor");

    let a_leto =
        Array::from_shape_vec([m, k], vec![1.0f32; m * k]).expect("benchmark input shape is valid");
    let b_leto =
        Array::from_shape_vec([k, n], vec![1.0f32; k * n]).expect("benchmark input shape is valid");

    let coeus_layout_a = coeus_core::Layout::new(vec![m, k].into());
    let coeus_layout_b = coeus_core::Layout::new(vec![k, n].into());
    let coeus_layout_out = coeus_core::Layout::new(vec![m, n].into());
    let coeus_a = vec![1.0f32; m * k];
    let coeus_b = vec![1.0f32; k * n];

    let mut group = c.benchmark_group("Matrix Multiplication (256x256)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ).expect("benchmark matrix multiplication"));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ).expect("benchmark matrix multiplication"));
        })
    });

    group.bench_function("Leto direct", |b| {
        b.iter_batched(
            || Array::zeros([m, n]),
            |mut out| {
                leto_ops::matmul(
                    black_box(&a_leto.view()),
                    black_box(&b_leto.view()),
                    &mut out.view_mut(),
                )
                .expect("benchmark layout contract is valid");
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("Coeus-Leto dispatch", |b| {
        b.iter_batched(
            || vec![0.0f32; m * n],
            |mut out| {
                coeus_leto::matmul_into(
                    black_box(&coeus_layout_a),
                    black_box(&coeus_a),
                    black_box(&coeus_layout_b),
                    black_box(&coeus_b),
                    black_box(&coeus_layout_out),
                    black_box(&mut out),
                )
                .expect("benchmark layout contract is valid");
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise_add,
    bench_matmul,
    provider::activation::bench_relu,
    provider::activation::bench_gelu,
    provider::activation::bench_sigmoid,
    provider::activation::bench_tanh,
    provider::activation::bench_silu,
    provider::reduction::bench_sum,
    provider::convolution::bench_conv1d,
    provider::convolution::bench_conv2d,
    provider::convolution::bench_conv_transpose2d,
    provider::pooling::bench_max_pool2d,
    provider::neural::bench_softmax,
    provider::neural::bench_attention,
    provider::neural::bench_layernorm,
);
criterion_main!(benches);
