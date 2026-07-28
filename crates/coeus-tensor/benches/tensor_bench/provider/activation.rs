//! Elementwise activation benchmarks.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, Criterion};

type SequentialUnary = fn(
    &Tensor<f32, SequentialBackend>,
    &SequentialBackend,
) -> Result<Tensor<f32, SequentialBackend>, coeus_core::BackendError>;
type MoiraiUnary = fn(
    &Tensor<f32, MoiraiBackend>,
    &MoiraiBackend,
) -> Result<Tensor<f32, MoiraiBackend>, coeus_core::BackendError>;

fn bench_unary(
    c: &mut Criterion,
    group_name: &str,
    sequential_operation: SequentialUnary,
    moirai_operation: MoiraiUnary,
) {
    const SIDE: usize = 1_024;
    let data: Vec<f32> = (0..SIDE * SIDE)
        .map(|index| index as f32 * 0.01 - 5.0)
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input = Tensor::<f32, SequentialBackend>::from_slice([SIDE, SIDE], &data).expect("construct tensor");
    let moirai_input = Tensor::<f32, MoiraiBackend>::from_slice([SIDE, SIDE], &data).expect("construct tensor");

    let mut group = c.benchmark_group(group_name);
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| {
            black_box(sequential_operation(
                black_box(&sequential_input),
                black_box(&sequential_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| {
            black_box(moirai_operation(
                black_box(&moirai_input),
                black_box(&moirai_backend),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_relu(c: &mut Criterion) {
    bench_unary(c, "ReLU (1024x1024)", coeus_ops::relu, coeus_ops::relu);
}

pub(crate) fn bench_gelu(c: &mut Criterion) {
    bench_unary(c, "GELU (1024x1024)", coeus_ops::gelu, coeus_ops::gelu);
}

pub(crate) fn bench_sigmoid(c: &mut Criterion) {
    bench_unary(
        c,
        "Sigmoid (1024x1024)",
        coeus_ops::sigmoid,
        coeus_ops::sigmoid,
    );
}

pub(crate) fn bench_tanh(c: &mut Criterion) {
    bench_unary(c, "Tanh (1024x1024)", coeus_ops::tanh, coeus_ops::tanh);
}

pub(crate) fn bench_silu(c: &mut Criterion) {
    bench_unary(c, "SiLU (1024x1024)", coeus_ops::silu, coeus_ops::silu);
}
