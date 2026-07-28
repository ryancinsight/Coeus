//! Softmax, attention, and normalization benchmarks.

use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_nn::{softmax, LayerNorm, Module};
use coeus_ops::scaled_dot_product_attention;
use coeus_tensor::Tensor;
use criterion::{black_box, Criterion};

pub(crate) fn bench_softmax(c: &mut Criterion) {
    const ROWS: usize = 256;
    const COLUMNS: usize = 1_024;
    let data: Vec<f32> = (0..ROWS * COLUMNS)
        .map(|index| (index as f32 * 0.001).sin())
        .collect();
    let sequential_input = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([ROWS, COLUMNS], &data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let moirai_input = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice([ROWS, COLUMNS], &data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Softmax (256x1024, axis=1)");
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| black_box(softmax(black_box(&sequential_input), 1)))
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| black_box(softmax(black_box(&moirai_input), 1)))
    });
    group.finish();
}

pub(crate) fn bench_attention(c: &mut Criterion) {
    const BATCH_HEADS: usize = 8;
    const SEQUENCE: usize = 64;
    const FEATURES: usize = 32;
    let scale = (FEATURES as f32).powf(-0.5);
    let query: Vec<f32> = (0..BATCH_HEADS * SEQUENCE * FEATURES)
        .map(|index| ((index as f32 + 1.0) * 0.013).sin())
        .collect();
    let key: Vec<f32> = (0..BATCH_HEADS * SEQUENCE * FEATURES)
        .map(|index| ((index as f32 + 3.0) * 0.017).cos())
        .collect();
    let value: Vec<f32> = (0..BATCH_HEADS * SEQUENCE * FEATURES)
        .map(|index| ((index as f32 + 5.0) * 0.011).sin())
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_query =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &query).expect("construct tensor");
    let sequential_key =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &key).expect("construct tensor");
    let sequential_value =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &value).expect("construct tensor");
    let moirai_query =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &query).expect("construct tensor");
    let moirai_key =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &key).expect("construct tensor");
    let moirai_value =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH_HEADS, SEQUENCE, FEATURES], &value).expect("construct tensor");

    let mut group = c.benchmark_group("Scaled dot-product attention (8x64x32)");
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| {
            black_box(scaled_dot_product_attention(
                black_box(&sequential_query),
                black_box(&sequential_key),
                black_box(&sequential_value),
                None,
                false,
                scale,
                black_box(&sequential_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| {
            black_box(scaled_dot_product_attention(
                black_box(&moirai_query),
                black_box(&moirai_key),
                black_box(&moirai_value),
                None,
                false,
                scale,
                black_box(&moirai_backend),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_layernorm(c: &mut Criterion) {
    const BATCH: usize = 4;
    const SEQUENCE: usize = 64;
    const FEATURES: usize = 128;
    let data: Vec<f32> = (0..BATCH * SEQUENCE * FEATURES)
        .map(|index| (index as f32 * 0.001) % 3.0 - 1.5)
        .collect();
    let sequential_input = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQUENCE, FEATURES], &data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let moirai_input = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, SEQUENCE, FEATURES], &data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let sequential_layer =
        LayerNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5).expect("construct layer norm");
    let moirai_layer =
        LayerNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5).expect("construct layer norm");

    let mut group = c.benchmark_group("LayerNorm (4x64x128)");
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| black_box(sequential_layer.forward(black_box(&sequential_input))))
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| black_box(moirai_layer.forward(black_box(&moirai_input))))
    });
    group.finish();
}
