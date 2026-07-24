//! rectifier forward benchmarks.

use super::*;

pub(crate) fn bench_relu_forward(c: &mut Criterion) {
    // ReLU activation on [BATCH=128, FEATURES=256] — largest normalization shape.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — ReLU forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(relu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_gelu_forward(c: &mut Criterion) {
    // GeLU activation on [BATCH=128, FEATURES=256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — GeLU forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(gelu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(gelu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_prelu_forward(c: &mut Criterion) {
    // PReLU on [BATCH x FEATURES] with the shared default alpha = 0.25. Inputs
    // are shifted negative so the parametric branch is exercised on ~half.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin() - 0.4)
        .collect();

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let w_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[0.25]),
        false,
    );
    let w_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![1], &[0.25]),
        false,
    );

    let mut group = c.benchmark_group("Coeus — PReLU forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(prelu(black_box(&x_seq), black_box(&w_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(prelu(black_box(&x_moirai), black_box(&w_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_leaky_relu_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — LeakyReLU forward (128x256, neg_slope=0.01)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(leaky_relu(black_box(&x_seq), 0.01)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(leaky_relu(black_box(&x_moirai), 0.01)))
    });
    group.finish();
}

pub(crate) fn bench_relu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - relu2 fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_gelu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - gelu fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::gelu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::gelu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_selu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - selu fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::selu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::selu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_elu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - elu fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::elu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::elu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_hardshrink_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - hardshrink(0.5) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardshrink(&x_seq, 0.5)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardshrink(&x_moirai, 0.5)))
    });
    group.finish();
}

pub(crate) fn bench_leaky_relu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - leaky_relu2(0.1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::leaky_relu(&x_seq, 0.1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::leaky_relu(&x_moirai, 0.1)))
    });
    group.finish();
}

pub(crate) fn bench_softshrink_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - softshrink(0.5) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softshrink(&x_seq, 0.5)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softshrink(&x_moirai, 0.5)))
    });
    group.finish();
}

pub(crate) fn bench_celu_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - celu(alpha=1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::celu(&x_seq, 1.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::celu(&x_moirai, 1.0)))
    });
    group.finish();
}

pub(crate) fn bench_prelu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let w_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[0.01]),
        false,
    );
    let w_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![1], &[0.01]),
        false,
    );
    let mut group = c.benchmark_group("Coeus - prelu2(0.01) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::prelu(&x_seq, &w_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::prelu(&x_moirai, &w_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_threshold_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - threshold(0.5,-0.5) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::threshold(&x_seq, 0.5, -0.5)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::threshold(&x_moirai, 0.5, -0.5)))
    });
    group.finish();
}

pub(crate) fn bench_relu3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.002 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - relu3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_relu4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.0025 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - relu4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_moirai))))
    });
    group.finish();
}
