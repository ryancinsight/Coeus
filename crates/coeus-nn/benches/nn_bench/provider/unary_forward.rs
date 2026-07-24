//! unary forward benchmarks.

use super::*;

pub(crate) fn bench_erf_forward(c: &mut Criterion) {
    // erf: [128, 256] — Gauss error function, used in GELU approximation.
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

    let mut group = c.benchmark_group("Coeus — erf forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::erf(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::erf(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_erfc_forward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - erfc forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::erfc(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::erfc(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_neg_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - neg forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::neg(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::neg(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sqrt2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - sqrt fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_abs2_forward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - abs fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_recip_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.1)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - recip forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::recip(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::recip(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sqrt3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| 0.25 + i as f32 * 0.0001)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - sqrt3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_abs3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.001 - 0.4)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - abs3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sqrt4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| 0.3 + i as f32 * 0.00008)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - sqrt4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_moirai))))
    });
    group.finish();
}
