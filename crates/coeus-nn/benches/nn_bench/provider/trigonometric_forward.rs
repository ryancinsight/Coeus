//! trigonometric forward benchmarks.

use super::*;

pub(crate) fn bench_sin_cos_forward(c: &mut Criterion) {
    // sin+cos fused: [128, 256] — typical in RoPE / positional encoding.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| i as f32 * 0.0031).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — sin+cos forward (128x256)");
    group.bench_function("Coeus Sequential (sin+cos)", |b| {
        b.iter(|| {
            let s = coeus_autograd::sin(black_box(&x_seq));
            let c2 = coeus_autograd::cos(black_box(&x_seq));
            black_box((s, c2))
        })
    });
    group.bench_function("Coeus Moirai (sin+cos)", |b| {
        b.iter(|| {
            let s = coeus_autograd::sin(black_box(&x_moirai));
            let c2 = coeus_autograd::cos(black_box(&x_moirai));
            black_box((s, c2))
        })
    });
    group.finish();
}

pub(crate) fn bench_tan_forward(c: &mut Criterion) {
    // tan: [128, 256] — important for rotation/phase ops in signal processing.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0021).sin() * 1.5) // stay in safe domain (|x| < π/2)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — tan forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tan(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tan(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_atan_forward(c: &mut Criterion) {
    // atan: [128, 256] — arctan is used in angle reconstruction and attention scores.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0027).cos() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — atan forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::atan(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::atan(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_asin_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0019).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - asin forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::asin(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::asin(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sinh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sinh forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sinh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sinh(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_cosh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cosh forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cosh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cosh(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_acos_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - acos forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::acos(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::acos(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sin3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.005 - 1.2)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sin3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sin(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sin(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_cos3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.005 - 1.2)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cos3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cos(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cos(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_sin4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.003 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sin4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sin(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sin(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_cos4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.003 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cos4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cos(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cos(black_box(&x_moirai))))
    });
    group.finish();
}
