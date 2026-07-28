//! tensor operation benchmarks.

use super::*;

pub(crate) fn bench_softmax_forward(c: &mut Criterion) {
    // Softmax forward (128x256, dim=1).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — Softmax forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_seq), 1).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_moirai), 1).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_glu_forward(c: &mut Criterion) {
    // GLU = x[:, :H] * sigmoid(x[:, H:]) — input [128, 512] → output [128, 256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES * 2))
        .map(|i| (i as f32 * 0.0027).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES * 2], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES * 2], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — GLU forward (128x512 → 128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::glu(black_box(&x_seq), 1).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::glu(black_box(&x_moirai), 1).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_softmin_forward(c: &mut Criterion) {
    // Softmin = softmax(-x), (128x256, dim=1).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — Softmin forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmin(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmin(black_box(&x_moirai), 1)))
    });
    group.finish();
}

pub(crate) fn bench_softmax2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - softmax fwd (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_seq), 1).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_moirai), 1).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_glu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - glu2 forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(&x_seq).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(&x_moirai).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_sign_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sign forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sign(&x_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sign(&x_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_softmax_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - softmax fwd+bwd (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::softmax(black_box(&x_seq), 1).expect("run operation");
            black_box(o).backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::softmax(black_box(&x_moirai), 1).expect("run operation");
            black_box(o).backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_softmin_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - softmin(dim=1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::softmin(black_box(&x_seq), 1);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::softmin(black_box(&x_moirai), 1);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}
