//! reduction backward benchmarks.

use super::*;

pub(crate) fn bench_cumsum_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cumsum fwd+bwd (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::cumsum(black_box(&x_seq), 1);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::cumsum(black_box(&x_moirai), 1);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_cumprod_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..FEATURES)
        .map(|i| 1.0 + (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cumprod fwd+bwd (256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::cumprod(black_box(&x_seq), 0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::cumprod(black_box(&x_moirai), 0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_prod_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..FEATURES)
        .map(|i| 1.0 + (i as f32 * 0.001).sin() * 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - prod fwd+bwd (256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::prod(black_box(&x_seq)).expect("run operation");
            black_box(o).backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::prod(black_box(&x_moirai)).expect("run operation");
            black_box(o).backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_std_backward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - std_dev fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::std_dev(black_box(&x_seq), true);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::std_dev(black_box(&x_moirai), true);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}
