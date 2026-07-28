//! trigonometric backward benchmarks.

use super::*;

pub(crate) fn bench_atan_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - atan fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::atan(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::atan(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_sinh_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sinh fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::sinh(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::sinh(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_cosh_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cosh fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::cosh(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::cosh(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_asinh_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - asinh fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::asinh(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::asinh(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_acosh_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| 1.1 + (i as f32 * 0.001).sin().abs() * 4.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - acosh fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::acosh(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::acosh(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_acos_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - acos fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::acos(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::acos(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_asin_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - asin fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::asin(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::asin(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_sin_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sin fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::sin(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::sin(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_cos_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - cos fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::cos(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::cos(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_tan_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 1.5)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - tan fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::tan(black_box(&x_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::tan(black_box(&x_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}
