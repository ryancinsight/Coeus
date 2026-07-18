//! exponential backward benchmarks.

use super::*;

pub(crate) fn bench_exp2_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 4.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - exp2 fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::exp2(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::exp2(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log2_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.1)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log2 fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log2(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log2(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_expm1_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - expm1 fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::expm1(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::expm1(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log1p_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log1p fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log1p(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log1p(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log10_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.1)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log10 fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log10(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log10(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_exp_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - exp fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::exp(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::exp(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log_softmax_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log_softmax fwd+bwd (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log_softmax(black_box(&x_seq), 1);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log_softmax(black_box(&x_moirai), 1);
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_log_sum_exp_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - log_sum_exp(axis=1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::log_sum_exp(black_box(&x_seq), 1);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::log_sum_exp(black_box(&x_moirai), 1);
            black_box(o).backward()
        })
    });
    group.finish();
}
