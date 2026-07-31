//! rectifier backward benchmarks.

use super::*;

pub(crate) fn bench_gelu_backward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - gelu fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::gelu(black_box(&x_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::gelu(black_box(&x_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_relu_backward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - relu fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::relu(black_box(&x_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::relu(black_box(&x_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_elu_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - elu fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::elu(black_box(&x_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::elu(black_box(&x_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_celu_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - celu fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::celu(black_box(&x_seq), 1.0);
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::celu(black_box(&x_moirai), 1.0);
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_selu_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - selu fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::selu(black_box(&x_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::selu(black_box(&x_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}
