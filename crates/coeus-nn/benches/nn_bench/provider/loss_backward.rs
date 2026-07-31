//! loss backward benchmarks.

use super::*;

pub(crate) fn bench_l1_loss_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let target_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - l1_loss fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::l1_loss(black_box(&x_seq), black_box(&t_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::l1_loss(black_box(&x_moirai), black_box(&t_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_bce_with_logits_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let target_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| if i % 2 == 0 { 1.0f32 } else { 0.0f32 })
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - bce_with_logits fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::bce_with_logits(black_box(&x_seq), black_box(&t_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::bce_with_logits(black_box(&x_moirai), black_box(&t_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_huber_loss_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let target_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - huber_loss(delta=1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::huber_loss(black_box(&x_seq), black_box(&t_seq), 1.0)
                .expect("invariant: benchmark shapes match and delta is positive");
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::huber_loss(black_box(&x_moirai), black_box(&t_moirai), 1.0)
                .expect("invariant: benchmark shapes match and delta is positive");
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_kl_div_backward(c: &mut Criterion) {
    let n = BATCH * FEATURES;
    let input_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.001).sin().abs() + 1e-4).ln())
        .collect();
    let target_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.002).cos().abs() + 1e-4) / (n as f32 * 1e-4 + 1.0))
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &target_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - kl_div fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::kl_divergence(black_box(&x_seq), black_box(&t_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::kl_divergence(black_box(&x_moirai), black_box(&t_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}
