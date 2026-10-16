use super::*;

fn class_logits(n: usize, c: usize) -> Vec<f32> {
    (0..(n * c)).map(|i| (i as f32 * 0.0041).sin()).collect()
}

fn class_vars(
    data: &[f32],
    n: usize,
    c: usize,
) -> (
    coeus_autograd::Var<f32, SequentialBackend>,
    coeus_autograd::Var<f32, MoiraiBackend>,
) {
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![n, c], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![n, c], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    (x_seq, x_moirai)
}

pub(crate) fn bench_bce_with_logits_loss(c: &mut Criterion) {
    // BCE-with-logits on [128, 10] logits vs [128, 10] targets.
    const N: usize = 128;
    const C: usize = 10;

    let logit_data = class_logits(N, C);
    let target_data: Vec<f32> = (0..(N * C))
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let (x_seq, x_moirai) = class_vars(&logit_data, N, C);
    let (t_seq, t_moirai) = class_vars(&target_data, N, C);

    let mut group = c.benchmark_group("Coeus — BCEWithLogitsLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::bce_with_logits(black_box(&x_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::bce_with_logits(black_box(&x_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_binary_cross_entropy_loss(c: &mut Criterion) {
    // Binary cross-entropy on [128, 10] probabilities vs [128, 10] targets.
    const N: usize = 128;
    const C: usize = 10;

    let prob_data: Vec<f32> = (0..(N * C))
        .map(|i| 0.5 + 0.4 * (i as f32 * 0.0041).sin())
        .collect();
    let target_data: Vec<f32> = (0..(N * C))
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let (x_seq, x_moirai) = class_vars(&prob_data, N, C);
    let (t_seq, t_moirai) = class_vars(&target_data, N, C);

    let mut group = c.benchmark_group("Coeus — BinaryCrossEntropyLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::binary_cross_entropy(black_box(&x_seq), black_box(&t_seq), 1e-7)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::binary_cross_entropy(black_box(&x_moirai), black_box(&t_moirai), 1e-7)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_nll_loss(c: &mut Criterion) {
    // NLL loss on [128, 10] log-probabilities vs class targets.
    const N: usize = 128;
    const C: usize = 10;

    let log_data: Vec<f32> = (0..(N * C))
        .map(|i| -((i % C) as f32 * 0.37 + 0.5))
        .collect();
    let targets: Vec<usize> = (0..N).map(|i| i % C).collect();
    let (x_seq, x_moirai) = class_vars(&log_data, N, C);

    let mut group = c.benchmark_group("Coeus — NLLLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nll_loss(black_box(&x_seq), black_box(&targets))
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nll_loss(black_box(&x_moirai), black_box(&targets))
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_multi_label_soft_margin_loss(c: &mut Criterion) {
    // Multi-label soft-margin on [128, 10] logits vs [128, 10] ±1 targets.
    const N: usize = 128;
    const C: usize = 10;

    let logit_data = class_logits(N, C);
    let target_data: Vec<f32> = (0..(N * C))
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (x_seq, x_moirai) = class_vars(&logit_data, N, C);
    let (t_seq, t_moirai) = class_vars(&target_data, N, C);

    let mut group = c.benchmark_group("Coeus — MultiLabelSoftMarginLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::multi_label_soft_margin_loss(black_box(&x_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::multi_label_soft_margin_loss(black_box(&x_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_kl_divergence_loss(c: &mut Criterion) {
    // KL divergence on [128, 10] log-probabilities vs [128, 10] probabilities.
    const N: usize = 128;
    const C: usize = 10;

    let log_data: Vec<f32> = (0..(N * C))
        .map(|i| -((i % C) as f32 * 0.37 + 0.5))
        .collect();
    let prob_data: Vec<f32> = (0..(N * C))
        .map(|i| 0.05 + 0.04 * (i as f32 * 0.0041).sin())
        .collect();
    let (x_seq, x_moirai) = class_vars(&log_data, N, C);
    let (t_seq, t_moirai) = class_vars(&prob_data, N, C);

    let mut group = c.benchmark_group("Coeus — KLDivergenceLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::kl_divergence(black_box(&x_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::kl_divergence(black_box(&x_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}
