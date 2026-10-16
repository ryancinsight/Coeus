use super::*;

fn margin_vars(
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

pub(crate) fn bench_multi_margin_loss(c: &mut Criterion) {
    // Multi-margin on [128, 10] scores vs class targets, margin 1.0.
    const N: usize = 128;
    const C: usize = 10;

    let score_data: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let targets: Vec<usize> = (0..N).map(|i| i % C).collect();
    let (x_seq, x_moirai) = margin_vars(&score_data, N, C);

    let mut group = c.benchmark_group("Coeus — MultiMarginLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::multi_margin(black_box(&x_seq), black_box(&targets), 2.0, 1.0)
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::multi_margin(black_box(&x_moirai), black_box(&targets), 2.0, 1.0)
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_multi_label_margin_loss(c: &mut Criterion) {
    // Multi-label margin on [128, 10] scores vs [128, 10] label indices.
    const N: usize = 128;
    const C: usize = 10;

    let score_data: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let labels: Vec<isize> = (0..(N * C)).map(|i| (i % C) as isize).collect();
    let (x_seq, x_moirai) = margin_vars(&score_data, N, C);

    let mut group = c.benchmark_group("Coeus — MultiLabelMarginLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::loss::multi_label_margin_loss(black_box(&x_seq), black_box(&labels))
                    .expect("invariant: benchmark shapes and labels are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::loss::multi_label_margin_loss(black_box(&x_moirai), black_box(&labels))
                    .expect("invariant: benchmark shapes and labels are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_soft_margin_loss(c: &mut Criterion) {
    // Soft-margin on [128, 10] inputs vs [128, 10] ±1 targets.
    const N: usize = 128;
    const C: usize = 10;

    let input_data: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let target_data: Vec<f32> = (0..(N * C))
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (x_seq, x_moirai) = margin_vars(&input_data, N, C);
    let (t_seq, t_moirai) = margin_vars(&target_data, N, C);

    let mut group = c.benchmark_group("Coeus — SoftMarginLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::soft_margin(black_box(&x_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::soft_margin(black_box(&x_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_margin_ranking_loss(c: &mut Criterion) {
    // Margin ranking on [128, 10] input pairs vs ±1 targets, margin 1.0.
    const N: usize = 128;
    const C: usize = 10;

    let data1: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let data2: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let targets: Vec<f32> = (0..(N * C))
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (x1_seq, x1_moirai) = margin_vars(&data1, N, C);
    let (x2_seq, x2_moirai) = margin_vars(&data2, N, C);

    let mut group = c.benchmark_group("Coeus — MarginRankingLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::margin_ranking_loss(
                    black_box(&x1_seq),
                    black_box(&x2_seq),
                    black_box(&targets),
                    1.0,
                )
                .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::margin_ranking_loss(
                    black_box(&x1_moirai),
                    black_box(&x2_moirai),
                    black_box(&targets),
                    1.0,
                )
                .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_hinge_embedding_loss(c: &mut Criterion) {
    // Hinge embedding on [128, 10] inputs vs ±1 targets, margin 1.0.
    const N: usize = 128;
    const C: usize = 10;

    let input_data: Vec<f32> = (0..(N * C)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let targets: Vec<f32> = (0..(N * C))
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (x_seq, x_moirai) = margin_vars(&input_data, N, C);

    let mut group = c.benchmark_group("Coeus — HingeEmbeddingLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::hinge_embedding_loss(black_box(&x_seq), black_box(&targets), 1.0)
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::hinge_embedding_loss(black_box(&x_moirai), black_box(&targets), 1.0)
                    .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.finish();
}
