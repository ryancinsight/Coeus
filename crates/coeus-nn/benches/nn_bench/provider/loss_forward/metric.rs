use super::*;

fn metric_vars(
    data: &[f32],
    n: usize,
    d: usize,
) -> (
    coeus_autograd::Var<f32, SequentialBackend>,
    coeus_autograd::Var<f32, MoiraiBackend>,
) {
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![n, d], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![n, d], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    (x_seq, x_moirai)
}

pub(crate) fn bench_pairwise_distance_loss(c: &mut Criterion) {
    // Pairwise p=2 distance on [128, 64] pairs, eps 1e-6.
    const N: usize = 128;
    const D: usize = 64;

    let data1: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let data2: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let (x1_seq, x1_moirai) = metric_vars(&data1, N, D);
    let (x2_seq, x2_moirai) = metric_vars(&data2, N, D);

    let mut group = c.benchmark_group("Coeus — PairwiseDistance (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::pairwise_distance(black_box(&x1_seq), black_box(&x2_seq), 2.0, 1e-6)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::pairwise_distance(
                    black_box(&x1_moirai),
                    black_box(&x2_moirai),
                    2.0,
                    1e-6,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_triplet_margin_loss(c: &mut Criterion) {
    // Triplet margin on [128, 64] triples, margin 1.0, p 2.0, eps 1e-6.
    const N: usize = 128;
    const D: usize = 64;

    let anchor_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let pos_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let neg_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0033).sin()).collect();
    let (a_seq, a_moirai) = metric_vars(&anchor_data, N, D);
    let (p_seq, p_moirai) = metric_vars(&pos_data, N, D);
    let (n_seq, n_moirai) = metric_vars(&neg_data, N, D);

    let mut group = c.benchmark_group("Coeus — TripletMarginLoss (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::triplet_margin_loss(
                    black_box(&a_seq),
                    black_box(&p_seq),
                    black_box(&n_seq),
                    1.0,
                    2.0,
                    1e-6,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::triplet_margin_loss(
                    black_box(&a_moirai),
                    black_box(&p_moirai),
                    black_box(&n_moirai),
                    1.0,
                    2.0,
                    1e-6,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_triplet_margin_with_distance_loss(c: &mut Criterion) {
    // Triplet margin with pairwise-distance delegate on [128, 64], margin 1.0.
    const N: usize = 128;
    const D: usize = 64;

    let anchor_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let pos_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let neg_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0033).sin()).collect();
    let (a_seq, a_moirai) = metric_vars(&anchor_data, N, D);
    let (p_seq, p_moirai) = metric_vars(&pos_data, N, D);
    let (n_seq, n_moirai) = metric_vars(&neg_data, N, D);

    let mut group = c.benchmark_group("Coeus — TripletMarginWithDistanceLoss (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::triplet_margin_with_distance_loss(
                    black_box(&a_seq),
                    black_box(&p_seq),
                    black_box(&n_seq),
                    |x: &coeus_autograd::Var<f32, SequentialBackend>,
                     y: &coeus_autograd::Var<f32, SequentialBackend>| {
                        coeus_nn::pairwise_distance(x, y, 2.0, 1e-6)
                    },
                    1.0,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::triplet_margin_with_distance_loss(
                    black_box(&a_moirai),
                    black_box(&p_moirai),
                    black_box(&n_moirai),
                    |x: &coeus_autograd::Var<f32, MoiraiBackend>,
                     y: &coeus_autograd::Var<f32, MoiraiBackend>| {
                        coeus_nn::pairwise_distance(x, y, 2.0, 1e-6)
                    },
                    1.0,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_cosine_embedding_loss(c: &mut Criterion) {
    // Cosine embedding on [128, 64] pairs vs ±1 targets, margin 0.5.
    const N: usize = 128;
    const D: usize = 64;

    let data1: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let data2: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let targets: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (x1_seq, x1_moirai) = metric_vars(&data1, N, D);
    let (x2_seq, x2_moirai) = metric_vars(&data2, N, D);

    let mut group = c.benchmark_group("Coeus — CosineEmbeddingLoss (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::cosine_embedding_loss(
                    black_box(&x1_seq),
                    black_box(&x2_seq),
                    black_box(&targets),
                    0.5,
                )
                .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::cosine_embedding_loss(
                    black_box(&x1_moirai),
                    black_box(&x2_moirai),
                    black_box(&targets),
                    0.5,
                )
                .expect("invariant: benchmark shapes and targets are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_cosine_similarity_loss(c: &mut Criterion) {
    // Cosine similarity on [128, 64] pairs along dim 1, eps 1e-8.
    const N: usize = 128;
    const D: usize = 64;

    let data1: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let data2: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let (x1_seq, x1_moirai) = metric_vars(&data1, N, D);
    let (x2_seq, x2_moirai) = metric_vars(&data2, N, D);

    let mut group = c.benchmark_group("Coeus — CosineSimilarity (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::cosine_similarity(black_box(&x1_seq), black_box(&x2_seq), 1, 1e-8)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::cosine_similarity(black_box(&x1_moirai), black_box(&x2_moirai), 1, 1e-8)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}
