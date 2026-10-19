//! regularization benchmarks: train-mode forward (seeded, deterministic).

use super::*;

const REG_BATCH: usize = 128;
const REG_FEATURES: usize = 256;

fn regularization_input() -> Vec<f32> {
    (0..(REG_BATCH * REG_FEATURES))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect()
}

pub(crate) fn bench_alpha_dropout_forward(c: &mut Criterion) {
    // AlphaDropout train-mode forward (seeded masking, p=0.5): [128, 256].
    let input_data = regularization_input();

    // Coeus: AlphaDropout::new is seeded (42) and trains by default.
    let mut alpha_seq = AlphaDropout::new(0.5);
    alpha_seq.set_training(true);
    let mut alpha_moirai = AlphaDropout::new(0.5);
    alpha_moirai.set_training(true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — AlphaDropout train forward (128x256, p=0.5)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                alpha_seq
                    .forward(black_box(&x_seq))
                    .expect("valid alpha dropout benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                alpha_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid alpha dropout benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_feature_alpha_dropout_forward(c: &mut Criterion) {
    // FeatureAlphaDropout train-mode forward (seeded masking, p=0.5): [128, 256].
    let input_data = regularization_input();

    let mut alpha_seq = FeatureAlphaDropout::new(0.5);
    alpha_seq.set_training(true);
    let mut alpha_moirai = FeatureAlphaDropout::new(0.5);
    alpha_moirai.set_training(true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — FeatureAlphaDropout train forward (128x256, p=0.5)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                alpha_seq
                    .forward(black_box(&x_seq))
                    .expect("valid feature alpha dropout benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                alpha_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid feature alpha dropout benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_gaussian_noise_forward(c: &mut Criterion) {
    // GaussianNoise train-mode forward (seeded noise, std=0.1): [128, 256].
    let input_data = regularization_input();

    let mut noise_seq = GaussianNoise::new(0.1);
    noise_seq.set_training(true);
    let mut noise_moirai = GaussianNoise::new(0.1);
    noise_moirai.set_training(true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![REG_BATCH, REG_FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — GaussianNoise train forward (128x256, std=0.1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                noise_seq
                    .forward(black_box(&x_seq))
                    .expect("valid gaussian noise benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                noise_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid gaussian noise benchmark input"),
            )
        })
    });
    group.finish();
}
