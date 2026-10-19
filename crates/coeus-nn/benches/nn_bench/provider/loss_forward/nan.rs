use super::*;

fn nan_data(n: usize, d: usize) -> Vec<f32> {
    (0..(n * d))
        .map(|i| {
            if i % 7 == 0 {
                f32::NAN
            } else {
                (i as f32 * 0.0041).sin()
            }
        })
        .collect()
}

pub(crate) fn bench_nansum_loss(c: &mut Criterion) {
    // NaN-aware sum over [128, 256] with every 7th element NaN.
    const N: usize = 128;
    const D: usize = 256;

    let data = nan_data(N, D);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N, D], &data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N, D], &data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — NanSum (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nansum(black_box(&x_seq)).expect("invariant: benchmark shape is valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nansum(black_box(&x_moirai))
                    .expect("invariant: benchmark shape is valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_nanmean_loss(c: &mut Criterion) {
    // NaN-aware mean over [128, 256] with every 7th element NaN.
    const N: usize = 128;
    const D: usize = 256;

    let data = nan_data(N, D);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N, D], &data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N, D], &data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — NanMean (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nanmean(black_box(&x_seq)).expect("invariant: benchmark shape is valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::nanmean(black_box(&x_moirai))
                    .expect("invariant: benchmark shape is valid"),
            )
        })
    });
    group.finish();
}
