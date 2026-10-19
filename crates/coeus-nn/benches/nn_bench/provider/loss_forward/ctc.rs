use super::*;

pub(crate) fn bench_ctc_loss(c: &mut Criterion) {
    // CTC loss on log-probs [T=50, N=4, C=10], target length 10, blank 0.
    const T: usize = 50;
    const N: usize = 4;
    const C: usize = 10;
    const TARGET_LEN: usize = 10;

    let log_data: Vec<f32> = (0..(T * N * C))
        .map(|i| -2.3 + 0.1 * (i as f32 * 0.0041).sin())
        .collect();
    let targets: Vec<usize> = (0..(N * TARGET_LEN)).map(|i| 1 + i % (C - 1)).collect();
    let input_lengths: Vec<usize> = vec![T; N];
    let target_lengths: Vec<usize> = vec![TARGET_LEN; N];

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![T, N, C], &log_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![T, N, C], &log_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — CTCLoss (T50xN4xC10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::ctc_loss(
                    black_box(&x_seq),
                    black_box(&targets),
                    black_box(&input_lengths),
                    black_box(&target_lengths),
                    0,
                )
                .expect("invariant: benchmark fixture satisfies CTC contracts"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::ctc_loss(
                    black_box(&x_moirai),
                    black_box(&targets),
                    black_box(&input_lengths),
                    black_box(&target_lengths),
                    0,
                )
                .expect("invariant: benchmark fixture satisfies CTC contracts"),
            )
        })
    });
    group.finish();
}
