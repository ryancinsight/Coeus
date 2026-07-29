//! loss forward benchmarks.

use super::*;

pub(crate) fn bench_cross_entropy_loss(c: &mut Criterion) {
    // Cross-entropy loss on logits [BATCH=128, num_classes=10].
    const CE_N: usize = 128;
    const CE_C: usize = 10;

    let logit_data: Vec<f32> = (0..(CE_N * CE_C))
        .map(|i| (i as f32 * 0.0041).sin())
        .collect();
    let targets: Vec<usize> = (0..CE_N).map(|i| i % CE_C).collect();

    // Coeus: cross_entropy_loss(logits, &targets).
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![CE_N, CE_C], &logit_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![CE_N, CE_C], &logit_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — CrossEntropyLoss (128x10)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(cross_entropy_loss(black_box(&x_seq), black_box(&targets))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(cross_entropy_loss(
                black_box(&x_moirai),
                black_box(&targets),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_mse_loss(c: &mut Criterion) {
    const MSE_N: usize = 128;
    const MSE_D: usize = 64;

    let pred_data: Vec<f32> = (0..(MSE_N * MSE_D))
        .map(|i| (i as f32 * 0.0037).sin())
        .collect();
    let target_data: Vec<f32> = (0..(MSE_N * MSE_D))
        .map(|i| (i as f32 * 0.0041).cos())
        .collect();

    let p_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MSE_N, MSE_D], &pred_data),
        false,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MSE_N, MSE_D], &target_data),
        false,
    );
    let p_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MSE_N, MSE_D], &pred_data),
        false,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MSE_N, MSE_D], &target_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — MSELoss (128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(mse_loss(black_box(&p_seq), black_box(&t_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(mse_loss(black_box(&p_moirai), black_box(&t_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_huber_loss(c: &mut Criterion) {
    // HuberLoss on [BATCH=128, D=64] predictions vs same-shape targets (delta=1.0).
    const H_N: usize = 128;
    const H_D: usize = 64;

    let pred_data: Vec<f32> = (0..(H_N * H_D))
        .map(|i| (i as f32 * 0.0043).sin())
        .collect();
    let tgt_data: Vec<f32> = (0..(H_N * H_D))
        .map(|i| (i as f32 * 0.0051).cos())
        .collect();

    let p_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![H_N, H_D], &pred_data),
        false,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![H_N, H_D], &tgt_data),
        false,
    );
    let p_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![H_N, H_D], &pred_data),
        false,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![H_N, H_D], &tgt_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — HuberLoss (128x64, delta=1.0)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                huber_loss(black_box(&p_seq), black_box(&t_seq), 1.0)
                    .expect("invariant: benchmark shapes match and delta is positive"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                huber_loss(black_box(&p_moirai), black_box(&t_moirai), 1.0)
                    .expect("invariant: benchmark shapes match and delta is positive"),
            )
        })
    });
    group.finish();
}
