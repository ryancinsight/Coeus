//! normalization forward benchmarks.

use super::*;

pub(crate) fn bench_layernorm_forward(c: &mut Criterion) {
    let ln_seq = LayerNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5)
        .expect("invariant: test operation succeeds");
    let ln_moirai = LayerNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — LayerNorm forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                ln_seq
                    .forward(black_box(&x_seq))
                    .expect("valid layer normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                ln_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid layer normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_rmsnorm_forward(c: &mut Criterion) {
    // RMSNorm forward on [BATCH=128, FEATURES=256] — same shape as LayerNorm row for direct
    // normalization-family comparison.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let rn_seq = RMSNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5)
        .expect("invariant: test operation succeeds");
    let rn_moirai = RMSNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — RMSNorm forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                rn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid RMS normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                rn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid RMS normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_batchnorm2d_eval_forward(c: &mut Criterion) {
    // BatchNorm2d eval-mode forward on [N=2, C=64, H=32, W=32].
    const BN_N: usize = 2;
    const BN_C: usize = 64;
    const BN_H: usize = 32;
    const BN_W: usize = 32;

    let input_data: Vec<f32> = (0..(BN_N * BN_C * BN_H * BN_W))
        .map(|i| (i % 31) as f32 * 0.01 - 0.15)
        .collect();

    let mut bn_seq = BatchNorm2d::<f32, SequentialBackend>::new(BN_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    let mut bn_moirai = BatchNorm2d::<f32, MoiraiBackend>::new(BN_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — BatchNorm2d eval forward (2x64x32x32)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                bn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                bn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_batchnorm1d_eval_forward(c: &mut Criterion) {
    // BatchNorm1d eval-mode forward on [N=16, C=128, L=256].
    const BN1_N: usize = 16;
    const BN1_C: usize = 128;
    const BN1_L: usize = 256;

    let input_data: Vec<f32> = (0..(BN1_N * BN1_C * BN1_L))
        .map(|i| (i % 29) as f32 * 0.01 - 0.14)
        .collect();

    let mut bn_seq = BatchNorm1d::<f32, SequentialBackend>::new(BN1_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    let mut bn_moirai = BatchNorm1d::<f32, MoiraiBackend>::new(BN1_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — BatchNorm1d eval forward (16x128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                bn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                bn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_batchnorm3d_eval_forward(c: &mut Criterion) {
    // BatchNorm3d eval-mode forward on [N=2, C=32, D=16, H=16, W=16].
    const BN3_N: usize = 2;
    const BN3_C: usize = 32;
    const BN3_D: usize = 16;
    const BN3_H: usize = 16;
    const BN3_W: usize = 16;

    let input_data: Vec<f32> = (0..(BN3_N * BN3_C * BN3_D * BN3_H * BN3_W))
        .map(|i| (i % 23) as f32 * 0.01 - 0.11)
        .collect();

    let mut bn_seq = BatchNorm3d::<f32, SequentialBackend>::new(BN3_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    let mut bn_moirai = BatchNorm3d::<f32, MoiraiBackend>::new(BN3_C, 1e-5, 0.1)
        .expect("invariant: test operation succeeds");
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — BatchNorm3d eval forward (2x32x16x16x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                bn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                bn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid batch normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_groupnorm_forward(c: &mut Criterion) {
    // GroupNorm forward on [N=8, C=32, H=16, W=16] with 8 groups.
    const GN_N: usize = 8;
    const GN_C: usize = 32;
    const GN_H: usize = 16;
    const GN_W: usize = 16;
    const GN_G: usize = 8;

    let input_data: Vec<f32> = (0..(GN_N * GN_C * GN_H * GN_W))
        .map(|i| (i % 37) as f32 * 0.02 - 0.36)
        .collect();

    let gn_seq = GroupNorm::<f32, SequentialBackend, GN_G>::new(GN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let gn_moirai = GroupNorm::<f32, MoiraiBackend, GN_G>::new(GN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — GroupNorm forward (8x32x16x16, g8)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                gn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid group normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                gn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid group normalization benchmark input"),
            )
        })
    });
    group.finish();
}

#[path = "normalization_forward/extended.rs"]
mod extended;
pub(crate) use extended::*;
