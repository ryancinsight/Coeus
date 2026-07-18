//! normalization forward benchmarks.

use super::*;

pub(crate) fn bench_layernorm_forward(c: &mut Criterion) {
    let ln_seq = LayerNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5);
    let ln_moirai = LayerNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );

    let mut group = c.benchmark_group("Coeus — LayerNorm forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(ln_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(ln_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_rmsnorm_forward(c: &mut Criterion) {
    // RMSNorm forward on [BATCH=128, FEATURES=256] — same shape as LayerNorm row for direct
    // normalization-family comparison.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let rn_seq = RMSNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5);
    let rn_moirai = RMSNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — RMSNorm forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(rn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(rn_moirai.forward(black_box(&x_moirai))))
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

    let mut bn_seq = BatchNorm2d::<f32, SequentialBackend>::new(BN_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm2d::<f32, MoiraiBackend>::new(BN_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — BatchNorm2d eval forward (2x64x32x32)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
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

    let mut bn_seq = BatchNorm1d::<f32, SequentialBackend>::new(BN1_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm1d::<f32, MoiraiBackend>::new(BN1_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — BatchNorm1d eval forward (16x128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
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

    let mut bn_seq = BatchNorm3d::<f32, SequentialBackend>::new(BN3_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm3d::<f32, MoiraiBackend>::new(BN3_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        ),
        false,
    );

    let mut group = c.benchmark_group("Coeus — BatchNorm3d eval forward (2x32x16x16x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
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

    let gn_seq = GroupNorm::<f32, SequentialBackend, GN_G>::new(GN_C, 1e-5);
    let gn_moirai = GroupNorm::<f32, MoiraiBackend, GN_G>::new(GN_C, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — GroupNorm forward (8x32x16x16, g8)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(gn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(gn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_instancenorm2d_forward(c: &mut Criterion) {
    // InstanceNorm2d forward on [N=2, C=32, H=16, W=16].
    const IN_N: usize = 2;
    const IN_C: usize = 32;
    const IN_H: usize = 16;
    const IN_W: usize = 16;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_H * IN_W))
        .map(|i| (i as f32 * 0.0027).cos())
        .collect();

    // Coeus: InstanceNorm2d::new(num_features, eps).
    let in_seq = InstanceNorm2d::<f32, SequentialBackend>::new(IN_C, 1e-5);
    let in_moirai = InstanceNorm2d::<f32, MoiraiBackend>::new(IN_C, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — InstanceNorm2d forward (2x32x16x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(in_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(in_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_local_response_norm_forward(c: &mut Criterion) {
    const LRN_N: usize = 8;
    const LRN_C: usize = 32;
    const LRN_H: usize = 16;
    const LRN_W: usize = 16;
    let input_data: Vec<f32> = (0..(LRN_N * LRN_C * LRN_H * LRN_W))
        .map(|index| (index as f32 * 0.0019).sin())
        .collect();
    let lrn = LocalResponseNorm::new(5);
    let input_sequential = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![LRN_N, LRN_C, LRN_H, LRN_W], &input_data),
        false,
    );
    let input_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![LRN_N, LRN_C, LRN_H, LRN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — LocalResponseNorm forward (8x32x16x16, size=5)");
    group.bench_function("Coeus Sequential", |bench| {
        bench.iter(|| black_box(lrn.forward(black_box(&input_sequential))))
    });
    group.bench_function("Coeus Moirai", |bench| {
        bench.iter(|| black_box(lrn.forward(black_box(&input_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_vector_norm_forward(c: &mut Criterion) {
    // vector_norm L2: [128, 256] global L2 norm.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — vector_norm L2 (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::norm(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::norm(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_group_norm_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * 8 * 16))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let layer_seq = GroupNorm::<f32, SequentialBackend, 2>::new(8, 1e-5);
    let layer_moirai = GroupNorm::<f32, MoiraiBackend, 2>::new(8, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, 8, 16], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, 8, 16], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - group_norm(G=2) fwd+bwd (128x8x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = layer_seq.forward(black_box(&x_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = layer_moirai.forward(black_box(&x_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}
