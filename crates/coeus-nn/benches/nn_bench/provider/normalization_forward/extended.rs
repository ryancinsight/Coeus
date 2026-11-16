use super::*;

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
    let in_seq = InstanceNorm2d::<f32, SequentialBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let in_moirai = InstanceNorm2d::<f32, MoiraiBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — InstanceNorm2d forward (2x32x16x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                in_seq
                    .forward(black_box(&x_seq))
                    .expect("valid instance normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                in_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid instance normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_instancenorm1d_forward(c: &mut Criterion) {
    // InstanceNorm1d forward on [N=2, C=32, L=64].
    const IN_N: usize = 2;
    const IN_C: usize = 32;
    const IN_L: usize = 64;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_L))
        .map(|i| (i as f32 * 0.0027).cos())
        .collect();

    // Coeus: InstanceNorm1d::new(num_features, eps).
    let in_seq = InstanceNorm1d::<f32, SequentialBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let in_moirai = InstanceNorm1d::<f32, MoiraiBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — InstanceNorm1d forward (2x32x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                in_seq
                    .forward(black_box(&x_seq))
                    .expect("valid instance normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                in_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid instance normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_instancenorm3d_forward(c: &mut Criterion) {
    // InstanceNorm3d forward on [N=2, C=8, D=8, H=8, W=8].
    const IN_N: usize = 2;
    const IN_C: usize = 8;
    const IN_D: usize = 8;
    const IN_H: usize = 8;
    const IN_W: usize = 8;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_D * IN_H * IN_W))
        .map(|i| (i as f32 * 0.0027).cos())
        .collect();

    // Coeus: InstanceNorm3d::new(num_features, eps).
    let in_seq = InstanceNorm3d::<f32, SequentialBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let in_moirai = InstanceNorm3d::<f32, MoiraiBackend>::new(IN_C, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![IN_N, IN_C, IN_D, IN_H, IN_W],
            &input_data,
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_D, IN_H, IN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — InstanceNorm3d forward (2x8x8x8x8)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                in_seq
                    .forward(black_box(&x_seq))
                    .expect("valid instance normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                in_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid instance normalization benchmark input"),
            )
        })
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
        Tensor::<f32, SequentialBackend>::from_slice(vec![LRN_N, LRN_C, LRN_H, LRN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let input_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![LRN_N, LRN_C, LRN_H, LRN_W], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — LocalResponseNorm forward (8x32x16x16, size=5)");
    group.bench_function("Coeus Sequential", |bench| {
        bench.iter(|| {
            black_box(
                lrn.forward(black_box(&input_sequential))
                    .expect("valid local response normalization benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |bench| {
        bench.iter(|| {
            black_box(
                lrn.forward(black_box(&input_moirai))
                    .expect("valid local response normalization benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_vector_norm_forward(c: &mut Criterion) {
    // vector_norm L2: [128, 256] global L2 norm.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).cos())
        .collect();
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

    let mut group = c.benchmark_group("Coeus — vector_norm L2 (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_autograd::norm(black_box(&x_seq))
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_autograd::norm(black_box(&x_moirai))
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_group_norm_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * 8 * 16))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let layer_seq = GroupNorm::<f32, SequentialBackend, 2>::new(8, 1e-5)
        .expect("invariant: test operation succeeds");
    let layer_moirai = GroupNorm::<f32, MoiraiBackend, 2>::new(8, 1e-5)
        .expect("invariant: test operation succeeds");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, 8, 16], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, 8, 16], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let mut group = c.benchmark_group("Coeus - group_norm(G=2) fwd+bwd (128x8x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = layer_seq
                .forward(black_box(&x_seq))
                .expect("valid group normalization benchmark input");
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = layer_moirai
                .forward(black_box(&x_moirai))
                .expect("valid group normalization benchmark input");
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}
