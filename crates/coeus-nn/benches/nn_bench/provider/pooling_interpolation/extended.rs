use super::*;

pub(crate) fn bench_adaptive_avg_pool1d_forward(c: &mut Criterion) {
    // AdaptiveAvgPool1d(8): input [8, 64, 49] → output [8, 64, 8].
    const AAP_N: usize = 8;
    const AAP_C: usize = 64;
    const AAP_L: usize = 49;

    let input_data: Vec<f32> = (0..(AAP_N * AAP_C * AAP_L))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::AdaptiveAvgPool1d::<f32, SequentialBackend>::new(8);
    let pool_moirai = coeus_nn::AdaptiveAvgPool1d::<f32, MoiraiBackend>::new(8);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AAP_N, AAP_C, AAP_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AAP_N, AAP_C, AAP_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — AdaptiveAvgPool1d(8) forward (8x64x49)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_adaptive_max_pool1d_forward(c: &mut Criterion) {
    // AdaptiveMaxPool1d(8): input [8, 64, 49] → output [8, 64, 8].
    const AMP_N: usize = 8;
    const AMP_C: usize = 64;
    const AMP_L: usize = 49;

    let input_data: Vec<f32> = (0..(AMP_N * AMP_C * AMP_L))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::AdaptiveMaxPool1d::<f32, SequentialBackend>::new(8);
    let pool_moirai = coeus_nn::AdaptiveMaxPool1d::<f32, MoiraiBackend>::new(8);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AMP_N, AMP_C, AMP_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AMP_N, AMP_C, AMP_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — AdaptiveMaxPool1d(8) forward (8x64x49)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_global_avg_pool1d_forward(c: &mut Criterion) {
    // GlobalAvgPool1d: input [8, 64, 49] → output [8, 64, 1].
    const G_N: usize = 8;
    const G_C: usize = 64;
    const G_L: usize = 49;

    let input_data: Vec<f32> = (0..(G_N * G_C * G_L))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::GlobalAvgPool1d::<f32, SequentialBackend>::new();
    let pool_moirai = coeus_nn::GlobalAvgPool1d::<f32, MoiraiBackend>::new();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![G_N, G_C, G_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![G_N, G_C, G_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — GlobalAvgPool1d forward (8x64x49)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_global_avg_pool2d_forward(c: &mut Criterion) {
    // GlobalAvgPool2d: input [8, 64, 7, 7] → output [8, 64, 1, 1].
    const G_N: usize = 8;
    const G_C: usize = 64;
    const G_HW: usize = 7;

    let input_data: Vec<f32> = (0..(G_N * G_C * G_HW * G_HW))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::GlobalAvgPool2d::<f32, SequentialBackend>::new();
    let pool_moirai = coeus_nn::GlobalAvgPool2d::<f32, MoiraiBackend>::new();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![G_N, G_C, G_HW, G_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![G_N, G_C, G_HW, G_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — GlobalAvgPool2d forward (8x64x7x7)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_global_avg_pool3d_forward(c: &mut Criterion) {
    // GlobalAvgPool3d: input [4, 32, 8, 8, 8] → output [4, 32, 1, 1, 1].
    const G_N: usize = 4;
    const G_C: usize = 32;
    const G_D: usize = 8;

    let input_data: Vec<f32> = (0..(G_N * G_C * G_D * G_D * G_D))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::GlobalAvgPool3d::<f32, SequentialBackend>::new();
    let pool_moirai = coeus_nn::GlobalAvgPool3d::<f32, MoiraiBackend>::new();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![G_N, G_C, G_D, G_D, G_D], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![G_N, G_C, G_D, G_D, G_D], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — GlobalAvgPool3d forward (4x32x8x8x8)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_global_max_pool2d_forward(c: &mut Criterion) {
    // GlobalMaxPool2d: input [8, 64, 7, 7] → output [8, 64, 1, 1].
    const G_N: usize = 8;
    const G_C: usize = 64;
    const G_HW: usize = 7;

    let input_data: Vec<f32> = (0..(G_N * G_C * G_HW * G_HW))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::GlobalMaxPool2d::<f32, SequentialBackend>::new();
    let pool_moirai = coeus_nn::GlobalMaxPool2d::<f32, MoiraiBackend>::new();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![G_N, G_C, G_HW, G_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![G_N, G_C, G_HW, G_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — GlobalMaxPool2d forward (8x64x7x7)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_global_max_pool3d_forward(c: &mut Criterion) {
    // GlobalMaxPool3d: input [4, 32, 8, 8, 8] → output [4, 32, 1, 1, 1].
    const G_N: usize = 4;
    const G_C: usize = 32;
    const G_D: usize = 8;

    let input_data: Vec<f32> = (0..(G_N * G_C * G_D * G_D * G_D))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let pool_seq = coeus_nn::GlobalMaxPool3d::<f32, SequentialBackend>::new();
    let pool_moirai = coeus_nn::GlobalMaxPool3d::<f32, MoiraiBackend>::new();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![G_N, G_C, G_D, G_D, G_D], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![G_N, G_C, G_D, G_D, G_D], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — GlobalMaxPool3d forward (4x32x8x8x8)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                pool_seq
                    .forward(black_box(&x_seq))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                pool_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid pooling benchmark input"),
            )
        })
    });
    group.finish();
}
