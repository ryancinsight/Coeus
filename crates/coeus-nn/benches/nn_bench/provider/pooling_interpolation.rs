//! pooling interpolation benchmarks.

use super::*;

pub(crate) fn bench_maxpool2d_forward(c: &mut Criterion) {
    // MaxPool2d forward on [N=8, C=16, H=32, W=32] with k=2, s=2.
    const MP_N: usize = 8;
    const MP_C: usize = 16;
    const MP_H: usize = 32;
    const MP_W: usize = 32;
    const MP_K: usize = 2;
    const MP_S: usize = 2;

    let input_data: Vec<f32> = (0..(MP_N * MP_C * MP_H * MP_W))
        .map(|i| (i as f32 * 0.0025).sin())
        .collect();

    let pool_seq = MaxPool2d::<f32, SequentialBackend>::with_params(MP_K, MP_S, 0, 1);
    let pool_moirai = MaxPool2d::<f32, MoiraiBackend>::with_params(MP_K, MP_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MP_N, MP_C, MP_H, MP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MP_N, MP_C, MP_H, MP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — MaxPool2d forward (8x16x32x32, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

pub(crate) fn bench_avgpool2d_forward(c: &mut Criterion) {
    // AvgPool2d forward on [N=8, C=16, H=32, W=32] with k=2, s=2.
    const AP_N: usize = 8;
    const AP_C: usize = 16;
    const AP_H: usize = 32;
    const AP_W: usize = 32;
    const AP_K: usize = 2;
    const AP_S: usize = 2;

    let input_data: Vec<f32> = (0..(AP_N * AP_C * AP_H * AP_W))
        .map(|i| (i as f32 * 0.0018).cos())
        .collect();

    let pool_seq = AvgPool2d::<f32, SequentialBackend>::with_params(AP_K, AP_S, 0, 1);
    let pool_moirai = AvgPool2d::<f32, MoiraiBackend>::with_params(AP_K, AP_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AP_N, AP_C, AP_H, AP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AP_N, AP_C, AP_H, AP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — AvgPool2d forward (8x16x32x32, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

/// MaxPool3d forward comparison across native providers.
pub(crate) fn bench_maxpool3d_forward(c: &mut Criterion) {
    // MaxPool3d forward on [N=4, C=8, D=16, H=16, W=16] with k=2, s=2.
    const MP3_N: usize = 4;
    const MP3_C: usize = 8;
    const MP3_D: usize = 16;
    const MP3_H: usize = 16;
    const MP3_W: usize = 16;
    const MP3_K: usize = 2;
    const MP3_S: usize = 2;

    let input_data: Vec<f32> = (0..(MP3_N * MP3_C * MP3_D * MP3_H * MP3_W))
        .map(|i| (i as f32 * 0.0013).sin())
        .collect();

    let pool_seq = MaxPool3d::<f32, SequentialBackend>::with_params(MP3_K, MP3_S, 0, 1);
    let pool_moirai = MaxPool3d::<f32, MoiraiBackend>::with_params(MP3_K, MP3_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W],
            &input_data,
        ).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W],
            &input_data,
        ).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — MaxPool3d forward (4x8x16x16x16, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

/// AvgPool3d forward comparison across native providers.
pub(crate) fn bench_avgpool3d_forward(c: &mut Criterion) {
    // AvgPool3d forward on [N=4, C=8, D=16, H=16, W=16] with k=2, s=2.
    const AP3_N: usize = 4;
    const AP3_C: usize = 8;
    const AP3_D: usize = 16;
    const AP3_H: usize = 16;
    const AP3_W: usize = 16;
    const AP3_K: usize = 2;
    const AP3_S: usize = 2;

    let input_data: Vec<f32> = (0..(AP3_N * AP3_C * AP3_D * AP3_H * AP3_W))
        .map(|i| (i as f32 * 0.0011).cos())
        .collect();

    let pool_seq = AvgPool3d::<f32, SequentialBackend>::with_params(AP3_K, AP3_S, 0, 1);
    let pool_moirai = AvgPool3d::<f32, MoiraiBackend>::with_params(AP3_K, AP3_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![AP3_N, AP3_C, AP3_D, AP3_H, AP3_W],
            &input_data,
        ).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![AP3_N, AP3_C, AP3_D, AP3_H, AP3_W],
            &input_data,
        ).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — AvgPool3d forward (4x8x16x16x16, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

/// Native-provider `interpolate_2d` comparison on `[N,C,H,W]`
/// upsampling to `2H x 2W`, parameterized by mode so nearest/bilinear share
/// one body.
pub(crate) fn bench_interpolate2d_forward(
    c: &mut Criterion,
    label: &str,
    coeus_mode: CoeusInterpolateMode,
) {
    const IN_N: usize = 8;
    const IN_C: usize = 16;
    const IN_H: usize = 32;
    const IN_W: usize = 32;
    const OUT_H: usize = 64;
    const OUT_W: usize = 64;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_H * IN_W))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    let x_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data).expect("construct tensor");
    let x_moirai =
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data).expect("construct tensor");

    let mut group = c.benchmark_group(label);
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(interpolate_2d(black_box(&x_seq), OUT_H, OUT_W, coeus_mode).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(interpolate_2d(
                black_box(&x_moirai),
                OUT_H,
                OUT_W,
                coeus_mode,
            ).expect("run operation"))
        })
    });
    group.finish();
}

pub(crate) fn bench_interpolate2d_nearest_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d nearest forward (8x16x32x32 -> 64x64)",
        CoeusInterpolateMode::Nearest,
    );
}

pub(crate) fn bench_interpolate2d_bilinear_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d bilinear forward (8x16x32x32 -> 64x64)",
        CoeusInterpolateMode::Bilinear,
    );
}

pub(crate) fn bench_maxpool1d_forward(c: &mut Criterion) {
    const MP1_N: usize = 8;
    const MP1_C: usize = 16;
    const MP1_L: usize = 128;
    const MP1_K: usize = 2;
    const MP1_S: usize = 2;
    let input_data: Vec<f32> = (0..(MP1_N * MP1_C * MP1_L))
        .map(|i| (i as f32 * 0.0019).sin())
        .collect();
    let pool_seq = MaxPool1d::<f32, SequentialBackend>::with_params(MP1_K, MP1_S, 0, 1);
    let pool_moirai = MaxPool1d::<f32, MoiraiBackend>::with_params(MP1_K, MP1_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MP1_N, MP1_C, MP1_L], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MP1_N, MP1_C, MP1_L], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — MaxPool1d forward (8x16x128, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

pub(crate) fn bench_avgpool1d_forward(c: &mut Criterion) {
    const AP1_N: usize = 8;
    const AP1_C: usize = 16;
    const AP1_L: usize = 128;
    const AP1_K: usize = 2;
    const AP1_S: usize = 2;
    let input_data: Vec<f32> = (0..(AP1_N * AP1_C * AP1_L))
        .map(|i| (i as f32 * 0.0023).cos())
        .collect();
    let pool_seq = CoeusAvgPool1d::<f32, SequentialBackend>::with_params(AP1_K, AP1_S, 0, 1);
    let pool_moirai = CoeusAvgPool1d::<f32, MoiraiBackend>::with_params(AP1_K, AP1_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AP1_N, AP1_C, AP1_L], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AP1_N, AP1_C, AP1_L], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — AvgPool1d forward (8x16x128, k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

pub(crate) fn bench_adaptive_avg_pool2d_forward(c: &mut Criterion) {
    // AdaptiveAvgPool2d(1,1): ResNet-style global pooling step.
    // Input [8, 64, 7, 7] → output [8, 64, 1, 1].
    const AAP_N: usize = 8;
    const AAP_C: usize = 64;
    const AAP_H: usize = 7;
    const AAP_W: usize = 7;

    let input_data: Vec<f32> = (0..(AAP_N * AAP_C * AAP_H * AAP_W))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    // Coeus AdaptiveAvgPool2d.
    let pool_seq = AdaptiveAvgPool2d::<f32, SequentialBackend>::square(1);
    let pool_moirai = AdaptiveAvgPool2d::<f32, MoiraiBackend>::square(1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AAP_N, AAP_C, AAP_H, AAP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AAP_N, AAP_C, AAP_H, AAP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — AdaptiveAvgPool2d(1,1) forward (8x64x7x7)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}

pub(crate) fn bench_adaptive_max_pool2d_forward(c: &mut Criterion) {
    // AdaptiveMaxPool2d(1,1): global max pooling step.
    // Input [8, 64, 7, 7] → output [8, 64, 1, 1].
    const AMP_N: usize = 8;
    const AMP_C: usize = 64;
    const AMP_H: usize = 7;
    const AMP_W: usize = 7;

    let input_data: Vec<f32> = (0..(AMP_N * AMP_C * AMP_H * AMP_W))
        .map(|i| (i as f32 * 0.0017).cos())
        .collect();

    // Coeus AdaptiveMaxPool2d.
    let pool_seq = coeus_nn::AdaptiveMaxPool2d::<f32, SequentialBackend>::square(1);
    let pool_moirai = coeus_nn::AdaptiveMaxPool2d::<f32, MoiraiBackend>::square(1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AMP_N, AMP_C, AMP_H, AMP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AMP_N, AMP_C, AMP_H, AMP_W], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — AdaptiveMaxPool2d(1,1) forward (8x64x7x7)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq)).expect("run forward")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai)).expect("run forward")))
    });
    group.finish();
}
