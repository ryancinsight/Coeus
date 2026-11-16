//! Interpolation forward benchmarks.

use super::*;

use coeus_core::FloatElement;

/// Native-provider `interpolate_2d` comparison on `[N,C,H,W]`
/// upsampling to `2H x 2W`, parameterized by mode so every mode shares one
/// body. The tap count and the tap arithmetic are the only per-mode differences
/// (1 for nearest, 2 for bilinear, 4 for bicubic, and one box for area), so the
/// rows are directly comparable at identical shapes and backends.
pub(crate) fn bench_interpolate2d_forward(
    c: &mut Criterion,
    label: &str,
    coeus_mode: CoeusResizeMode,
) {
    const IN_N: usize = 8;
    const IN_C: usize = 16;
    const IN_H: usize = 32;
    const IN_W: usize = 32;
    const OUT_H: usize = 64;
    const OUT_W: usize = 64;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_H * IN_W))
        .map(|i| (f32::from_count(i) * 0.0021).sin())
        .collect();

    let x_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data)
            .expect("invariant: test backend operation succeeds");
    let x_moirai =
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data)
            .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group(label);
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                interpolate_2d(black_box(&x_seq), OUT_H, OUT_W, coeus_mode)
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                interpolate_2d(black_box(&x_moirai), OUT_H, OUT_W, coeus_mode)
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_interpolate2d_nearest_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d nearest forward (8x16x32x32 -> 64x64)",
        CoeusResizeMode::Nearest,
    );
}

pub(crate) fn bench_interpolate2d_bilinear_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d bilinear forward (8x16x32x32 -> 64x64)",
        CoeusResizeMode::Bilinear,
    );
}

pub(crate) fn bench_interpolate2d_bicubic_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d bicubic forward (8x16x32x32 -> 64x64)",
        CoeusResizeMode::Bicubic,
    );
}

pub(crate) fn bench_interpolate2d_area_forward(c: &mut Criterion) {
    bench_interpolate2d_forward(
        c,
        "Coeus — interpolate_2d area forward (8x16x32x32 -> 64x64)",
        CoeusResizeMode::Area,
    );
}

pub(crate) fn bench_interpolate1d_nearest_forward(c: &mut Criterion) {
    const IN_N: usize = 8;
    const IN_C: usize = 16;
    const IN_L: usize = 128;
    const OUT_L: usize = 256;

    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_L))
        .map(|i| (f32::from_count(i) * 0.0021).sin())
        .collect();

    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_L], &input_data)
        .expect("invariant: test backend operation succeeds");
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_L], &input_data)
        .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — interpolate_1d nearest forward (8x16x128 -> 256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                interpolate_1d(black_box(&x_seq), OUT_L, CoeusResizeMode::Nearest)
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                interpolate_1d(black_box(&x_moirai), OUT_L, CoeusResizeMode::Nearest)
                    .expect("invariant: test operation succeeds"),
            )
        })
    });
    group.finish();
}
