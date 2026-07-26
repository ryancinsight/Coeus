//! Convolution benchmarks.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::ConvOps;
use coeus_tensor::Tensor;
use criterion::{black_box, Criterion};

pub(crate) fn bench_conv1d(c: &mut Criterion) {
    const BATCH: usize = 2;
    const INPUT_CHANNELS: usize = 8;
    const LENGTH: usize = 128;
    const OUTPUT_CHANNELS: usize = 16;
    const KERNEL: usize = 3;
    const OUTPUT_LENGTH: usize = LENGTH - KERNEL + 1;

    let input: Vec<f32> = (0..BATCH * INPUT_CHANNELS * LENGTH)
        .map(|index| (index as f32 * 0.01).sin())
        .collect();
    let weights: Vec<f32> = (0..OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL)
        .map(|index| index as f32 * 0.001)
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, INPUT_CHANNELS, LENGTH], &input);
    let sequential_weights = Tensor::<f32, SequentialBackend>::from_slice(
        [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL],
        &weights,
    );
    let moirai_input =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, INPUT_CHANNELS, LENGTH], &input);
    let moirai_weights = Tensor::<f32, MoiraiBackend>::from_slice(
        [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL],
        &weights,
    );

    let mut group = c.benchmark_group("Conv1d (2x8x128, kernel=3)");
    group.bench_function("Coeus Sequential", |bencher| {
        let mut output =
            Tensor::<f32, SequentialBackend>::zeros([BATCH, OUTPUT_CHANNELS, OUTPUT_LENGTH]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            sequential_backend.conv1d(
                black_box(sequential_input.storage()),
                black_box(sequential_input.layout()),
                black_box(sequential_weights.storage()),
                black_box(sequential_weights.layout()),
                None,
                1,
                0,
                1,
                output.storage_mut(),
                &output_layout,
            );
            black_box(output.storage());
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        let mut output =
            Tensor::<f32, MoiraiBackend>::zeros([BATCH, OUTPUT_CHANNELS, OUTPUT_LENGTH]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            moirai_backend.conv1d(
                black_box(moirai_input.storage()),
                black_box(moirai_input.layout()),
                black_box(moirai_weights.storage()),
                black_box(moirai_weights.layout()),
                None,
                1,
                0,
                1,
                output.storage_mut(),
                &output_layout,
            );
            black_box(output.storage());
        })
    });
    group.finish();
}

pub(crate) fn bench_conv2d(c: &mut Criterion) {
    const BATCH: usize = 1;
    const INPUT_CHANNELS: usize = 4;
    const SIDE: usize = 16;
    const OUTPUT_CHANNELS: usize = 8;
    const KERNEL: usize = 3;
    const OUTPUT_SIDE: usize = SIDE - KERNEL + 1;

    let input: Vec<f32> = (0..BATCH * INPUT_CHANNELS * SIDE * SIDE)
        .map(|index| index as f32 * 0.01)
        .collect();
    let weights: Vec<f32> = (0..OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL * KERNEL)
        .map(|index| index as f32 * 0.001)
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, INPUT_CHANNELS, SIDE, SIDE], &input);
    let sequential_weights = Tensor::<f32, SequentialBackend>::from_slice(
        [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL, KERNEL],
        &weights,
    );
    let moirai_input =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, INPUT_CHANNELS, SIDE, SIDE], &input);
    let moirai_weights = Tensor::<f32, MoiraiBackend>::from_slice(
        [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL, KERNEL],
        &weights,
    );

    let mut group = c.benchmark_group("Conv2d (1x4x16x16, kernel=3)");
    group.bench_function("Coeus Sequential", |bencher| {
        let mut output = Tensor::<f32, SequentialBackend>::zeros([
            BATCH,
            OUTPUT_CHANNELS,
            OUTPUT_SIDE,
            OUTPUT_SIDE,
        ]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            sequential_backend.conv2d(
                black_box(sequential_input.storage()),
                black_box(sequential_input.layout()),
                black_box(sequential_weights.storage()),
                black_box(sequential_weights.layout()),
                None,
                1,
                0,
                1,
                output.storage_mut(),
                &output_layout,
            );
            black_box(output.storage());
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        let mut output =
            Tensor::<f32, MoiraiBackend>::zeros([BATCH, OUTPUT_CHANNELS, OUTPUT_SIDE, OUTPUT_SIDE]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            moirai_backend.conv2d(
                black_box(moirai_input.storage()),
                black_box(moirai_input.layout()),
                black_box(moirai_weights.storage()),
                black_box(moirai_weights.layout()),
                None,
                1,
                0,
                1,
                output.storage_mut(),
                &output_layout,
            );
            black_box(output.storage());
        })
    });
    group.finish();
}

pub(crate) fn bench_conv_transpose2d(c: &mut Criterion) {
    const BATCH: usize = 1;
    const INPUT_CHANNELS: usize = 4;
    const SIDE: usize = 16;
    const OUTPUT_CHANNELS: usize = 8;
    const KERNEL: usize = 3;
    const STRIDE: usize = 2;
    const PADDING: usize = 1;
    const OUTPUT_PADDING: usize = 1;
    const DILATION: usize = 1;

    let input: Vec<f32> = (0..BATCH * INPUT_CHANNELS * SIDE * SIDE)
        .map(|index| index as f32 * 0.01)
        .collect();
    let weights: Vec<f32> = (0..INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL * KERNEL)
        .map(|index| index as f32 * 0.001)
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, INPUT_CHANNELS, SIDE, SIDE], &input);
    let sequential_weights = Tensor::<f32, SequentialBackend>::from_slice(
        [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL, KERNEL],
        &weights,
    );
    let moirai_input =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, INPUT_CHANNELS, SIDE, SIDE], &input);
    let moirai_weights = Tensor::<f32, MoiraiBackend>::from_slice(
        [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL, KERNEL],
        &weights,
    );

    let mut group = c.benchmark_group("ConvTranspose2d (1x4x16x16, kernel=3, stride=2)");
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| {
            black_box(coeus_ops::conv_transpose2d(
                black_box(&sequential_input),
                black_box(&sequential_weights),
                None,
                STRIDE,
                PADDING,
                OUTPUT_PADDING,
                DILATION,
                black_box(&sequential_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| {
            black_box(coeus_ops::conv_transpose2d(
                black_box(&moirai_input),
                black_box(&moirai_weights),
                None,
                STRIDE,
                PADDING,
                OUTPUT_PADDING,
                DILATION,
                black_box(&moirai_backend),
            ))
        })
    });
    group.finish();
}
