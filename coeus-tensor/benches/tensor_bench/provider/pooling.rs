//! Pooling benchmarks.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::PoolOps;
use coeus_tensor::Tensor;
use criterion::{black_box, Criterion};

pub(crate) fn bench_max_pool2d(c: &mut Criterion) {
    const BATCH: usize = 1;
    const CHANNELS: usize = 8;
    const SIDE: usize = 32;
    const KERNEL: usize = 2;
    const STRIDE: usize = 2;
    const OUTPUT_SIDE: usize = (SIDE - KERNEL) / STRIDE + 1;

    let input: Vec<f32> = (0..BATCH * CHANNELS * SIDE * SIDE)
        .map(|index| (index as f32 * 0.05).sin())
        .collect();
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, CHANNELS, SIDE, SIDE], &input);
    let moirai_input =
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, CHANNELS, SIDE, SIDE], &input);

    let mut group = c.benchmark_group("MaxPool2d (1x8x32x32, kernel=2, stride=2)");
    group.bench_function("Coeus Sequential", |bencher| {
        let mut output =
            Tensor::<f32, SequentialBackend>::zeros([BATCH, CHANNELS, OUTPUT_SIDE, OUTPUT_SIDE]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            sequential_backend.max_pool2d(
                black_box(sequential_input.storage()),
                black_box(sequential_input.layout()),
                KERNEL,
                STRIDE,
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
            Tensor::<f32, MoiraiBackend>::zeros([BATCH, CHANNELS, OUTPUT_SIDE, OUTPUT_SIDE]);
        let output_layout = output.layout().clone();
        bencher.iter(|| {
            moirai_backend.max_pool2d(
                black_box(moirai_input.storage()),
                black_box(moirai_input.layout()),
                KERNEL,
                STRIDE,
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
