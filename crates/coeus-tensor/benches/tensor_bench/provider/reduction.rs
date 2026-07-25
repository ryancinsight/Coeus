//! Reduction benchmarks.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, Criterion};

pub(crate) fn bench_sum(c: &mut Criterion) {
    const SIDE: usize = 1_024;
    let data = vec![1.0f32; SIDE * SIDE];
    let sequential_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let sequential_input = Tensor::<f32, SequentialBackend>::from_slice([SIDE, SIDE], &data);
    let moirai_input = Tensor::<f32, MoiraiBackend>::from_slice([SIDE, SIDE], &data);

    let mut group = c.benchmark_group("Sum axis=1 (1024x1024)");
    group.bench_function("Coeus Sequential", |bencher| {
        bencher.iter(|| {
            black_box(
                coeus_ops::sum_axis(
                    black_box(&sequential_input),
                    1,
                    black_box(&sequential_backend),
                )
                .expect("valid benchmark reduction"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |bencher| {
        bencher.iter(|| {
            black_box(
                coeus_ops::sum_axis(black_box(&moirai_input), 1, black_box(&moirai_backend))
                    .expect("valid benchmark reduction"),
            )
        })
    });
    group.finish();
}
