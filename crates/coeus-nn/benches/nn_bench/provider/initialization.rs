//! Parameter initialization benchmarks.

use super::*;

pub(crate) fn bench_uniform_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS]),
        false,
    );
    let mut moirai = Var::new(Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS]), false);

    let mut group = c.benchmark_group("Coeus — uniform initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::uniform_with_seed(
                black_box(&mut sequential),
                black_box(-1.0),
                black_box(1.0),
                black_box(SEED),
            );
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::uniform_with_seed(
                black_box(&mut moirai),
                black_box(-1.0),
                black_box(1.0),
                black_box(SEED),
            );
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}
