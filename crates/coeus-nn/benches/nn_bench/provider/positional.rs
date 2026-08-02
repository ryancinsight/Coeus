//! positional-encoding benchmarks.

use super::*;

pub(crate) fn bench_sinusoidal_encoding_forward(c: &mut Criterion) {
    const BATCH: usize = 8;
    const SEQUENCE: usize = 64;
    const MODEL: usize = 256;

    let encoding_sequential = SinusoidalEncoding::<f32, SequentialBackend>::new(SEQUENCE, MODEL);
    let encoding_moirai = SinusoidalEncoding::<f32, MoiraiBackend>::new(SEQUENCE, MODEL);
    let input_sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([BATCH, SEQUENCE, MODEL]),
        false,
    );
    let input_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([BATCH, SEQUENCE, MODEL]),
        false,
    );

    let mut group = c.benchmark_group("Coeus — sinusoidal encoding forward (8x64x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                encoding_sequential
                    .forward(black_box(&input_sequential))
                    .expect("valid sinusoidal encoding benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                encoding_moirai
                    .forward(black_box(&input_moirai))
                    .expect("valid sinusoidal encoding benchmark input"),
            )
        })
    });
    group.finish();
}
