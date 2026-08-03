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

pub(crate) fn bench_rotary_embedding_forward(c: &mut Criterion) {
    const BATCH: usize = 8;
    const SEQUENCE: usize = 64;
    const HEADS: usize = 8;
    const HEAD_DIMENSION: usize = 32;

    let encoding_sequential =
        RotaryEmbedding::<f32, SequentialBackend>::new(SEQUENCE, HEAD_DIMENSION, 10_000.0);
    let encoding_moirai =
        RotaryEmbedding::<f32, MoiraiBackend>::new(SEQUENCE, HEAD_DIMENSION, 10_000.0);
    let input_sequential = Var::new(
        Tensor::<f32, SequentialBackend>::ones([BATCH, SEQUENCE, HEADS, HEAD_DIMENSION]),
        false,
    );
    let input_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones([BATCH, SEQUENCE, HEADS, HEAD_DIMENSION]),
        false,
    );

    let mut group = c.benchmark_group("Coeus — rotary embedding forward (8x64x8x32)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                encoding_sequential
                    .forward(black_box(&input_sequential))
                    .expect("valid rotary embedding benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                encoding_moirai
                    .forward(black_box(&input_moirai))
                    .expect("valid rotary embedding benchmark input"),
            )
        })
    });
    group.finish();
}
