//! Parameter initialization benchmarks.

use super::*;

pub(crate) fn bench_uniform_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — uniform initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::uniform_with_seed(
                black_box(&mut sequential),
                black_box(-1.0),
                black_box(1.0),
                black_box(SEED),
            )
            .expect("valid sequential uniform initializer benchmark fixture");
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
            )
            .expect("valid parallel uniform initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}

pub(crate) fn bench_normal_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — normal initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::normal_with_seed(
                black_box(&mut sequential),
                black_box(0.0),
                black_box(1.0),
                black_box(SEED),
            )
            .expect("valid sequential normal initializer benchmark fixture");
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::normal_with_seed(
                black_box(&mut moirai),
                black_box(0.0),
                black_box(1.0),
                black_box(SEED),
            )
            .expect("valid parallel normal initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}

pub(crate) fn bench_xavier_uniform_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — xavier-uniform initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::xavier_uniform_with_seed(
                black_box(&mut sequential),
                black_box(COLUMNS),
                black_box(ROWS),
                black_box(SEED),
            )
            .expect("valid sequential xavier-uniform initializer benchmark fixture");
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::xavier_uniform_with_seed(
                black_box(&mut moirai),
                black_box(COLUMNS),
                black_box(ROWS),
                black_box(SEED),
            )
            .expect("valid parallel xavier-uniform initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}

pub(crate) fn bench_xavier_normal_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — xavier-normal initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::xavier_normal_with_seed(
                black_box(&mut sequential),
                black_box(COLUMNS),
                black_box(ROWS),
                black_box(SEED),
            )
            .expect("valid sequential xavier-normal initializer benchmark fixture");
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::xavier_normal_with_seed(
                black_box(&mut moirai),
                black_box(COLUMNS),
                black_box(ROWS),
                black_box(SEED),
            )
            .expect("valid parallel xavier-normal initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}

pub(crate) fn bench_kaiming_uniform_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — kaiming-uniform initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::kaiming_uniform_with_seed(
                black_box(&mut sequential),
                black_box(COLUMNS),
                black_box(SEED),
            )
            .expect("valid sequential kaiming-uniform initializer benchmark fixture");
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::kaiming_uniform_with_seed(
                black_box(&mut moirai),
                black_box(COLUMNS),
                black_box(SEED),
            )
            .expect("valid parallel kaiming-uniform initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}

pub(crate) fn bench_kaiming_normal_initializer(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLUMNS: usize = 1024;
    const SEED: u64 = 42;

    let mut sequential = Var::new(
        Tensor::<f32, SequentialBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::zeros([ROWS, COLUMNS])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — kaiming-normal initializer (1024x1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            coeus_nn::init::kaiming_normal_with_seed(
                black_box(&mut sequential),
                black_box(COLUMNS),
                black_box(SEED),
            )
            .expect("valid sequential kaiming-normal initializer benchmark fixture");
            black_box(sequential.tensor.as_slice().first().copied())
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            coeus_nn::init::kaiming_normal_with_seed(
                black_box(&mut moirai),
                black_box(COLUMNS),
                black_box(SEED),
            )
            .expect("valid parallel kaiming-normal initializer benchmark fixture");
            black_box(moirai.tensor.as_slice().first().copied())
        })
    });
    group.finish();
}
