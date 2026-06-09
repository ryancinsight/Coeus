use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let shape = vec![size, size];

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());
    let b_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());

    let mut group = c.benchmark_group("Elementwise Add (1024x1024)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let _res = coeus_ops::add(&a_seq, &b_seq, &seq_backend);
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let _res = coeus_ops::add(&a_moirai, &b_moirai, &moirai_backend);
        })
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    // 256x256 matmul to keep bench times reasonable
    let m = 256;
    let k = 256;
    let n = 256;

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(vec![m, k]);
    let b_seq = Tensor::<f32, SequentialBackend>::ones(vec![k, n]);

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![m, k]);
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![k, n]);

    let mut group = c.benchmark_group("Matrix Multiplication (256x256)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let _res = coeus_ops::matmul(&a_seq, &b_seq, &seq_backend);
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let _res = coeus_ops::matmul(&a_moirai, &b_moirai, &moirai_backend);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_elementwise_add, bench_matmul);
criterion_main!(benches);
