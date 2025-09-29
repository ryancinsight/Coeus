use criterion::{criterion_group, criterion_main, Criterion};
use coeus_backend::dispatch::{select_backend, BackendKind};
use coeus_backend::Backend;

fn bench_matmul(c: &mut Criterion) {
    let backend = select_backend(BackendKind::Gpu).expect("backend selection");
    let size = 4096;
    let a_vec = vec![1.0f32; size * size];
    let b_vec = vec![1.0f32; size * size];
    let a = backend.create_tensor_data(a_vec, vec![size, size]).expect("create tensor data");
    let b = backend.create_tensor_data(b_vec, vec![size, size]).expect("create tensor data");

    c.bench_function("vulkan_matmul_4096", |bencher| {
        bencher.iter(|| {
            let _result = backend.matmul(&a, &b);
        })
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
