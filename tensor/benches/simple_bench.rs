use coeus_tensor::{Tensor, CpuBackend};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simple_tensor_creation(c: &mut Criterion) {
    c.bench_function("tensor_creation_1000", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            black_box(Tensor::from_vec(CpuBackend::default(), data, vec![1000]).unwrap());
        });
    });
}

fn bench_matrix_ops(c: &mut Criterion) {
    c.bench_function("matrix_addition_100", |b| {
        let a_data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..10000).map(|i| (i + 1) as f32).collect();
        let a = Tensor::from_vec(CpuBackend::default(), a_data, vec![100, 100]).unwrap();
        let b_tensor = Tensor::from_vec(CpuBackend::default(), b_data, vec![100, 100]).unwrap();

        b.iter(|| {
            let _ = black_box(&a + &b_tensor);
        });
    });
}

criterion_group!(benches, simple_tensor_creation, bench_matrix_ops);
criterion_main!(benches);
