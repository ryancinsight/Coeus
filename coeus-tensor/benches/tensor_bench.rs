use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leto::Array;
use nalgebra::DMatrix;
use ndarray::Array2;
use rayon::prelude::*;

fn bench_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let shape = vec![size, size];

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());
    let b_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());

    let a_ndarray = Array2::<f32>::ones((size, size));
    let b_ndarray = Array2::<f32>::ones((size, size));

    let a_nalgebra = DMatrix::<f32>::from_element(size, size, 1.0);
    let b_nalgebra = DMatrix::<f32>::from_element(size, size, 1.0);

    let a_rayon = vec![1.0f32; size * size];
    let b_rayon = vec![1.0f32; size * size];

    let mut group = c.benchmark_group("Elementwise Add (1024x1024)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ));
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            black_box(black_box(&a_ndarray) + black_box(&b_ndarray));
        })
    });

    group.bench_function("nalgebra", |b| {
        b.iter(|| {
            black_box(black_box(&a_nalgebra) + black_box(&b_nalgebra));
        })
    });

    group.bench_function("rayon slice", |b| {
        b.iter(|| {
            black_box(
                black_box(&a_rayon)
                    .par_iter()
                    .zip(black_box(&b_rayon))
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f32>>(),
            );
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

    let a_ndarray = Array2::<f32>::ones((m, k));
    let b_ndarray = Array2::<f32>::ones((k, n));

    let a_nalgebra = DMatrix::<f32>::from_element(m, k, 1.0);
    let b_nalgebra = DMatrix::<f32>::from_element(k, n, 1.0);

    let a_leto = Array::from_shape_vec([m, k], vec![1.0f32; m * k]).unwrap();
    let b_leto = Array::from_shape_vec([k, n], vec![1.0f32; k * n]).unwrap();

    let coeus_layout_a = coeus_core::Layout::new(vec![m, k].into());
    let coeus_layout_b = coeus_core::Layout::new(vec![k, n].into());
    let coeus_layout_out = coeus_core::Layout::new(vec![m, n].into());
    let coeus_a = vec![1.0f32; m * k];
    let coeus_b = vec![1.0f32; k * n];

    let mut group = c.benchmark_group("Matrix Multiplication (256x256)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ));
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
        })
    });

    group.bench_function("nalgebra", |b| {
        b.iter(|| {
            black_box(black_box(&a_nalgebra) * black_box(&b_nalgebra));
        })
    });

    group.bench_function("Leto direct", |b| {
        b.iter_batched(
            || Array::zeros([m, n]),
            |mut out| {
                leto_ops::matmul(
                    black_box(&a_leto.view()),
                    black_box(&b_leto.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("Coeus-Leto dispatch", |b| {
        b.iter_batched(
            || vec![0.0f32; m * n],
            |mut out| {
                coeus_leto::matmul_into(
                    black_box(&coeus_layout_a),
                    black_box(&coeus_a),
                    black_box(&coeus_layout_b),
                    black_box(&coeus_b),
                    black_box(&coeus_layout_out),
                    black_box(&mut out),
                )
                .unwrap();
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_elementwise_add, bench_matmul);
criterion_main!(benches);
