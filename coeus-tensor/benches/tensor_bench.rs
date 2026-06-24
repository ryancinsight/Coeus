use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leto::Array;

// ── Burn NdArray (dev/bench only) ─────────────────────────────────────────
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor as BurnTensor, TensorData};
type BurnB = NdArray<f32>;

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

    // Leto and direct layouts
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

// ── Burn NdArray comparison benchmarks ────────────────────────────────────

fn bench_burn_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = vec![1.0f32; size * size];

    let a_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let b_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — Elementwise Add (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(a_burn.clone() + b_burn.clone()))
    });

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::ones(vec![size, size]);
    let b_seq = Tensor::<f32, SequentialBackend>::ones(vec![size, size]);
    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![size, size]);
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![size, size]);

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_matmul(c: &mut Criterion) {
    let (m, k, n) = (256, 256, 256);
    let device = NdArrayDevice::default();
    let a_data: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.001).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.001).collect();

    let a_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(a_data.clone(), [m, k]), &device);
    let b_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(b_data.clone(), [k, n]), &device);

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![m, k], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![k, n], &b_data);
    let a_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![m, k], &a_data);
    let b_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![k, n], &b_data);

    let mut group = c.benchmark_group("Burn vs Coeus — Matmul (256x256)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(a_burn.clone().matmul(b_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_relu(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = (0..size * size).map(|x| x as f32 * 0.01 - 5.0).collect();

    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![size, size], &data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![size, size], &data);

    let mut group = c.benchmark_group("Burn vs Coeus — ReLU (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(x_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::relu(black_box(&x_seq), black_box(&seq_backend))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::relu(
                black_box(&x_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_sum(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = vec![1.0f32; size * size];

    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![size, size], &data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![size, size], &data);

    let mut group = c.benchmark_group("Burn vs Coeus — Sum axis=1 (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(x_burn.clone().sum_dim(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::sum_axis(
                black_box(&x_seq),
                1,
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::sum_axis(
                black_box(&x_moirai),
                1,
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise_add,
    bench_matmul,
    bench_burn_elementwise_add,
    bench_burn_matmul,
    bench_burn_relu,
    bench_burn_sum,
);
criterion_main!(benches);
