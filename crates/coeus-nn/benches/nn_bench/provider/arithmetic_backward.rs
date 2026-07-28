//! arithmetic backward benchmarks.

use super::*;

pub(crate) fn bench_matmul_backward(c: &mut Criterion) {
    // matmul [128,256] @ [256,128] fwd+bwd
    const M: usize = BATCH;
    const K: usize = FEATURES;
    const N: usize = BATCH;
    let a_data: Vec<f32> = (0..(M * K)).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..(K * N)).map(|i| (i as f32 * 0.001).cos()).collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![M, K], &a_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![K, N], &b_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![M, K], &a_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![K, N], &b_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - matmul fwd+bwd (128x256@256x128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::matmul(black_box(&a_seq), black_box(&b_seq));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::matmul(black_box(&a_moirai), black_box(&b_moirai));
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_pow_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - pow(2) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::pow(black_box(&x_seq), 2.0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::pow(black_box(&x_moirai), 2.0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_scalar_mul_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - scalar_mul(3.0) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::scalar_mul(black_box(&x_seq), 3.0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::scalar_mul(black_box(&x_moirai), 3.0);
            black_box(o).expect("run operation").backward().expect("run backward")
        })
    });
    group.finish();
}
