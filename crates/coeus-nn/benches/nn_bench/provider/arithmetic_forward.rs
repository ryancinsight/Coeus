//! arithmetic forward benchmarks.

use super::*;

pub(crate) fn bench_bmm_forward(c: &mut Criterion) {
    // bmm: [32, 64, 128] × [32, 128, 64] — head attention pattern.
    const BMM_B: usize = 32;
    const BMM_M: usize = 64;
    const BMM_K: usize = 128;
    const BMM_N: usize = 64;
    let a_data: Vec<f32> = (0..(BMM_B * BMM_M * BMM_K))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BMM_B * BMM_K * BMM_N))
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BMM_B, BMM_M, BMM_K], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BMM_B, BMM_K, BMM_N], &b_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BMM_B, BMM_M, BMM_K], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BMM_B, BMM_K, BMM_N], &b_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — bmm forward (32x64x128 @ 32x128x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::matmul(black_box(&a_seq), black_box(&b_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_pow_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - pow(3) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::pow(black_box(&x_seq), 3.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::pow(black_box(&x_moirai), 3.0)))
    });
    group.finish();
}

pub(crate) fn bench_mul_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos())
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - mul forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::mul(&a_seq, &b_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::mul(&a_moirai, &b_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_div_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 4.0 + 0.1)
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos().abs() + 0.5)
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - div forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::div(&a_seq, &b_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::div(&a_moirai, &b_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_add_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos())
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - add forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::add(&a_seq, &b_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::add(&a_moirai, &b_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_sub_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos())
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - sub forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sub(&a_seq, &b_seq)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sub(&a_moirai, &b_moirai)))
    });
    group.finish();
}

pub(crate) fn bench_pow2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - pow2(2.0) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::pow(&x_seq, 2.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::pow(&x_moirai, 2.0)))
    });
    group.finish();
}

/// Binary elementwise operation across Coeus providers on identical inputs.
pub(crate) fn bench_binary(
    c: &mut Criterion,
    name: &str,
    coeus_seq: impl Fn(
        &Var<f32, SequentialBackend>,
        &Var<f32, SequentialBackend>,
    ) -> Var<f32, SequentialBackend>,
    coeus_moirai: impl Fn(&Var<f32, MoiraiBackend>, &Var<f32, MoiraiBackend>) -> Var<f32, MoiraiBackend>,
) {
    let a: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    // `+ 1.5` keeps the second operand strictly positive so `remainder`'s
    // divisor never hits zero.
    let b: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0027).cos() + 1.5)
        .collect();
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b),
        false,
    );

    let mut group = c.benchmark_group(format!("Coeus — {name} forward (128x256)"));
    group.bench_function("Coeus Sequential", |bn| {
        bn.iter(|| black_box(coeus_seq(black_box(&a_seq), black_box(&b_seq))))
    });
    group.bench_function("Coeus Moirai", |bn| {
        bn.iter(|| black_box(coeus_moirai(black_box(&a_moirai), black_box(&b_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_maximum_forward(c: &mut Criterion) {
    bench_binary(
        c,
        "maximum",
        coeus_autograd::maximum,
        coeus_autograd::maximum,
    );
}

pub(crate) fn bench_minimum_forward(c: &mut Criterion) {
    bench_binary(
        c,
        "minimum",
        coeus_autograd::minimum,
        coeus_autograd::minimum,
    );
}

pub(crate) fn bench_remainder_forward(c: &mut Criterion) {
    bench_binary(
        c,
        "remainder",
        coeus_autograd::remainder,
        coeus_autograd::remainder,
    );
}
