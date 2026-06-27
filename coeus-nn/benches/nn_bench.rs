//! Layer-level forward-pass benchmarks: Coeus vs Burn NdArray.
//!
//! Complements `coeus-tensor/benches/tensor_bench.rs` (tensor primitives) by
//! timing whole `nn` layer forward passes against Burn's reference NdArray
//! backend on identical shapes. Each `Burn vs Coeus` group runs the same logical
//! computation three ways — Burn NdArray, Coeus `SequentialBackend`, and Coeus
//! `MoiraiBackend` — so the relative cost of the Coeus autograd-graph-building
//! forward is directly comparable to Burn's eager forward.
//!
//! `burn` is a dev/bench-only dependency (production dependency policy is
//! enforced by the `dependency_policy` test). These benchmarks measure the real
//! production layer code; the harness body is never tuned to move the number.
//!
//! Run one group:
//!   `cargo bench -p coeus-nn --bench nn_bench -- Linear`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_nn::{Conv2d, LayerNorm, Linear, Module};
use coeus_tensor::Tensor;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::nn::conv::Conv2dConfig;
use burn::nn::{LayerNormConfig, LinearConfig, PaddingConfig2d};
use burn::tensor::{Tensor as BurnTensor, TensorData};
type BurnB = NdArray<f32>;

// Shared workload: batch of `BATCH` vectors of width `FEATURES`.
const BATCH: usize = 128;
const FEATURES: usize = 256;

fn bench_linear_forward(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; BATCH * FEATURES];

    // Burn: LinearConfig(FEATURES -> FEATURES), input [BATCH, FEATURES].
    let burn_linear = LinearConfig::new(FEATURES, FEATURES).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    // Coeus: same dims; forward builds the autograd graph (production path).
    let lin_seq = Linear::<f32, SequentialBackend>::new(FEATURES, FEATURES, true);
    let lin_moirai = Linear::<f32, MoiraiBackend>::new(FEATURES, FEATURES, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Linear forward (128x256 -> 256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_linear.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(lin_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(lin_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_layernorm_forward(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; BATCH * FEATURES];

    let burn_ln = LayerNormConfig::new(FEATURES).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let ln_seq = LayerNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5);
    let ln_moirai = LayerNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — LayerNorm forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_ln.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(ln_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(ln_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_conv2d_forward(c: &mut Criterion) {
    // Workload: [N=8, C_in=16, 32x32] -> Conv2d(16 -> 16, k=3, stride=1, no pad),
    // bias disabled on both sides for a like-for-like forward comparison.
    const N: usize = 8;
    const C: usize = 16;
    const HW: usize = 32;
    const K: usize = 3;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; N * C * HW * HW];

    // Burn: stride 1, Valid padding (== no padding), matching Coeus defaults.
    let burn_conv = Conv2dConfig::new([C, C], [K, K])
        .with_bias(false)
        .with_padding(PaddingConfig2d::Valid)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 4> =
        BurnTensor::from_data(TensorData::new(input_data.clone(), [N, C, HW, HW]), &device);

    // Coeus: Conv2d::new defaults to stride=1, padding=0, dilation=1.
    let conv_seq = Conv2d::<f32, SequentialBackend>::new(C, C, K, false);
    let conv_moirai = Conv2d::<f32, MoiraiBackend>::new(C, C, K, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![N, C, HW, HW]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![N, C, HW, HW]),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Conv2d forward (8x16x32x32, k3)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_conv.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(conv_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(conv_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_forward,
    bench_layernorm_forward,
    bench_conv2d_forward
);
criterion_main!(benches);
