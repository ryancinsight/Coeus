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

/// One Coeus-vs-Burn Conv2d forward comparison for a square `[n, ch, hw, hw]`
/// input through `Conv2d(ch -> ch, k, stride 1, no pad, no bias)`. Bias is
/// disabled on both sides for a like-for-like forward; Coeus `Conv2d::new`
/// defaults (stride 1, padding 0, dilation 1) match Burn's `Valid` padding.
fn bench_conv2d_shape(c: &mut Criterion, label: &str, n: usize, ch: usize, hw: usize, k: usize) {
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; n * ch * hw * hw];

    let burn_conv = Conv2dConfig::new([ch, ch], [k, k])
        .with_bias(false)
        .with_padding(PaddingConfig2d::Valid)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [n, ch, hw, hw]),
        &device,
    );

    let conv_seq = Conv2d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv2d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, hw, hw]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, hw, hw]),
        false,
    );

    let mut group = c.benchmark_group(label);
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

fn bench_conv2d_forward(c: &mut Criterion) {
    // Small shape: per-output-row input band c_in*kh*w = 16*3*32 ≈ 6 KB is
    // L1-resident, so the AXPY kernel is bounded by call/dispatch overhead.
    bench_conv2d_shape(
        c,
        "Burn vs Coeus — Conv2d forward (8x16x32x32, k3)",
        8,
        16,
        32,
        3,
    );
    // Large shape: band c_in*kh*w = 128*3*32 ≈ 49 KB spills L1, so cross-channel
    // input reuse (each input window feeds every output channel) is the lever —
    // this is the regime that would justify a channel-batched (axpy_rows) tile.
    bench_conv2d_shape(
        c,
        "Burn vs Coeus — Conv2d forward (2x128x32x32, k3)",
        2,
        128,
        32,
        3,
    );
}

criterion_group!(
    benches,
    bench_linear_forward,
    bench_layernorm_forward,
    bench_conv2d_forward
);
criterion_main!(benches);
