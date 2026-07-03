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
use coeus_nn::{
    cross_entropy_loss, gelu, huber_loss, leaky_relu, mse_loss, prelu, relu, sigmoid, silu, tanh,
    AdaptiveAvgPool2d, AvgPool1d as CoeusAvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d,
    BatchNorm3d, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose3d, Dropout, Embedding,
    EmbeddingBag, EmbeddingBagMode, GroupNorm, Gru as CoeusGru, InstanceNorm2d, LayerNorm, Linear,
    Lstm, MaxPool1d, MaxPool2d, Module, MultiHeadAttention, NullMask, RMSNorm, SwiGlu,
    TransformerEncoderLayer,
};
use coeus_tensor::Tensor;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::nn::attention::{MhaInput, MultiHeadAttentionConfig};
use burn::nn::conv::Conv1dConfig;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv3dConfig;
use burn::nn::loss::{
    CrossEntropyLoss, CrossEntropyLossConfig, HuberLoss, HuberLossConfig, MseLoss, Reduction,
};
// Burn 0.16 re-exports `lstm::*` from `nn::rnn` but not `gru::*`, so `GruConfig`
// is only reachable by its submodule path (unlike the flattened LSTM types).
use burn::nn::gru::GruConfig;
use burn::nn::pool::MaxPool1dConfig;
use burn::nn::transformer::{TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{
    BatchNormConfig, DropoutConfig, GroupNormConfig, InstanceNormConfig, LayerNormConfig,
    LinearConfig, LstmConfig, PaddingConfig1d, PaddingConfig2d, PaddingConfig3d, RmsNormConfig,
    SwiGluConfig,
};
use burn::tensor::{Int, Tensor as BurnTensor, TensorData};
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

fn bench_rmsnorm_forward(c: &mut Criterion) {
    // RMSNorm forward on [BATCH=128, FEATURES=256] — same shape as LayerNorm row for direct
    // normalization-family comparison.
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let burn_rn = RmsNormConfig::new(FEATURES).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let rn_seq = RMSNorm::<f32, SequentialBackend>::new(FEATURES, 1e-5);
    let rn_moirai = RMSNorm::<f32, MoiraiBackend>::new(FEATURES, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — RMSNorm forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_rn.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(rn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(rn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_batchnorm2d_eval_forward(c: &mut Criterion) {
    // BatchNorm2d eval-mode forward on [N=2, C=64, H=32, W=32].
    // Burn NdArray BatchNorm runs in eval mode; Coeus is explicitly set to eval.
    const BN_N: usize = 2;
    const BN_C: usize = 64;
    const BN_H: usize = 32;
    const BN_W: usize = 32;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(BN_N * BN_C * BN_H * BN_W))
        .map(|i| (i % 31) as f32 * 0.01 - 0.15)
        .collect();

    let burn_bn: burn::nn::BatchNorm<BurnB, 2> =
        BatchNormConfig::new(BN_C).init::<BurnB, 2>(&device);
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BN_N, BN_C, BN_H, BN_W]),
        &device,
    );

    let mut bn_seq = BatchNorm2d::<f32, SequentialBackend>::new(BN_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm2d::<f32, MoiraiBackend>::new(BN_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN_N, BN_C, BN_H, BN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — BatchNorm2d eval forward (2x64x32x32)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_bn.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_batchnorm1d_eval_forward(c: &mut Criterion) {
    // BatchNorm1d eval-mode forward on [N=16, C=128, L=256].
    // Burn NdArray BatchNorm runs in eval mode; Coeus is explicitly set to eval.
    const BN1_N: usize = 16;
    const BN1_C: usize = 128;
    const BN1_L: usize = 256;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(BN1_N * BN1_C * BN1_L))
        .map(|i| (i % 29) as f32 * 0.01 - 0.14)
        .collect();

    let burn_bn: burn::nn::BatchNorm<BurnB, 1> =
        BatchNormConfig::new(BN1_C).init::<BurnB, 1>(&device);
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BN1_N, BN1_C, BN1_L]),
        &device,
    );

    let mut bn_seq = BatchNorm1d::<f32, SequentialBackend>::new(BN1_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm1d::<f32, MoiraiBackend>::new(BN1_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BN1_N, BN1_C, BN1_L], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — BatchNorm1d eval forward (16x128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_bn.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_batchnorm3d_eval_forward(c: &mut Criterion) {
    // BatchNorm3d eval-mode forward on [N=2, C=32, D=16, H=16, W=16].
    // Burn NdArray BatchNorm runs in eval mode; Coeus is explicitly set to eval.
    const BN3_N: usize = 2;
    const BN3_C: usize = 32;
    const BN3_D: usize = 16;
    const BN3_H: usize = 16;
    const BN3_W: usize = 16;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(BN3_N * BN3_C * BN3_D * BN3_H * BN3_W))
        .map(|i| (i % 23) as f32 * 0.01 - 0.11)
        .collect();

    let burn_bn: burn::nn::BatchNorm<BurnB, 3> =
        BatchNormConfig::new(BN3_C).init::<BurnB, 3>(&device);
    let x_burn: BurnTensor<BurnB, 5> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BN3_N, BN3_C, BN3_D, BN3_H, BN3_W]),
        &device,
    );

    let mut bn_seq = BatchNorm3d::<f32, SequentialBackend>::new(BN3_C, 1e-5, 0.1);
    let mut bn_moirai = BatchNorm3d::<f32, MoiraiBackend>::new(BN3_C, 1e-5, 0.1);
    bn_seq.set_training(false);
    bn_moirai.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![BN3_N, BN3_C, BN3_D, BN3_H, BN3_W],
            &input_data,
        ),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — BatchNorm3d eval forward (2x32x16x16x16)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_bn.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(bn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_groupnorm_forward(c: &mut Criterion) {
    // GroupNorm forward on [N=8, C=32, H=16, W=16] with 8 groups.
    const GN_N: usize = 8;
    const GN_C: usize = 32;
    const GN_H: usize = 16;
    const GN_W: usize = 16;
    const GN_G: usize = 8;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(GN_N * GN_C * GN_H * GN_W))
        .map(|i| (i % 37) as f32 * 0.02 - 0.36)
        .collect();

    let burn_gn = GroupNormConfig::new(GN_G, GN_C).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [GN_N, GN_C, GN_H, GN_W]),
        &device,
    );

    let gn_seq = GroupNorm::<f32, SequentialBackend, GN_G>::new(GN_C, 1e-5);
    let gn_moirai = GroupNorm::<f32, MoiraiBackend, GN_G>::new(GN_C, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![GN_N, GN_C, GN_H, GN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — GroupNorm forward (8x32x16x16, g8)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_gn.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(gn_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(gn_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_maxpool2d_forward(c: &mut Criterion) {
    // MaxPool2d forward on [N=8, C=16, H=32, W=32] with k=2, s=2.
    const MP_N: usize = 8;
    const MP_C: usize = 16;
    const MP_H: usize = 32;
    const MP_W: usize = 32;
    const MP_K: usize = 2;
    const MP_S: usize = 2;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(MP_N * MP_C * MP_H * MP_W))
        .map(|i| (i as f32 * 0.0025).sin())
        .collect();

    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [MP_N, MP_C, MP_H, MP_W]),
        &device,
    );

    let pool_seq = MaxPool2d::<f32, SequentialBackend>::with_params(MP_K, MP_S, 0, 1);
    let pool_moirai = MaxPool2d::<f32, MoiraiBackend>::with_params(MP_K, MP_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MP_N, MP_C, MP_H, MP_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MP_N, MP_C, MP_H, MP_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — MaxPool2d forward (8x16x32x32, k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::module::max_pool2d(
                black_box(x_burn.clone()),
                [MP_K, MP_K],
                [MP_S, MP_S],
                [0, 0],
                [1, 1],
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_avgpool2d_forward(c: &mut Criterion) {
    // AvgPool2d forward on [N=8, C=16, H=32, W=32] with k=2, s=2.
    const AP_N: usize = 8;
    const AP_C: usize = 16;
    const AP_H: usize = 32;
    const AP_W: usize = 32;
    const AP_K: usize = 2;
    const AP_S: usize = 2;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(AP_N * AP_C * AP_H * AP_W))
        .map(|i| (i as f32 * 0.0018).cos())
        .collect();

    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [AP_N, AP_C, AP_H, AP_W]),
        &device,
    );

    let pool_seq = AvgPool2d::<f32, SequentialBackend>::with_params(AP_K, AP_S, 0, 1);
    let pool_moirai = AvgPool2d::<f32, MoiraiBackend>::with_params(AP_K, AP_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AP_N, AP_C, AP_H, AP_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AP_N, AP_C, AP_H, AP_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — AvgPool2d forward (8x16x32x32, k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::module::avg_pool2d(
                black_box(x_burn.clone()),
                [AP_K, AP_K],
                [AP_S, AP_S],
                [0, 0],
                false,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

/// One Coeus-vs-Burn Conv1d forward comparison for `[n, ch, len]` input through
/// `Conv1d(ch -> ch, k, stride 1, no pad, no bias)`.
fn bench_conv1d_shape(c: &mut Criterion, label: &str, n: usize, ch: usize, len: usize, k: usize) {
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; n * ch * len];

    let burn_conv = Conv1dConfig::new(ch, ch, k)
        .with_bias(false)
        .with_padding(PaddingConfig1d::Valid)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(input_data.clone(), [n, ch, len]), &device);

    let conv_seq = Conv1d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv1d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, len]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, len]), false);

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

fn bench_conv1d_forward(c: &mut Criterion) {
    bench_conv1d_shape(
        c,
        "Burn vs Coeus — Conv1d forward (8x32x256, k3)",
        8,
        32,
        256,
        3,
    );
}

fn bench_conv1d_forward_backward(c: &mut Criterion) {
    // Conv1d forward + backward: [8, 32, 256], k3, no bias.
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    const FB1_N: usize = 8;
    const FB1_C: usize = 32;
    const FB1_L: usize = 256;
    const FB1_K: usize = 3;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(FB1_N * FB1_C * FB1_L))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();

    let burn_conv_fwd = Conv1dConfig::new(FB1_C, FB1_C, FB1_K)
        .with_bias(false)
        .with_padding(PaddingConfig1d::Valid)
        .init::<AB>(&device);
    let x_burn_ad: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [FB1_N, FB1_C, FB1_L]),
        &device,
    )
    .require_grad();

    let conv_seq = Conv1d::<f32, SequentialBackend>::new(FB1_C, FB1_C, FB1_K, false);
    let conv_moirai = Conv1d::<f32, MoiraiBackend>::new(FB1_C, FB1_C, FB1_K, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FB1_N, FB1_C, FB1_L], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FB1_N, FB1_C, FB1_L], &input_data),
        true,
    );

    let mut group =
        c.benchmark_group("Burn vs Coeus — Conv1d forward+backward (8x32x256, k3, no bias)");
    group.bench_function("Burn NdArray (autodiff)", |b| {
        b.iter(|| {
            let grads = burn_conv_fwd
                .forward(black_box(x_burn_ad.clone()))
                .sum()
                .backward();
            black_box(grads)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            conv_seq.zero_grad();
            x_seq.zero_grad();
            let out = conv_seq.forward(black_box(&x_seq));
            coeus_autograd::sum(&out).backward();
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            conv_moirai.zero_grad();
            x_moirai.zero_grad();
            let out = conv_moirai.forward(black_box(&x_moirai));
            coeus_autograd::sum(&out).backward();
        })
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

/// One Coeus-vs-Burn Conv3d forward comparison for cubic `[n, ch, d, h, w]`
/// input through `Conv3d(ch -> ch, k, stride 1, no pad, no bias)`.
fn bench_conv3d_shape(c: &mut Criterion, label: &str, n: usize, ch: usize, dim: usize, k: usize) {
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![1.0f32; n * ch * dim * dim * dim];

    let burn_conv = Conv3dConfig::new([ch, ch], [k, k, k])
        .with_bias(false)
        .with_padding(PaddingConfig3d::Valid)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 5> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [n, ch, dim, dim, dim]),
        &device,
    );

    let conv_seq = Conv3d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv3d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, dim, dim, dim]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, dim, dim, dim]),
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

fn bench_conv3d_forward(c: &mut Criterion) {
    bench_conv3d_shape(
        c,
        "Burn vs Coeus — Conv3d forward (2x8x16x16x16, k3)",
        2,
        8,
        16,
        3,
    );
}

fn bench_mha_forward(c: &mut Criterion) {
    // Self-attention forward on a realistic transformer block:
    // [batch=8, seq=64, d_model=256] with 8 heads (d_head=32).
    const B: usize = 8;
    const SEQ: usize = 64;
    const D: usize = 256;
    const H: usize = 8;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![0.02f32; B * SEQ * D];

    let burn_mha = MultiHeadAttentionConfig::new(D, H).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(input_data.clone(), [B, SEQ, D]), &device);

    let mha_seq = MultiHeadAttention::<f32, SequentialBackend, H, NullMask>::new(D, true);
    let mha_moirai = MultiHeadAttention::<f32, MoiraiBackend, H, NullMask>::new(D, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![B, SEQ, D]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![B, SEQ, D]), false);

    let mut group = c.benchmark_group("Burn vs Coeus — MHA self-attn forward (8x64x256, 8 heads)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_mha.forward(MhaInput::self_attn(black_box(x_burn.clone())))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(mha_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(mha_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_transformer_encoder_forward(c: &mut Criterion) {
    // One Pre-LN transformer encoder layer (self-attn + FFN + 2 LayerNorms +
    // residuals) on [batch=8, seq=64, d_model=256], d_ff=1024, 8 heads. Burn uses
    // a single-layer encoder with norm-first and dropout disabled to match.
    const B: usize = 8;
    const SEQ: usize = 64;
    const D: usize = 256;
    const D_FF: usize = 1024;
    const H: usize = 8;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = vec![0.02f32; B * SEQ * D];

    let burn_enc = TransformerEncoderConfig::new(D, D_FF, 1, H)
        .with_norm_first(true)
        .with_dropout(0.0)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(input_data.clone(), [B, SEQ, D]), &device);

    let enc_seq = TransformerEncoderLayer::<f32, SequentialBackend, H, NullMask>::new(D, D_FF, 0.0);
    let enc_moirai = TransformerEncoderLayer::<f32, MoiraiBackend, H, NullMask>::new(D, D_FF, 0.0);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![B, SEQ, D]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![B, SEQ, D]), false);

    let mut group = c
        .benchmark_group("Burn vs Coeus — Transformer encoder layer forward (8x64x256, d_ff=1024)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_enc.forward(TransformerEncoderInput::new(black_box(x_burn.clone()))))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(enc_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(enc_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_embedding_forward(c: &mut Criterion) {
    // Embedding lookup on [batch=2, seq=16] into [vocab=4096, d_model=256].
    // Burn uses integer index tensors; Coeus routes through the same embedding
    // forward path used by the module via `forward_indices`.
    const EMB_BATCH: usize = 2;
    const EMB_SEQ: usize = 16;
    const EMB_VOCAB: usize = 4096;
    const EMB_DIM: usize = 256;

    let device = NdArrayDevice::default();
    let indices: [[i32; EMB_SEQ]; EMB_BATCH] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ];
    let idx_data: Vec<f32> = indices
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v as f32)
        .collect();

    let burn_weight: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(vec![1.0f32; EMB_VOCAB * EMB_DIM], [EMB_VOCAB, EMB_DIM]),
        &device,
    );
    let burn_indices: BurnTensor<BurnB, 2, Int> = BurnTensor::from_ints(indices, &device);

    let emb_seq = Embedding::<f32, SequentialBackend>::new(EMB_VOCAB, EMB_DIM);
    let emb_moirai = Embedding::<f32, MoiraiBackend>::new(EMB_VOCAB, EMB_DIM);
    let idx_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![EMB_BATCH, EMB_SEQ], &idx_data);
    let idx_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![EMB_BATCH, EMB_SEQ], &idx_data);

    let mut group =
        c.benchmark_group("Burn vs Coeus — Embedding lookup forward (2x16, vocab=4096, d=256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::module::embedding(
                black_box(burn_weight.clone()),
                black_box(burn_indices.clone()),
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(emb_seq.forward_indices(black_box(&idx_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(emb_moirai.forward_indices(black_box(&idx_moirai))))
    });
    group.finish();
}

fn bench_linear_forward_backward(c: &mut Criterion) {
    // Full autograd cycle — forward + sum-loss + backward — for Linear
    // [128x256 -> 256], the part the forward-only groups don't measure. Burn
    // uses its Autodiff<NdArray> backend over the same manual linear expression.
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::default();

    let xb: BurnTensor<AB, 2> = BurnTensor::from_data(
        TensorData::new(vec![0.01f32; BATCH * FEATURES], [BATCH, FEATURES]),
        &device,
    )
    .require_grad();
    let wb: BurnTensor<AB, 2> = BurnTensor::from_data(
        TensorData::new(vec![0.01f32; FEATURES * FEATURES], [FEATURES, FEATURES]),
        &device,
    )
    .require_grad();
    let bb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(vec![0.0f32; FEATURES], [FEATURES]), &device)
            .require_grad();

    let lin_seq = Linear::<f32, SequentialBackend>::new(FEATURES, FEATURES, true);
    let lin_moirai = Linear::<f32, MoiraiBackend>::new(FEATURES, FEATURES, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        true,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Linear forward+backward (128x256 -> 256)");
    group.bench_function("Burn NdArray (autodiff)", |b| {
        b.iter(|| {
            let grads = (xb.clone().matmul(wb.clone().transpose()) + bb.clone().unsqueeze::<2>())
                .sum()
                .backward();
            black_box(wb.grad(&grads));
        })
    });
    // Coeus accumulates grads into the leaf Vars; zero them each iteration so the
    // measured backward work is identical across iterations (zero_grad is O(params)).
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            lin_seq.zero_grad();
            x_seq.zero_grad();
            coeus_autograd::sum(&lin_seq.forward(black_box(&x_seq))).backward();
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            lin_moirai.zero_grad();
            x_moirai.zero_grad();
            coeus_autograd::sum(&lin_moirai.forward(black_box(&x_moirai))).backward();
        })
    });
    group.finish();
}

fn bench_lstm_forward(c: &mut Criterion) {
    // LSTM sequence forward on [batch=4, seq=32, input=64] → hidden=128.
    // Modest size keeps the run tractable while exercising the full unroll path.
    const LSTM_BATCH: usize = 4;
    const LSTM_SEQ: usize = 32;
    const LSTM_IN: usize = 64;
    const LSTM_H: usize = 128;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(LSTM_BATCH * LSTM_SEQ * LSTM_IN))
        .map(|i| (i as f32 * 0.0017).cos())
        .collect();

    // Burn: LstmConfig(d_input, d_hidden, bias=true).
    let burn_lstm = LstmConfig::new(LSTM_IN, LSTM_H, true).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [LSTM_BATCH, LSTM_SEQ, LSTM_IN]),
        &device,
    );

    // Coeus: Lstm::new(input_size, hidden_size).
    let lstm_seq = Lstm::<f32, SequentialBackend>::new(LSTM_IN, LSTM_H);
    let lstm_moirai = Lstm::<f32, MoiraiBackend>::new(LSTM_IN, LSTM_H);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![LSTM_BATCH, LSTM_SEQ, LSTM_IN],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![LSTM_BATCH, LSTM_SEQ, LSTM_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — LSTM forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_lstm.forward(black_box(x_burn.clone()), None)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(lstm_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(lstm_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_gru_forward(c: &mut Criterion) {
    // GRU sequence forward on [batch=4, seq=32, input=64] → hidden=128.
    // Same shape as the LSTM row so the recurrent-family comparison is direct.
    // GRU has 3 gates vs LSTM's 4, but the same compute shape (one projection per
    // timestep); the unroll loop + cat/reshape output stacking costs dominate.
    const GRU_BATCH: usize = 4;
    const GRU_SEQ: usize = 32;
    const GRU_IN: usize = 64;
    const GRU_H: usize = 128;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(GRU_BATCH * GRU_SEQ * GRU_IN))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();

    // Burn: GruConfig(d_input, d_hidden, bias=true).
    let burn_gru = GruConfig::new(GRU_IN, GRU_H, true).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [GRU_BATCH, GRU_SEQ, GRU_IN]),
        &device,
    );

    // Coeus: Gru::new(input_size, hidden_size).
    let gru_seq = CoeusGru::<f32, SequentialBackend>::new(GRU_IN, GRU_H);
    let gru_moirai = CoeusGru::<f32, MoiraiBackend>::new(GRU_IN, GRU_H);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![GRU_BATCH, GRU_SEQ, GRU_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![GRU_BATCH, GRU_SEQ, GRU_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — GRU forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_gru.forward(black_box(x_burn.clone()), None)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(gru_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(gru_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_swiglu_forward(c: &mut Criterion) {
    // SwiGLU forward on [batch=32, d_input=256] → d_output=512 — the FFN-style
    // projection shape. Two parallel d_input→d_output linear projections plus a
    // SiLU gate and an element-wise product; the two matmuls dominate.
    const SG_BATCH: usize = 32;
    const SG_IN: usize = 256;
    const SG_OUT: usize = 512;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(SG_BATCH * SG_IN))
        .map(|i| (i as f32 * 0.0017).sin())
        .collect();

    // Burn: SwiGluConfig(d_input, d_output), bias defaults to false.
    let burn_swiglu = SwiGluConfig::new(SG_IN, SG_OUT).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [SG_BATCH, SG_IN]),
        &device,
    );

    // Coeus: SwiGlu::new(d_input, d_output, bias=false).
    let sg_seq = SwiGlu::<f32, SequentialBackend>::new(SG_IN, SG_OUT, false);
    let sg_moirai = SwiGlu::<f32, MoiraiBackend>::new(SG_IN, SG_OUT, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![SG_BATCH, SG_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![SG_BATCH, SG_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — SwiGLU forward (32 batch, in=256 out=512)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_swiglu.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(sg_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(sg_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_instancenorm2d_forward(c: &mut Criterion) {
    // InstanceNorm2d forward on [N=2, C=32, H=16, W=16].
    const IN_N: usize = 2;
    const IN_C: usize = 32;
    const IN_H: usize = 16;
    const IN_W: usize = 16;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(IN_N * IN_C * IN_H * IN_W))
        .map(|i| (i as f32 * 0.0027).cos())
        .collect();

    // Burn: InstanceNormConfig(num_channels).
    let burn_in = InstanceNormConfig::new(IN_C).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [IN_N, IN_C, IN_H, IN_W]),
        &device,
    );

    // Coeus: InstanceNorm2d::new(num_features, eps).
    let in_seq = InstanceNorm2d::<f32, SequentialBackend>::new(IN_C, 1e-5);
    let in_moirai = InstanceNorm2d::<f32, MoiraiBackend>::new(IN_C, 1e-5);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![IN_N, IN_C, IN_H, IN_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — InstanceNorm2d forward (2x32x16x16)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_in.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(in_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(in_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_cross_entropy_loss(c: &mut Criterion) {
    // Cross-entropy loss on logits [BATCH=128, num_classes=10].
    const CE_N: usize = 128;
    const CE_C: usize = 10;

    let device = NdArrayDevice::default();
    let logit_data: Vec<f32> = (0..(CE_N * CE_C))
        .map(|i| (i as f32 * 0.0041).sin())
        .collect();
    let targets: Vec<usize> = (0..CE_N).map(|i| i % CE_C).collect();
    let targets_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();

    // Burn: CrossEntropyLossConfig(no pad).forward(logits [N,C], targets [N, Int]).
    let burn_ce: CrossEntropyLoss<BurnB> = CrossEntropyLossConfig::new().init(&device);
    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(logit_data.clone(), [CE_N, CE_C]), &device);
    let t_burn: BurnTensor<BurnB, 1, Int> =
        BurnTensor::from_data(TensorData::new(targets_i64, [CE_N]), &device);

    // Coeus: cross_entropy_loss(logits, &targets).
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![CE_N, CE_C], &logit_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![CE_N, CE_C], &logit_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — CrossEntropyLoss (128x10)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_ce.forward(black_box(x_burn.clone()), black_box(t_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(cross_entropy_loss(black_box(&x_seq), black_box(&targets))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(cross_entropy_loss(
                black_box(&x_moirai),
                black_box(&targets),
            ))
        })
    });
    group.finish();
}

fn bench_mse_loss(c: &mut Criterion) {
    const MSE_N: usize = 128;
    const MSE_D: usize = 64;

    let device = NdArrayDevice::default();
    let pred_data: Vec<f32> = (0..(MSE_N * MSE_D))
        .map(|i| (i as f32 * 0.0037).sin())
        .collect();
    let target_data: Vec<f32> = (0..(MSE_N * MSE_D))
        .map(|i| (i as f32 * 0.0041).cos())
        .collect();

    let burn_mse = MseLoss::new();
    let p_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(pred_data.clone(), [MSE_N, MSE_D]), &device);
    let t_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(target_data.clone(), [MSE_N, MSE_D]),
        &device,
    );
    let p_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MSE_N, MSE_D], &pred_data),
        false,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MSE_N, MSE_D], &target_data),
        false,
    );
    let p_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MSE_N, MSE_D], &pred_data),
        false,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MSE_N, MSE_D], &target_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — MSELoss (128x64)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_mse.forward(
                black_box(p_burn.clone()),
                black_box(t_burn.clone()),
                Reduction::Mean,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(mse_loss(black_box(&p_seq), black_box(&t_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(mse_loss(black_box(&p_moirai), black_box(&t_moirai))))
    });
    group.finish();
}

fn bench_huber_loss(c: &mut Criterion) {
    // HuberLoss on [BATCH=128, D=64] predictions vs same-shape targets (delta=1.0).
    const H_N: usize = 128;
    const H_D: usize = 64;

    let device = NdArrayDevice::default();
    let pred_data: Vec<f32> = (0..(H_N * H_D))
        .map(|i| (i as f32 * 0.0043).sin())
        .collect();
    let tgt_data: Vec<f32> = (0..(H_N * H_D))
        .map(|i| (i as f32 * 0.0051).cos())
        .collect();

    let burn_hl: HuberLoss = HuberLossConfig::new(1.0).init();
    let p_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(pred_data.clone(), [H_N, H_D]), &device);
    let t_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(tgt_data.clone(), [H_N, H_D]), &device);

    let p_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![H_N, H_D], &pred_data),
        false,
    );
    let t_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![H_N, H_D], &tgt_data),
        false,
    );
    let p_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![H_N, H_D], &pred_data),
        false,
    );
    let t_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![H_N, H_D], &tgt_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — HuberLoss (128x64, delta=1.0)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_hl.forward(
                black_box(p_burn.clone()),
                black_box(t_burn.clone()),
                Reduction::Mean,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(huber_loss(black_box(&p_seq), black_box(&t_seq), 1.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(huber_loss(black_box(&p_moirai), black_box(&t_moirai), 1.0)))
    });
    group.finish();
}

fn bench_relu_forward(c: &mut Criterion) {
    // ReLU activation on [BATCH=128, FEATURES=256] — largest normalization shape.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — ReLU forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(relu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_gelu_forward(c: &mut Criterion) {
    // GeLU activation on [BATCH=128, FEATURES=256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();

    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — GeLU forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::gelu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(gelu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(gelu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_prelu_forward(c: &mut Criterion) {
    // PReLU on [BATCH x FEATURES] with the shared default alpha = 0.25. Inputs
    // are shifted negative so the parametric branch is exercised on ~half.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin() - 0.4)
        .collect();
    let device = NdArrayDevice::default();

    // Burn: PReluConfig defaults num_parameters=1, alpha=0.25.
    let burn_prelu = burn::nn::PReluConfig::new().init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — PReLU forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_prelu.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(prelu(black_box(&x_seq), 0.25)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(prelu(black_box(&x_moirai), 0.25)))
    });
    group.finish();
}

fn bench_sigmoid_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Sigmoid forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::sigmoid(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(sigmoid(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(sigmoid(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_tanh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Tanh forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::tanh(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(tanh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(tanh(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_silu_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — SiLU forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::silu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(silu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(silu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_maxpool1d_forward(c: &mut Criterion) {
    const MP1_N: usize = 8;
    const MP1_C: usize = 16;
    const MP1_L: usize = 128;
    const MP1_K: usize = 2;
    const MP1_S: usize = 2;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(MP1_N * MP1_C * MP1_L))
        .map(|i| (i as f32 * 0.0019).sin())
        .collect();
    let burn_mp1 = MaxPool1dConfig::new(MP1_K).with_stride(MP1_S).init();
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [MP1_N, MP1_C, MP1_L]),
        &device,
    );
    let pool_seq = MaxPool1d::<f32, SequentialBackend>::with_params(MP1_K, MP1_S, 0, 1);
    let pool_moirai = MaxPool1d::<f32, MoiraiBackend>::with_params(MP1_K, MP1_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![MP1_N, MP1_C, MP1_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![MP1_N, MP1_C, MP1_L], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — MaxPool1d forward (8x16x128, k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_mp1.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_avgpool1d_forward(c: &mut Criterion) {
    const AP1_N: usize = 8;
    const AP1_C: usize = 16;
    const AP1_L: usize = 128;
    const AP1_K: usize = 2;
    const AP1_S: usize = 2;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(AP1_N * AP1_C * AP1_L))
        .map(|i| (i as f32 * 0.0023).cos())
        .collect();
    let burn_ap1 = burn::nn::pool::AvgPool1dConfig::new(AP1_K)
        .with_stride(AP1_S)
        .init();
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [AP1_N, AP1_C, AP1_L]),
        &device,
    );
    let pool_seq = CoeusAvgPool1d::<f32, SequentialBackend>::with_params(AP1_K, AP1_S, 0, 1);
    let pool_moirai = CoeusAvgPool1d::<f32, MoiraiBackend>::with_params(AP1_K, AP1_S, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AP1_N, AP1_C, AP1_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AP1_N, AP1_C, AP1_L], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — AvgPool1d forward (8x16x128, k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_ap1.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}
fn bench_embeddingbag_sum(c: &mut Criterion) {
    // EmbeddingBag sum-mode forward: 16 bags × 100 tokens each, vocab=200, dim=64.
    // Burn 0.16 has no dedicated EmbeddingBag; the equivalent is Embedding::forward + sum_dim.
    const EB_VOCAB: usize = 200;
    const EB_DIM: usize = 64;
    const EB_BAGS: usize = 16;
    const EB_BAG_SIZE: usize = 100;

    let device = NdArrayDevice::default();

    // Build deterministic indices: each bag cycles through vocab.
    let flat_indices: Vec<usize> = (0..(EB_BAGS * EB_BAG_SIZE)).map(|i| i % EB_VOCAB).collect();
    let offsets: Vec<usize> = (0..EB_BAGS).map(|b| b * EB_BAG_SIZE).collect();
    let idx_i64_2d: Vec<i64> = flat_indices.iter().map(|&x| x as i64).collect();

    // Burn: Embedding + sum_dim (equivalent to EmbeddingBag sum).
    let burn_emb = burn::nn::EmbeddingConfig::new(EB_VOCAB, EB_DIM).init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 2, Int> =
        BurnTensor::from_data(TensorData::new(idx_i64_2d, [EB_BAGS, EB_BAG_SIZE]), &device);

    // Coeus EmbeddingBag.
    let eb_seq =
        EmbeddingBag::<f32, SequentialBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Sum);
    let eb_moirai =
        EmbeddingBag::<f32, MoiraiBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Sum);

    let mut group = c.benchmark_group(
        "Burn vs Coeus — EmbeddingBag sum (16 bags × 100 tokens, vocab=200 dim=64)",
    );
    group.bench_function("Burn NdArray (Embedding + sum_dim)", |b| {
        b.iter(|| {
            let embedded = burn_emb.forward(black_box(x_burn.clone()));
            black_box(embedded.sum_dim(1))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                eb_seq.forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                eb_moirai.forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.finish();
}

fn bench_adaptive_avg_pool2d_forward(c: &mut Criterion) {
    // AdaptiveAvgPool2d(1,1): ResNet-style global pooling step.
    // Input [8, 64, 7, 7] → output [8, 64, 1, 1].
    const AAP_N: usize = 8;
    const AAP_C: usize = 64;
    const AAP_H: usize = 7;
    const AAP_W: usize = 7;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(AAP_N * AAP_C * AAP_H * AAP_W))
        .map(|i| (i as f32 * 0.0021).sin())
        .collect();

    // Burn: AdaptiveAvgPool2d — init() has no type parameter.
    let burn_pool = burn::nn::pool::AdaptiveAvgPool2dConfig::new([1, 1]).init();
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [AAP_N, AAP_C, AAP_H, AAP_W]),
        &device,
    );

    // Coeus AdaptiveAvgPool2d.
    let pool_seq = AdaptiveAvgPool2d::<f32, SequentialBackend>::square(1);
    let pool_moirai = AdaptiveAvgPool2d::<f32, MoiraiBackend>::square(1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AAP_N, AAP_C, AAP_H, AAP_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AAP_N, AAP_C, AAP_H, AAP_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — AdaptiveAvgPool2d(1,1) forward (8x64x7x7)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_pool.forward::<BurnB>(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_adaptive_max_pool2d_forward(c: &mut Criterion) {
    // AdaptiveMaxPool2d(1,1): global max pooling step.
    // Input [8, 64, 7, 7] → output [8, 64, 1, 1].
    // Burn equivalent: AdaptiveAvgPool2d output (AdaptiveMaxPool2d not in Burn 0.16 public API,
    // so compare against the Burn MaxPool2d with kernel matching spatial dims).
    const AMP_N: usize = 8;
    const AMP_C: usize = 64;
    const AMP_H: usize = 7;
    const AMP_W: usize = 7;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(AMP_N * AMP_C * AMP_H * AMP_W))
        .map(|i| (i as f32 * 0.0017).cos())
        .collect();

    // Burn: MaxPool2d with kernel = spatial dim (approximates global max).
    let burn_pool = burn::nn::pool::MaxPool2dConfig::new([AMP_H, AMP_W])
        .with_strides([1, 1])
        .with_padding(burn::nn::PaddingConfig2d::Valid)
        .init();
    let x_burn: BurnTensor<BurnB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [AMP_N, AMP_C, AMP_H, AMP_W]),
        &device,
    );

    // Coeus AdaptiveMaxPool2d.
    let pool_seq = coeus_nn::AdaptiveMaxPool2d::<f32, SequentialBackend>::square(1);
    let pool_moirai = coeus_nn::AdaptiveMaxPool2d::<f32, MoiraiBackend>::square(1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![AMP_N, AMP_C, AMP_H, AMP_W], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![AMP_N, AMP_C, AMP_H, AMP_W], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — AdaptiveMaxPool2d(1,1) forward (8x64x7x7)");
    group.bench_function("Burn NdArray (MaxPool2d k=7)", |b| {
        b.iter(|| black_box(burn_pool.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(pool_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(pool_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_dropout_forward(c: &mut Criterion) {
    // Dropout eval-mode forward (no masking, p=0.5): [128, 256].
    // In eval mode both Burn and Coeus pass through unchanged (identity).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();
    let device = NdArrayDevice::default();
    let burn_do = DropoutConfig::new(0.5).init();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    // Coeus: Dropout is not generic over B; uses f32 Var with default backend.
    let mut do_layer = Dropout::new(0.5);
    do_layer.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Dropout eval forward (128x256, p=0.5)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_do.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(do_layer.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(do_layer.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_leaky_relu_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group =
        c.benchmark_group("Burn vs Coeus — LeakyReLU forward (128x256, neg_slope=0.01)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::leaky_relu(
                black_box(x_burn.clone()),
                0.01,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(leaky_relu(black_box(&x_seq), 0.01)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(leaky_relu(black_box(&x_moirai), 0.01)))
    });
    group.finish();
}

fn bench_mish_forward(c: &mut Criterion) {
    // Mish: x * tanh(softplus(x)) — matches Burn `activation::mish`.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — Mish forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::mish(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::mish(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::mish(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_conv_transpose1d_forward(c: &mut Criterion) {
    // ConvTranspose1d: [B=4, C_in=32, L=16] → [B=4, C_out=16, L_out=32]
    const CT_B: usize = 4;
    const CT_CIN: usize = 32;
    const CT_COUT: usize = 16;
    const CT_L: usize = 16;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(CT_B * CT_CIN * CT_L))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    // Burn ConvTranspose1d: [CT_CIN → CT_COUT, kernel=2, stride=2]
    let burn_ct = burn::nn::conv::ConvTranspose1dConfig::new([CT_CIN, CT_COUT], 2)
        .with_stride(2)
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [CT_B, CT_CIN, CT_L]),
        &device,
    );
    let ct_seq = ConvTranspose1d::<f32, SequentialBackend>::with_params(
        CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true,
    );
    let ct_moirai =
        ConvTranspose1d::<f32, MoiraiBackend>::with_params(CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![CT_B, CT_CIN, CT_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![CT_B, CT_CIN, CT_L], &input_data),
        false,
    );
    let mut group =
        c.benchmark_group("Burn vs Coeus — ConvTranspose1d forward (4x32x16, cin32→cout16 k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_ct.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(ct_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(ct_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_conv2d_forward_backward(c: &mut Criterion) {
    // Conv2d forward + backward: [4, 32, 16, 16], k3, no bias.
    // Measures full autograd cycle vs Burn's Autodiff backend.
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    const FB_N: usize = 4;
    const FB_C: usize = 32;
    const FB_HW: usize = 16;
    const FB_K: usize = 3;

    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(FB_N * FB_C * FB_HW * FB_HW))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();

    // Burn: Conv2d with autodiff backend.
    let burn_conv_fwd = Conv2dConfig::new([FB_C, FB_C], [FB_K, FB_K])
        .with_bias(false)
        .with_padding(PaddingConfig2d::Valid)
        .init::<AB>(&device);
    let x_burn_ad: BurnTensor<AB, 4> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [FB_N, FB_C, FB_HW, FB_HW]),
        &device,
    )
    .require_grad();

    // Coeus: Conv2d with tracked Var.
    let conv_seq = Conv2d::<f32, SequentialBackend>::new(FB_C, FB_C, FB_K, false);
    let conv_moirai = Conv2d::<f32, MoiraiBackend>::new(FB_C, FB_C, FB_K, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FB_N, FB_C, FB_HW, FB_HW], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FB_N, FB_C, FB_HW, FB_HW], &input_data),
        true,
    );

    let mut group =
        c.benchmark_group("Burn vs Coeus — Conv2d forward+backward (4x32x16x16, k3, no bias)");
    group.bench_function("Burn NdArray (autodiff)", |b| {
        b.iter(|| {
            let grads = burn_conv_fwd
                .forward(black_box(x_burn_ad.clone()))
                .sum()
                .backward();
            black_box(grads)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            conv_seq.zero_grad();
            x_seq.zero_grad();
            let out = conv_seq.forward(black_box(&x_seq));
            coeus_autograd::sum(&out).backward();
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            conv_moirai.zero_grad();
            x_moirai.zero_grad();
            let out = conv_moirai.forward(black_box(&x_moirai));
            coeus_autograd::sum(&out).backward();
        })
    });
    group.finish();
}

fn bench_conv_transpose3d_forward(c: &mut Criterion) {
    // ConvTranspose3d: [B=2, C_in=8, D=4, H=4, W=4] → [B=2, C_out=4, D=8, H=8, W=8]
    const CT3_B: usize = 2;
    const CT3_CIN: usize = 8;
    const CT3_COUT: usize = 4;
    const CT3_D: usize = 4;
    const CT3_H: usize = 4;
    const CT3_W: usize = 4;
    let device = NdArrayDevice::default();
    let input_data: Vec<f32> = (0..(CT3_B * CT3_CIN * CT3_D * CT3_H * CT3_W))
        .map(|i| (i as f32 * 0.013).sin())
        .collect();

    // Burn ConvTranspose3d: kernel=2, stride=2
    let burn_ct3 = burn::nn::conv::ConvTranspose3dConfig::new([CT3_CIN, CT3_COUT], [2, 2, 2])
        .with_stride([2, 2, 2])
        .init::<BurnB>(&device);
    let x_burn: BurnTensor<BurnB, 5> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [CT3_B, CT3_CIN, CT3_D, CT3_H, CT3_W]),
        &device,
    );
    let ct3_seq = ConvTranspose3d::<f32, SequentialBackend>::with_params(
        CT3_CIN, CT3_COUT, 2, 2, 0, 0, 1, true,
    );
    let ct3_moirai =
        ConvTranspose3d::<f32, MoiraiBackend>::with_params(CT3_CIN, CT3_COUT, 2, 2, 0, 0, 1, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![CT3_B, CT3_CIN, CT3_D, CT3_H, CT3_W],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![CT3_B, CT3_CIN, CT3_D, CT3_H, CT3_W],
            &input_data,
        ),
        false,
    );
    let mut group =
        c.benchmark_group("Burn vs Coeus — ConvTranspose3d forward (2x8x4x4x4, cin8→cout4 k2 s2)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_ct3.forward(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(ct3_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(ct3_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_softmax_forward(c: &mut Criterion) {
    // Softmax forward (128x256, dim=1).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Softmax forward (128x256, dim=1)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::softmax(
                black_box(x_burn.clone()),
                1,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_log_sigmoid_forward(c: &mut Criterion) {
    // LogSigmoid = log(sigmoid(x)) = -softplus(-x).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — LogSigmoid forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::log_sigmoid(black_box(
                x_burn.clone(),
            )))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::log_sigmoid(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::log_sigmoid(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_softplus_forward(c: &mut Criterion) {
    // Softplus = log(1 + exp(x)), beta=1 (Burn's default).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — Softplus forward (128x256, beta=1)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::softplus(
                black_box(x_burn.clone()),
                1.0,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_glu_forward(c: &mut Criterion) {
    // GLU = x[:, :H] * sigmoid(x[:, H:]) — input [128, 512] → output [128, 256].
    // Burn has no explicit GLU module; compare against manually assembled
    // Burn tensor ops (mul + sigmoid).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES * 2))
        .map(|i| (i as f32 * 0.0027).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES * 2]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES * 2], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES * 2], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — GLU forward (128x512 → 128x256)");
    group.bench_function("Burn NdArray (manual split+mul+sigmoid)", |b| {
        b.iter(|| {
            let chunks = black_box(x_burn.clone()).chunk(2, 1);
            let a = chunks[0].clone();
            let b_ = chunks[1].clone();
            black_box(a * burn::tensor::activation::sigmoid(b_))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::glu(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::glu(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_softmin_forward(c: &mut Criterion) {
    // Softmin = softmax(-x), (128x256, dim=1).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — Softmin forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (softmax(-x))", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::softmax(
                black_box(x_burn.clone()).neg(),
                1,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmin(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmin(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_diff_forward(c: &mut Criterion) {
    // torch.diff(x, n=1) — first-order discrete difference along last dim.
    // Burn has no direct diff; compare against manual x[1:] - x[:-1] via slice/sub.
    let input_data: Vec<f32> = (0..BATCH * FEATURES)
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &NdArrayDevice::default(),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Burn vs Coeus — diff(n=1) forward (128x256)");
    group.bench_function("Burn NdArray (manual slice/sub)", |b| {
        b.iter(|| {
            let x_ = black_box(x_burn.clone());
            let n = x_.dims()[1];
            let a = x_.clone().slice([0..BATCH, 1..n]);
            let b_ = x_.slice([0..BATCH, 0..n - 1]);
            black_box(a - b_)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::diff(black_box(&x_seq), 1, 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::diff(black_box(&x_moirai), 1, 1)))
    });
    group.finish();
}

fn bench_nansum_forward(c: &mut Criterion) {
    // nansum: [128, 256] matrix — sum ignoring NaN values.
    let mut input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();
    // Inject 5% NaN to stress the NaN-mask path.
    for i in (0..input_data.len()).step_by(20) {
        input_data[i] = f32::NAN;
    }
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    // Burn does not expose a nansum equivalent, so compare against Burn sum.
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — nansum forward (128x256, 5% NaN)");
    group.bench_function("Burn NdArray (sum, no NaN guard)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sum()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::nansum(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::nansum(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_tril_forward(c: &mut Criterion) {
    // tril: [256, 256] lower-triangular mask, diagonal=0.
    const SZ: usize = 256;
    let input_data: Vec<f32> = (0..(SZ * SZ)).map(|i| i as f32 * 0.001).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![SZ, SZ], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![SZ, SZ], &input_data),
        false,
    );

    // Burn: no direct tril, use mask_lower (burn 0.16 tensor method).
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(input_data.clone(), [SZ, SZ]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — tril forward (256x256)");
    group.bench_function("Burn NdArray (equal_elem as mask baseline)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).equal_elem(0.0f32)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tril(black_box(&x_seq), 0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tril(black_box(&x_moirai), 0)))
    });
    group.finish();
}

fn bench_topk_forward(c: &mut Criterion) {
    // topk k=16 along dim=1 on [128, 256].
    const TOPK: usize = 16;
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);

    // Burn topk is not in the public API; compare against Burn sort (full sort upper bound).
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — topk(k=16) forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (sort descending, upper bound)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sort_descending(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_seq), TOPK, 1, true)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_moirai), TOPK, 1, true)))
    });
    group.finish();
}

fn bench_cumsum_forward(c: &mut Criterion) {
    // cumsum along dim=1 on [128, 256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — cumsum forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (sum baseline, no cumsum in 0.16)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sum_dim(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cumsum(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cumsum(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_roll_forward(c: &mut Criterion) {
    // roll shift=32 along dim=1 on [128, 256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    // Burn has no roll; compare against Burn narrow+cat (upper-bound equivalent).
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — roll(shift=32,dim=1) forward (128x256)");
    group.bench_function("Burn NdArray (narrow+cat baseline)", |b| {
        b.iter(|| {
            let a = black_box(x_burn.clone())
                .clone()
                .narrow(1, FEATURES - 32, 32);
            let b2 = black_box(x_burn.clone())
                .clone()
                .narrow(1, 0, FEATURES - 32);
            black_box(BurnTensor::<BurnB, 2>::cat(vec![a, b2], 1))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::roll(
                black_box(&x_seq),
                &[32isize],
                &[1usize],
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::roll(
                black_box(&x_moirai),
                &[32isize],
                &[1usize],
            ))
        })
    });
    group.finish();
}

fn bench_bmm_forward(c: &mut Criterion) {
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

    let device = NdArrayDevice::default();
    let a_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(a_data.clone(), [BMM_B, BMM_M, BMM_K]),
        &device,
    );
    let b_burn: BurnTensor<BurnB, 3> = BurnTensor::from_data(
        TensorData::new(b_data.clone(), [BMM_B, BMM_K, BMM_N]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — bmm forward (32x64x128 @ 32x128x64)");
    group.bench_function("Burn NdArray (matmul)", |b| {
        b.iter(|| black_box(black_box(a_burn.clone()).matmul(black_box(b_burn.clone()))))
    });
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

fn bench_log_sum_exp_forward(c: &mut Criterion) {
    // logsumexp dim=1 on [128, 256] — numerically stable softmax log-normalizer.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0019).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — log_sum_exp forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (exp+sum+log as proxy)", |b| {
        b.iter(|| {
            let exp_x = black_box(x_burn.clone()).exp();
            black_box(exp_x.sum_dim(1).log())
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log_sum_exp(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log_sum_exp(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_sdp_attention_forward(c: &mut Criterion) {
    // Scaled dot-product attention: [4, 8, 64] Q×K×V (4 heads, seq=8, d_k=64).
    // The output alloc uses alloc_on: every position written by the kernel.
    const SA_B: usize = 4;
    const SA_S: usize = 8;
    const SA_D: usize = 64;
    let q_data: Vec<f32> = (0..(SA_B * SA_S * SA_D))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let k_data = q_data.clone();
    let v_data: Vec<f32> = (0..(SA_B * SA_S * SA_D))
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    let backend_seq = SequentialBackend;
    let q_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![SA_B, SA_S, SA_D], &q_data);
    let k_seq = q_seq.clone();
    let v_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![SA_B, SA_S, SA_D], &v_data);

    let backend_moirai = MoiraiBackend;
    let q_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![SA_B, SA_S, SA_D], &q_data);
    let k_moirai = q_moirai.clone();
    let v_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![SA_B, SA_S, SA_D], &v_data);

    let scale = 1.0f32 / (SA_D as f32).sqrt();

    // Burn: MultiHeadAttention with SA_B heads is a superset; use raw matmul chain
    let device = NdArrayDevice::default();
    let q_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(q_data.clone(), [SA_B, SA_S, SA_D]), &device);
    let kt_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(k_data.clone(), [SA_B, SA_D, SA_S]), &device);
    let v_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(v_data.clone(), [SA_B, SA_S, SA_D]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — sdp_attention forward (4x8x64)");
    group.bench_function("Burn NdArray (QK^T/scale@V proxy)", |b| {
        b.iter(|| {
            let scores = black_box(q_burn.clone()).matmul(black_box(kt_burn.clone())) * scale;
            let weights = burn::tensor::activation::softmax(scores, 2);
            black_box(weights.matmul(black_box(v_burn.clone())))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::scaled_dot_product_attention(
                black_box(&q_seq),
                black_box(&k_seq),
                black_box(&v_seq),
                None,
                false,
                scale,
                &backend_seq,
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::scaled_dot_product_attention(
                black_box(&q_moirai),
                black_box(&k_moirai),
                black_box(&v_moirai),
                None,
                false,
                scale,
                &backend_moirai,
            ))
        })
    });
    group.finish();
}

fn bench_nanmean_forward(c: &mut Criterion) {
    // nanmean: [128, 256] with 5% NaN injection.
    let mut input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0029).sin())
        .collect();
    for i in (0..input_data.len()).step_by(20) {
        input_data[i] = f32::NAN;
    }
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    // Filter NaN before passing to Burn (Burn sum is the proxy).
    let clean: Vec<f32> = input_data
        .iter()
        .map(|&x| if x.is_nan() { 0.0 } else { x })
        .collect();
    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(clean, [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus — nanmean forward (128x256, 5% NaN)");
    group.bench_function("Burn NdArray (mean of pre-cleaned, no NaN guard)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).mean()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::nanmean(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::nanmean(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_gather_forward(c: &mut Criterion) {
    // gather dim=1 on [128, 256] with indices covering full column range.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0017).sin())
        .collect();
    let idx_f32_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| ((i * 7 + 13) % FEATURES) as f32)
        .collect();

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let idx_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &idx_f32_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let idx_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &idx_f32_data),
        false,
    );

    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let idx_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(idx_f32_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — gather forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (select_rows proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()) * black_box(idx_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::gather(
                black_box(&x_seq),
                1,
                black_box(&idx_seq),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::gather(
                black_box(&x_moirai),
                1,
                black_box(&idx_moirai),
            ))
        })
    });
    group.finish();
}

fn bench_softplus_activation(c: &mut Criterion) {
    // softplus: [128, 256] — F.softplus(x, beta=1, threshold=20).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0037).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — softplus forward (128x256)");
    group.bench_function("Burn NdArray (log(1+exp(x)))", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).exp().add_scalar(1.0f32).log()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_vector_norm_forward(c: &mut Criterion) {
    // vector_norm L2: [128, 256] global L2 norm.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — vector_norm L2 (128x256)");
    group.bench_function("Burn NdArray (sum(x^2).sqrt as proxy)", |b| {
        b.iter(|| {
            let x = black_box(x_burn.clone());
            let sq = x.clone() * x;
            black_box(sq.sum().sqrt())
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::norm(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::norm(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_erf_forward(c: &mut Criterion) {
    // erf: [128, 256] — Gauss error function, used in GELU approximation.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — erf forward (128x256)");
    group.bench_function("Burn NdArray (tanh approx proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).tanh()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::erf(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::erf(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_sin_cos_forward(c: &mut Criterion) {
    // sin+cos fused: [128, 256] — typical in RoPE / positional encoding.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| i as f32 * 0.0031).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — sin+cos forward (128x256)");
    group.bench_function("Burn NdArray (sin+cos sequential)", |b| {
        b.iter(|| {
            let s = black_box(x_burn.clone()).sin();
            let c2 = black_box(x_burn.clone()).cos();
            black_box((s, c2))
        })
    });
    group.bench_function("Coeus Sequential (sin+cos)", |b| {
        b.iter(|| {
            let s = coeus_autograd::sin(black_box(&x_seq));
            let c2 = coeus_autograd::cos(black_box(&x_seq));
            black_box((s, c2))
        })
    });
    group.bench_function("Coeus Moirai (sin+cos)", |b| {
        b.iter(|| {
            let s = coeus_autograd::sin(black_box(&x_moirai));
            let c2 = coeus_autograd::cos(black_box(&x_moirai));
            black_box((s, c2))
        })
    });
    group.finish();
}

// Reference is consumed by `criterion_group!` macro below; the macro
// expansion does not propagate usage info, so suppress `dead_code` here.
#[allow(dead_code, reason = "referenced by criterion_group! macro below")]
fn bench_tan_forward(c: &mut Criterion) {
    // tan: [128, 256] — important for rotation/phase ops in signal processing.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0021).sin() * 1.5) // stay in safe domain (|x| < π/2)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — tan forward (128x256)");
    group.bench_function("Burn NdArray (sin/cos proxy)", |b| {
        b.iter(|| {
            let s = black_box(x_burn.clone()).sin();
            let c2 = black_box(x_burn.clone()).cos();
            black_box(s / c2)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tan(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tan(black_box(&x_moirai))))
    });
    group.finish();
}

// Reference is consumed by `criterion_group!` macro below; the macro
// expansion does not propagate usage info, so suppress `dead_code` here.
#[allow(dead_code, reason = "referenced by criterion_group! macro below")]
fn bench_atan_forward(c: &mut Criterion) {
    // atan: [128, 256] — arctan is used in angle reconstruction and attention scores.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0027).cos() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — atan forward (128x256)");
    group.bench_function("Burn NdArray (1/(1+x^2) recip proxy)", |b| {
        b.iter(|| {
            let x = black_box(x_burn.clone());
            let xsq = x.clone() * x;
            black_box(xsq.add_scalar(1.0f32).recip())
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::atan(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::atan(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_clamp_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - clamp(-1,1) forward (128x256)");
    group.bench_function("Burn NdArray (clamp_min+clamp_max)", |b| {
        b.iter(|| {
            black_box(
                black_box(x_burn.clone())
                    .clamp_min(-1.0f32)
                    .clamp_max(1.0f32),
            )
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::clamp(black_box(&x_seq), -1.0, 1.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::clamp(black_box(&x_moirai), -1.0, 1.0)))
    });
    group.finish();
}

fn bench_asin_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0019).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - asin forward (128x256)");
    group.bench_function("Burn NdArray (sin proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sin()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::asin(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::asin(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_erfc_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - erfc forward (128x256)");
    group.bench_function("Burn NdArray (1-tanh proxy)", |b| {
        b.iter(|| {
            let t = black_box(x_burn.clone()).tanh();
            black_box(t.mul_scalar(-1.0f32).add_scalar(1.0f32))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::erfc(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::erfc(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_exp_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - exp forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).exp()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::exp(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::exp(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_log_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - log forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).log()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_neg_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - neg forward (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).neg()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::neg(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::neg(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_std_forward(c: &mut Criterion) {
    // std (unbiased): [128, 256] — variance reduction used in normalization.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0017).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - std forward (128x256)");
    group.bench_function("Burn NdArray (mean proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).mean()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::std_dev(&x_seq, true)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::std_dev(&x_moirai, true)))
    });
    group.finish();
}

fn bench_sinh_forward(c: &mut Criterion) {
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
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - sinh forward (128x256)");
    group.bench_function("Burn NdArray (exp proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).exp()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sinh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sinh(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_cosh_forward(c: &mut Criterion) {
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
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - cosh forward (128x256)");
    group.bench_function("Burn NdArray (exp+recip proxy)", |b| {
        b.iter(|| {
            let e = black_box(x_burn.clone()).exp();
            let ne = black_box(x_burn.clone()).neg().exp();
            black_box((e + ne) / 2.0f32)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cosh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cosh(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_log2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - log2 forward (128x256)");
    group.bench_function("Burn NdArray (log/ln2 proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).log().div_scalar(2.0f32.ln())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log2(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log2(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_log10_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - log10 forward (128x256)");
    group.bench_function("Burn NdArray (log/ln10 proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).log().div_scalar(10.0f32.ln())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log10(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log10(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_relu2_forward(c: &mut Criterion) {
    // relu [128,256] — explicit Burn vs Coeus parity row (Burn has relu_forward already)
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - relu2 fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::relu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_tanh2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - tanh fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::tanh(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_sigmoid2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - sigmoid fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::sigmoid(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_gelu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - gelu fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::gelu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::gelu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::gelu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_atanh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - atanh forward (128x256)");
    group.bench_function("Burn NdArray (log proxy)", |b| {
        b.iter(|| {
            let one = BurnTensor::<BurnB, 2>::ones_like(black_box(&x_burn));
            black_box(
                ((one.clone() + black_box(x_burn.clone())) / (one - black_box(x_burn.clone())))
                    .log()
                    * 0.5f32,
            )
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::atanh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::atanh(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_expm1_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - expm1 forward (128x256)");
    group.bench_function("Burn NdArray (exp-1 proxy)", |b| {
        b.iter(|| {
            let one = BurnTensor::<BurnB, 2>::ones_like(black_box(&x_burn));
            black_box(black_box(x_burn.clone()).exp() - one)
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::expm1(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::expm1(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_log1p_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - log1p forward (128x256)");
    group.bench_function("Burn NdArray (log(1+x) proxy)", |b| {
        b.iter(|| {
            let one = BurnTensor::<BurnB, 2>::ones_like(black_box(&x_burn));
            black_box((one + black_box(x_burn.clone())).log())
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log1p(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log1p(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_silu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - silu2 fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::silu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::silu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::silu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_softmax2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - softmax fwd (128x256, dim=1)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::softmax(
                black_box(x_burn.clone()),
                1,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softmax(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_sqrt2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.01)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - sqrt fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sqrt()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sqrt(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_abs2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - abs fwd (128x256)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).abs()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::abs(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_selu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - selu fwd (128x256)");
    group.bench_function("Burn NdArray (elu proxy)", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::selu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::selu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_exp2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 4.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - exp2 forward (128x256)");
    group.bench_function("Burn NdArray (exp proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).exp()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::exp2(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::exp2(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_hardsigmoid2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - hardsigmoid fwd (128x256)");
    group.bench_function("Burn NdArray (sigmoid proxy)", |b| {
        b.iter(|| black_box(burn::tensor::activation::sigmoid(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_log_softmax2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - log_softmax fwd (128x256, dim=1)");
    group.bench_function("Burn NdArray (softmax.log)", |b| {
        b.iter(|| {
            black_box(burn::tensor::activation::log_softmax(
                black_box(x_burn.clone()),
                1,
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::log_softmax(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::log_softmax(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_lgamma_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.5)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - lgamma forward (128x256)");
    group.bench_function("Burn NdArray (log proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).log()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::lgamma_forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::lgamma_forward(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_pow_forward(c: &mut Criterion) {
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
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - pow(3) forward (128x256)");
    group.bench_function("Burn NdArray (clamp_min proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).clamp_min(-10.0f32)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::pow(black_box(&x_seq), 3.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::pow(black_box(&x_moirai), 3.0)))
    });
    group.finish();
}

fn bench_recip_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).cos().abs() + 0.1)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - recip forward (128x256)");
    group.bench_function("Burn NdArray (recip)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).recip()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::recip(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::recip(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_conv2d_fwd_bwd(c: &mut Criterion) {
    // Conv2d (8,16,k=3) forward+backward: [4,8,16,16]
    const N: usize = 4;
    const C_IN: usize = 8;
    const C_OUT: usize = 16;
    const K: usize = 3;
    const H: usize = 16;
    const W: usize = 16;
    let _w_data: Vec<f32> = (0..(C_OUT * C_IN * K * K))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let inp_data: Vec<f32> = (0..(N * C_IN * H * W))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let device = NdArrayDevice::default();
    let inp_burn: BurnTensor<BurnB, 4> =
        BurnTensor::from_data(TensorData::new(inp_data.clone(), [N, C_IN, H, W]), &device);
    let burn_conv = burn::nn::conv::Conv2dConfig::new([C_IN, C_OUT], [K, K])
        .with_bias(false)
        .init::<BurnB>(&device);

    let inp_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N, C_IN, H, W], &inp_data),
        true,
    );
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N, C_IN, H, W], &inp_data),
        true,
    );
    let conv_seq = coeus_nn::Conv2d::<f32, SequentialBackend>::new(C_IN, C_OUT, K, false);
    let conv_moirai = coeus_nn::Conv2d::<f32, MoiraiBackend>::new(C_IN, C_OUT, K, false);
    use coeus_nn::Module;

    let mut group = c.benchmark_group("Burn vs Coeus - Conv2d(8,16,k=3) fwd+bwd (4x8x16x16)");
    group.bench_function("Burn NdArray (fwd only)", |b| {
        b.iter(|| black_box(burn_conv.forward(black_box(inp_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let out = conv_seq.forward(black_box(&inp_seq));
            black_box(out).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let out = conv_moirai.forward(black_box(&inp_moirai));
            black_box(out).backward()
        })
    });
    group.finish();
}

fn bench_scatter_add_forward(c: &mut Criterion) {
    // scatter_add dim=1 on [128,256] with random indices covering full range
    const S: usize = BATCH;
    const D: usize = FEATURES;
    let src_data: Vec<f32> = (0..(S * D)).map(|i| (i as f32 * 0.001).sin()).collect();
    let idx_f32: Vec<f32> = (0..(S * D)).map(|i| ((i * 7 + 13) % D) as f32).collect();
    let base_data = vec![0.0f32; S * D];
    let base_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &base_data);
    let idx_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &idx_f32);
    let src_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &src_data);
    let base_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &base_data);
    let idx_moirai = coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &idx_f32);
    let src_moirai = coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &src_data);
    let device = NdArrayDevice::default();
    let src_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(src_data.clone(), [S, D]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus - scatter_add forward (128x256, dim=1)");
    group.bench_function("Burn NdArray (sum proxy)", |b| {
        b.iter(|| black_box(black_box(src_burn.clone()).sum()))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::scatter_add(
                black_box(&base_seq),
                1,
                black_box(&idx_seq),
                black_box(&src_seq),
                &SequentialBackend,
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::scatter_add(
                black_box(&base_moirai),
                1,
                black_box(&idx_moirai),
                black_box(&src_moirai),
                &MoiraiBackend,
            ))
        })
    });
    group.finish();
}

fn bench_argmax2_forward(c: &mut Criterion) {
    // argmax dim=1 on [128,256]
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let x_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(
        vec![BATCH, FEATURES],
        &input_data,
    );
    let x_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus - argmax(dim=1) forward (128x256)");
    group.bench_function("Burn NdArray (argmax)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).argmax(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::argmax(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::argmax(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_topk2_forward(c: &mut Criterion) {
    // topk k=32 on [128,256] dim=1
    const K2: usize = 32;
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.009).cos())
        .collect();
    let x_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(
        vec![BATCH, FEATURES],
        &input_data,
    );
    let x_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus - topk(k=32,dim=1) forward (128x256)");
    group.bench_function("Burn NdArray (sort descending)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sort_descending(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_seq), K2, 1, true)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_moirai), K2, 1, true)))
    });
    group.finish();
}

fn bench_mean_axis_forward(c: &mut Criterion) {
    // mean_axis dim=1 on [128,256]
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus - mean_axis(dim=1) forward (128x256)");
    group.bench_function("Burn NdArray (mean_dim)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).mean_dim(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::mean_axis(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::mean_axis(black_box(&x_moirai), 1)))
    });
    group.finish();
}

fn bench_elu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - elu fwd (128x256)");
    group.bench_function("Burn NdArray (relu proxy)", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(black_box(x_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::elu(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::elu(black_box(&x_moirai))))
    });
    group.finish();
}

fn bench_cumsum_dim0_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - cumsum(dim=0) forward (128x256)");
    group.bench_function("Burn NdArray (cumsum dim=0 n/a, sum proxy)", |b| {
        b.iter(|| black_box(black_box(x_burn.clone()).sum_dim(0)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::cumsum(black_box(&x_seq), 0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::cumsum(black_box(&x_moirai), 0)))
    });
    group.finish();
}

fn bench_where_cond_forward(c: &mut Criterion) {
    let cond_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos())
        .collect();
    let cond_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &cond_data),
        false,
    );
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let cond_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &cond_data),
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
    let device = NdArrayDevice::default();
    let a_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(a_data.clone(), [BATCH, FEATURES]), &device);
    let b_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(b_data.clone(), [BATCH, FEATURES]), &device);
    let c_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(
        TensorData::new(cond_data.clone(), [BATCH, FEATURES]),
        &device,
    );
    let mut group = c.benchmark_group("Burn vs Coeus - where_cond forward (128x256)");
    group.bench_function("Burn NdArray (mask_where proxy)", |b| {
        b.iter(|| {
            let mask = black_box(c_burn.clone()).equal_elem(1.0f32);
            black_box(black_box(a_burn.clone()).mask_where(mask, black_box(b_burn.clone())))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::where_cond(
                black_box(&cond_seq),
                black_box(&a_seq),
                black_box(&b_seq),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::where_cond(
                black_box(&cond_moirai),
                black_box(&a_moirai),
                black_box(&b_moirai),
            ))
        })
    });
    group.finish();
}

fn bench_conv1d2_forward(c: &mut Criterion) {
    // Conv1d(16,32,k=3): [8,16,64] — second conv1d row with different shape
    const N1: usize = 8;
    const C_IN1: usize = 16;
    const C_OUT1: usize = 32;
    const K1: usize = 3;
    const L1: usize = 64;
    let inp_data: Vec<f32> = (0..(N1 * C_IN1 * L1))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let device = NdArrayDevice::default();
    let inp_burn: BurnTensor<BurnB, 3> =
        BurnTensor::from_data(TensorData::new(inp_data.clone(), [N1, C_IN1, L1]), &device);
    let burn_conv = burn::nn::conv::Conv1dConfig::new(C_IN1, C_OUT1, K1)
        .with_bias(false)
        .with_padding(PaddingConfig1d::Valid)
        .init::<BurnB>(&device);
    let inp_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data),
        false,
    );
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data),
        false,
    );
    let conv_seq = coeus_nn::Conv1d::<f32, SequentialBackend>::new(C_IN1, C_OUT1, K1, false);
    let conv_moirai = coeus_nn::Conv1d::<f32, MoiraiBackend>::new(C_IN1, C_OUT1, K1, false);
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Burn vs Coeus - Conv1d(16,32,k=3) fwd (8x16x64)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_conv.forward(black_box(inp_burn.clone()))))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(conv_seq.forward(black_box(&inp_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(conv_moirai.forward(black_box(&inp_moirai))))
    });
    group.finish();
}

fn bench_acos_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.001).sin() * 0.9).collect();
    let x_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(input_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - acos forward (128x256)");
    group.bench_function("Burn NdArray (acos not in 0.16, cos proxy)", |b| { b.iter(|| black_box(black_box(x_burn.clone()).cos())) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::acos(black_box(&x_seq)))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::acos(black_box(&x_moirai)))) });
    group.finish();
}

fn bench_sum_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.003).sin()).collect();
    let x_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(input_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - sum forward (128x256)");
    group.bench_function("Burn NdArray", |b| { b.iter(|| black_box(black_box(x_burn.clone()).sum())) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::sum(black_box(&x_seq)))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::sum(black_box(&x_moirai)))) });
    group.finish();
}

fn bench_linear_fwd_bwd(c: &mut Criterion) {
    // Linear(256,512) fwd+bwd: [128,256]
    const IN_F: usize = FEATURES;
    const OUT_F: usize = 512;
    let inp_data: Vec<f32> = (0..(BATCH * IN_F)).map(|i| (i as f32 * 0.002).sin()).collect();
    let _w_data: Vec<f32> = (0..(OUT_F * IN_F)).map(|i| (i as f32 * 0.001).cos()).collect();
    let device = NdArrayDevice::default();
    let inp_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(inp_data.clone(), [BATCH, IN_F]), &device);
    let burn_lin = burn::nn::LinearConfig::new(IN_F, OUT_F).with_bias(false).init::<BurnB>(&device);
    let inp_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, IN_F], &inp_data), true);
    let inp_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, IN_F], &inp_data), true);
    let lin_seq = coeus_nn::Linear::<f32, SequentialBackend>::new(IN_F, OUT_F, false);
    let lin_moirai = coeus_nn::Linear::<f32, MoiraiBackend>::new(IN_F, OUT_F, false);
    let mut group = c.benchmark_group("Burn vs Coeus - Linear(256,512) fwd+bwd (128x256)");
    group.bench_function("Burn NdArray (fwd only)", |b| { b.iter(|| black_box(burn_lin.forward(black_box(inp_burn.clone())))) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| { let o = lin_seq.forward(black_box(&inp_seq)); black_box(o).backward() }) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| { let o = lin_moirai.forward(black_box(&inp_moirai)); black_box(o).backward() }) });
    group.finish();
}

fn bench_prod_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.0001).sin() + 1.0001).collect();
    let x_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(input_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - prod forward (128x256)");
    group.bench_function("Burn NdArray (sum proxy)", |b| { b.iter(|| black_box(black_box(x_burn.clone()).sum())) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::prod(black_box(&x_seq)))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::prod(black_box(&x_moirai)))) });
    group.finish();
}


fn bench_var_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.003).sin()).collect();
    let x_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(input_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - var(unbiased) forward (128x256)");
    group.bench_function("Burn NdArray (mean+sum proxy)", |b| { b.iter(|| black_box(black_box(x_burn.clone()).mean())) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::var(&x_seq, true))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::var(&x_moirai, true))) });
    group.finish();
}

fn bench_hardshrink_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.002).sin() * 2.0).collect();
    let x_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data), false);
    let device = NdArrayDevice::default();
    let x_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(input_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - hardshrink(0.5) forward (128x256)");
    group.bench_function("Burn NdArray (relu proxy)", |b| { b.iter(|| black_box(burn::tensor::activation::relu(black_box(x_burn.clone())))) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::hardshrink(&x_seq, 0.5))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::hardshrink(&x_moirai, 0.5))) });
    group.finish();
}

fn bench_mul_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.003).sin()).collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.005).cos()).collect();
    let a_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data), false);
    let b_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data), false);
    let a_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data), false);
    let b_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data), false);
    let device = NdArrayDevice::default();
    let a_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(a_data.clone(), [BATCH, FEATURES]), &device);
    let b_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(b_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - mul forward (128x256)");
    group.bench_function("Burn NdArray", |b| { b.iter(|| black_box(black_box(a_burn.clone()) * black_box(b_burn.clone()))) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::mul(&a_seq, &b_seq))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::mul(&a_moirai, &b_moirai))) });
    group.finish();
}

fn bench_div_forward(c: &mut Criterion) {
    let a_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.003).sin() * 4.0 + 0.1).collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| (i as f32 * 0.005).cos().abs() + 0.5).collect();
    let a_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data), false);
    let b_seq = Var::new(Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data), false);
    let a_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data), false);
    let b_moirai = Var::new(Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data), false);
    let device = NdArrayDevice::default();
    let a_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(a_data.clone(), [BATCH, FEATURES]), &device);
    let b_burn: BurnTensor<BurnB, 2> = BurnTensor::from_data(TensorData::new(b_data.clone(), [BATCH, FEATURES]), &device);
    let mut group = c.benchmark_group("Burn vs Coeus - div forward (128x256)");
    group.bench_function("Burn NdArray", |b| { b.iter(|| black_box(black_box(a_burn.clone()) / black_box(b_burn.clone()))) });
    group.bench_function("Coeus Sequential", |b| { b.iter(|| black_box(coeus_autograd::div(&a_seq, &b_seq))) });
    group.bench_function("Coeus Moirai", |b| { b.iter(|| black_box(coeus_autograd::div(&a_moirai, &b_moirai))) });
    group.finish();
}


criterion_group!(
    benches,
    bench_linear_forward,
    bench_layernorm_forward,
    bench_rmsnorm_forward,
    bench_batchnorm1d_eval_forward,
    bench_batchnorm2d_eval_forward,
    bench_batchnorm3d_eval_forward,
    bench_groupnorm_forward,
    bench_maxpool2d_forward,
    bench_avgpool2d_forward,
    bench_conv1d_forward,
    bench_conv1d_forward_backward,
    bench_conv2d_forward,
    bench_conv2d_forward_backward,
    bench_conv3d_forward,
    bench_conv_transpose1d_forward,
    bench_conv_transpose3d_forward,
    bench_mha_forward,
    bench_transformer_encoder_forward,
    bench_embedding_forward,
    bench_embeddingbag_sum,
    bench_linear_forward_backward,
    bench_lstm_forward,
    bench_gru_forward,
    bench_swiglu_forward,
    bench_glu_forward,
    bench_softmin_forward,
    bench_diff_forward,
    bench_softmax_forward,
    bench_adaptive_avg_pool2d_forward,
    bench_instancenorm2d_forward,
    bench_cross_entropy_loss,
    bench_mse_loss,
    bench_huber_loss,
    bench_relu_forward,
    bench_prelu_forward,
    bench_gelu_forward,
    bench_sigmoid_forward,
    bench_tanh_forward,
    bench_silu_forward,
    bench_leaky_relu_forward,
    bench_mish_forward,
    bench_log_sigmoid_forward,
    bench_softplus_forward,
    bench_dropout_forward,
    bench_maxpool1d_forward,
    bench_avgpool1d_forward,
    bench_adaptive_max_pool2d_forward,
    bench_nansum_forward,
    bench_tril_forward,
    bench_topk_forward,
    bench_cumsum_forward,
    bench_roll_forward,
    bench_bmm_forward,
    bench_log_sum_exp_forward,
    bench_sdp_attention_forward,
    bench_nanmean_forward,
    bench_gather_forward,
    bench_softplus_activation,
    bench_vector_norm_forward,
    bench_erf_forward,
    bench_sin_cos_forward,
    bench_tan_forward,
    bench_atan_forward,
    bench_clamp_forward,
    bench_asin_forward,
    bench_erfc_forward,
    bench_std_forward,
    bench_exp_forward,
    bench_log_forward,
    bench_neg_forward,
    bench_sinh_forward,
    bench_cosh_forward,
    bench_log2_forward,
    bench_log10_forward,
    bench_relu2_forward,
    bench_tanh2_forward,
    bench_sigmoid2_forward,
    bench_gelu2_forward,
    bench_atanh_forward,
    bench_expm1_forward,
    bench_log1p_forward,
    bench_silu2_forward,
    bench_softmax2_forward,
    bench_sqrt2_forward,
    bench_abs2_forward,
    bench_selu2_forward,
    bench_exp2_forward,
    bench_hardsigmoid2_forward,
    bench_log_softmax2_forward,
    bench_lgamma_forward,
    bench_pow_forward,
    bench_recip_forward,
    bench_conv2d_fwd_bwd,
    bench_scatter_add_forward,
    bench_argmax2_forward,
    bench_topk2_forward,
    bench_mean_axis_forward,
    bench_elu2_forward,
    bench_cumsum_dim0_forward,
    bench_where_cond_forward,
    bench_conv1d2_forward,
    bench_acos_forward,
    bench_sum_forward,
    bench_linear_fwd_bwd,
    bench_prod_forward,
    bench_var_forward,
    bench_hardshrink_forward,
    bench_mul_forward,
    bench_div_forward
);
criterion_main!(benches);
