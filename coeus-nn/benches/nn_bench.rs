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
    BatchNorm2d, Conv1d, Conv2d, Conv3d, Embedding, LayerNorm, Linear, Module, MultiHeadAttention,
    NullMask, TransformerEncoderLayer,
};
use coeus_tensor::Tensor;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::nn::attention::{MhaInput, MultiHeadAttentionConfig};
use burn::nn::conv::Conv1dConfig;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv3dConfig;
use burn::nn::transformer::{TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{
    BatchNormConfig, LayerNormConfig, LinearConfig, PaddingConfig1d, PaddingConfig2d,
    PaddingConfig3d,
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

criterion_group!(
    benches,
    bench_linear_forward,
    bench_layernorm_forward,
    bench_batchnorm2d_eval_forward,
    bench_conv1d_forward,
    bench_conv2d_forward,
    bench_conv3d_forward,
    bench_mha_forward,
    bench_transformer_encoder_forward,
    bench_embedding_forward,
    bench_linear_forward_backward
);
criterion_main!(benches);
