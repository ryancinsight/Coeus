//! Layer-level forward-pass benchmarks across Coeus providers.
//!
//! Complements `crates/coeus-tensor/benches/tensor_bench.rs` (tensor primitives) by
//! timing whole `nn` layer forward passes through `SequentialBackend` and
//! `MoiraiBackend` on identical shapes. These benchmarks measure the real
//! production layer code; the harness body is never tuned to move the number.
//!
//! Run one group:
//!   `cargo bench -p coeus-nn --bench nn_bench -- Linear`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_nn::{
    cross_entropy_loss, gelu, huber_loss, interpolate_2d, leaky_relu, mse_loss, prelu, relu,
    sigmoid, silu, tanh, AdaptiveAvgPool2d, AvgPool1d as CoeusAvgPool1d, AvgPool2d, AvgPool3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d, Bidirectional, Bilinear, Conv1d, Conv2d, Conv3d,
    ConvTranspose1d, ConvTranspose3d, Dropout, Embedding, EmbeddingBag, EmbeddingBagMode,
    GroupNorm, Gru as CoeusGru, InstanceNorm2d, InterpolateMode as CoeusInterpolateMode, LayerNorm,
    Linear, LocalResponseNorm, Lstm, MaxPool1d, MaxPool2d, MaxPool3d, Module, ModuleExt,
    MultiHeadAttention, NullMask, RMSNorm, RNNCell, ReLU, Rnn, RnnNonlinearity, RotaryEmbedding,
    Sequential, SinusoidalEncoding, SwiGlu, TransformerEncoderLayer,
};
use coeus_tensor::Tensor;

// Shared workload: batch of `BATCH` vectors of width `FEATURES`.
const BATCH: usize = 128;
const FEATURES: usize = 256;

#[path = "nn_bench/provider/arithmetic_backward.rs"]
mod arithmetic_backward;
#[path = "nn_bench/provider/arithmetic_forward.rs"]
mod arithmetic_forward;
#[path = "nn_bench/provider/attention.rs"]
mod attention;
#[path = "nn_bench/provider/convolution.rs"]
mod convolution;
#[path = "nn_bench/provider/dense.rs"]
mod dense;
#[path = "nn_bench/provider/exponential_backward.rs"]
mod exponential_backward;
#[path = "nn_bench/provider/exponential_forward.rs"]
mod exponential_forward;
#[path = "nn_bench/provider/gating_backward.rs"]
mod gating_backward;
#[path = "nn_bench/provider/gating_forward.rs"]
mod gating_forward;
#[path = "nn_bench/provider/indexing.rs"]
mod indexing;
#[path = "nn_bench/provider/initialization.rs"]
mod initialization;
#[path = "nn_bench/provider/loss_backward.rs"]
mod loss_backward;
#[path = "nn_bench/provider/loss_forward.rs"]
mod loss_forward;
#[path = "nn_bench/provider/normalization_backward.rs"]
mod normalization_backward;
#[path = "nn_bench/provider/normalization_forward.rs"]
mod normalization_forward;
#[path = "nn_bench/provider/pooling_interpolation.rs"]
mod pooling_interpolation;
#[path = "nn_bench/provider/positional.rs"]
mod positional;
#[path = "nn_bench/provider/rectifier_backward.rs"]
mod rectifier_backward;
#[path = "nn_bench/provider/rectifier_forward.rs"]
mod rectifier_forward;
#[path = "nn_bench/provider/reduction_backward.rs"]
mod reduction_backward;
#[path = "nn_bench/provider/reduction_forward.rs"]
mod reduction_forward;
#[path = "nn_bench/provider/sequence.rs"]
mod sequence;
#[path = "nn_bench/provider/tensor_operation.rs"]
mod tensor_operation;
#[path = "nn_bench/provider/trigonometric_backward.rs"]
mod trigonometric_backward;
#[path = "nn_bench/provider/trigonometric_forward.rs"]
mod trigonometric_forward;
#[path = "nn_bench/provider/unary_backward.rs"]
mod unary_backward;
#[path = "nn_bench/provider/unary_forward.rs"]
mod unary_forward;

use arithmetic_backward::*;
use arithmetic_forward::*;
use attention::*;
use convolution::*;
use dense::*;
use exponential_backward::*;
use exponential_forward::*;
use gating_backward::*;
use gating_forward::*;
use indexing::*;
use initialization::*;
use loss_backward::*;
use loss_forward::*;
use normalization_backward::*;
use normalization_forward::*;
use pooling_interpolation::*;
use positional::*;
use rectifier_backward::*;
use rectifier_forward::*;
use reduction_backward::*;
use reduction_forward::*;
use sequence::*;
use tensor_operation::*;
use trigonometric_backward::*;
use trigonometric_forward::*;
use unary_backward::*;
use unary_forward::*;

criterion_group!(
    benches,
    bench_maximum_forward,
    bench_minimum_forward,
    bench_remainder_forward,
    bench_uniform_initializer,
    bench_sinusoidal_encoding_forward,
    bench_rotary_embedding_forward,
    bench_sequential_composition_forward,
    bench_linear_forward,
    bench_layernorm_forward,
    bench_rmsnorm_forward,
    bench_batchnorm1d_eval_forward,
    bench_batchnorm2d_eval_forward,
    bench_batchnorm3d_eval_forward,
    bench_groupnorm_forward,
    bench_maxpool2d_forward,
    bench_avgpool2d_forward,
    bench_maxpool3d_forward,
    bench_avgpool3d_forward,
    bench_interpolate2d_nearest_forward,
    bench_interpolate2d_bilinear_forward,
    bench_bilinear_forward,
    bench_conv1d_forward,
    bench_conv1d_forward_backward,
    bench_conv2d_forward,
    bench_conv2d_forward_backward,
    bench_conv3d_forward,
    bench_conv_transpose1d_forward,
    bench_conv_transpose3d_forward,
    bench_mha_forward,
    bench_mha_cross_attention_forward,
    bench_transformer_encoder_forward,
    bench_embedding_forward,
    bench_embeddingbag_sum,
    bench_linear_forward_backward,
    bench_lstm_forward,
    bench_gru_forward,
    bench_rnn_forward,
    bench_rnn_cell_forward,
    bench_bidirectional_rnn_forward,
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
    bench_local_response_norm_forward,
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
    bench_pow2_forward,
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
    bench_var2_forward,
    bench_hardshrink_forward,
    bench_mul_forward,
    bench_div_forward,
    bench_add_forward,
    bench_sub_forward,
    bench_glu2_forward,
    bench_leaky_relu2_forward,
    bench_softshrink_forward,
    bench_softsign_forward,
    bench_sign_forward,
    bench_hardsigmoid_forward,
    bench_celu_forward,
    bench_tanh_backward,
    bench_sigmoid_backward,
    bench_softplus2_forward,
    bench_hardswish_forward,
    bench_gelu_backward,
    bench_silu_backward,
    bench_relu_backward,
    bench_prelu2_forward,
    bench_hardtanh_forward,
    bench_threshold_forward,
    bench_mish_backward,
    bench_softsign_backward,
    bench_elu_backward,
    bench_celu_backward,
    bench_selu_backward,
    bench_exp2_backward,
    bench_log2_backward,
    bench_atan_backward,
    bench_sinh_backward,
    bench_cosh_backward,
    bench_erf_backward,
    bench_erfc_backward,
    bench_atanh_backward,
    bench_asinh_backward,
    bench_acosh_backward,
    bench_expm1_backward,
    bench_log1p_backward,
    bench_log10_backward,
    bench_acos_backward,
    bench_asin_backward,
    bench_sin_backward,
    bench_cos_backward,
    bench_tan_backward,
    bench_sqrt_backward,
    bench_log_backward,
    bench_exp_backward,
    bench_softmax_backward,
    bench_log_softmax_backward,
    bench_cumsum_backward,
    bench_matmul_backward,
    bench_sqrt_backward2,
    bench_abs_backward,
    bench_recip_backward,
    bench_neg_backward,
    bench_pow_backward,
    bench_flip_backward,
    bench_permute_backward,
    bench_tile_backward,
    bench_l1_loss_backward,
    bench_bce_with_logits_backward,
    bench_huber_loss_backward,
    bench_kl_div_backward,
    bench_clamp_backward,
    bench_neg_backward2,
    bench_cumprod_backward,
    bench_prod_backward,
    bench_std_backward,
    bench_norm_backward,
    bench_group_norm_forward,
    bench_abs_backward2,
    bench_sort_backward,
    bench_log_sum_exp_backward,
    bench_softmin_backward,
    bench_scalar_mul_backward,
    bench_exp3_forward,
    bench_log3_forward,
    bench_tanh3_forward,
    bench_sigmoid3_forward,
    bench_relu3_forward,
    bench_sqrt3_forward,
    bench_abs3_forward,
    bench_sin3_forward,
    bench_cos3_forward,
    bench_exp4_forward,
    bench_log4_forward,
    bench_sin4_forward,
    bench_cos4_forward,
    bench_tanh4_forward,
    bench_sigmoid4_forward,
    bench_relu4_forward,
    bench_sqrt4_forward
);
criterion_main!(benches);
