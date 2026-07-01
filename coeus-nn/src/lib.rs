//! Neural network layer module system built on [`coeus_autograd`].
//!
//! # Module trait
//! [`Module<T, B>`](module::Module) is the core abstraction: `forward(&self, input: &Var<T, B>) -> Var<T, B>`.
//!
//! # Layer families
//! - **Linear** — [`Linear`], weight + optional bias, Xavier/Kaiming init via [`init`].
//! - **Convolution** — [`Conv1d`], [`Conv2d`], [`Conv3d`] with stride/padding/dilation.
//! - **Normalization** — [`LayerNorm`], [`RMSNorm`], [`BatchNorm1d/2d/3d`](BatchNorm2d), [`GroupNorm`], [`InstanceNorm1d`], [`InstanceNorm2d`], [`InstanceNorm3d`].
//! - **Pooling** — [`MaxPool2d`], [`AvgPool2d`], [`MaxPool3d`], [`AvgPool3d`].
//! - **Attention** — [`MultiHeadAttention`], [`ScaledDotProductAttention`] with [`CausalMask`] / [`NullMask`].
//! - **Transformer** — [`TransformerEncoder`], [`TransformerDecoder`], [`FeedForward`] blocks.
//! - **Positional** — [`SinusoidalEncoding`], [`RotaryEmbedding`].
//! - **Composites** — [`Sequential`], [`StaticSeq`], [`Dropout`], [`Embedding`], [`Softmax`].

// ── Coeus NN ──
// Neural network building blocks.
#![deny(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::get_first,
    clippy::manual_range_contains,
    clippy::type_complexity
)]

/// Activation functions (ReLU, GeLU, SiLU, etc.).
pub mod activation;
/// Attention mechanisms (MHA, SDP, masks).
pub mod attention;
/// Bilinear interaction layer.
pub mod bilinear;
/// Convolution layers (1D/2D/3D and transposed variants).
pub mod conv;
/// Dropout regularization layer.
pub mod dropout;
/// Embedding lookup layer.
pub mod embedding;
/// EmbeddingBag aggregation layer.
pub mod embeddingbag;
/// Weight initialization utilities (Xavier, Kaiming).
pub mod init;
/// Spatial interpolation operations.
pub mod interpolate;
/// Fully-connected linear layer.
pub mod linear;
/// Loss functions (cross-entropy, MSE, etc.).
pub mod loss;
/// Core `Module` trait for neural network layers.
pub mod module;
/// Normalization layers (LayerNorm, BatchNorm, GroupNorm, etc.).
pub mod normalization;
/// Learnable parameter wrapper.
pub mod parameter;
/// Pooling layers (max, average, global).
pub mod pool;
/// Positional encoding layers (sinusoidal, RoPE).
pub mod positional;
/// Regularization layers (AlphaDropout, FeatureAlphaDropout, GaussianNoise, LocalResponseNorm).
pub mod regularization;
/// Recurrent layers (LSTM, GRU).
pub mod rnn;
/// Sequential and static-sequence module containers.
pub mod sequential;
/// Softmax layer and functional softmax.
pub mod softmax;
/// SwiGLU gated feed-forward unit.
pub mod swiglu;
/// Transformer encoder, decoder, and sub-layers.
pub mod transformer;

pub use activation::{
    celu, elu, gelu, gelu_tanh, glu, hardshrink, hardsigmoid, hardswish, hardtanh, leaky_relu,
    log_sigmoid, mish, prelu, relu, sigmoid, silu, softplus, softshrink, softsign, tanh,
    tanhshrink, threshold, Celu, CeluOp, GeLU, GeLUTanh, Hardshrink, HardshrinkOp, Hardsigmoid,
    HardsigmoidOp, Hardswish, HardswishOp, Hardtanh, HardtanhOp, LeakyReLU, LogSigmoid, Mish,
    PReLU, ReLU, SiLU, Sigmoid, Softplus, Softshrink, SoftshrinkOp, Softsign, SoftsignOp, Tanh,
    Tanhshrink, Threshold, ThresholdNode, ELU, GLU,
};
pub use attention::{
    multi_head_attention_cross, AttentionMask, CausalMask, MhaProjectionParams, MultiHeadAttention,
    NullMask, ScaledDotProductAttention,
};
pub use bilinear::{bilinear, Bilinear};
pub use conv::{
    Conv, Conv1d, Conv2d, Conv3d, ConvDim, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    Dim1D, Dim2D, Dim3D, Fold1d, Fold2d, Unfold1d, Unfold2d,
};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use embeddingbag::{EmbeddingBag, EmbeddingBagMode};
pub use init::{kaiming_uniform, xavier_uniform};
pub use interpolate::{interpolate_1d, interpolate_2d, InterpolateMode};
pub use linear::Linear;
pub use loss::{
    bce_with_logits, binary_cross_entropy, cosine_embedding_loss, cosine_similarity,
    cross_entropy_loss, ctc_loss, gaussian_nll_loss, hinge_embedding_loss, huber_loss,
    kl_divergence, l1_loss, margin_ranking_loss, mse_loss, multi_label_soft_margin_loss,
    multi_margin, nll_loss, pairwise_distance, poisson_nll, smooth_l1_loss, soft_margin,
    triplet_margin_loss, triplet_margin_with_distance_loss,
};
pub use module::Module;
pub use normalization::{
    group_norm, layer_norm, rms_norm, BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LayerNorm, RMSNorm,
};
pub use parameter::Parameter;
pub use pool::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, AvgPool3d, GlobalAvgPool1d, GlobalAvgPool2d, GlobalAvgPool3d, GlobalMaxPool2d,
    GlobalMaxPool3d, MaxPool1d, MaxPool2d, MaxPool3d,
};
pub use positional::{RotaryEmbedding, SinusoidalEncoding};
pub use regularization::{AlphaDropout, FeatureAlphaDropout, GaussianNoise, LocalResponseNorm};
pub use rnn::{Bidirectional, GRUCell, Gru, LSTMCell, Lstm, RNNCell, Rnn, RnnNonlinearity};
pub use sequential::{ModuleExt, Sequential, StaticSeq};
pub use softmax::{softmax, Softmax};
pub use swiglu::SwiGlu;
pub use transformer::{
    feed_forward, transformer_decoder_layer, transformer_encoder_layer, FeedForward, Transformer,
    TransformerDecoder, TransformerDecoderLayer, TransformerDecoderLayerParams, TransformerEncoder,
    TransformerEncoderLayer, TransformerEncoderLayerParams,
};
