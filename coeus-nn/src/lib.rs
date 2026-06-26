//! Neural network layer module system built on [`coeus_autograd`].
//!
//! # Module trait
//! [`Module<T, B>`](module::Module) is the core abstraction: `forward(&self, input: &Var<T, B>) -> Var<T, B>`.
//!
//! # Layer families
//! - **Linear** — [`Linear`], weight + optional bias, Xavier/Kaiming init via [`init`].
//! - **Convolution** — [`Conv1d`], [`Conv2d`], [`Conv3d`] with stride/padding/dilation.
//! - **Normalization** — [`LayerNorm`], [`RMSNorm`], [`BatchNorm1d/2d/3d`](BatchNorm2d), [`GroupNorm`], [`InstanceNorm1d/2d`](InstanceNorm2d).
//! - **Pooling** — [`MaxPool2d`], [`AvgPool2d`], [`MaxPool3d`], [`AvgPool3d`].
//! - **Attention** — [`MultiHeadAttention`], [`ScaledDotProductAttention`] with [`CausalMask`] / [`NullMask`].
//! - **Transformer** — [`TransformerEncoder`], [`TransformerDecoder`], [`FeedForward`] blocks.
//! - **Positional** — [`SinusoidalEncoding`], [`RotaryEmbedding`].
//! - **Composites** — [`Sequential`], [`StaticSeq`], [`Dropout`], [`Embedding`], [`Softmax`].

// ── Coeus NN ──
// Neural network building blocks.
#![allow(
    clippy::needless_range_loop,
    clippy::get_first,
    clippy::manual_range_contains,
    clippy::type_complexity
)]

pub mod activation;
pub mod attention;
pub mod bilinear;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod init;
pub mod interpolate;
pub mod linear;
pub mod loss;
pub mod module;
pub mod normalization;
pub mod parameter;
pub mod pool;
pub mod positional;
pub mod rnn;
pub mod sequential;
pub mod softmax;
pub mod transformer;

pub use activation::{
    elu, gelu, gelu_tanh, leaky_relu, mish, relu, sigmoid, silu, softplus, tanh, GeLU, GeLUTanh,
    LeakyReLU, Mish, ReLU, SiLU, Sigmoid, Softplus, Tanh, ELU,
};
pub use attention::{
    AttentionMask, CausalMask, MultiHeadAttention, NullMask, ScaledDotProductAttention,
};
pub use bilinear::Bilinear;
pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use init::{kaiming_uniform, xavier_uniform};
pub use interpolate::{interpolate_1d, interpolate_2d, InterpolateMode};
pub use linear::Linear;
pub use loss::{
    binary_cross_entropy, cosine_embedding_loss, cross_entropy_loss, huber_loss, mse_loss, nll_loss,
};
pub use module::Module;
pub use normalization::{
    group_norm, BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm1d, InstanceNorm2d,
    LayerNorm, RMSNorm,
};
pub use parameter::Parameter;
pub use pool::{
    AvgPool2d, AvgPool3d, GlobalAvgPool1d, GlobalAvgPool2d, GlobalAvgPool3d, GlobalMaxPool2d,
    GlobalMaxPool3d, MaxPool2d, MaxPool3d,
};
pub use positional::{RotaryEmbedding, SinusoidalEncoding};
pub use rnn::{GRUCell, Gru, LSTMCell, Lstm};
pub use sequential::{ModuleExt, Sequential, StaticSeq};
pub use softmax::{softmax, Softmax};
pub use transformer::{
    FeedForward, Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer,
};
