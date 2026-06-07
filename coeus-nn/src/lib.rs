// ── Coeus NN ──
// Neural network building blocks.
#![allow(clippy::needless_range_loop, clippy::get_first, clippy::manual_range_contains, clippy::type_complexity)]

pub mod module;
pub mod parameter;
pub mod linear;
pub mod conv;
pub mod embedding;
pub mod pool;
pub mod activation;
pub mod normalization;
pub mod dropout;
pub mod softmax;
pub mod loss;
pub mod init;
pub mod attention;
pub mod positional;
pub mod transformer;
pub mod sequential;

pub use module::Module;
pub use parameter::Parameter;
pub use linear::Linear;
pub use embedding::Embedding;
pub use conv::{Conv1d, Conv2d, Conv3d};
pub use pool::{AvgPool2d, MaxPool2d, AvgPool3d, MaxPool3d};
pub use activation::{relu, gelu, sigmoid, tanh, silu, mish, leaky_relu, elu, softplus, gelu_tanh,
                     ReLU, Sigmoid, Tanh, GeLU, SiLU, Mish, LeakyReLU, ELU, Softplus, GeLUTanh};
pub use normalization::{BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, RMSNorm,
                        GroupNorm, InstanceNorm1d, InstanceNorm2d};
pub use dropout::Dropout;
pub use softmax::{softmax, Softmax};
pub use loss::{mse_loss, cross_entropy_loss, binary_cross_entropy, nll_loss, huber_loss, cosine_embedding_loss};
pub use init::{xavier_uniform, kaiming_uniform};
pub use attention::{AttentionMask, CausalMask, NullMask, ScaledDotProductAttention, MultiHeadAttention};
pub use positional::{SinusoidalEncoding, RotaryEmbedding};
pub use transformer::{FeedForward, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer};
pub use sequential::{Sequential, StaticSeq, ModuleExt};
