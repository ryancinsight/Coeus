// ── Tracked neural network and loss operations module ──

/// Scaled dot-product attention and masking types.
pub mod attention;
/// Convolution layers (1D, 2D, 3D, transpose).
pub mod conv;
/// Dropout layer with tracked random masking.
pub mod dropout;
/// Log-softmax operation.
pub mod log_softmax;
/// Loss functions (BCE, cross-entropy, NLL, Huber, cosine embedding).
pub mod loss;
/// Normalization layers (batchnorm, layernorm, rmsnorm).
pub mod normalization;
/// Pooling layers (max-pool, avg-pool 2D/3D).
pub mod pool;
/// Softmax operation.
pub mod softmax;

pub use attention::{sdp_attention, AttentionMask, CausalMask, NullMask};
pub use conv::{conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d};
pub use dropout::dropout;
pub use log_softmax::log_softmax;
pub use loss::{
    bce_with_logits, binary_cross_entropy, cosine_embedding_loss, cross_entropy_loss, huber_loss,
    kl_divergence, l1_loss, margin_ranking_loss, nll_loss, poisson_nll,
};
pub use normalization::{batchnorm1d, batchnorm2d, batchnorm3d, layernorm, rmsnorm, BatchNormArgs};
pub use pool::{avg_pool2d, avg_pool3d, max_pool2d, max_pool3d};
pub use softmax::softmax;
