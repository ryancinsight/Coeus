// ── Tracked autograd ops ──

pub mod arithmetic;
pub mod activation;
pub mod linalg;
pub mod nn;
pub mod embedding;
pub mod shape;
pub mod var_ops;
pub mod reduction;

pub use arithmetic::{add, sub, mul, div, sum, mean, sum_axis, mean_axis, scalar_mul, scalar_add, scalar_sub, scalar_div};
pub use activation::{relu, sigmoid, tanh, gelu, silu, mish, exp, log, elu, softplus, gelu_tanh, leaky_relu,
                     neg, abs, sqrt, pow, clamp};
pub use reduction::{max_axis, min_axis, log_sum_exp};

pub use linalg::{matmul, transpose_2d, sparse_matmul};
pub use nn::{layernorm, rmsnorm, batchnorm1d, batchnorm2d, batchnorm3d, conv1d, conv2d, conv3d, max_pool2d, avg_pool2d, max_pool3d, avg_pool3d, softmax, dropout, cross_entropy_loss, binary_cross_entropy, nll_loss, huber_loss, sdp_attention, AttentionMask, CausalMask, NullMask, log_softmax, cosine_embedding_loss};

pub use embedding::embedding;
pub use shape::{reshape, permute, slice, contiguous, cat, split, pad, squeeze, unsqueeze, transpose, cumsum};
