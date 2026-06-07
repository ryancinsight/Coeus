// ── Coeus Autograd ──
// Automatic differentiation engine with computational graph.
#![allow(clippy::type_complexity, clippy::needless_range_loop, clippy::get_first)]

pub mod var;
pub mod node;
pub mod backward;
pub mod ops;

pub use var::Var;
pub use node::BackwardNode;
pub use ops::{
    add, sub, mul, div, matmul, sparse_matmul, relu, sigmoid, tanh, gelu, silu, mish, transpose_2d,
    sum, mean, sum_axis, mean_axis, exp, log, layernorm, rmsnorm, batchnorm1d, batchnorm2d, batchnorm3d,
    conv1d, conv2d, conv3d, max_pool2d, avg_pool2d, max_pool3d, avg_pool3d, softmax, dropout,
    cross_entropy_loss, binary_cross_entropy, nll_loss, huber_loss, cosine_embedding_loss,
    embedding, sdp_attention, AttentionMask, CausalMask, NullMask,
    reshape, permute, slice, contiguous, cat, split, pad, squeeze, unsqueeze, transpose, cumsum,
    elu, softplus, gelu_tanh, leaky_relu, log_softmax,
    // New unary math ops
    neg, abs, sqrt, pow, clamp,
    // New scalar arithmetic
    scalar_mul, scalar_add, scalar_sub, scalar_div,
    // New axis reductions
    max_axis, min_axis, log_sum_exp,
};
