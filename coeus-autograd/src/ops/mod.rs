// ── Tracked autograd ops ──

/// Activation functions (ReLU, sigmoid, GELU, etc.).
pub mod activation;
/// Arithmetic operations (add, sub, mul, div, reductions, scalar ops).
pub mod arithmetic;
/// Embedding lookup operations.
pub mod embedding;
/// Linear algebra operations (matmul, sparse matmul, transpose).
pub mod linalg;
/// Neural network layers (conv, pooling, normalization, loss, attention).
pub mod nn;
/// Reduction operations (norm, log-sum-exp, min/max over axes).
pub mod reduction;
/// Shape manipulation operations (reshape, permute, cat, split, etc.).
pub mod shape;
/// Variable-level operation helpers.
pub mod var_ops;

pub use activation::{
    abs, ceil, celu, clamp, cos, elu, exp, floor, gelu, gelu_tanh, hardshrink, hardsigmoid,
    hardswish, hardtanh, leaky_relu, log, mish, neg, pack_pairs, pow, prelu, recip, relu, round,
    sigmoid, sign, silu, sin, softplus, softshrink, softsign, sqrt, tanh, threshold, trunc,
};
pub use arithmetic::{
    add, div, mean, mean_axis, mul, scalar_add, scalar_div, scalar_mul, scalar_sub, sub, sum,
    sum_axis,
};
pub use reduction::{log_sum_exp, max_axis, min_axis, norm, norm_p, norm_p_axis};

pub use arithmetic::VarScalarExt;
pub use linalg::{matmul, sparse_matmul, sparse_matmul_coo, transpose_2d};
pub use nn::{
    avg_pool2d, avg_pool3d, batchnorm1d, batchnorm2d, batchnorm3d, binary_cross_entropy, conv1d,
    conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, cosine_embedding_loss,
    cross_entropy_loss, dropout, huber_loss, kl_divergence, layernorm, log_softmax,
    margin_ranking_loss, max_pool2d, max_pool3d, nll_loss, rmsnorm, sdp_attention, softmax,
    AttentionMask, BatchNormArgs, CausalMask, NullMask,
};

pub use embedding::{embedding, embedding_with_padding_idx};
pub use shape::{
    broadcast_to, cat, contiguous, cumprod, cumsum, diag, diagonal, einsum, einsum3, flip, gather,
    index_select, masked_fill, pad, permute, reshape, roll, slice, split, squeeze, stack, tile,
    transpose, tril, triu, unsqueeze, where_cond,
};
