// ── Tracked autograd ops ──

/// Activation functions (ReLU, sigmoid, GELU, etc.).
pub mod activation;
/// Arithmetic operations (add, sub, mul, div, reductions, scalar ops).
pub mod arithmetic;
/// Embedding lookup operations.
pub mod embedding;
/// Differentiable coordinate-grid interpolation.
pub mod interpolation;
/// Linear algebra operations (matmul, sparse matmul, transpose).
pub mod linalg;
/// Neural network layers (conv, pooling, normalization, loss, attention).
pub mod nn;
/// Reduction operations (norm, log-sum-exp, min/max over axes).
pub mod reduction;
/// Differentiable selective scan (Mamba/S6 linear state-space recurrence).
pub mod scan;
/// Shape manipulation operations (reshape, permute, cat, split, etc.).
pub mod shape;
/// Variable-level operation helpers.
pub mod var_ops;

pub use activation::{
    abs, acos, acosh, asin, asinh, atan, atanh, ceil, celu, clamp, cos, cosh, elu, erf, erfc, exp,
    exp2, expm1, floor, gelu, gelu_tanh, hardshrink, hardsigmoid, hardswish, hardtanh, leaky_relu,
    lgamma_forward, log, log10, log1p, log2, mish, neg, pack_pairs, pow, prelu, recip, relu, round,
    selu, sigmoid, sign, silu, sin, sinh, softplus, softshrink, softsign, sqrt, tan, tanh,
    threshold, trunc,
};
pub use arithmetic::{
    add, div, eq, ge, gt, le, lt, maximum, mean, mean_axis, minimum, mul, nanmean, nansum, ne,
    remainder, scalar_add, scalar_div, scalar_mul, scalar_sub, sub, sum, sum_axis,
};
pub use reduction::{
    log_sum_exp, max_axis, min_axis, norm, norm_p, norm_p_axis, prod, sort, std_dev, std_dev_axis,
    std_mean, std_mean_axis, topk, var, var_axis, var_mean, var_mean_axis,
};

pub use arithmetic::VarScalarExt;
pub use linalg::{matmul, sparse_matmul, sparse_matmul_coo, transpose_2d};
pub use nn::{
    avg_pool1d, avg_pool2d, avg_pool3d, batchnorm1d, batchnorm2d, batchnorm3d, bce_with_logits,
    binary_cross_entropy, causal_softmax, conv1d, conv2d, conv3d, conv_transpose1d,
    conv_transpose2d, conv_transpose3d, cosine_embedding_loss, cosine_similarity,
    cross_entropy_loss, ctc_loss, dropout, fold1d, fold2d, huber_loss, kl_divergence, l1_loss,
    layernorm, log_softmax, margin_ranking_loss, masked_softmax, max_pool1d, max_pool2d,
    max_pool3d, multi_label_margin_loss, multi_margin, nll_loss, pairwise_distance, poisson_nll,
    rmsnorm, sdp_attention, smooth_l1_loss, soft_margin, softmax, softmin, unfold1d, unfold2d,
    AttentionMask, BatchNormArgs, CausalMask, NullMask,
};

pub use embedding::{embedding, embedding_with_padding_idx};
pub use interpolation::{grid_sample_3d, linear_interpolation};
pub use scan::selective_scan;
pub use shape::{
    broadcast_to, cat, contiguous, cumprod, cumsum, diag, diagonal, diff, einsum, einsum3, flatten,
    flip, gather, index_put, index_select, masked_fill, movedim, pad, permute, reshape, roll,
    scatter_add, slice, split, squeeze, stack, swapaxes, tile, transpose, tril, triu, unsqueeze,
    where_cond,
};
