// ── Tracked autograd ops ──

/// Activation functions (ReLU, sigmoid, GELU, etc.).
pub mod activation;
/// Arithmetic operations (add, sub, mul, div, reductions, scalar ops).
pub mod arithmetic;
/// Embedding lookup operations.
pub mod embedding;
/// Fast Fourier Transform operations backed by Apollo FFT.
pub mod fft;
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
    avg_pool1d, avg_pool2d, avg_pool3d, batchnorm1d, batchnorm2d, batchnorm3d, bce_with_logits,
    binary_cross_entropy, conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d,
    conv_transpose3d, cosine_embedding_loss, cosine_similarity, cross_entropy_loss, dropout,
    fold1d, fold2d, huber_loss, kl_divergence, l1_loss, layernorm, log_softmax,
    margin_ranking_loss, max_pool1d, max_pool2d, max_pool3d, multi_margin, nll_loss,
    pairwise_distance, poisson_nll, rmsnorm, sdp_attention, smooth_l1_loss, soft_margin, softmax,
    unfold1d, unfold2d, AttentionMask, BatchNormArgs, CausalMask, NullMask,
};

pub use embedding::{embedding, embedding_with_padding_idx};
pub use fft::{
    fft_1d, fft_1d_var, fft_energy, ifft_1d, ifft_1d_var, Fft1DNode, FftScalar, Ifft1DNode,
};
pub use shape::{
    broadcast_to, cat, contiguous, cumprod, cumsum, diag, diagonal, einsum, einsum3, flip, gather,
    index_select, masked_fill, pad, permute, reshape, roll, slice, split, squeeze, stack, tile,
    transpose, tril, triu, unsqueeze, where_cond,
};
