// ── Tracked autograd ops ──

pub mod activation;
pub mod arithmetic;
pub mod embedding;
pub mod linalg;
pub mod nn;
pub mod reduction;
pub mod shape;
pub mod var_ops;

pub use activation::{
    abs, clamp, cos, elu, exp, gelu, gelu_tanh, leaky_relu, log, mish, neg, pow, relu, sigmoid,
    silu, sin, softplus, sqrt, tanh,
};
pub use arithmetic::{
    add, div, mean, mean_axis, mul, scalar_add, scalar_div, scalar_mul, scalar_sub, sub, sum,
    sum_axis,
};
pub use reduction::{log_sum_exp, max_axis, min_axis};

pub use linalg::{matmul, sparse_matmul, transpose_2d};
pub use nn::{
    avg_pool2d, avg_pool3d, batchnorm1d, batchnorm2d, batchnorm3d, binary_cross_entropy, conv1d,
    conv2d, conv3d, cosine_embedding_loss, cross_entropy_loss, dropout, huber_loss, layernorm,
    log_softmax, max_pool2d, max_pool3d, nll_loss, rmsnorm, sdp_attention, softmax, AttentionMask,
    BatchNorm1dArgs, BatchNorm2dArgs, BatchNorm3dArgs, CausalMask, NullMask,
};

pub use embedding::embedding;
pub use shape::{
    broadcast_to, cat, contiguous, cumsum, flip, gather, masked_fill, pad, permute, reshape, roll,
    slice, split, squeeze, stack, transpose, tril, triu, unsqueeze, where_cond,
};
