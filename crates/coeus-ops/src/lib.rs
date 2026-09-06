//! Tensor operation kernels and backend dispatch for the Coeus stack.
//!
//! # Operation families
//! - **Elementwise** — [`unary`] and [`binary`] kernels dispatched via [`ElementwiseOps::elementwise_unary`] / [`ElementwiseOps::elementwise_binary`].
//! - **Linear algebra** — [`matmul()`], [`sparse`] SpMM/SpMV.
//! - **Reductions** — [`reduction`]: `sum`, `mean`, `max/min`, `argmax/argmin`, `topk`, cumulative sum/product scans, plus vector arithmetic `dot` (flat inner product) and `cross` (per-channel 3-vector cross along `dim`), plus matrix norms `frobenius_norm` / `frobenius_norm_batched` (compose on `norm` for `torch.linalg.matrix_norm(A, ord='fro')`).
//! - **Convolution** — 1-D/2-D/3-D forward+backward routed through `BackendOps::conv1d`/`conv2d`/`conv3d`.
//! - **Pooling** — max and average pooling (2-D/3-D) with backward gradients.
//! - **Attention** — [`attention::scaled_dot_product_attention`] with causal/padding mask support.
//! - **Fused expressions** — [`fuse`] lazy expression DAG evaluated in a single pass on CPU.
//! - **Optimizer steps** — provider-dispatched `sgd_step`, `adam_step`,
//!   `adamw_step`, `rmsprop_step`, and `adagrad_step` on [`OptimizerOps`].

// ── Coeus Ops ──
// Tensor operations: unary, binary, matmul, reductions, FFT.
#![allow(clippy::needless_range_loop)]
#![deny(missing_docs)]
/// Backend dispatch trait and CPU implementation.
pub mod backend_ops;
pub(crate) mod ptr;
pub use backend_ops::{
    AttentionOps, AttentionScalar, BackendOps, BinaryOp, ConvOps, ConvolutionBackward,
    ConvolutionForward, CpuBackend, CrossEntropyOps, ElementwiseOps, FiniteDifference3DOps,
    FiniteDifference3DScheme, FiniteDifferenceAxis, MatmulOps, OptimizerOps, OptimizerStateRef,
    OptimizerStepRule, OptimizerStepValidation, PoolOps, RandomInitOps, ReductionOp, ReductionOps,
    RotateHalfOps, ScalarPowerOps, StaggeredPairOps, UnaryOp, UnfoldFoldOps,
};
/// Element-wise binary operations (add, sub, mul, div).
pub mod binary;
/// Embedding lookup and backward pass.
pub mod embedding;
/// Coordinate-grid interpolation operations.
pub mod interpolation;
/// Matrix multiplication kernels (matmul, bmm, outer).
pub mod matmul;
/// Reduction operations (sum, mean, max, min, norms, variance).
pub mod reduction;
/// Provider-selected half-vector rotation.
pub mod rotate_half;
/// Shape manipulation operations (cat, stack, reshape, flip, sort, etc.).
pub mod shape;
/// Sparse tensor operations (SpMM, SpMV, format conversions).
pub mod sparse;
/// Element-wise unary operations (activations, math functions).
pub mod unary;

pub use unary::{
    abs, abs_assign, acos, acosh, asin, asinh, atan, atanh, causal_softmax, ceil, ceil_assign, cos,
    cos_assign, cosh, elementwise_unary, elementwise_unary_assign, elementwise_unary_to, elu,
    elu_assign, erf, erfc, exp, exp2, exp_assign, expm1, floor, floor_assign, gelu, gelu_assign,
    gelu_tanh, gelu_tanh_assign, glu, leaky_relu, leaky_relu_assign, lgamma, log, log10, log1p,
    log2, log_assign, log_softmax_axis, masked_softmax, mish, mish_assign, neg, neg_assign,
    pow_scalar, recip, recip_assign, relu, relu_assign, round, round_assign, sigmoid,
    sigmoid_assign, sign, sign_assign, silu, silu_assign, sin, sin_assign, sinh, softplus,
    softplus_assign, sqrt, sqrt_assign, tan, tanh, tanh_assign, trunc, trunc_assign,
};

pub use binary::{
    add, add_assign, div, div_assign, elementwise_binary, elementwise_binary_to, eq, ge, gt, le,
    lt, mul, mul_assign, ne, sub, sub_assign,
};
pub use embedding::{embedding, embedding_backward, embedding_backward_with_padding_idx};
pub use interpolation::{
    linear_interpolation, linear_interpolation_backward, BoundaryPolicy, Dimension,
    InterpolationError, InterpolationGradients, Replicate, SupportedDimension,
};
pub use matmul::{bmm, matmul, matmul_accumulate, outer};
pub use reduction::{
    amax, amin, argmax, argmin, cross, cumprod, cumsum, dot, frobenius_norm,
    frobenius_norm_batched, max_axis, mean, mean_axis, min_axis, norm, norm_p, norm_p_axis,
    norm_p_tensor, prod, prod_axis, prod_tensor, std_dev, std_dev_axis, std_mean, std_mean_axis,
    suffix_prod, suffix_sum, sum, sum_axis, topk, var, var_axis, var_mean, var_mean_axis,
};
pub use rotate_half::rotate_half;
pub use shape::{
    broadcast_to, cat, chunk, diag, diagonal, einsum, einsum3, flip, gather, index_put,
    index_select, masked_fill, masked_select, meshgrid, nonzero, one_hot, pad, repeat_interleave,
    roll, scatter_add, sort, split, stack, tile, tril, triu, where_cond,
};
pub use sparse::{
    coo_to_csr, coo_to_dense, csr_to_dense, dense_to_coo, dense_to_csr, spmm, spmm_backward_dense,
    spmm_backward_values, spmv,
};

/// Device-neutral fused expression DAG with CPU-addressable evaluation support.
pub mod fuse;
pub use fuse::{
    evaluate_fused_cpu, evaluate_fused_reduce_cpu, scalar, CpuExprNode, Expr, ExprNode,
    TensorExprExt,
};

/// Scaled dot-product attention with causal and padding mask support.
pub mod attention;
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_backward};

/// Transposed convolution operations (1-D, 2-D, and 3-D).
pub mod conv_transpose;
pub use conv_transpose::{conv_transpose1d, conv_transpose2d, conv_transpose3d};

/// Adaptive pooling operations (avg and max in 1D/2D).
pub mod adaptive_pool;
pub use adaptive_pool::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_max_pool1d, adaptive_max_pool2d,
};

/// Tensor constructors (linspace, logspace, geomspace).
pub mod constructors;
pub use constructors::{geomspace, linspace, logspace};

/// Sliding-window extraction and adjoint accumulation.
pub mod unfold_fold;
pub use unfold_fold::{fold1d, fold2d, unfold1d, unfold2d};
