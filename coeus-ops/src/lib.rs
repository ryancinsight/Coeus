//! Tensor operation kernels and backend dispatch for the Coeus stack.
//!
//! # Operation families
//! - **Elementwise** — [`unary`] and [`binary`] kernels dispatched via [`BackendOps::elementwise_unary`] / [`BackendOps::elementwise_binary`].
//! - **Linear algebra** — [`matmul()`], [`sparse`] SpMM/SpMV, and FFT via Bluestein/Cooley-Tukey.
//! - **Reductions** — [`reduction`]: `sum`, `mean`, `max/min`, `argmax/argmin`, `topk`, `cumsum`, plus vector arithmetic `dot` (flat inner product) and `cross` (per-channel 3-vector cross along `dim`), plus matrix norms `frobenius_norm` / `frobenius_norm_batched` (compose on `norm` for `torch.linalg.matrix_norm(A, ord='fro')`).
//! - **Convolution** — 1-D/2-D/3-D forward+backward routed through `BackendOps::conv1d`/`conv2d`/`conv3d`.
//! - **Pooling** — max and average pooling (2-D/3-D) with backward gradients.
//! - **Attention** — [`attention::scaled_dot_product_attention`] with causal/padding mask support.
//! - **Fused expressions** — [`fuse`] lazy expression DAG evaluated in a single pass on CPU.
//! - **Optimizer steps** — fused `sgd_step`, `adam_step`, `adamw_step`, `rmsprop_step`, `adagrad_step` on [`BackendOps`].

// ── Coeus Ops ──
// Tensor operations: unary, binary, matmul, reductions, FFT.
#![allow(clippy::needless_range_loop)]
/// Backend dispatch trait and CPU implementation.
pub mod backend_ops;
pub(crate) mod ptr;
pub use backend_ops::{BackendOps, BinaryOp, CpuBackend, ReductionOp, UnaryOp};
/// Element-wise binary operations (add, sub, mul, div).
pub mod binary;
/// Embedding lookup and backward pass.
pub mod embedding;
/// Matrix multiplication kernels (matmul, bmm, outer).
pub mod matmul;
/// Reduction operations (sum, mean, max, min, norms, variance).
pub mod reduction;
/// Shape manipulation operations (cat, stack, reshape, flip, sort, etc.).
pub mod shape;
/// Sparse tensor operations (SpMM, SpMV, format conversions).
pub mod sparse;
/// Element-wise unary operations (activations, math functions).
pub mod unary;

pub use unary::{
    abs, abs_assign, causal_softmax, ceil, ceil_assign, cos, cos_assign, elementwise_unary,
    elementwise_unary_assign, elementwise_unary_to, elu, elu_assign, exp, exp_assign, floor,
    floor_assign, gelu, gelu_assign, gelu_tanh, gelu_tanh_assign, glu, leaky_relu,
    leaky_relu_assign, log, log_assign, log_softmax_axis, masked_softmax, mish, mish_assign, neg,
    neg_assign, recip, recip_assign, relu, relu_assign, round, round_assign, sigmoid,
    sigmoid_assign, sign, sign_assign, silu, silu_assign, sin, sin_assign, softplus,
    softplus_assign, sqrt, sqrt_assign, tanh, tanh_assign, trunc, trunc_assign,
};

pub use binary::{
    add, add_assign, div, div_assign, elementwise_binary, elementwise_binary_to, mul, mul_assign,
    sub, sub_assign,
};
pub use embedding::{embedding, embedding_backward, embedding_backward_with_padding_idx};
pub use matmul::{bmm, matmul, matmul_accumulate, outer};
pub use reduction::{
    amax, amin, argmax, argmin, cross, cumprod, cumsum, dot, frobenius_norm,
    frobenius_norm_batched, max_axis, mean, mean_axis, min_axis, norm, norm_p, norm_p_axis, prod,
    std_dev, std_dev_axis, std_mean, std_mean_axis, suffix_sum, sum, sum_axis, topk, var, var_axis,
    var_mean, var_mean_axis,
};
pub use shape::{
    broadcast_to, cat, chunk, diag, diagonal, einsum, einsum3, flip, gather, index_put,
    index_select, masked_fill, masked_select, meshgrid, nonzero, one_hot, pad, repeat_interleave,
    roll, scatter_add, sort, split, stack, tile, tril, triu, where_cond,
};
pub use sparse::{
    coo_to_csr, coo_to_dense, csr_to_dense, dense_to_coo, dense_to_csr, spmm, spmm_backward_dense,
    spmm_backward_values, spmv,
};

/// Fused expression evaluation DAG for single-pass CPU computation.
pub mod fuse;
pub use fuse::{
    evaluate_fused_cpu, evaluate_fused_reduce_cpu, scalar, Expr, ExprNode, TensorExprExt,
};

/// Scaled dot-product attention with causal and padding mask support.
pub mod attention;
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_backward};

/// Transposed convolution operations (1-D and 2-D).
pub mod conv_transpose;
pub use conv_transpose::{conv_transpose1d, conv_transpose2d};

/// Tensor constructors (linspace, logspace, geomspace).
pub mod constructors;
pub use constructors::{geomspace, linspace, logspace};
