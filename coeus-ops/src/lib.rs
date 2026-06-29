//! Tensor operation kernels and backend dispatch for the Coeus stack.
//!
//! # Operation families
//! - **Elementwise** — [`unary`] and [`binary`] kernels dispatched via [`ElementwiseOps::elementwise_unary`] / [`ElementwiseOps::elementwise_binary`].
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
#![deny(missing_docs)]
/// Backend dispatch trait and CPU implementation.
pub mod backend_ops;
pub(crate) mod ptr;
pub use backend_ops::{
    AttentionOps, BackendOps, BinaryOp, ConvOps, CpuBackend, ElementwiseOps, MatmulOps,
    OptimizerOps, PoolOps, ReductionOp, ReductionOps, UnaryOp, UnfoldFoldOps,
};
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

/// Unfold 1D: extracts sliding windows from `[N, C, L]` into `[N, C*kernel_size, L_out]`.
///
/// Equivalent to `torch.nn.Unfold` in 1D.  Output size:
/// `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`.
pub fn unfold1d<T: coeus_core::Scalar, B: BackendOps<T> + Default>(
    input: &coeus_tensor::Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    backend: &B,
) -> coeus_tensor::Tensor<T, B> {
    let shape = input.shape();
    let (n, c, l) = (shape[0], shape[1], shape[2]);
    let l_out = (l + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let ck = c * kernel_size;
    let mut out = coeus_tensor::Tensor::zeros_on([n, ck, l_out], backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.unfold1d(
        input.storage(),
        input.layout(),
        kernel_size,
        stride,
        padding,
        dilation,
        out_storage,
        out_layout,
    );
    out
}

/// Fold 1D: accumulates `[N, C*kernel_size, L_out]` back into `[N, C, output_size]`.
///
/// Inverse (adjoint) of `unfold1d`; overlapping window contributions are summed.
pub fn fold1d<T: coeus_core::Scalar, B: BackendOps<T> + Default>(
    input: &coeus_tensor::Tensor<T, B>,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    backend: &B,
) -> coeus_tensor::Tensor<T, B> {
    let shape = input.shape();
    let n = shape[0];
    let c = shape[1] / kernel_size;
    let mut out = coeus_tensor::Tensor::zeros_on([n, c, output_size], backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.fold1d(
        input.storage(),
        input.layout(),
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
        out_storage,
        out_layout,
    );
    out
}

/// Unfold 2D: extracts sliding windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
///
/// Equivalent to `torch.nn.Unfold`.
#[allow(clippy::too_many_arguments)]
pub fn unfold2d<T: coeus_core::Scalar, B: BackendOps<T> + Default>(
    input: &coeus_tensor::Tensor<T, B>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    backend: &B,
) -> coeus_tensor::Tensor<T, B> {
    let shape = input.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = (h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    let w_out = (w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    let ckk = c * kernel_h * kernel_w;
    let l_out = h_out * w_out;
    let mut out = coeus_tensor::Tensor::zeros_on([n, ckk, l_out], backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.unfold2d(
        input.storage(),
        input.layout(),
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        out_storage,
        out_layout,
    );
    out
}

/// Fold 2D: accumulates `[N, C*kH*kW, H_out*W_out]` back into `[N, C, output_h, output_w]`.
///
/// Inverse (adjoint) of `unfold2d`; overlapping window contributions are summed.
#[allow(clippy::too_many_arguments)]
pub fn fold2d<T: coeus_core::Scalar, B: BackendOps<T> + Default>(
    input: &coeus_tensor::Tensor<T, B>,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    backend: &B,
) -> coeus_tensor::Tensor<T, B> {
    let shape = input.shape();
    let n = shape[0];
    let c = shape[1] / (kernel_h * kernel_w);
    let mut out = coeus_tensor::Tensor::zeros_on([n, c, output_h, output_w], backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.fold2d(
        input.storage(),
        input.layout(),
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        out_storage,
        out_layout,
    );
    out
}
