// ── Coeus Ops ──
// Tensor operations: unary, binary, matmul, reductions, FFT.
#![allow(clippy::needless_range_loop)]

pub(crate) mod ptr;
pub mod backend_ops;
pub use backend_ops::{BackendOps, BinaryOp, UnaryOp, ReductionOp, CpuBackend};
pub mod unary;
pub mod binary;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod fft;
pub mod sparse;
pub mod embedding;

pub use unary::{
    sin, cos, exp, log, relu, gelu, sigmoid, tanh, silu, mish, neg, abs, sqrt, elementwise_unary,
    sin_assign, cos_assign, exp_assign, log_assign, relu_assign, gelu_assign, sigmoid_assign, tanh_assign, silu_assign, mish_assign, neg_assign, abs_assign, sqrt_assign, elementwise_unary_assign,
    elementwise_unary_to,
    elu, elu_assign, softplus, softplus_assign, gelu_tanh, gelu_tanh_assign, leaky_relu, leaky_relu_assign,
    log_softmax_axis,
};

pub use sparse::{
    spmv, spmm, spmm_backward_values, spmm_backward_dense,
    dense_to_coo, coo_to_dense, coo_to_csr, dense_to_csr, csr_to_dense,
};
pub use binary::{add, sub, mul, div, add_assign, sub_assign, mul_assign, div_assign, elementwise_binary, elementwise_binary_to};
pub use matmul::matmul;
pub use reduction::{sum, mean, sum_axis, mean_axis, max_axis, min_axis, cumsum, suffix_sum, topk, argmax, argmin};
pub use shape::{cat, split, pad};
pub use fft::{fft_1d, ifft_1d, FftScalar};
pub use embedding::{embedding, embedding_backward};

pub mod fuse;
pub use fuse::{Expr, ExprNode, TensorExprExt, scalar, evaluate_fused_cpu, evaluate_fused_reduce_cpu};

pub mod attention;
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_backward};
