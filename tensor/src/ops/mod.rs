//! Tensor operations modules.
//!
//! This module contains all tensor operations organized by category.

pub mod activation;
pub mod arithmetic;
pub mod classification;

pub mod comparison;
pub mod conv;
pub mod creation;
pub mod cast;
pub mod dispatch; // Unified storage dispatch
pub mod indexing; // Indexing operations (gather, scatter, etc)
pub mod linalg;
pub mod math;
pub mod pooling;
pub mod reduction;
pub mod rnn;
pub mod shape;

pub mod sparse; // Sparse specific ops

pub mod bitwise; // Bitwise operations
pub use bitwise::{logical_and, logical_or, logical_xor, logical_not};
pub mod inplace; // In-place operations

pub use inplace::*;

pub mod tensor_ops; // Generic tensor ops (stack, etc)

// Re-export unified dispatch trait
pub use dispatch::TensorStorageOps;

// Re-export convenience functions from the new hierarchy
pub use activation::{gelu, leaky_relu, relu, sigmoid, tanh};
pub use arithmetic::{add, div, mul, neg, sub};
pub use classification::{cross_entropy, nll_loss, softmax, log_softmax};
pub use linalg::{addbmm, addmm, addmv, addr, baddbmm, bilinear, bmm, cholesky, eig, eigh, matmul, matrix_exp, matrix_power, mv, outer, qr, svd};
pub use math::{
    abs, acos, acosh, asin, asinh, atan, atan2, atanh, ceil, clamp_, clamp_max_, clamp_min_,
    copysign, cos, cosh, erf, erfc, erfinv, exp, exp2, expm1, floor, fmod, frac, hypot, ldexp,
    lerp, lerp_scalar, log, log10, log1p, log2, nan_to_num, norm, pairwise_distance, pow,
    rad2deg, reciprocal, remainder, renorm, round, rsqrt, sign, signbit, sin, sinh, sort,
    sqrt, square, tan, topk, trunc, unique, cosine_similarity, deg2rad, cumsum, cumprod,
};
pub use comparison::{allclose, eq, ge, gt, isclose, isnan, isinf, isfinite, le, lt, ne};
pub use comparison::where_cond::where_cond;
pub use arithmetic::scalar::{add_scalar, div_scalar, mul_scalar, sub_scalar};
// pub use comparison::scalar::{eq_scalar, ge_scalar, gt_scalar, le_scalar, lt_scalar, ne_scalar};
// pub use comparison::maximum_scalar::maximum_scalar;
// pub use comparison::minimum_scalar::minimum_scalar;
pub use reduction::{all, any, argmax, argmin, max, mean, min, std, sum, var};
pub use shape::{cat, flatten, permute, reshape, squeeze, transpose, unsqueeze};

pub use indexing::{gather, index_select, scatter};
pub use rnn::rnn;

// Re-export old logic temporarily if needed (or prefer new hierarchy)
pub use tensor_ops::{concatenate_tensors, stack_tensors};
