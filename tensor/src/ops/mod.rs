//! Tensor operations modules.
//!
//! This module contains all tensor operations organized by category.

pub mod activation;
pub mod arithmetic;
pub mod classification;

pub mod comparison;
pub mod conv;
pub mod creation;
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
pub mod inplace; // In-place operations

pub mod tensor_ops; // Generic tensor ops (stack, etc)

// Re-export unified dispatch trait
pub use dispatch::TensorStorageOps;

// Re-export convenience functions from the new hierarchy
pub use activation::{gelu, leaky_relu, relu, sigmoid, tanh};
pub use arithmetic::{add, div, mul, neg, sub};
pub use classification::{cross_entropy, nll_loss, softmax};
pub use linalg::{addmm, addr, bmm, cholesky, eig, eigh, matmul, matrix_exp, matrix_power, mv, qr, svd};
pub use math::{
    abs, acos, acosh, asin, asinh, atan, atan2, atanh, ceil, clamp_, clamp_max_, clamp_min_,
    copysign, cos, cosh, cumprod, cumsum, deg2rad, erf, erfc, erfinv, exp, exp2, expm1, floor,
    fmod, frac, hypot, ldexp, lerp, lerp_scalar, log, log10, log1p, log2, nan_to_num, pow, rad2deg,
    reciprocal, remainder, renorm, round, rsqrt, sign, signbit, sin, sinh, sort, sqrt, square, tan,
    topk, trunc, unique,
};
pub use reduction::{all, any, argmax, argmin, max, mean, min, std, sum, var};
pub use shape::{cat, flatten, permute, reshape, squeeze, transpose, unsqueeze};

pub use indexing::{gather, index_select, scatter};
pub use rnn::rnn;

// Re-export old logic temporarily if needed (or prefer new hierarchy)
pub use tensor_ops::{concatenate_tensors, stack_tensors};
