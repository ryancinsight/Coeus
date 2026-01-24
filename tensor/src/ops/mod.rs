//! Tensor operations modules.
//!
//! This module contains all tensor operations organized by category.

pub mod activation;
pub mod arithmetic;
pub mod classification;

pub mod comparison;
pub mod creation;
pub mod dispatch; // Unified storage dispatch
pub mod indexing; // Indexing operations (gather, scatter, etc)
pub mod shape;
pub mod linalg;
pub mod math;
pub mod reduction;
pub mod rnn;

pub mod sparse; // Sparse specific ops

pub mod bitwise; // Bitwise operations
pub mod inplace; // In-place operations

pub mod tensor_ops; // Generic tensor ops (stack, etc)


// Re-export unified dispatch trait
pub use dispatch::TensorStorageOps;

// Re-export convenience functions from the new hierarchy
pub use arithmetic::{add, div, mul, neg, sub};
pub use activation::{gelu, leaky_relu, relu, sigmoid, tanh};
pub use classification::{cross_entropy, nll_loss, softmax};
pub use shape::{cat, flatten, reshape, squeeze, transpose, unsqueeze};
pub use linalg::{addmm, addr, bmm, matmul, mv};
pub use math::{
    abs, acos, acosh, asin, asinh, atan, atan2, atanh, ceil, cos, cosh, cumprod, cumsum, deg2rad,
    erf, erfc, erfinv, exp, exp2, expm1, floor, frac, log, log1p, log2, log10, nan_to_num, pow,
    rad2deg, reciprocal, round, rsqrt, sign, signbit, sin, sinh, sort, sqrt, tan, topk, trunc, unique,
};
pub use inplace::{abs_, add_, div_, fill_, mul_, sub_, zero_};
pub use reduction::{all, any, max, mean, min, std, sum, var};

pub use indexing::{gather, index_select, scatter};
pub use rnn::rnn;

// Re-export old logic temporarily if needed (or prefer new hierarchy)
pub use tensor_ops::{concatenate_tensors, stack_tensors};

