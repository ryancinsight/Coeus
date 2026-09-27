//! 2-D batch normalization (`[N, C, H, W]` inputs).
//!
//! [`BatchNorm2d`] is the fixed-rank alias for the generic
//! [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) implementation.

use coeus_core::MoiraiBackend;

/// 2D Batch Normalization for `[N, C, H, W]` inputs.
///
/// Normalizes over the N, H, W dimensions (per-channel mean/variance).
/// Running stats are updated during each forward call.
///
/// Alias for `BatchNorm<T, B, 2>`; see
/// [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) for the shared implementation.
pub type BatchNorm2d<T, B = MoiraiBackend> = super::batchnorm::BatchNorm<T, B, 2>;
