//! 3-D batch normalization (`[N, C, D, H, W]` inputs).
//!
//! [`BatchNorm3d`] is the fixed-rank alias for the generic
//! [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) implementation.

use coeus_core::MoiraiBackend;

/// 3D Batch Normalization for `[N, C, D, H, W]` inputs.
///
/// Normalizes over the N, D, H, W dimensions (per-channel mean/variance).
/// Running stats are updated during each forward call.
///
/// Alias for `BatchNorm<T, B, 3>`; see
/// [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) for the shared implementation.
pub type BatchNorm3d<T, B = MoiraiBackend> = super::batchnorm::BatchNorm<T, B, 3>;
