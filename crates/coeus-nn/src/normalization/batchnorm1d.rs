//! 1-D batch normalization (`[N, C]` or `[N, C, L]` inputs).
//!
//! [`BatchNorm1d`] is the fixed-rank alias for the generic
//! [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) implementation.

use coeus_core::MoiraiBackend;

/// 1D Batch Normalization for `[N, C, L]` inputs.
///
/// Normalizes over the N, L dimensions (per-channel mean/variance).
/// Running stats are updated during each forward call in training mode.
/// In eval mode (`is_training = false`) uses running_mean/running_var.
///
/// Alias for `BatchNorm<T, B, 1>`; PyTorch `nn.BatchNorm1d`'s degenerate
/// `[N, C]` (no-spatial) form is accepted by the shared implementation.
pub type BatchNorm1d<T, B = MoiraiBackend> = super::batchnorm::BatchNorm<T, B, 1>;
