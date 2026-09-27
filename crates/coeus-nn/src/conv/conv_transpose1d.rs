//! 1-D transposed (fractional-stride) convolution.

use coeus_core::MoiraiBackend;

/// 1-D Transposed Convolution layer.
///
/// Weight convention: `[C_in, C_out, K]` (groups=1; opposite of Conv1d).
/// The forward pass delegates through `ConvOps::conv_transpose1d`; CPU
/// backends execute in Leto and accelerator backends execute in Hephaestus.
///
/// Alias for `ConvTranspose<T, B, 1>`.
pub type ConvTranspose1d<T, B = MoiraiBackend> = super::conv_transpose::ConvTranspose<T, B, 1>;
