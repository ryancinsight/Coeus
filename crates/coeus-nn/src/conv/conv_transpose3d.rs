//! 3-D transposed (fractional-stride) convolution.

use coeus_core::MoiraiBackend;

/// 3-D Transposed Convolution layer.
///
/// Weight convention: `[C_in, C_out, KD, KH, KW]` (groups=1; the in/out
/// channel order is reversed relative to the regular Conv3d).
/// Dispatch follows the selected backend's provider-owned convolution kernel.
///
/// Alias for `ConvTranspose<T, B, 3>`.
pub type ConvTranspose3d<T, B = MoiraiBackend> = super::conv_transpose::ConvTranspose<T, B, 3>;
