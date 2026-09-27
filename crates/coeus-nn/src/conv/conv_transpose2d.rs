//! 2-D transposed (fractional-stride) convolution.

use coeus_core::MoiraiBackend;

/// 2-D Transposed Convolution layer.
///
/// Weight convention: `[C_in, C_out, KH, KW]` (groups=1; opposite of Conv2d).
///
/// Alias for `ConvTranspose<T, B, 2>`.
pub type ConvTranspose2d<T, B = MoiraiBackend> = super::conv_transpose::ConvTranspose<T, B, 2>;
