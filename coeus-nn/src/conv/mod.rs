/// Dimension strategy traits and ZST markers for the generic [`Conv`] layer.
pub mod dim;

mod conv_nd;
mod conv_transpose1d;
mod conv_transpose2d;

pub use conv_nd::{Conv, ConvParams};
pub use dim::{ConvDim, Dim1D, Dim2D, Dim3D};

/// 1D convolution layer; alias for `Conv<T, B, Dim1D>`.
pub type Conv1d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim1D>;
/// 2D convolution layer; alias for `Conv<T, B, Dim2D>`.
pub type Conv2d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim2D>;
/// 3D convolution layer; alias for `Conv<T, B, Dim3D>`.
pub type Conv3d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim3D>;

pub use conv_transpose1d::ConvTranspose1d;
pub use conv_transpose2d::ConvTranspose2d;
