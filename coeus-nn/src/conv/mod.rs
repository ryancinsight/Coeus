/// Dimension strategy traits and ZST markers for the generic [`Conv`] layer.
pub mod dim;

mod conv_nd;
mod depthwise3d;
mod conv_transpose1d;
mod conv_transpose2d;
mod conv_transpose3d;
mod unfold_fold;

pub use conv_nd::{Conv, ConvParams};
pub use depthwise3d::DepthwiseConv3d;
pub use dim::{ConvDim, Dim1D, Dim2D, Dim3D};

/// 1D convolution layer; alias for `Conv<T, B, Dim1D>`.
pub type Conv1d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim1D>;
/// 2D convolution layer; alias for `Conv<T, B, Dim2D>`.
pub type Conv2d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim2D>;
/// 3D convolution layer; alias for `Conv<T, B, Dim3D>`.
pub type Conv3d<T, B = coeus_core::MoiraiBackend> = Conv<T, B, Dim3D>;

pub use conv_transpose1d::ConvTranspose1d;
pub use conv_transpose2d::ConvTranspose2d;
pub use conv_transpose3d::ConvTranspose3d;

pub use unfold_fold::{Fold1d, Fold2d, Unfold1d, Unfold2d};
