mod bn1d;
mod bn2d;
mod bn3d;

pub use bn1d::{batchnorm1d, BatchNorm1dArgs, BatchNorm1dNode};
pub use bn2d::{batchnorm2d, BatchNorm2dArgs, BatchNorm2dNode};
pub use bn3d::{batchnorm3d, BatchNorm3dArgs, BatchNorm3dNode};
