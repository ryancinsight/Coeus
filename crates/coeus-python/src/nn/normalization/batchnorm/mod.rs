//! Python BatchNorm bindings partitioned by input dimensionality.

mod batchnorm1d;
mod batchnorm2d;
mod batchnorm3d;

pub use batchnorm1d::PyBatchNorm1d;
pub use batchnorm2d::PyBatchNorm2d;
pub use batchnorm3d::PyBatchNorm3d;
