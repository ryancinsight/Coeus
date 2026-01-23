//! Batch Normalization layers for neural networks.

pub mod core;

#[path = "1d.rs"]
pub mod batch1d;
#[path = "2d.rs"]
pub mod batch2d;
#[path = "3d.rs"]
pub mod batch3d;

pub use batch1d::BatchNorm1d;
pub use batch2d::BatchNorm2d;
pub use batch3d::BatchNorm3d;
pub use core::{BatchNormBase, BatchNormOps};
