//! CUDA unfold/fold dispatch split by dimensionality and validation concern.

mod dispatch;
mod one_d;
mod source;
mod two_d;
mod validation;

pub use one_d::{dispatch_fold1d, dispatch_unfold1d};
pub use two_d::{dispatch_fold2d, dispatch_unfold2d};
