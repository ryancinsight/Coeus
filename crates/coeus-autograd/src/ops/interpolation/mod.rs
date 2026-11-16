//! Differentiable dimension-generic coordinate-grid interpolation.

mod grid_sample;
mod linear;

pub use grid_sample::grid_sample_3d;
pub use linear::{linear_interpolation, LinearInterpolationNode};
