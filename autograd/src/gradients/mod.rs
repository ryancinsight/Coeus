//! Gradient computation implementations for automatic differentiation
//!
//! This module contains modular implementations of backward pass functions
//! for different categories of operations.

pub mod arithmetic;
pub mod elementwise;
pub mod trigonometric;
pub mod rounding;
pub mod reduction;
pub mod shape_ops;
pub mod matrix_ops;
pub mod activation;
pub mod special;

// Re-export for internal use
pub use arithmetic::*;
pub use elementwise::*;
pub use trigonometric::*;
pub use rounding::*;
pub use reduction::*;
pub use shape_ops::*;
pub use matrix_ops::*;
pub use activation::*;
pub use special::*;
