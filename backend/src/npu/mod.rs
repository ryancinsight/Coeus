//! NPU (Neural Processing Unit) backend implementation
//!
//! This module provides NPU-specific implementations of backend operations.
//! NPU operations are optimized for neural network inference and training.

pub mod arithmetic;
pub mod linear_algebra;
pub mod activation;
pub mod reduction;

// Re-export commonly used items
pub use arithmetic::*;
pub use linear_algebra::*;
pub use activation::*;
pub use reduction::*;
