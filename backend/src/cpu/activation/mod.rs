//! CPU activation function primitives
//!
//! Activation function primitives optimized for CPU execution.
//! These operations provide the foundation for neural network activations.

pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod math_ops;

// Re-export operations for convenience
pub use relu::{relu_primitive, relu_strided_primitive, relu_csr_primitive};
pub use sigmoid::{sigmoid_primitive, sigmoid_strided_primitive};
pub use tanh::{tanh_primitive, tanh_strided_primitive, tanh_csr_primitive};
pub use math_ops::*;