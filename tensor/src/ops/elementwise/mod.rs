//! Element-wise tensor operations
//!
//! This module provides element-wise arithmetic operations between tensors,
//! including broadcasting support for operations between tensors of different shapes.
//!
//! ## Supported Operations
//!
//! - **Addition**: `tensor1 + tensor2` or `tensor.add(&other)`
//! - **Subtraction**: `tensor1 - tensor2` or `tensor.sub(&other)`
//! - **Multiplication**: `tensor1 * tensor2` or `tensor.mul(&other)`
//! - **Division**: `tensor1 / tensor2` or `tensor.div(&other)`
//! - **Negation**: `-tensor` or `tensor.neg()`
//!
//! ## Broadcasting
//!
//! Broadcasting allows operations between tensors of different shapes by automatically
//! expanding the smaller tensor to match the larger one:
//!
//! ```rust
//! use coeus_tensor::{Tensor, Add};
//!
//! // Scalar + Vector broadcasting
//! let scalar = Tensor::scalar(1.0);
//! let vector = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
//!
//! let result = scalar.add(&vector).unwrap();
//! // Result: [2.0, 3.0, 4.0]
//! ```
//!
//! ## Broadcasting Rules
//!
//! 1. **Dimensions Match**: Shapes are compared element-wise from the right
//! 2. **Dimension Size 1**: A dimension of size 1 can be broadcast to any size
//! 3. **Missing Dimensions**: Missing dimensions are treated as size 1
//!
//! ## References
//!
//! - [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
//! - [PyTorch Broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html)

pub mod ops;

pub use ops::*;
