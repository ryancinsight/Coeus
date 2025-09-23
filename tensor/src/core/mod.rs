//! Core tensor structure and basic data management
//!
//! This module contains the fundamental `Tensor` struct and its basic operations
//! for data storage, shape management, and memory layout.
//!
//! ## Core Concepts
//!
//! ### Tensor Structure
//! A tensor is a multi-dimensional array with the following components:
//! - **Data**: Contiguous memory buffer storing tensor elements
//! - **Shape**: Dimensions of the tensor (e.g., `[3, 4]` for a 3×4 matrix)
//! - **Device**: Memory location (CPU/GPU)
//! - **Layout**: Memory layout (contiguous, transposed, etc.)
//! - **Dtype**: Data type of tensor elements (imported from `coeus_core`)
//!
//! ### Memory Layout
//! Tensors use row-major (C-style) layout by default:
//!
//! ```text
//! Shape: [2, 3]
//! Data:  [1, 2, 3, 4, 5, 6]
//!
//! Layout in memory:
//! [[1, 2, 3],
//!  [4, 5, 6]]
//! ```
//!
//! ## Basic Operations
//!
//! - **Creation**: `zeros`, `ones`, `eye`, `from_vec`
//! - **Properties**: `shape`, `ndim`, `numel`
//! - **Data Access**: `data`, `data_mut`
//!
//! ## References
//!
//! - [NumPy Array Basics](https://numpy.org/doc/stable/user/basics.html)
//! - [PyTorch Tensor Basics](https://pytorch.org/docs/stable/tensors.html)
//! - [Tensor Memory Layout](https://pytorch.org/blog/tensor-memory-layout/)

pub mod activations;
pub mod arithmetic;
pub mod arithmetic_ops;
pub mod creation;
pub mod indexing_ops;
pub mod matrix_ops;
pub mod reduction_ops;
pub mod shape_ops;
pub mod tensor;
pub mod tensor_ops;


pub use tensor::{
    apply_pending_gradients, store_pending_gradient, with_autograd_context, HessianTensorIter,
    Tensor,
};
// pub use tensor_ops::*; // Temporarily disabled to avoid unused import warning
