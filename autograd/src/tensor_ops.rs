//! Automatic differentiation tensor operations
//!
//! This module provides tensor operations that automatically construct the computation graph
//! for gradient computation. These functions mirror the operations in `coeus_tensor::arithmetic`
//! but attach `Function` objects to enable automatic differentiation.

extern crate alloc;

use crate::Result;
use coeus_dtype::float::Float32;
use coeus_tensor::{CpuBackend, DenseStorage, Tensor};

/// Element-wise addition with automatic differentiation
///
/// This function performs element-wise addition and automatically attaches
/// an `AddFunction` to the result tensor if either input requires gradients.
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with automatic differentiation support
///
/// # Examples
///
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
/// use coeus_dtype::float::Float32;
/// use coeus_autograd::tensor_ops::add;
///
/// let x = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let y = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(3.0), Float32::new(4.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let z = add(&x, &y).unwrap();
/// assert!(z.grad_fn().is_some()); // Has AddBackward function attached
/// ```
pub fn add(
    lhs: &Tensor<CpuBackend, DenseStorage<Float32>, Float32>,
    rhs: &Tensor<CpuBackend, DenseStorage<Float32>, Float32>,
) -> Result<Tensor<CpuBackend, DenseStorage<Float32>, Float32>> {
    // Use the ops module to get proper grad_fn setting
    Ok(crate::ops::add(lhs, rhs))
}
