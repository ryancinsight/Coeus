//! Automatic differentiation tensor operations
//!
//! This module provides tensor operations that automatically construct the computation graph
//! for gradient computation. These functions mirror the operations in `tensor::arithmetic`
//! but attach `Function` objects to enable automatic differentiation.

extern crate alloc;

use crate::{functions::*, Result};
use dtype::DataType;
use tensor::{CpuBackend, DenseStorage, Tensor};
use alloc::{sync::Arc, vec::Vec};

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
/// use tensor::{Tensor, CpuBackend, DenseStorage};
/// use dtype::float::Float32;
/// use autograd::tensor_ops::add;
///
/// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(3.0), Float32::new(4.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let z = add(&x, &y).unwrap();
/// assert!(z.grad_fn().is_some()); // Has AddBackward function attached
/// ```
#[allow(clippy::missing_errors_doc)]
pub fn add(
    lhs: &Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>,
    rhs: &Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>,
) -> Result<Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>> {
    use dtype::float::Float32;

    // Perform the addition operation
    let result = lhs + rhs;

    // Create computation graph if gradients are required
    if lhs.requires_grad() || rhs.requires_grad() {
        let add_fn = Arc::new(AddFunction::new(
            Arc::new(lhs.clone()),
            Arc::new(rhs.clone()),
        ));
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some("add".to_string()));
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Matrix multiplication with automatic differentiation
///
/// This function performs matrix multiplication and automatically attaches
/// a `MatMulFunction` to the result tensor if either input requires gradients.
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with automatic differentiation support
#[allow(clippy::missing_errors_doc)]
pub fn matmul(
    lhs: &Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>,
    rhs: &Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>,
) -> Result<Tensor<CpuBackend<dtype::float::Float32>, DenseStorage<dtype::float::Float32>, dtype::float::Float32>> {
    // Perform the matrix multiplication
    let result = lhs.matmul(rhs).map_err(|e| crate::AutogradError::TensorError(e))?;

    // Create computation graph if gradients are required
    if lhs.requires_grad() || rhs.requires_grad() {
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some("matmul".to_string()));
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}
