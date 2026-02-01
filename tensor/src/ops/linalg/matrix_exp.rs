//! Matrix exponentiation operations
//!
//! Provides matrix exponential and power computation for tensors.

use crate::Backend;
use crate::{Result, Tensor, TensorError};
use storage::DenseStorage;

/// Computes the matrix exponential e^A.
///
/// Uses the scaling and squaring method with Padé approximation.
///
/// # Arguments
/// * `self` - Input tensor (must be a 2D square matrix)
///
/// # Returns
/// Tensor containing e^A (same shape as input)
///
/// # Errors
/// Returns error if input is not a 2D square matrix
pub fn matrix_exp<B: Backend>(
    tensor: &Tensor<B, DenseStorage<B::Data>, B::Data>,
) -> Result<Tensor<B, DenseStorage<B::Data>, B::Data>> {
    let shape = tensor.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TensorError::ShapeError {
            expected: 2,
            actual: dims.len(),
            message: "matrix_exp requires a 2D tensor".to_string(),
        });
    }

    let n = dims[0];
    if dims[1] != n {
        return Err(TensorError::ShapeError {
            expected: n,
            actual: dims[1],
            message: "matrix_exp requires a square matrix".to_string(),
        });
    }

    if n == 0 {
        return Tensor::from_vec_with_backend(vec![], &[0, 0], tensor.backend.clone());
    }

    #[cfg(feature = "cpu")]
    {
        let device = tensor.backend.clone();
        let data = tensor.as_slice();

        // Prepare output array
        let mut result = vec![B::Data::default(); n * n];

        use backend::cpu::linear_algebra::matrix_exp_primitive;
        matrix_exp_primitive(data, &mut result, n)
            .map_err(|e| TensorError::BackendError(e.to_string()))?;
        
        let result_storage = DenseStorage::from_vec(result, &[n, n])?;

        Ok(Tensor::from_storage(result_storage, device))
    }

    #[cfg(not(feature = "cpu"))]
    {
        Err(TensorError::BackendError(
            "matrix_exp not implemented for this backend".to_string(),
        ))
    }
}

/// Computes A^n for an integer n.
///
/// Uses exponentiation by squaring for efficient computation.
/// Supports negative powers (computes inverse first).
///
/// # Arguments
/// * `self` - Input tensor (must be a 2D square matrix)
/// * `n` - Integer exponent (can be negative)
///
/// # Returns
/// Tensor containing A^n (same shape as input)
///
/// # Errors
/// Returns error if input is not a 2D square matrix or if matrix is singular (for negative powers)
pub fn matrix_power<B: Backend>(
    tensor: &Tensor<B, DenseStorage<B::Data>, B::Data>,
    n: i64,
) -> Result<Tensor<B, DenseStorage<B::Data>, B::Data>> {
    let shape = tensor.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TensorError::ShapeError {
            expected: 2,
            actual: dims.len(),
            message: "matrix_power requires a 2D tensor".to_string(),
        });
    }

    let matrix_n = dims[0];
    if dims[1] != matrix_n {
        return Err(TensorError::ShapeError {
            expected: matrix_n,
            actual: dims[1],
            message: "matrix_power requires a square matrix".to_string(),
        });
    }

    if matrix_n == 0 {
        return Tensor::from_vec_with_backend(vec![], &[0, 0], tensor.backend.clone());
    }

    #[cfg(feature = "cpu")]
    {
        let device = tensor.backend.clone();
        let data = tensor.as_slice();

        // Prepare output array
        let mut result = vec![B::Data::default(); matrix_n * matrix_n];

        use backend::cpu::linear_algebra::matrix_power_primitive;
        matrix_power_primitive(data, &mut result, matrix_n, n)
            .map_err(|e| TensorError::BackendError(e.to_string()))?;

        let result_storage = DenseStorage::from_vec(result, &[matrix_n, matrix_n])?;

        Ok(Tensor::from_storage(result_storage, device))
    }

    #[cfg(not(feature = "cpu"))]
    {
        Err(TensorError::BackendError(
            "matrix_power not implemented for this backend".to_string(),
        ))
    }
}
