//! Eigendecomposition operations
//!
//! Provides eigenvalue and eigenvector computation for tensors.

use crate::Backend;
use crate::{Result, Tensor, TensorError};
use storage::DenseStorage;

/// Computes eigenvalues and eigenvectors of a general matrix.
///
/// # Arguments
/// * `self` - Input tensor (must be a 2D square matrix)
///
/// # Returns
/// Tuple of (eigenvalues, eigenvectors) where:
/// - eigenvalues: Tensor of shape [n] containing eigenvalues
/// - eigenvectors: Tensor of shape [n, n] containing eigenvectors as columns
///
/// # Errors
/// Returns error if input is not a 2D square matrix
pub fn eig<B: Backend>(
    tensor: &Tensor<B, DenseStorage<B::Data>, B::Data>,
) -> Result<(
    Tensor<B, DenseStorage<B::Data>, B::Data>,
    Tensor<B, DenseStorage<B::Data>, B::Data>,
)> {
    let shape = tensor.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TensorError::ShapeError {
            expected: 2,
            actual: dims.len(),
            message: "eig requires a 2D tensor".to_string(),
        });
    }

    let n = dims[0];
    if dims[1] != n {
        return Err(TensorError::ShapeError {
            expected: n,
            actual: dims[1],
            message: "eig requires a square matrix".to_string(),
        });
    }

    if n == 0 {
        return Ok((
            Tensor::from_vec(vec![], &[0])?,
            Tensor::from_vec(vec![], &[0, 0])?,
        ));
    }

    Err(TensorError::BackendError(
        "eig not yet implemented for this backend".to_string(),
    ))
}

/// Computes eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix.
///
/// This is more efficient and numerically stable than `eig` for symmetric matrices.
///
/// # Arguments
/// * `self` - Input tensor (must be a 2D square matrix, symmetric/Hermitian)
///
/// # Returns
/// Tuple of (eigenvalues, eigenvectors) where:
/// - eigenvalues: Tensor of shape [n] containing eigenvalues (ascending order)
/// - eigenvectors: Tensor of shape [n, n] containing orthonormal eigenvectors as columns
///
/// # Errors
/// Returns error if input is not a 2D square matrix
pub fn eigh<B: Backend>(
    tensor: &Tensor<B, DenseStorage<B::Data>, B::Data>,
) -> Result<(
    Tensor<B, DenseStorage<B::Data>, B::Data>,
    Tensor<B, DenseStorage<B::Data>, B::Data>,
)> {
    let shape = tensor.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TensorError::ShapeError {
            expected: 2,
            actual: dims.len(),
            message: "eigh requires a 2D tensor".to_string(),
        });
    }

    let n = dims[0];
    if dims[1] != n {
        return Err(TensorError::ShapeError {
            expected: n,
            actual: dims[1],
            message: "eigh requires a square matrix".to_string(),
        });
    }

    if n == 0 {
        return Ok((
            Tensor::from_vec(vec![], &[0])?,
            Tensor::from_vec(vec![], &[0, 0])?,
        ));
    }

    Err(TensorError::BackendError(
        "eigh not yet implemented for this backend".to_string(),
    ))
}
