//! Matrix multiplication operation

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Compute matrix multiplication with another tensor.
pub fn matmul<B, T, S>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + 'static,
{
    let lhs_dense = lhs.to_dense_generic()?;
    let rhs_dense = rhs.to_dense_generic()?;
    
    let lhs_shape = lhs_dense.shape().dims();
    let rhs_shape = rhs_dense.shape().dims();

    // Validate 2D matrices
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(TensorError::ShapeError {
            expected: 2,
            actual: std::cmp::max(lhs_shape.len(), rhs_shape.len()),
            message: format!("Matrix multiplication requires 2D matrices, got lhs={:?}, rhs={:?}", lhs_shape, rhs_shape),
        });
    }

    if lhs_shape[1] != rhs_shape[0] {
        return Err(TensorError::ShapeError {
            expected: lhs_shape[1],
            actual: rhs_shape[0],
            message: format!("Matrix inner dimensions must match: {} != {}", lhs_shape[1], rhs_shape[0]),
        });
    }

    let m = lhs_shape[0];
    let n = lhs_shape[1];
    let p = rhs_shape[1];

    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();

    let mut result_data = vec![T::zero(); m * p];
    for i in 0..m {
        for j in 0..p {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + lhs_data[i * n + k] * rhs_data[k * p + j];
            }
            result_data[i * p + j] = sum;
        }
    }

    let mut result = Tensor::from_vec_with_backend(result_data, &[m, p], lhs.backend.clone())?;

    if crate::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        let grad_fn = crate::functions::linalg::MatMulFunction::new(Arc::new(lhs.clone()), Arc::new(rhs.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }


    Ok(result)
}
