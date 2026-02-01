//! Batch matrix multiplication.

use crate::functions::linalg::BMMFunction;
use crate::{Result, Tensor, TensorError};
use alloc::sync::Arc;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Batch matrix multiplication.
pub fn bmm<B, T, S>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType
        + Clone
        + Copy
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + 'static,
    S: Storage<T>
        + StorageToDense<T>
        + StorageFromVec<T>
        + Clone
        + Send
        + Sync
        + crate::ops::dispatch::TensorStorageOps<T>
        + 'static,
{
    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    if lhs_shape.len() != 3 || rhs_shape.len() != 3 {
        return Err(TensorError::ShapeError {
            expected: 3,
            actual: lhs_shape.len(),
            message: format!(
                "bmm: both tensors must be 3D, got lhs={:?}, rhs={:?}",
                lhs_shape, rhs_shape
            ),
        });
    }

    if lhs_shape[0] != rhs_shape[0] {
        return Err(TensorError::ShapeError {
            expected: lhs_shape[0],
            actual: rhs_shape[0],
            message: format!(
                "bmm: batch dimensions must match: {} != {}",
                lhs_shape[0], rhs_shape[0]
            ),
        });
    }

    let b = lhs_shape[0];
    let m = lhs_shape[1];
    let n = lhs_shape[2];
    let p = rhs_shape[2];

    if n != rhs_shape[1] {
        return Err(TensorError::ShapeError {
            expected: n,
            actual: rhs_shape[1],
            message: format!(
                "bmm: matrix inner dimensions must match: {} != {}",
                n, rhs_shape[1]
            ),
        });
    }

    let lhs_dense = lhs.to_dense_generic()?;
    let rhs_dense = rhs.to_dense_generic()?;
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    let mut result_data = Vec::with_capacity(b * m * p);

    // Naive implementation for now, should delegate to optimized backend in future
    for i in 0..b {
        let lhs_offset = i * m * n;
        let rhs_offset = i * n * p;

        for row in 0..m {
            for col in 0..p {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum
                        + lhs_data[lhs_offset + row * n + k] * rhs_data[rhs_offset + k * p + col];
                }
                result_data.push(sum);
            }
        }
    }

    let mut result = Tensor::from_vec_with_backend(result_data, &[b, m, p], lhs.backend.clone())?;

    if crate::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        let grad_fn = BMMFunction::new(Arc::new(lhs.clone()), Arc::new(rhs.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
