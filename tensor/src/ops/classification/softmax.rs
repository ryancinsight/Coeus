//! Softmax activation function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::sync::Arc;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use crate::functions::SoftmaxFunction;

/// Applies the Softmax function to an n-dimensional input Tensor.
///
/// Softmax is defined as:
/// `f_i(x) = exp(x_i) / Σ_j exp(x_j)`
pub fn softmax<B, T, S>(
    input: &Tensor<B, S, T>,
    dim: i64,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + FloatExt + PartialOrd + Clone + 'static,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
{
    let input_dense = input.to_dense_generic()?;
    let dims = input_dense.shape().dims();
    let ndim = dims.len();

    let actual_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };

    if actual_dim >= ndim {
        return Err(crate::TensorError::ShapeError {
            expected: ndim,
            actual: actual_dim,
            message: format!("Softmax dim {actual_dim} out of range for {ndim}-D tensor"),
        });
    }

    let input_data = input_dense.as_slice();
    let mut result_data = vec![T::zero(); input_data.len()];

    let dim_size = dims[actual_dim];
    let outer_size: usize = dims[..actual_dim].iter().product();
    let inner_size: usize = dims[actual_dim + 1..].iter().product();

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let offset = outer * dim_size * inner_size + inner;
            
            // Find max for numerical stability
            let mut max_val = T::neg_infinity();
            for d in 0..dim_size {
                let idx = offset + d * inner_size;
                if input_data[idx] > max_val {
                    max_val = input_data[idx];
                }
            }

            // Sum exp(x - max)
            let mut sum_exp = T::zero();
            for d in 0..dim_size {
                let idx = offset + d * inner_size;
                let val = (input_data[idx] - max_val).exp();
                result_data[idx] = val;
                sum_exp = sum_exp + val;
            }

            // Normalize
            for d in 0..dim_size {
                let idx = offset + d * inner_size;
                result_data[idx] = result_data[idx] / sum_exp;
            }
        }
    }

    let mut result = Tensor::from_vec_with_backend(
        result_data,
        dims,
        input_dense.backend().clone(),
    )?;

    if crate::tensor_core::grad_enabled() && input_dense.requires_grad() {
        let grad_fn = SoftmaxFunction::new(Arc::new(input_dense.clone()), Arc::new(result.clone()), actual_dim);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

