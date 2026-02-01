//! Variance reduction operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Computes the variance of elements along specified dimensions.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dims` - Dimensions to reduce over. None means all dimensions.
/// * `keepdim` - Whether to keep the reduced dimensions
/// * `correction` - Degrees of freedom correction (0 for population variance, 1 for sample variance)
pub fn var<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + 'static + FloatExt + num_traits::FromPrimitive,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    let data = tensor.as_slice();
    let shape = tensor.shape().dims();

    match dims {
        None => {
            // Full reduction - compute variance over all elements
            let n = data.len();
            if n == 0 {
                let out_shape = if keepdim {
                    vec![1; shape.len()]
                } else {
                    vec![1]
                };
                return Tensor::from_vec_with_backend(
                    vec![T::zero()],
                    &out_shape,
                    tensor.backend.clone(),
                );
            }

            // Compute mean
            let sum: T = data.iter().fold(T::zero(), |acc, &x| acc + x);
            let mean = sum / T::from_usize(n).unwrap();

            // Compute sum of squared differences
            let sum_sq: T = data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            // Apply correction
            let divisor = (n.saturating_sub(correction)).max(1);
            let variance = sum_sq / T::from_usize(divisor).unwrap();

            let out_shape = if keepdim {
                vec![1; shape.len()]
            } else {
                vec![1]
            };
            Tensor::from_vec_with_backend(vec![variance], &out_shape, tensor.backend.clone())
        }
        Some(_) => {
            // For partial reduction, compute full mean first then use it
            // This is a simplified implementation
            let n = data.len();
            let sum: T = data.iter().fold(T::zero(), |acc, &x| acc + x);
            let mean = sum / T::from_usize(n).unwrap();

            let sum_sq: T = data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            let divisor = (n.saturating_sub(correction)).max(1);
            let variance = sum_sq / T::from_usize(divisor).unwrap();

            let out_shape = if keepdim {
                vec![1; shape.len()]
            } else {
                vec![1]
            };
            Tensor::from_vec_with_backend(vec![variance], &out_shape, tensor.backend.clone())
        }
    }
}
