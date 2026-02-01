//! LeakyReLU activation function

use crate::functions::LeakyReluFunction;
use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::FromPrimitive;
use std::sync::Arc;
use storage::{DenseStorage, StorageFromVec, StorageToDense};

/// Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
///
/// Formula: `max(α * x, x)` where α is the negative slope
pub fn leaky_relu<B, T, S>(
    input: &Tensor<B, S, T>,
    negative_slope: f64,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + PartialOrd + Clone + FromPrimitive + 'static + dtype::traits::FloatExt,
    S: StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();

    let slope_t = T::from_f64(negative_slope).ok_or_else(|| {
        crate::TensorError::BackendError(format!(
            "{}",
            backend::BackendError::InvalidInput(
                "Failed to convert negative_slope to dtype".to_string()
            )
        ))
    })?;

    let result_data = data
        .iter()
        .map(|&x| if x > T::zero() { x } else { x * slope_t })
        .collect();

    let mut result =
        Tensor::from_vec_with_backend(result_data, input.shape().dims(), input.backend().clone())?;

    if crate::tensor_core::grad_enabled() && input_dense.requires_grad() {
        let grad_fn = LeakyReluFunction::new(Arc::new(input_dense.clone()), slope_t);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
