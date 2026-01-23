//! Tanh activation function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::sync::Arc;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use crate::functions::TanhFunction;

/// Applies the Hyperbolic Tangent (tanh) activation function.
///
/// Formula: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
pub fn tanh<B, T, S>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + FloatExt + Clone + 'static,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let result_data = data.iter().map(|&x| x.tanh()).collect();

    let mut result = Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?;

    if crate::tensor_core::grad_enabled() && input_dense.requires_grad() {
        let grad_fn = TanhFunction::new(Arc::new(input_dense.clone()), Arc::new(result.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
