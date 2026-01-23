//! Sigmoid activation function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::sync::Arc;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use crate::functions::SigmoidFunction;

/// Applies the Sigmoid activation function.
///
/// Formula: `1 / (1 + exp(-x))`
pub fn sigmoid<B, T, S>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone + 'static,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    let mut result_data = Vec::with_capacity(data.len());

    let one = T::from(1.0).unwrap();
    for &val in data {
        let neg_val = -val;
        let exp_neg = neg_val.exp();
        let denom = one + exp_neg;
        result_data.push(one / denom);
    }

    let mut result = Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?;

    if crate::tensor_core::grad_enabled() && input_dense.requires_grad() {
        let grad_fn = SigmoidFunction::new(Arc::new(input_dense.clone()), Arc::new(result.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }


    Ok(result)
}
