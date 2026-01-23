//! GELU activation function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::sync::Arc;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use crate::functions::GeluFunction;

/// Applies the Gaussian Error Linear Unit (GELU) activation function.
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
pub fn gelu<B, T, S>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone + 'static + num_traits::FromPrimitive,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
{
    let input_dense = input.to_dense_generic()?;
    let data = input_dense.as_slice();
    
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let mut result_data = Vec::with_capacity(data.len());
    let sqrt_2_pi = T::from_f64((2.0 / std::f64::consts::PI).sqrt()).unwrap();
    let point_zero_four = T::from_f64(0.044715).unwrap();
    let zero_five = T::from_f64(0.5).unwrap();
    let one = T::from_f64(1.0).unwrap();

    for &x in data {
        let x_3 = x * x * x;
        let inner = sqrt_2_pi * (x + point_zero_four * x_3);
        let val = zero_five * x * (one + inner.tanh());
        result_data.push(val);
    }

    let mut result = Tensor::from_vec_with_backend(
        result_data,
        input.shape().dims(),
        input.backend().clone(),
    )?;

    if crate::tensor_core::grad_enabled() && input_dense.requires_grad() {
        let grad_fn = GeluFunction::new(Arc::new(input_dense.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}
