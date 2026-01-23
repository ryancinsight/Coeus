//! Element-wise inverse error function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::{FromPrimitive, ToPrimitive};
use storage::{Storage, StorageFromVec};

/// Element-wise inverse error function
pub fn erfinv<
    T: DataType + FromPrimitive + ToPrimitive,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let data = tensor
        .as_slice()
        .iter()
        .map(|&x| {
            let x_f64 = x.to_f64().unwrap_or(0.0);
            let erfinv_f64 = statrs::function::erf::erf_inv(x_f64);
            T::from_f64(erfinv_f64).unwrap_or(T::zero())
        })
        .collect();
    let mut result =
        Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if tensor.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}
