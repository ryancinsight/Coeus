//! Replace NaN, infinity values with numbers

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec};

/// Replace NaN, infinity values with numbers.
pub fn nan_to_num<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    nan: Option<T>,
    posinf: Option<T>,
    neginf: Option<T>,
) -> Result<Tensor<B, S, T>> {
    let nan_val = nan.unwrap_or_else(T::zero);
    let posinf_val = posinf.unwrap_or_else(T::max_value);
    let neginf_val = neginf.unwrap_or_else(T::min_value);

    let data = tensor
        .as_slice()
        .iter()
        .map(|&x| {
            if x.is_nan() {
                nan_val
            } else if x.is_infinite() && x > T::zero() {
                posinf_val
            } else if x.is_infinite() && x < T::zero() {
                neginf_val
            } else {
                x
            }
        })
        .collect();

    Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())
}
