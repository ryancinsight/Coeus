//! Element-wise absolute value

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Element-wise absolute value
///
/// Uses arithmetic operations to compute absolute value without requiring Signed trait.
/// Formula: abs(x) = x if x >= 0, -x otherwise (implemented as 0 - x)
pub fn abs<
    T: DataType + PartialOrd + std::ops::Sub<Output = T> + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    let zero = T::zero();
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|&x| if x < zero { zero - x } else { x })
        .collect();
    
    let mut result = Tensor::from_vec_with_backend(data, tensor.shape().dims(), tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        result = result.requires_grad_(true);
    }

    Ok(result)
}
