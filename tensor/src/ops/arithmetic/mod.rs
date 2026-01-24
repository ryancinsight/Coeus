//! Arithmetic operations module


mod add;
mod div;
mod maximum;
mod minimum;
mod mul;
mod neg;
mod sub;

pub use add::add;
pub use div::div;
pub use maximum::maximum;
pub use minimum::minimum;
pub use mul::mul;
pub use neg::neg;
pub use sub::sub;

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Broadcasting binary operation helper
pub fn broadcast_binary_op<
    T: DataType,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
    F: Fn(T, T) -> T,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
    op: F,
) -> Result<Tensor<B, S, T>> {
    let out_shape = crate::ops::shape::broadcast_shapes(a.shape().dims(), b.shape().dims())?;
    
    let a_data = crate::ops::shape::broadcast_tensor_data(a.as_slice(), a.shape().dims(), &out_shape)?;
    let b_data = crate::ops::shape::broadcast_tensor_data(b.as_slice(), b.shape().dims(), &out_shape)?;
    
    let data = a_data.into_iter().zip(b_data).map(|(x, y)| op(x, y)).collect();
    Tensor::from_vec_with_backend(data, &out_shape, a.backend.clone())
}

/// Helper to broadcast tensor data to a target shape
pub use crate::ops::shape::broadcast_tensor_data;

/// Helper to broadcast a tensor to a specific shape
pub fn broadcast_to<T, B, S>(tensor: &Tensor<B, S, T>, shape: &[usize]) -> Result<Tensor<B, S, T>>
where
    T: DataType + Copy,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
{
    let data = crate::ops::shape::broadcast_tensor_data(tensor.as_slice(), tensor.shape().dims(), shape)?;
    Tensor::from_vec_with_backend(data, shape, tensor.backend.clone())
}

