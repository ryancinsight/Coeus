//! Element-wise comparison operations
//! 
//! This module provides element-wise comparison operations for tensors:
//! - eq, ne: Equal, Not Equal
//! - gt, ge: Greater Than, Greater Expected
//! - lt, le: Less Than, Less Expected
//! 
//! These operations return a tensor of the same shape with binary 0/1 values
//! (represented as the input data type T, typically 0.0/1.0 for floats).

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};
use num_traits::{Num, FromPrimitive};

/// Element-wise equality comparison
pub fn eq<
    T: DataType + PartialEq + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x == y)
}

/// Element-wise inequality comparison
pub fn ne<
    T: DataType + PartialEq + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x != y)
}

/// Element-wise greater than comparison
pub fn gt<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x > y)
}

/// Element-wise greater than or equal comparison
pub fn ge<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x >= y)
}

/// Element-wise less than comparison
pub fn lt<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x < y)
}

/// Element-wise less than or equal comparison
pub fn le<
    T: DataType + PartialOrd + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>> {
    compare(a, b, |x, y| x <= y)
}

/// Helper function for comparison operations
/// Returns T (1 for true, 0 for false) to conform to Tensor<B, S, T>
fn compare<
    T: DataType + num_traits::One + num_traits::Zero,
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    F: Fn(&T, &T) -> bool,
>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>,
    op: F,
) -> Result<Tensor<B, S, T>> {
    // For now, only support same-shape tensors for simplicity
    // Broadcasting can be added later following arithmetic.rs pattern
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
             expected: a.shape().dims().to_vec(),
             actual: b.shape().dims().to_vec(),
             operation: "comparison",
        });
    }

    let data = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| if op(x, y) { T::one() } else { T::zero() })
        .collect();

    let result = Tensor::from_vec(data, a.shape().dims())?;
    
    // Comparison results don't require gradients
    Ok(result)
}
