//! Outer product of two vectors.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, traits::FloatExt};
use storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
use crate::ops::linalg::matmul;

/// Outer product of vectors vec1 and vec2.
/// If vec1 is (M) and vec2 is (N), result is (M, N).
pub fn addr<B, T, S>(
    vec1: &Tensor<B, S, T>,
    vec2: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + 'static,
{
    // vec1: [M]
    // vec2: [N]
    // result: [M, N] = vec1.unsqueeze(1) @ vec2.unsqueeze(0)
    
    let shape1 = vec1.shape().dims();
    let shape2 = vec2.shape().dims();
    
    if shape1.len() != 1 || shape2.len() != 1 {
         return Err(crate::TensorError::ShapeError {
            expected: 1,
            actual: if shape1.len() != 1 { shape1.len() } else { shape2.len() },
            message: format!("addr: both inputs must be 1D, got {:?} and {:?}", shape1, shape2),
        });
    }
    
    // unsqueeze returns DenseStorage.
    let v1 = crate::ops::shape::unsqueeze::unsqueeze(vec1, 1)?; // [M, 1]
    let v2 = crate::ops::shape::unsqueeze::unsqueeze(vec2, 0)?; // [1, N]
    
    // matmul expects same storage type. Both are DenseStorage.
    // Explicitly call generic matmul.
    matmul(&v1, &v2)
}
