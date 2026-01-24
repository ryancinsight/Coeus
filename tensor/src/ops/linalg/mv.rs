//! Matrix-vector multiplication.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, traits::FloatExt};
use storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
use crate::ops::linalg::matmul;

/// Matrix-vector multiplication.
/// Performs matrix multiplication of a matrix and a vector.
/// If matrix is (M, N) and vector is (N), result is (M).
pub fn mv<B, T, S>(
    mat: &Tensor<B, S, T>,
    vec: &Tensor<B, S, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + 'static,
{
    // mat: [M, N]
    // vec: [N]
    // We treat vec as [N, 1], perform matmul -> [M, 1], then squeeze -> [M]
    
    let vec_shape = vec.shape().dims();
    if vec_shape.len() != 1 {
         return Err(crate::TensorError::ShapeError {
            expected: 1,
            actual: vec_shape.len(),
            message: format!("mv: vector must be 1D, got {:?}", vec_shape),
        });
    }
    
    // unsqueeze(1) -> [N, 1] (Dense)
    let vec_col = crate::ops::shape::unsqueeze::unsqueeze(vec, 1)?;
    
    // mat needs to be converted to Dense to match vec_col for matmul
    let mat_dense = mat.to_dense_generic()?;
    
    // matmul(mat_dense, vec_col) -> [M, 1] (Dense)
    // Both inputs are DenseStorage, so matmul works.
    let prod = matmul(&mat_dense, &vec_col)?;
    
    // squeeze(1) -> [M] (Dense)
    crate::ops::shape::squeeze::squeeze(&prod, 1)
}
